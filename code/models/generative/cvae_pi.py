"""
cvae_pi.py -- Physics-Informed cVAE (M2/M3/FULL model).

Extends CVAEBase with:
  - Soft physics loss (ordering + bounds penalties in ELBO) -> M2
  - Hard isotonic projection on decoder output                -> M3
  - Both -> FULL (used with MF data and AL)

Loss:
  L_total = L_recon + β·L_KL + λ_order·L_order + λ_bounds·L_bounds

The hard projection (IsotonicProjectionLayer) is applied AFTER the soft
penalty training: the projection ensures r1>=r2>=r3>=r4 at inference regardless
of the penalty weight, while the soft penalty guides gradient descent toward
the feasible region during training.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import (
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    CVAE_LATENT_DIM, CVAE_ENCODER_HIDDEN, CVAE_DECODER_HIDDEN,
    CVAE_BATCH_SIZE, CVAE_MAX_EPOCHS, CVAE_LR, CVAE_BETA_KL,
    CVAE_LAMBDA_ORDER, CVAE_LAMBDA_BOUNDS, CVAE_PATIENCE,
    CVAE_VAL_FRAC, RESULTS_DIR, GLOBAL_SEED
)
from utils.data_utils import (CeSRRScaler, build_condition,
                              train_test_split_stratified, load_processed)
from utils.train_utils import EarlyStopping, DEVICE, set_seed, save_checkpoint
from models.generative.cvae_base import CVAEBase, elbo_loss
from models.generative.projection_layer import IsotonicProjectionLayer
from models.surrogate.loss_physics import PhysicsLoss
from evaluation.metrics import geometry_feasibility_rate

log = logging.getLogger(__name__)


class PICVAENet(nn.Module):
    """
    Physics-Informed cVAE network.

    Identical to CVAEBase except the decoder is followed by an optional
    IsotonicProjectionLayer (hard constraint).

    The soft physics penalty is applied externally in the training loop
    (not inside this module) so that it can be toggled in ablations.
    """

    def __init__(
        self,
        x_dim: int = 9,
        c_dim: int = 2,
        latent_dim: int = CVAE_LATENT_DIM,
        encoder_hidden: list = None,
        decoder_hidden: list = None,
        use_hard_projection: bool = True,
        min_gap: float = 0.05,
    ):
        super().__init__()
        self.base = CVAEBase(x_dim, c_dim, latent_dim, encoder_hidden, decoder_hidden)
        self.use_projection = use_hard_projection
        self.projector = IsotonicProjectionLayer(min_gap=min_gap) if use_hard_projection else None
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        x_recon_raw, mu_z, logvar_z = self.base(x, c)
        if self.use_projection and self.projector is not None:
            x_recon = self.projector(x_recon_raw)
        else:
            x_recon = x_recon_raw
        return x_recon, x_recon_raw, mu_z, logvar_z

    def sample(self, c: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Sample n geometry proposals for each condition vector in c."""
        B = c.size(0)
        c_rep = c.repeat_interleave(n, dim=0)
        z = torch.randn(B * n, self.latent_dim, device=c.device)
        x_raw = self.base.decoder(z, c_rep)
        if self.use_projection and self.projector is not None:
            x = self.projector(x_raw)
        else:
            x = x_raw
        return x.view(B, n, -1)


class PICVAEModel:
    """
    Wrapper for Physics-Informed cVAE with full ablation control.

    Parameters
    ----------
    use_soft_physics     : add L_order + L_bounds to ELBO (M2/M3)
    use_hard_projection  : apply IsotonicProjectionLayer (M3/FULL)
    lambda_order         : weight for ordering penalty
    lambda_bounds        : weight for bounds penalty
    """

    def __init__(
        self,
        use_soft_physics: bool = True,
        use_hard_projection: bool = True,
        latent_dim: int = CVAE_LATENT_DIM,
        encoder_hidden: list = None,
        decoder_hidden: list = None,
        beta_kl: float = CVAE_BETA_KL,
        lambda_order: float = CVAE_LAMBDA_ORDER,
        lambda_bounds: float = CVAE_LAMBDA_BOUNDS,
    ):
        self.use_soft_physics    = use_soft_physics
        self.use_hard_projection = use_hard_projection
        self.latent_dim          = latent_dim
        self.encoder_hidden      = encoder_hidden or CVAE_ENCODER_HIDDEN
        self.decoder_hidden      = decoder_hidden or CVAE_DECODER_HIDDEN
        self.beta_kl             = beta_kl
        self.lambda_order        = lambda_order
        self.lambda_bounds       = lambda_bounds

        self.scaler: CeSRRScaler | None = None
        self._model: PICVAENet | None = None
        self._phys_loss: PhysicsLoss | None = None

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> "PICVAEModel":
        set_seed(GLOBAL_SEED)
        self.scaler = CeSRRScaler().fit(df_train)

        X_tr = torch.tensor(self.scaler.transform_features(df_train), dtype=torch.float32)
        C_tr = torch.tensor(build_condition(df_train, self.scaler), dtype=torch.float32)

        if df_val is None:
            n_val = max(1, int(CVAE_VAL_FRAC * len(df_train)))
            X_val = X_tr[-n_val:].to(DEVICE)
            C_val = C_tr[-n_val:].to(DEVICE)
            X_tr  = X_tr[:-n_val]
            C_tr  = C_tr[:-n_val]
        else:
            X_val = torch.tensor(
                self.scaler.transform_features(df_val), dtype=torch.float32, device=DEVICE)
            C_val = torch.tensor(
                build_condition(df_val, self.scaler), dtype=torch.float32, device=DEVICE)

        loader = DataLoader(
            TensorDataset(X_tr, C_tr),
            batch_size=CVAE_BATCH_SIZE, shuffle=True,
        )

        self._model = PICVAENet(
            latent_dim=self.latent_dim,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
            use_hard_projection=self.use_hard_projection,
        ).to(DEVICE)

        if self.use_soft_physics:
            self._phys_loss = PhysicsLoss(
                lambda_order=self.lambda_order,
                lambda_bounds=self.lambda_bounds,
            )

        opt   = torch.optim.Adam(self._model.parameters(), lr=CVAE_LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15)
        es    = EarlyStopping(patience=CVAE_PATIENCE)

        model_name = (
            "PI-cVAE (soft+hard)" if (self.use_soft_physics and self.use_hard_projection)
            else "PI-cVAE (soft only)" if self.use_soft_physics
            else "cVAE-base"
        )
        log.info("Training %s (%d samples)...", model_name, len(X_tr))

        for epoch in range(1, CVAE_MAX_EPOCHS + 1):
            self._model.train()
            for xb, cb in loader:
                xb, cb = xb.to(DEVICE), cb.to(DEVICE)
                opt.zero_grad()

                x_recon, x_recon_raw, mu_z, logvar_z = self._model(xb, cb)

                # ELBO loss (on projected output for reconstruction)
                elbo, _, _ = elbo_loss(x_recon, xb, mu_z, logvar_z, self.beta_kl)

                # Physics penalty on raw pre-projection output.
                # x_recon_raw is in [0,1] MinMax-scaled space -- correct input
                # for loss_physics functions (see loss_physics.py docstring).
                phys_total = torch.tensor(0.0, device=DEVICE)
                if self.use_soft_physics and self._phys_loss is not None:
                    phys_total, _ = self._phys_loss(x_recon_raw)

                total_loss = elbo + phys_total
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                opt.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                x_recon_v, x_raw_v, mu_v, logvar_v = self._model(X_val, C_val)
                val_elbo, _, _ = elbo_loss(x_recon_v, X_val, mu_v, logvar_v, self.beta_kl)
                val_loss = val_elbo.item()

            sched.step(val_loss)
            if epoch % 50 == 0:
                log.info("  Epoch %d | val_ELBO=%.5f", epoch, val_loss)
            if es(val_loss, self._model):
                break
        es.restore_best(self._model)
        return self

    def sample_geometry(self, df: pd.DataFrame, n_samples: int = 10) -> np.ndarray:
        C = torch.tensor(
            build_condition(df, self.scaler), dtype=torch.float32, device=DEVICE)
        self._model.eval()
        with torch.no_grad():
            x_s = self._model.sample(C, n=n_samples)   # (N, n_samples, 9) scaled
        x_flat = x_s.view(-1, 9).cpu().numpy()
        return self.scaler.inverse_transform_features(x_flat)

    def evaluate(self, df_test: pd.DataFrame, n_samples: int = 10) -> dict:
        X_samples = self.sample_geometry(df_test, n_samples=n_samples)
        feas = geometry_feasibility_rate(X_samples)
        diversity = float(np.std(X_samples))
        return {
            "model": "PI-cVAE",
            "use_soft_physics": self.use_soft_physics,
            "use_hard_projection": self.use_hard_projection,
            "feasibility_rate": feas,
            "geometry_diversity_std": diversity,
        }

    def get_uncertainty(self, df: pd.DataFrame, n_mc: int = 50) -> np.ndarray:
        """
        MC-Dropout uncertainty: run encoder n_mc times with dropout active,
        compute variance across z samples.

        Handles unlabelled pool candidates (NaN freq_ghz) by filling with
        the training data median frequency before building the condition.

        Returns
        -------
        (N,) array of uncertainty scores (sum of variance across z dims)
        """
        df_c = df.copy()
        if df_c["freq_ghz"].isna().any():
            # Unlabelled pool: fill NaN freq with midpoint of training freq range
            median_freq = float(self.scaler.tgt_scaler.inverse_transform([[0.5]])[0, 0])
            df_c["freq_ghz"] = df_c["freq_ghz"].fillna(median_freq)

        C = torch.tensor(
            build_condition(df_c, self.scaler), dtype=torch.float32, device=DEVICE)
        X = torch.tensor(
            self.scaler.transform_features(df_c), dtype=torch.float32, device=DEVICE)

        # Activate dropout for MC sampling
        self._model.train()
        mu_samples = []
        with torch.no_grad():
            for _ in range(n_mc):
                mu_z, _ = self._model.base.encoder(X, C)
                mu_samples.append(mu_z.cpu().numpy())
        self._model.eval()

        mu_stack = np.stack(mu_samples, axis=0)   # (n_mc, N, latent_dim)
        return mu_stack.var(axis=0).sum(axis=-1)  # (N,)

    def save(self, path: Path = None):
        path = path or (RESULTS_DIR / "cvae_pi.pt")
        save_checkpoint(self._model, path, extra={
            "use_soft": self.use_soft_physics,
            "use_hard": self.use_hard_projection,
        })


if __name__ == "__main__":
    from utils.fast_mode import apply_fast_mode
    apply_fast_mode()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_processed("all")
    df_tr, df_te = train_test_split_stratified(df, test_frac=0.2)

    # M3: soft + hard
    m = PICVAEModel(use_soft_physics=True, use_hard_projection=True)
    m.fit(df_tr)
    print(json.dumps(m.evaluate(df_te), indent=2))
    m.save()
