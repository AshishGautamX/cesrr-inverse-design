"""
cvae_base.py -- Standard cVAE (ablation B4): no physics constraints.

Conditional Variational Autoencoder mapping:
  Encoder : (geometry, freq, rotation) -> (μ_z, log σ2_z)
  Decoder : (z, freq, rotation) -> geometry

Loss = MSE reconstruction + β · KL divergence
     = ELBO (evidence lower bound)

This is the "ablation control" -- it shows what physics constraints add.
Physics constraints are added in cvae_pi.py (M2/M3).
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import (
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    CVAE_LATENT_DIM, CVAE_ENCODER_HIDDEN, CVAE_DECODER_HIDDEN,
    CVAE_BATCH_SIZE, CVAE_MAX_EPOCHS, CVAE_LR, CVAE_BETA_KL,
    CVAE_PATIENCE, CVAE_VAL_FRAC, RESULTS_DIR, GLOBAL_SEED
)
from utils.data_utils import (CeSRRScaler, build_condition,
                              train_test_split_stratified, load_processed)
from utils.train_utils import EarlyStopping, DEVICE, set_seed, save_checkpoint, build_mlp
from evaluation.metrics import geometry_feasibility_rate

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Network modules
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    q(z | x, c) -- encodes geometry x conditioned on c = [freq_scaled, rot].

    Input  : (B, 9 + 2) = geometry + condition
    Output : (μ_z, log_var_z), each shape (B, latent_dim)
    """

    def __init__(
        self,
        x_dim: int = 9,
        c_dim: int = 2,
        latent_dim: int = CVAE_LATENT_DIM,
        hidden: list = None,
    ):
        super().__init__()
        hidden = hidden or CVAE_ENCODER_HIDDEN
        in_dim = x_dim + c_dim
        self.shared = build_mlp(in_dim, hidden[-1], hidden[:-1], dropout_p=0.1)
        self.mu_head     = nn.Linear(hidden[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden[-1], latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        h = self.shared(torch.cat([x, c], dim=-1))
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    """
    p(x | z, c) -- decodes geometry from latent z conditioned on c.

    Input  : (B, latent_dim + 2)
    Output : (B, 9) scaled geometry
    """

    def __init__(
        self,
        latent_dim: int = CVAE_LATENT_DIM,
        c_dim: int = 2,
        x_dim: int = 9,
        hidden: list = None,
    ):
        super().__init__()
        hidden = hidden or CVAE_DECODER_HIDDEN
        in_dim = latent_dim + c_dim
        self.net = build_mlp(in_dim, x_dim, hidden, dropout_p=0.1)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, c], dim=-1))


# -----------------------------------------------------------------------------
# Base cVAE model
# -----------------------------------------------------------------------------

class CVAEBase(nn.Module):
    """Standard cVAE (no physics penalties)."""

    def __init__(
        self,
        x_dim: int = 9,
        c_dim: int = 2,
        latent_dim: int = CVAE_LATENT_DIM,
        encoder_hidden: list = None,
        decoder_hidden: list = None,
    ):
        super().__init__()
        self.encoder = Encoder(x_dim, c_dim, latent_dim, encoder_hidden)
        self.decoder = Decoder(latent_dim, c_dim, x_dim, decoder_hidden)
        self.latent_dim = latent_dim

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = μ + ε·σ where ε ~ N(0,I)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu_z, logvar_z = self.encoder(x, c)
        z = self.reparameterise(mu_z, logvar_z)
        x_recon = self.decoder(z, c)
        return x_recon, mu_z, logvar_z

    def sample(self, c: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Sample n geometry proposals for each condition vector in c."""
        B = c.size(0)
        c_rep = c.repeat_interleave(n, dim=0)   # (B*n, c_dim)
        z = torch.randn(B * n, self.latent_dim, device=c.device)
        x = self.decoder(z, c_rep)
        return x.view(B, n, -1)                 # (B, n, 9)


def elbo_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    mu_z: torch.Tensor,
    log_var_z: torch.Tensor,
    beta: float = CVAE_BETA_KL,
) -> tuple:
    """
    ELBO = -E[log p(x|z,c)] + β·KL(q(z|x,c) || p(z))

    Reconstruction term: MSE (equivalent to -log N(x|x_recon, I))
    KL term: closed form for diagonal Gaussians.
    """
    recon_loss = F.mse_loss(x_recon, x_true, reduction="mean")
    kl_loss    = -0.5 * torch.mean(
        1 + log_var_z - mu_z.pow(2) - log_var_z.exp()
    )
    total = recon_loss + beta * kl_loss
    return total, recon_loss.item(), kl_loss.item()


# -----------------------------------------------------------------------------
# Wrapper with sklearn-style interface
# -----------------------------------------------------------------------------

class CVAEBaseModel:
    """Wrapper providing fit / sample / evaluate interface."""

    def __init__(
        self,
        latent_dim: int = CVAE_LATENT_DIM,
        encoder_hidden: list = None,
        decoder_hidden: list = None,
        beta_kl: float = CVAE_BETA_KL,
    ):
        self.latent_dim      = latent_dim
        self.encoder_hidden  = encoder_hidden or CVAE_ENCODER_HIDDEN
        self.decoder_hidden  = decoder_hidden or CVAE_DECODER_HIDDEN
        self.beta_kl         = beta_kl
        self.scaler: CeSRRScaler | None = None
        self._model: CVAEBase | None = None

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> "CVAEBaseModel":
        set_seed(GLOBAL_SEED)
        self.scaler = CeSRRScaler().fit(df_train)

        X_train = torch.tensor(
            self.scaler.transform_features(df_train), dtype=torch.float32)
        C_train = torch.tensor(
            build_condition(df_train, self.scaler), dtype=torch.float32)

        if df_val is None:
            n_val   = max(1, int(CVAE_VAL_FRAC * len(df_train)))
            X_val   = X_train[-n_val:].to(DEVICE)
            C_val   = C_train[-n_val:].to(DEVICE)
            X_train = X_train[:-n_val]
            C_train = C_train[:-n_val]
        else:
            X_val = torch.tensor(
                self.scaler.transform_features(df_val), dtype=torch.float32, device=DEVICE)
            C_val = torch.tensor(
                build_condition(df_val, self.scaler), dtype=torch.float32, device=DEVICE)

        loader = DataLoader(
            TensorDataset(X_train, C_train),
            batch_size=CVAE_BATCH_SIZE, shuffle=True,
        )

        self._model = CVAEBase(
            latent_dim=self.latent_dim,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
        ).to(DEVICE)

        opt   = torch.optim.Adam(self._model.parameters(), lr=CVAE_LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15)
        es    = EarlyStopping(patience=CVAE_PATIENCE)

        log.info("Training cVAE-Base (%d samples)...", len(X_train))
        for epoch in range(1, CVAE_MAX_EPOCHS + 1):
            self._model.train()
            for xb, cb in loader:
                xb, cb = xb.to(DEVICE), cb.to(DEVICE)
                opt.zero_grad()
                x_recon, mu_z, logvar_z = self._model(xb, cb)
                loss, _, _ = elbo_loss(x_recon, xb, mu_z, logvar_z, self.beta_kl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                opt.step()

            self._model.eval()
            with torch.no_grad():
                x_recon_v, mu_v, logvar_v = self._model(X_val, C_val)
                val_loss, _, _ = elbo_loss(x_recon_v, X_val, mu_v, logvar_v, self.beta_kl)
                val_loss = val_loss.item()
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
            x_s = self._model.sample(C, n=n_samples)   # (N, n_samples, 9)
        x_flat = x_s.view(-1, 9).cpu().numpy()
        return self.scaler.inverse_transform_features(x_flat)

    def evaluate(self, df_test: pd.DataFrame, n_samples: int = 10) -> dict:
        X_samples = self.sample_geometry(df_test, n_samples)
        feas = geometry_feasibility_rate(X_samples)
        return {"model": "cVAE_Base", "feasibility_rate": feas}

    def save(self, path: Path = None):
        path = path or (RESULTS_DIR / "cvae_base.pt")
        save_checkpoint(self._model, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_processed("all")
    df_tr, df_te = train_test_split_stratified(df, test_frac=0.2)
    m = CVAEBaseModel()
    m.fit(df_tr)
    print(json.dumps(m.evaluate(df_te), indent=2))
