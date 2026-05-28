"""
tandem_network.py -- Tandem neural network inverse design baseline (B3).

Architecture (Liu et al. 2018, ACS Photonics 5:1365, DOI: 10.1021/acsphotonics.7b01377):
  Stage 1: Train a forward network  F: geometry -> freq    (freeze after)
  Stage 2: Train an inverse network I: freq    -> geometry
           Loss = MSE(F(I(freq_target)), freq_target)
           Backprop flows through I only (F is frozen)

This resolves the one-to-many mapping problem by not computing loss in
geometry space (where the mapping is ambiguous) but in frequency space
(which is unique given a geometry).
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import (
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    TANDEM_HIDDEN, TANDEM_LR, TANDEM_MAX_EPOCHS,
    TANDEM_PATIENCE, TANDEM_BATCH_SIZE, RESULTS_DIR, GLOBAL_SEED
)
from utils.data_utils import (CeSRRScaler, build_condition,
                              train_test_split_stratified, load_processed)
from utils.train_utils import (train_model, build_mlp, DEVICE,
                               set_seed, save_checkpoint)
from evaluation.metrics import compute_regression_metrics, geometry_feasibility_rate

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Network definitions
# -----------------------------------------------------------------------------

class ForwardNet(nn.Module):
    """Geometry + rotation -> scaled frequency."""

    def __init__(self, in_dim: int = 10, hidden: list = None):
        super().__init__()
        self.net = build_mlp(in_dim, 1, hidden or TANDEM_HIDDEN, dropout_p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InverseNet(nn.Module):
    """Frequency + rotation -> scaled geometry (9 dims)."""

    def __init__(self, in_dim: int = 2, out_dim: int = 9, hidden: list = None):
        super().__init__()
        self.net = build_mlp(in_dim, out_dim, hidden or TANDEM_HIDDEN, dropout_p=0.1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


# -----------------------------------------------------------------------------
# Tandem wrapper
# -----------------------------------------------------------------------------

class TandemNetwork:
    """
    Tandem architecture.

    fit() runs two training stages sequentially:
      1. Train ForwardNet (geometry+rot -> freq)
      2. Train InverseNet with ForwardNet frozen (freq+rot -> geometry)
    """

    def __init__(self, hidden: list = None):
        self.hidden  = hidden or TANDEM_HIDDEN
        self.scaler: CeSRRScaler | None = None
        self.fwd_net: ForwardNet | None = None
        self.inv_net: InverseNet | None = None

    # -- Stage 1: train forward network ---------------------------------------

    def _train_forward(
        self, X_train, y_train, X_val, y_val
    ) -> ForwardNet:
        """X = [geom_scaled | rot], y = freq_scaled."""
        model = ForwardNet(in_dim=X_train.shape[1], hidden=self.hidden).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=TANDEM_LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)
        loss_fn = nn.MSELoss()

        log.info("Stage 1: training ForwardNet (%d samples)...", len(X_train))
        train_model(
            model, opt, loss_fn,
            X_train, y_train, X_val, y_val,
            max_epochs=TANDEM_MAX_EPOCHS, batch_size=TANDEM_BATCH_SIZE,
            patience=TANDEM_PATIENCE, scheduler=sched,
        )
        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # -- Stage 2: train inverse network ---------------------------------------

    def _train_inverse(
        self, C_train, X_geom_train, C_val, X_geom_val
    ) -> InverseNet:
        """
        C = condition [freq_scaled, rot_binary].
        Loss = ||ForwardNet([InverseNet(C) | rot]) - freq_target||2
        """
        fwd = self.fwd_net.to(DEVICE)
        model = InverseNet(in_dim=2, out_dim=9, hidden=self.hidden).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=TANDEM_LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)

        def tandem_loss(C_batch, _ignored_target):
            """Compute forward loss: MSE in frequency space."""
            geom_pred = model(C_batch)              # (B, 9)
            rot_col   = C_batch[:, 1:2]             # rotation flag (B,1)
            fwd_input = torch.cat([geom_pred, rot_col], dim=1)  # (B, 10)
            freq_pred = fwd(fwd_input)              # (B, 1)
            freq_tgt  = C_batch[:, 0:1]             # scaled frequency
            return nn.functional.mse_loss(freq_pred, freq_tgt)

        # Custom training loop (loss fn has different signature)
        from torch.utils.data import DataLoader, TensorDataset
        from utils.train_utils import EarlyStopping

        Ct  = torch.tensor(C_train, dtype=torch.float32, device=DEVICE)
        Cv  = torch.tensor(C_val,   dtype=torch.float32, device=DEVICE)
        loader = DataLoader(TensorDataset(Ct), batch_size=TANDEM_BATCH_SIZE, shuffle=True)
        es = EarlyStopping(patience=TANDEM_PATIENCE)

        log.info("Stage 2: training InverseNet (tandem loss)...")
        for epoch in range(1, TANDEM_MAX_EPOCHS + 1):
            model.train()
            for (Cb,) in loader:
                opt.zero_grad()
                loss = tandem_loss(Cb, None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = tandem_loss(Cv, None).item()
            if sched:
                sched.step(val_loss)
            if epoch % 50 == 0:
                log.info("  Epoch %d | val_loss=%.5f", epoch, val_loss)
            if es(val_loss, model):
                break
        es.restore_best(model)
        return model

    # -- Public API ------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> "TandemNetwork":
        set_seed(GLOBAL_SEED)
        self.scaler = CeSRRScaler().fit(df_train)

        # Feature matrices
        X_geom = self.scaler.transform_features(df_train)   # (N, 9)
        rot     = (df_train[ROTATION_COL] == 180).values.reshape(-1, 1).astype(float)
        X_fwd   = np.hstack([X_geom, rot])                  # (N, 10)
        y_freq  = self.scaler.transform_target(df_train)    # (N,)
        C       = build_condition(df_train, self.scaler)     # (N, 2)

        if df_val is None:
            n_val = max(1, int(0.15 * len(df_train)))
            X_fwd_v, y_freq_v = X_fwd[-n_val:], y_freq[-n_val:]
            X_fwd,   y_freq   = X_fwd[:-n_val], y_freq[:-n_val]
            C_v  = C[-n_val:];  C = C[:-n_val]
            X_geom_v = X_geom[-n_val:]; X_geom = X_geom[:-n_val]
        else:
            X_geom_v = self.scaler.transform_features(df_val)
            rot_v    = (df_val[ROTATION_COL] == 180).values.reshape(-1, 1).astype(float)
            X_fwd_v  = np.hstack([X_geom_v, rot_v])
            y_freq_v = self.scaler.transform_target(df_val)
            C_v      = build_condition(df_val, self.scaler)

        self.fwd_net = self._train_forward(X_fwd, y_freq, X_fwd_v, y_freq_v)
        self.inv_net = self._train_inverse(C, X_geom, C_v, X_geom_v)
        return self

    def predict_geometry(self, df: pd.DataFrame) -> np.ndarray:
        C = build_condition(df, self.scaler)
        Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)
        self.inv_net.eval()
        with torch.no_grad():
            X_scaled = self.inv_net(Ct).cpu().numpy()
        return self.scaler.inverse_transform_features(X_scaled)

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        X_pred = self.predict_geometry(df_test)
        X_true = df_test[UNIT_CELL_FEATURES].values
        feas   = geometry_feasibility_rate(X_pred)
        return {
            "model": "Tandem_MLP",
            "feasibility_rate": feas,
            "mean_geometry_MAE_mm": float(np.abs(X_pred - X_true).mean()),
        }

    def save(self, path: Path = None):
        path = path or (RESULTS_DIR / "tandem_network.pt")
        save_checkpoint(self.inv_net, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_processed("all")
    df_tr, df_te = train_test_split_stratified(df, test_frac=0.2)
    m = TandemNetwork()
    m.fit(df_tr)
    print(json.dumps(m.evaluate(df_te), indent=2))
