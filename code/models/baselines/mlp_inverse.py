"""
mlp_inverse.py -- Simplest baseline (B0): direct MLP inverse mapping.

Maps target frequency (+ rotation flag) -> geometry.
Trained directly on (freq, rot) -> (r1..p) pairs.

Known limitation: fails on one-to-many mapping -- this is intentionally the
weakest baseline, included to demonstrate why more sophisticated models
(tandem, cVAE) are necessary.
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
from utils.data_utils import CeSRRScaler, build_condition, train_test_split_stratified, load_processed
from utils.train_utils import train_model, build_mlp, DEVICE, set_seed, save_checkpoint
from evaluation.metrics import compute_regression_metrics, geometry_feasibility_rate

log = logging.getLogger(__name__)


class MLPInverse(nn.Module):
    """Direct inverse MLP: (freq_scaled, rotation_binary) -> geometry_scaled."""

    def __init__(
        self,
        in_dim: int = 2,        # [freq_scaled, rotation_binary]
        out_dim: int = 9,       # UNIT_CELL_FEATURES
        hidden: list = None,
    ):
        super().__init__()
        hidden = hidden or TANDEM_HIDDEN
        self.net = build_mlp(in_dim, out_dim, hidden, dropout_p=0.1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class MLPInverseModel:
    """
    Scikit-learn-style wrapper for the direct MLP inverse model.
    Provides fit / predict / evaluate interface consistent with other baselines.
    """

    def __init__(self, hidden: list = None):
        self.hidden  = hidden or TANDEM_HIDDEN
        self.scaler: CeSRRScaler | None = None
        self._model: MLPInverse | None = None

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> "MLPInverseModel":
        set_seed(GLOBAL_SEED)
        self.scaler = CeSRRScaler().fit(df_train)

        # Build condition (freq+rot) and targets (geometry)
        C_train = build_condition(df_train, self.scaler)   # (N, 2)
        X_train = self.scaler.transform_features(df_train)  # (N, 9) -- target

        if df_val is None:
            n_val = max(1, int(0.15 * len(df_train)))
            C_val = C_train[-n_val:]
            X_val = X_train[-n_val:]
            C_train = C_train[:-n_val]
            X_train = X_train[:-n_val]
        else:
            C_val = build_condition(df_val, self.scaler)
            X_val = self.scaler.transform_features(df_val)

        self._model = MLPInverse(in_dim=2, out_dim=9, hidden=self.hidden)
        opt = torch.optim.Adam(self._model.parameters(), lr=TANDEM_LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)
        loss_fn = nn.MSELoss()

        train_model(
            self._model, opt, loss_fn,
            C_train, X_train, C_val, X_val,
            max_epochs=TANDEM_MAX_EPOCHS,
            batch_size=TANDEM_BATCH_SIZE,
            patience=TANDEM_PATIENCE,
            scheduler=sched,
        )
        return self

    def predict_geometry(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted geometry (mm scale) for each row in df."""
        C = build_condition(df, self.scaler)
        C_t = torch.tensor(C, dtype=torch.float32, device=DEVICE)
        self._model.eval()
        with torch.no_grad():
            X_scaled = self._model(C_t).cpu().numpy()
        return self.scaler.inverse_transform_features(X_scaled)

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        X_pred = self.predict_geometry(df_test)
        X_true = df_test[UNIT_CELL_FEATURES].values
        mae_per_dim = np.abs(X_pred - X_true).mean(axis=0)
        feat_mae    = dict(zip(UNIT_CELL_FEATURES, mae_per_dim))
        feas        = geometry_feasibility_rate(X_pred)
        return {
            "model": "MLP_Inverse_Direct",
            "feasibility_rate": feas,
            "mean_geometry_MAE_mm": float(np.abs(X_pred - X_true).mean()),
            "per_param_MAE": feat_mae,
        }

    def save(self, path: Path = None):
        path = path or (RESULTS_DIR / "mlp_inverse.pt")
        save_checkpoint(self._model, path, extra={"hidden": self.hidden})

if __name__ == "__main__":
    from utils.fast_mode import apply_fast_mode
    apply_fast_mode()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_processed("all")
    df_tr, df_te = train_test_split_stratified(df, test_frac=0.2)
    m = MLPInverseModel()
    m.fit(df_tr)
    print(json.dumps(m.evaluate(df_te), indent=2))
