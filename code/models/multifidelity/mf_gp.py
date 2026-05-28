"""
mf_gp.py -- Multi-Fidelity Gaussian Process (M1 novel model).

Implements Co-Kriging (Kennedy & O'Hagan 2000) combining:
  - LF data : analytical CeSRR LC model predictions (cheap, ≈200 samples)
  - HF data : CST-verified unit-cell simulations (expensive, ≈111 samples)

Model:
  f_HF(x) = rho · f_LF(x) + δ(x)

where:
  rho    : scalar correlation factor (learned from data)
  f_LF : GP trained on LF data (global trend)
  δ    : discrepancy GP trained on HF residuals (local correction)

This is the first application of multi-fidelity learning to CeSRR SWS.

Reference:
  Kennedy, M. C. & O'Hagan, A. (2000) Bayesian calibration of computer models.
  J. R. Statist. Soc. B, 63(3):425-464.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import (
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    GP_MATERN_NU, MF_GP_N_RESTARTS, RESULTS_DIR, GLOBAL_SEED
)
from utils.data_utils import (CeSRRScaler, load_combined_hf_lf,
                              train_test_split_stratified, load_processed)
from evaluation.metrics import compute_regression_metrics

log = logging.getLogger(__name__)


def _build_X(df: pd.DataFrame, scaler: CeSRRScaler) -> np.ndarray:
    """Scaled geometry (9) + rotation binary (1) -> (N, 10)."""
    X = scaler.transform_features(df)
    rot = (df[ROTATION_COL] == 180).values.reshape(-1, 1).astype(float)
    return np.hstack([X, rot])


def _make_kernel(nu: float = GP_MATERN_NU) -> object:
    return (
        ConstantKernel(1.0, (1e-3, 1e3)) *
        Matern(nu=nu, length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
        WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )


class MultiFidelityGP:
    """
    Co-Kriging two-level multi-fidelity GP.

    Training procedure:
      1. Fit GP_lf on low-fidelity (analytical) data
      2. Compute rho = cov(HF, LF_predictions) / var(LF_predictions)
      3. Compute HF residuals: r_i = y_HF_i - rho·GP_lf.predict(x_HF_i)
      4. Fit GP_delta on (x_HF, r) residuals

    Prediction:
      ŷ_HF(x) = rho·GP_lf(x) + GP_delta(x)
    """

    def __init__(
        self,
        nu: float = GP_MATERN_NU,
        n_restarts: int = MF_GP_N_RESTARTS,
    ):
        self.nu = nu
        self.n_restarts = n_restarts
        self.scaler: CeSRRScaler | None = None
        self._gp_lf: GaussianProcessRegressor | None = None
        self._gp_delta: GaussianProcessRegressor | None = None
        self._rho: float = 1.0

    def fit(
        self,
        df_lf: pd.DataFrame,
        df_hf: pd.DataFrame,
    ) -> "MultiFidelityGP":
        """
        Parameters
        ----------
        df_lf : LF dataset (analytical oracle predictions)
        df_hf : HF dataset (CST-verified records)
        """
        # Fit scaler on combined data for consistent normalisation
        df_all = pd.concat([df_lf, df_hf], ignore_index=True)
        self.scaler = CeSRRScaler().fit(df_all)

        # -- Step 1: train GP_lf on low-fidelity data ----------------------
        X_lf = _build_X(df_lf, self.scaler)
        y_lf = self.scaler.transform_target(df_lf)

        self._gp_lf = GaussianProcessRegressor(
            kernel=_make_kernel(self.nu),
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=GLOBAL_SEED,
        )
        log.info("Fitting GP_lf on %d LF samples...", len(df_lf))
        self._gp_lf.fit(X_lf, y_lf)
        log.info("  GP_lf LML: %.3f", self._gp_lf.log_marginal_likelihood_value_)

        # -- Step 2: compute rho ---------------------------------------------
        X_hf = _build_X(df_hf, self.scaler)
        y_hf = self.scaler.transform_target(df_hf)

        lf_pred_at_hf = self._gp_lf.predict(X_hf)   # LF prediction at HF points

        # rho = least-squares scaling factor
        # min ||y_hf - rho·lf_pred||2 -> rho = (lf·hf) / (lf·lf)
        self._rho = float(
            np.dot(lf_pred_at_hf, y_hf) /
            (np.dot(lf_pred_at_hf, lf_pred_at_hf) + 1e-12)
        )
        log.info("  Estimated rho (LF-HF correlation): %.4f", self._rho)

        # -- Step 3: compute HF residuals ----------------------------------
        residuals = y_hf - self._rho * lf_pred_at_hf

        # -- Step 4: train GP_delta on residuals ---------------------------
        self._gp_delta = GaussianProcessRegressor(
            kernel=_make_kernel(self.nu),
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=GLOBAL_SEED,
        )
        log.info("Fitting GP_delta on %d HF residuals...", len(df_hf))
        self._gp_delta.fit(X_hf, residuals)
        log.info("  GP_delta LML: %.3f", self._gp_delta.log_marginal_likelihood_value_)

        return self

    def predict(
        self,
        df: pd.DataFrame,
        return_std: bool = False,
    ) -> np.ndarray | tuple:
        """
        Predict HF frequency using the co-kriging formula.

        ŷ_HF(x) = rho·GP_lf(x) + GP_delta(x)

        Returns freq in GHz.
        """
        X = _build_X(df, self.scaler)

        if return_std:
            mu_lf,  std_lf  = self._gp_lf.predict(X, return_std=True)
            mu_del, std_del = self._gp_delta.predict(X, return_std=True)
            y_scaled = self._rho * mu_lf + mu_del
            # Combined variance (approximate, assuming independence)
            std_scaled = np.sqrt((self._rho * std_lf) ** 2 + std_del ** 2)
            freq_range = (self.scaler.tgt_scaler.data_max_[0] -
                          self.scaler.tgt_scaler.data_min_[0])
            y_pred = self.scaler.inverse_transform_target(y_scaled)
            y_std  = std_scaled * freq_range
            return y_pred, y_std
        else:
            mu_lf  = self._gp_lf.predict(X, return_std=False)
            mu_del = self._gp_delta.predict(X, return_std=False)
            y_scaled = self._rho * mu_lf + mu_del
            return self.scaler.inverse_transform_target(y_scaled)

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        y_pred = self.predict(df_test)
        y_true = df_test[TARGET_COL].values
        return compute_regression_metrics(y_true, y_pred, model_name="MF_GP_CoKriging")

    def save(self, path: Path = None):
        path = path or (RESULTS_DIR / "mf_gp.pkl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("MF-GP saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "MultiFidelityGP":
        return joblib.load(path)


if __name__ == "__main__":
    from utils.fast_mode import apply_fast_mode
    apply_fast_mode()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df_hf, df_lf = load_combined_hf_lf()
    df_hf_tr, df_hf_te = train_test_split_stratified(df_hf, test_frac=0.2)

    mf = MultiFidelityGP()
    mf.fit(df_lf, df_hf_tr)

    print("\n-- MF-GP Evaluation (HF test set) --")
    print(json.dumps(mf.evaluate(df_hf_te), indent=2))
    mf.save()
