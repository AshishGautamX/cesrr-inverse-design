"""
gp_surrogate.py -- Gaussian Process surrogate model (PRIMARY baseline B1).

Uses scikit-learn GaussianProcessRegressor with a Matérn-5/2 kernel.
The Matérn-5/2 kernel assumes twice-differentiable responses -- physically
appropriate for smooth EM resonance curves.

Provides both:
  - Forward surrogate: geometry -> predicted frequency (for evaluation)
  - Uncertainty estimation: σ2 at each prediction point (for AL)

Reference:
  Rasmussen & Williams (2006) Gaussian Processes for Machine Learning.
  Katkevičius et al. (2022) Electronics 11(15):2360 (ANN review context).
"""

import sys
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
    GP_MATERN_NU, GP_N_RESTARTS, RESULTS_DIR, GLOBAL_SEED
)
from utils.data_utils import CeSRRScaler, build_condition
from evaluation.metrics import compute_regression_metrics

log = logging.getLogger(__name__)


class GPSurrogate:
    """
    Gaussian Process surrogate for CeSRR frequency prediction.

    Input  : scaled geometry features + rotation binary flag -> (N, 10)
    Output : predicted frequency (GHz) + standard deviation
    """

    def __init__(
        self,
        nu: float = GP_MATERN_NU,
        n_restarts: int = GP_N_RESTARTS,
        noise_level_bounds: tuple = (1e-5, 1e-1),
    ):
        self.nu = nu
        self.n_restarts = n_restarts
        self.scaler: CeSRRScaler | None = None
        self._gp: GaussianProcessRegressor | None = None

        # Kernel: Constant amplitude x Matern + WhiteKernel for noise
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
            Matern(nu=nu, length_scale=1.0,
                   length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=1e-3,
                        noise_level_bounds=noise_level_bounds)
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=GLOBAL_SEED,
        )

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """Build input matrix: scaled geometry + rotation binary."""
        X_geom = self.scaler.transform_features(df)          # (N, 9)
        rot    = (df[ROTATION_COL] == 180).values.reshape(-1, 1).astype(float)
        return np.hstack([X_geom, rot])                       # (N, 10)

    def fit(self, df_train: pd.DataFrame) -> "GPSurrogate":
        """Fit the GP on a training DataFrame."""
        self.scaler = CeSRRScaler().fit(df_train)
        X = self._prepare_X(df_train)
        y = self.scaler.transform_target(df_train)
        log.info("Fitting GP on %d samples (dim=%d)...", len(df_train), X.shape[1])
        self._gp.fit(X, y)
        log.info("GP fitted. Log-marginal-likelihood: %.3f",
                 self._gp.log_marginal_likelihood_value_)
        return self

    def predict(
        self, df: pd.DataFrame, return_std: bool = False
    ) -> np.ndarray | tuple:
        """
        Predict frequency for df rows.

        Returns
        -------
        y_pred in GHz, or (y_pred, y_std) if return_std=True.
        """
        X = self._prepare_X(df)
        if return_std:
            y_scaled, std_scaled = self._gp.predict(X, return_std=True)
            y_pred = self.scaler.inverse_transform_target(y_scaled)
            # Scale std back (MinMax scaler: std_original ≈ std_scaled * range)
            freq_range = (self.scaler.tgt_scaler.data_max_[0] -
                          self.scaler.tgt_scaler.data_min_[0])
            y_std = std_scaled * freq_range
            return y_pred, y_std
        else:
            y_scaled = self._gp.predict(X, return_std=False)
            return self.scaler.inverse_transform_target(y_scaled)

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """Return metrics dict on test set."""
        y_pred = self.predict(df_test)
        y_true = df_test[TARGET_COL].values
        return compute_regression_metrics(y_true, y_pred, model_name="GP_Matern")

    def cross_validate(self, df: pd.DataFrame, n_folds: int = 5) -> dict:
        """
        5-fold cross-validated R² and MAE on a dataset.

        This is the key evidence that results are not overfitting to the 111
        HF samples. Reviewers will demand this for any <200 sample paper.

        Returns
        -------
        dict with 'cv_R2_mean', 'cv_R2_std', 'cv_MAE_mean', 'cv_MAE_std'
        """
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
        r2_scores, mae_scores = [], []

        X_all = df[UNIT_CELL_FEATURES].values
        rot   = (df[ROTATION_COL] == 180).values.reshape(-1, 1).astype(float)
        y_all = df[TARGET_COL].values

        log.info("Running %d-fold CV on %d samples...", n_folds, len(df))

        for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X_all)):
            df_tr_cv = df.iloc[tr_idx]
            df_te_cv = df.iloc[te_idx]

            gp_cv = GPSurrogate(nu=self.nu, n_restarts=self.n_restarts)
            gp_cv.fit(df_tr_cv)
            metrics = gp_cv.evaluate(df_te_cv)
            r2_scores.append(metrics["R2"])
            mae_scores.append(metrics["MAE_GHz"])
            log.info("  Fold %d/%d: R2=%.4f, MAE=%.4f GHz",
                     fold_i + 1, n_folds, metrics["R2"], metrics["MAE_GHz"])

        result = {
            "cv_R2_mean":  round(float(np.mean(r2_scores)),  4),
            "cv_R2_std":   round(float(np.std(r2_scores)),   4),
            "cv_MAE_mean": round(float(np.mean(mae_scores)), 4),
            "cv_MAE_std":  round(float(np.std(mae_scores)),  4),
            "n_folds":     n_folds,
            "n_samples":   len(df),
        }
        log.info("CV result: R2=%.4f±%.4f, MAE=%.4f±%.4f GHz",
                 result["cv_R2_mean"], result["cv_R2_std"],
                 result["cv_MAE_mean"], result["cv_MAE_std"])
        return result


    def save(self, path: Path = None):
        path = path or (RESULTS_DIR / "gp_surrogate.pkl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Ensure pickle stores the canonical module path, not '__main__'.
        # Without this, loading from any other script fails with:
        #   AttributeError: Can't get attribute 'GPSurrogate' on <module '__main__' ...>
        _orig_module = self.__class__.__module__
        self.__class__.__module__ = "models.baselines.gp_surrogate"
        joblib.dump(self, path)
        self.__class__.__module__ = _orig_module
        log.info("GP surrogate saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "GPSurrogate":
        obj = joblib.load(path)
        log.info("GP surrogate loaded: %s", path)
        return obj


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from utils.fast_mode import apply_fast_mode
    apply_fast_mode()
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from utils.data_utils import load_processed, train_test_split_stratified

    df = load_processed("hf")
    df_train, df_test = train_test_split_stratified(df, test_frac=0.2)

    gp = GPSurrogate()
    gp.fit(df_train)

    metrics = gp.evaluate(df_test)
    print("\n-- GP Surrogate Metrics (HF data) --")
    print(json.dumps(metrics, indent=2))

    # 5-fold CV on the full HF dataset (111 samples) for reviewers
    cv_results = gp.cross_validate(df, n_folds=5)
    print("\n-- GP 5-Fold CV (111 HF samples) --")
    print(json.dumps(cv_results, indent=2))

    # Save CV results to disk
    cv_out = RESULTS_DIR / "gp_cv_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cv_out, "w") as f:
        json.dump({**metrics, **cv_results}, f, indent=2)
    log.info("CV results saved: %s", cv_out)

    gp.save()

