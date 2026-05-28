"""
rigid_inverse.py — RIGID baseline (B2): Random Forest surrogate + MCMC inverse.

RIGID strategy (adapted from 2024 small-data EM design literature):
  1. Train a Random Forest (RF) as a fast forward surrogate: geometry → freq
  2. At inference time, use MCMC (Metropolis-Hastings) to sample geometries
     that produce the target frequency according to the RF surrogate.

This approach is specifically designed for <250 sample regimes where neural
networks may overfit but ensemble methods remain robust.

The MCMC proposal explores the geometry space conditioned on the target
frequency by accepting/rejecting based on the RF-predicted frequency error.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import (
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    PARAM_BOUNDS, RESULTS_DIR, GLOBAL_SEED
)
from utils.data_utils import (CeSRRScaler, train_test_split_stratified, load_processed)
from evaluation.metrics import compute_regression_metrics, geometry_feasibility_rate

log = logging.getLogger(__name__)


class RIGIDInverse:
    """
    Random Forest forward surrogate + Metropolis-Hastings inverse sampler.

    Methods
    -------
    fit(df_train)       : fit RF on (geometry, rotation) → freq
    predict_forward(df) : RF forward prediction (for metrics)
    sample_inverse(df, n_chains, n_steps) : MCMC sampling for inverse design
    evaluate(df_test)   : compute feasibility rate and forward error
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = None,
        n_chains: int = 10,
        n_mcmc_steps: int = 500,
        mcmc_step_scale: float = 0.05,   # proposal std as fraction of param range
    ):
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.n_chains        = n_chains
        self.n_mcmc_steps    = n_mcmc_steps
        self.mcmc_step_scale = mcmc_step_scale

        self._rf: RandomForestRegressor | None = None
        self.scaler: CeSRRScaler | None = None
        self._param_ranges: np.ndarray | None = None  # (9, 2) lo/hi for MCMC

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        """Geometry features + rotation binary, raw (not scaled for RF)."""
        X = df[UNIT_CELL_FEATURES].values.astype(float)
        rot = (df[ROTATION_COL] == 180).values.reshape(-1, 1).astype(float)
        return np.hstack([X, rot])  # (N, 10)

    def fit(self, df_train: pd.DataFrame) -> "RIGIDInverse":
        """Train Random Forest forward model."""
        self.scaler = CeSRRScaler().fit(df_train)
        X = self._build_X(df_train)
        y = df_train[TARGET_COL].values

        self._rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=GLOBAL_SEED,
            n_jobs=-1,
        )
        self._rf.fit(X, y)

        # Store param ranges from training data for MCMC proposal scaling
        lo = np.array([PARAM_BOUNDS[c][0] for c in UNIT_CELL_FEATURES])
        hi = np.array([PARAM_BOUNDS[c][1] for c in UNIT_CELL_FEATURES])
        self._param_ranges = np.stack([lo, hi], axis=1)

        oob = self._rf.oob_score_ if hasattr(self._rf, "oob_score_") else None
        log.info("RF fitted. OOB R²: %s", f"{oob:.4f}" if oob else "N/A")
        return self

    def predict_forward(self, df: pd.DataFrame) -> np.ndarray:
        """RF forward prediction: geometry → freq (GHz)."""
        X = self._build_X(df)
        return self._rf.predict(X)

    def _rf_predict_single(self, geom: np.ndarray, rotation: float) -> float:
        """Forward prediction for a single geometry vector (9,)."""
        x = np.append(geom, rotation).reshape(1, -1)
        return self._rf.predict(x)[0]

    def _ordering_ok(self, geom: np.ndarray) -> bool:
        """r1 > r2 > r3 > r4."""
        return bool(geom[0] > geom[1] > geom[2] > geom[3])

    def _bounds_ok(self, geom: np.ndarray) -> bool:
        """All params within PARAM_BOUNDS."""
        lo = self._param_ranges[:, 0]
        hi = self._param_ranges[:, 1]
        return bool(np.all(geom >= lo) and np.all(geom <= hi))

    def _metropolis_hastings(
        self,
        target_freq: float,
        rotation: float,
        rng: np.random.Generator,
    ) -> list:
        """
        Run Metropolis-Hastings to sample geometries giving target_freq.

        Energy: E(geom) = (RF(geom) - target_freq)²
        We seek samples from P(geom) ∝ exp(-β·E(geom)).

        Returns list of accepted geometry arrays (9,).
        """
        beta = 100.0   # inverse temperature — higher → tighter acceptance

        # Initialise from random valid point
        lo = self._param_ranges[:, 0]
        hi = self._param_ranges[:, 1]
        accepted_samples = []
        max_init_attempts = 200

        geom = None
        for _ in range(max_init_attempts):
            candidate = rng.uniform(lo, hi)
            if self._ordering_ok(candidate):
                geom = candidate
                break
        if geom is None:
            return []

        E_current = (self._rf_predict_single(geom, rotation) - target_freq) ** 2

        # Step scale: fraction of each param's range
        step_std = (hi - lo) * self.mcmc_step_scale

        n_accept = 0
        for step in range(self.n_mcmc_steps):
            # Propose
            proposal = geom + rng.normal(0, step_std)
            proposal = np.clip(proposal, lo, hi)

            if not self._ordering_ok(proposal):
                continue

            E_prop = (self._rf_predict_single(proposal, rotation) - target_freq) ** 2

            # Metropolis acceptance
            log_alpha = -beta * (E_prop - E_current)
            if log_alpha >= 0 or rng.random() < np.exp(log_alpha):
                geom = proposal
                E_current = E_prop
                n_accept += 1
                accepted_samples.append(geom.copy())

        log.debug("MCMC acceptance rate: %.2f%%", 100 * n_accept / self.n_mcmc_steps)
        return accepted_samples

    def sample_inverse(
        self,
        df: pd.DataFrame,
        n_chains: int = None,
    ) -> np.ndarray:
        """
        Sample geometry candidates for each row in df via MCMC.

        Returns
        -------
        (N × n_chains, 9) array with best geometry proposal per (sample, chain).
        """
        n_chains = n_chains or self.n_chains
        rng = np.random.default_rng(GLOBAL_SEED)
        all_geoms = []

        for _, row in df.iterrows():
            target_freq = row[TARGET_COL]
            rotation    = float(row[ROTATION_COL] == 180)
            chain_best  = []

            for _ in range(n_chains):
                samples = self._metropolis_hastings(target_freq, rotation, rng)
                if samples:
                    # Best sample = lowest energy (closest to target freq)
                    energies = [
                        (self._rf_predict_single(s, rotation) - target_freq) ** 2
                        for s in samples
                    ]
                    best = samples[int(np.argmin(energies))]
                    chain_best.append(best)

            if chain_best:
                all_geoms.extend(chain_best)
            else:
                # Fallback: random valid geometry
                lo = self._param_ranges[:, 0]
                hi = self._param_ranges[:, 1]
                all_geoms.append(rng.uniform(lo, hi))

        return np.array(all_geoms)

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        y_pred   = self.predict_forward(df_test)
        y_true   = df_test[TARGET_COL].values
        fwd_met  = compute_regression_metrics(y_true, y_pred, model_name="RIGID_RF_Forward")

        geoms    = self.sample_inverse(df_test, n_chains=5)
        feas     = geometry_feasibility_rate(geoms)

        return {**fwd_met, "feasibility_rate_mcmc": feas}

    def save(self, path: Path = None):
        path = path or (RESULTS_DIR / "rigid_inverse.pkl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("RIGID model saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "RIGIDInverse":
        return joblib.load(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_processed("hf")
    df_tr, df_te = train_test_split_stratified(df, test_frac=0.2)
    m = RIGIDInverse()
    m.fit(df_tr)
    print(json.dumps(m.evaluate(df_te), indent=2))
    m.save()
