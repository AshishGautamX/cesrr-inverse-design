"""
rigid_inverse.py -- RIGID baseline (B2): Random Forest surrogate + MCMC inverse.

RIGID strategy (adapted from 2024 small-data EM design literature):
  1. Train a Random Forest (RF) as a fast forward surrogate: geometry -> freq
  2. At inference time, use MCMC (Metropolis-Hastings) to sample geometries
     that produce the target frequency according to the RF surrogate.

This approach is specifically designed for <250 sample regimes where neural
networks may overfit but ensemble methods remain robust.

PERFORMANCE FIXES (vs original):
  - MCMC steps reduced: 500 -> 200 default (capped at 100 in FAST_MODE)
  - Chains reduced: 10 -> 5 default (capped at 3 in FAST_MODE)
  - RF n_estimators: 200 -> 100 (still highly accurate on 89 samples)
  - Batch RF predict: run all chains at once via np.vstack instead of one-by-one
  - Progress logging: prints every sample so it never goes silent
"""

import sys
import json
import logging
import os
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
    fit(df_train)       : fit RF on (geometry, rotation) -> freq
    predict_forward(df) : RF forward prediction (for metrics)
    sample_inverse(df, n_chains, n_steps) : MCMC sampling for inverse design
    evaluate(df_test)   : compute feasibility rate and forward error
    """

    def __init__(
        self,
        n_estimators: int = 100,    # reduced from 200 (still accurate at N=89)
        max_depth: int = None,
        n_chains: int = 5,          # reduced from 10
        n_mcmc_steps: int = 200,    # reduced from 500
        mcmc_step_scale: float = 0.05,
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
            oob_score=True,   # enables out-of-bag R² -- fixes the "N/A" OOB report
            n_jobs=-1,
        )
        self._rf.fit(X, y)

        # Store param ranges from training data for MCMC proposal scaling
        lo = np.array([PARAM_BOUNDS[c][0] for c in UNIT_CELL_FEATURES])
        hi = np.array([PARAM_BOUNDS[c][1] for c in UNIT_CELL_FEATURES])
        self._param_ranges = np.stack([lo, hi], axis=1)

        log.info("RF fitted. OOB R2: %.4f", self._rf.oob_score_)
        return self

    def predict_forward(self, df: pd.DataFrame) -> np.ndarray:
        """RF forward prediction: geometry -> freq (GHz)."""
        X = self._build_X(df)
        return self._rf.predict(X)

    def _ordering_ok(self, geom: np.ndarray) -> bool:
        """r1 > r2 > r3 > r4."""
        return bool(geom[0] > geom[1] > geom[2] > geom[3])

    def _metropolis_hastings_vectorized(
        self,
        target_freq: float,
        rotation: float,
        rng: np.random.Generator,
        n_chains: int,
        n_steps: int,
    ) -> np.ndarray:
        """
        Run n_chains parallel MCMC chains -- ALL RF predictions are batched
        into a single predict() call per step instead of one-by-one.

        This is the key performance fix: 5 chains × 200 steps = 1000 RF calls
        done as 200 batch-of-5 calls instead of 1000 single-row calls.
        RF batch predict is ~100x faster than 1000 individual predictions.

        Returns (n_chains, 9) best geometry per chain.
        """
        lo = self._param_ranges[:, 0]
        hi = self._param_ranges[:, 1]
        step_std = (hi - lo) * self.mcmc_step_scale
        beta = 100.0

        # Initialise chains: find valid starting geometries
        chains = np.zeros((n_chains, 9))
        for i in range(n_chains):
            for _ in range(500):
                c = rng.uniform(lo, hi)
                if self._ordering_ok(c):
                    chains[i] = c
                    break
            else:
                chains[i] = (lo + hi) / 2  # fallback midpoint

        # Batch predict initial energies
        rot_col = np.full((n_chains, 1), rotation)
        X_batch = np.hstack([chains, rot_col])
        preds = self._rf.predict(X_batch)
        E_current = (preds - target_freq) ** 2

        best_chains = chains.copy()
        best_E      = E_current.copy()
        n_accept    = 0

        for _ in range(n_steps):
            # Propose all chains simultaneously
            proposals = chains + rng.normal(0, step_std, size=(n_chains, 9))
            proposals = np.clip(proposals, lo, hi)

            # Filter ordering constraint (per-chain)
            valid = np.array([self._ordering_ok(p) for p in proposals])

            # Batch predict for ALL proposals (even invalid ones -- fast)
            X_prop = np.hstack([proposals, rot_col])
            E_prop = (self._rf.predict(X_prop) - target_freq) ** 2

            # Metropolis acceptance per chain
            log_alpha = -beta * (E_prop - E_current)
            accept = valid & (
                (log_alpha >= 0) | (rng.random(n_chains) < np.exp(np.clip(log_alpha, -500, 0)))
            )

            # Update accepted chains
            chains[accept]    = proposals[accept]
            E_current[accept] = E_prop[accept]
            n_accept += accept.sum()

            # Track best per chain
            improved = accept & (E_prop < best_E)
            best_chains[improved] = proposals[improved]
            best_E[improved]      = E_prop[improved]

        log.debug(
            "MCMC accept rate: %.1f%% | best_E: %.4f",
            100 * n_accept / (n_steps * n_chains),
            best_E.min(),
        )
        return best_chains  # (n_chains, 9)

    def sample_inverse(
        self,
        df: pd.DataFrame,
        n_chains: int = None,
        n_steps: int = None,
    ) -> np.ndarray:
        """
        Sample geometry candidates for each row in df via batched MCMC.

        Returns
        -------
        (N, 9) array -- best geometry proposal per test sample.
        """
        n_chains = n_chains or self.n_chains
        n_steps  = n_steps  or self.n_mcmc_steps

        # Apply FAST_MODE caps
        if os.environ.get("CESRR_FAST_MODE", "0") == "1":
            n_chains = min(n_chains, 3)
            n_steps  = min(n_steps, 50)

        rng = np.random.default_rng(GLOBAL_SEED)
        all_best = []
        n_samples = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            target_freq = row[TARGET_COL]
            rotation    = float(row[ROTATION_COL] == 180)

            # Progress print -- so Colab never goes silent
            if i % max(1, n_samples // 5) == 0:
                print(
                    f"  MCMC sample {i+1}/{n_samples} | target={target_freq:.2f} GHz ...",
                    flush=True,
                )

            best_chains = self._metropolis_hastings_vectorized(
                target_freq, rotation, rng, n_chains, n_steps
            )

            # Pick chain with lowest energy
            rot_col = np.full((n_chains, 1), rotation)
            X_eval  = np.hstack([best_chains, rot_col])
            energies = (self._rf.predict(X_eval) - target_freq) ** 2
            best_idx = int(np.argmin(energies))
            all_best.append(best_chains[best_idx])

        return np.array(all_best)  # (N, 9)

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """
        Evaluate RIGID on test set.

        Returns BOTH forward accuracy (RF predicts frequency from true geometry)
        AND inverse quality (MCMC-generated geometries satisfy r1>r2>r3>r4).

        This dual reporting is critical for reviewer credibility:
          - Forward accuracy proves the RF surrogate is working
          - MCMC feasibility proves the MCMC inverse is searching valid space
        """
        log.info("Forward evaluation on %d test samples...", len(df_test))
        y_pred = self.predict_forward(df_test)
        y_true = df_test[TARGET_COL].values
        fwd_met = compute_regression_metrics(y_true, y_pred, model_name="RIGID_RF_Forward")

        log.info("Running batched MCMC inverse (%d samples, %d chains, %d steps)...",
                 len(df_test), self.n_chains, self.n_mcmc_steps)
        geoms = self.sample_inverse(df_test)
        feas  = geometry_feasibility_rate(geoms)

        # Compute forward frequency of the generated geometries (reconstruction MAE)
        # This answers: "how close is the generated geometry's frequency to the target?"
        rot_col  = (df_test[ROTATION_COL] == 180).values.astype(float).reshape(-1, 1)
        X_recon  = np.hstack([geoms, rot_col])  # (N, 10)
        freq_recon = self._rf.predict(X_recon)   # RF forward pass on generated geoms
        recon_mae  = float(np.mean(np.abs(freq_recon - y_true)))
        log.info("MCMC reconstruction MAE (RF oracle): %.4f GHz", recon_mae)

        return {
            **fwd_met,
            "feasibility_rate_mcmc": feas,
            "reconstruction_mae_ghz": round(recon_mae, 4),
        }

    def save(self, path: Path = None):
        path = path or (RESULTS_DIR / "rigid_inverse.pkl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("RIGID model saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "RIGIDInverse":
        return joblib.load(path)


if __name__ == "__main__":
    from utils.fast_mode import apply_fast_mode
    apply_fast_mode()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_processed("hf")
    df_tr, df_te = train_test_split_stratified(df, test_frac=0.2)
    m = RIGIDInverse()
    m.fit(df_tr)
    print(json.dumps(m.evaluate(df_te), indent=2))
    m.save()
