"""
lf_oracle.py -- Low-fidelity oracle wrapping the analytical CeSRR LC model.

The oracle interface is used by:
  - 06_build_datasets.py  (label LHS pool with LF frequencies)
  - active_learning/al_loop.py  (cheap labelling during AL rounds)

The oracle is intentionally approximate; its role is to provide a
physically-grounded low-fidelity signal for multi-fidelity GP training.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from analytical.cesrr_lc_model import cesrr_resonant_frequency, batch_predict
from utils.config import FREQ_BOUNDS, UNIT_CELL_FEATURES

log = logging.getLogger(__name__)


class LFOracle:
    """
    Wraps the analytical CeSRR LC model as a callable oracle.

    Provides a consistent interface for:
      - Single-point prediction (scalar output)
      - Batch prediction (DataFrame input)
      - Validity checking (returns None for unphysical inputs)
    """

    def __init__(self, noise_std: float = 0.0):
        """
        Parameters
        ----------
        noise_std : optional Gaussian noise σ (GHz) added to predictions.
                    Set > 0 to simulate model uncertainty for testing.
                    In production (multi-fidelity training), keep at 0.
        """
        self.noise_std = noise_std
        log.debug("LFOracle initialised (noise_std=%.4f GHz)", noise_std)

    def predict_freq(
        self,
        r1: float, r2: float, r3: float, r4: float,
        t: float,
        d1: float = None, d2: float = None,
        h: float = None, p: float = None,
    ) -> Optional[float]:
        """
        Predict the resonant frequency for a single CeSRR design.

        Returns None if inputs are physically invalid or the predicted
        frequency is outside FREQ_BOUNDS.
        """
        f0 = cesrr_resonant_frequency(
            r1=r1, r2=r2, r3=r3, r4=r4, t=t,
            d1=d1, d2=d2, h=h, p=p,
        )
        if f0 is None:
            return None
        if not (FREQ_BOUNDS[0] <= f0 <= FREQ_BOUNDS[1]):
            return None

        if self.noise_std > 0:
            f0 += np.random.normal(0, self.noise_std)
            if not (FREQ_BOUNDS[0] <= f0 <= FREQ_BOUNDS[1]):
                return None

        return float(f0)

    def predict_batch(self, df) -> np.ndarray:
        """
        Predict frequencies for all rows in a DataFrame.

        Returns
        -------
        np.ndarray of shape (N,) with NaN for invalid/out-of-range rows.
        """
        raw = batch_predict(df)

        # Apply bounds filter
        lo, hi = FREQ_BOUNDS
        raw[(raw < lo) | (raw > hi)] = np.nan

        if self.noise_std > 0:
            noise_mask = ~np.isnan(raw)
            raw[noise_mask] += np.random.normal(0, self.noise_std, noise_mask.sum())

        return raw

    def label_pool(self, df_pool) -> "pd.DataFrame":
        """
        Label an unlabelled LHS pool with oracle predictions.
        Drops rows where prediction is NaN (unphysical).

        Parameters
        ----------
        df_pool : DataFrame with columns matching UNIT_CELL_FEATURES

        Returns
        -------
        Labelled DataFrame with 'freq_ghz' column populated.
        """
        import pandas as pd
        df = df_pool.copy()
        freqs = self.predict_batch(df)
        df["freq_ghz"] = freqs
        n_before = len(df)
        df = df.dropna(subset=["freq_ghz"]).reset_index(drop=True)
        log.info(
            "Oracle labelled %d/%d pool candidates (%.1f%% valid)",
            len(df), n_before, 100 * len(df) / n_before,
        )
        return df


# -----------------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    oracle = LFOracle()
    f = oracle.predict_freq(r1=42.8, r2=40.8, r3=34.8, r4=20.0, t=1.0)
    print(f"Single-point oracle: {f:.3f} GHz" if f else "INVALID")
