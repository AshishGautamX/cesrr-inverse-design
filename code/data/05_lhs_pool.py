"""
05_lhs_pool.py — Generate a Latin Hypercube Sampling unlabelled pool for
                 active learning and low-fidelity oracle evaluation.

Produces:
  - D_lhs_pool.csv : ~200-300 physics-valid candidates (after ordering filter)

Usage:
    python code/data/05_lhs_pool.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))
from utils.config import (
    DATA_PROC_DIR,
    UNIT_CELL_FEATURES, PARAM_BOUNDS,
    LHS_N_CANDIDATES, LHS_RANDOM_SEED
)


def generate_lhs_pool(
    n: int = LHS_N_CANDIDATES,
    seed: int = LHS_RANDOM_SEED,
    enforce_ordering: bool = True,
) -> pd.DataFrame:
    """
    Generate n physics-valid candidates via Latin Hypercube Sampling.

    Parameters
    ----------
    n                 : number of LHS samples to draw initially
    seed              : random seed
    enforce_ordering  : if True, filter to r1>r2>r3>r4 only

    Returns
    -------
    DataFrame of valid candidates (columns = UNIT_CELL_FEATURES)
    """
    params = UNIT_CELL_FEATURES
    lo = np.array([PARAM_BOUNDS[p][0] for p in params])
    hi = np.array([PARAM_BOUNDS[p][1] for p in params])

    # Use scrambled LHS for better space coverage
    sampler = qmc.LatinHypercube(d=len(params), scramble=True, seed=seed)
    raw = sampler.random(n=n)
    scaled = qmc.scale(raw, lo, hi)

    df = pd.DataFrame(scaled, columns=params)

    if enforce_ordering:
        ordering_mask = (
            (df["r1"] > df["r2"]) &
            (df["r2"] > df["r3"]) &
            (df["r3"] > df["r4"])
        )
        n_before = len(df)
        df = df[ordering_mask].reset_index(drop=True)
        log.info(
            f"Ordering filter: {n_before} → {len(df)} candidates "
            f"({100*len(df)/n_before:.1f}% valid)"
        )

    # Tag as unlabelled (no freq_ghz assigned — will be filled by oracle or CST)
    df["freq_ghz"] = np.nan
    df["rotation"] = 0      # default; AL can query both rotations
    df["source"] = "lhs_pool"

    return df


def main():
    log.info(f"Generating LHS pool: n={LHS_N_CANDIDATES}, seed={LHS_RANDOM_SEED}")
    df_pool = generate_lhs_pool()

    out = DATA_PROC_DIR / "D_lhs_pool.csv"
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    df_pool.to_csv(out, index=False)

    print(f"\n── LHS Pool ──")
    print(f"  Requested : {LHS_N_CANDIDATES} samples")
    print(f"  Valid     : {len(df_pool)} candidates (after ordering filter)")
    print(f"  Columns   : {list(df_pool.columns)}")
    print(f"\n✅ Step 05 complete → {out}")


if __name__ == "__main__":
    main()
