"""
04_augment_physics.py — Physics-guided Gaussian perturbation augmentation.

Strategy (from plan, grounded in MDPI 2024 / IEEE 2024 precedent):
  For each validated record, generate AUG_N_PER_SAMPLE × 3 candidates by
  applying ±2% Gaussian noise to all geometric parameters, then filter:
    - Keep only those satisfying r1>r2>r3>r4
    - Keep only those within PARAM_BOUNDS
    - Keep only those with freq estimated by analytical oracle within FREQ_BOUNDS

  Result: ~111 × 4 ≈ 444 new synthetic records + 111 originals = ~555 total.

Usage:
    python code/data/04_augment_physics.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))
from utils.config import (
    DATA_PROC_DIR,
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    PARAM_BOUNDS, FREQ_BOUNDS,
    AUG_SIGMA_FRAC, AUG_N_PER_SAMPLE, AUG_OVERSAMPLE_FACTOR, AUG_RANDOM_SEED
)


# ─────────────────────────────────────────────────────────────────────────────
# Constraint checks (local, without importing data_utils to avoid circular dep)
# ─────────────────────────────────────────────────────────────────────────────

def _ordering_ok(r: dict) -> bool:
    return r["r1"] > r["r2"] > r["r3"] > r["r4"]


def _bounds_ok(r: dict) -> bool:
    for col, (lo, hi) in PARAM_BOUNDS.items():
        if col in r and not (lo <= r[col] <= hi):
            return False
    return True


def _is_valid(r: dict) -> bool:
    return _ordering_ok(r) and _bounds_ok(r)


# ─────────────────────────────────────────────────────────────────────────────
# Core augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_record(
    row: pd.Series,
    n_target: int,
    sigma_frac: float,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate up to n_target valid augmented copies of a single record.

    Parameters
    ----------
    row       : original record (pd.Series)
    n_target  : desired number of valid augmented copies
    sigma_frac: fractional standard deviation for Gaussian noise (e.g. 0.02)
    rng       : numpy random generator for reproducibility

    Returns
    -------
    List of dicts (each is one augmented record with all original columns)
    """
    geom_cols = [c for c in UNIT_CELL_FEATURES if c in row.index]
    row_dict  = row.to_dict()
    accepted  = []
    attempts  = 0
    max_tries = n_target * AUG_OVERSAMPLE_FACTOR * 10  # safety limit

    while len(accepted) < n_target and attempts < max_tries:
        attempts += 1
        # Multiplicative Gaussian noise (preserves scale)
        noise = rng.normal(loc=1.0, scale=sigma_frac, size=len(geom_cols))
        candidate = row_dict.copy()
        for col, nfac in zip(geom_cols, noise):
            candidate[col] = row_dict[col] * nfac

        if _is_valid(candidate):
            # Frequency is kept from parent record (same topology, similar freq)
            # The analytical oracle in 06_build_datasets re-estimates if needed
            candidate["augmented"] = True
            candidate["parent_idx"] = row_dict.get("_idx", -1)
            accepted.append(candidate)

    return accepted


def physics_guided_augment(
    df: pd.DataFrame,
    n_per_sample: int = AUG_N_PER_SAMPLE,
    sigma_frac: float = AUG_SIGMA_FRAC,
    seed: int = AUG_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Apply physics-guided augmentation to all rows in df.

    Returns a DataFrame of synthetic-only records (do NOT include originals here;
    they are concatenated in 06_build_datasets.py).
    """
    rng = np.random.default_rng(seed)
    all_augmented = []

    for idx, row in df.reset_index(drop=True).iterrows():
        row_copy = row.copy()
        row_copy["_idx"] = idx
        new_recs = augment_record(row_copy, n_per_sample, sigma_frac, rng)
        all_augmented.extend(new_recs)

    df_aug = pd.DataFrame(all_augmented)

    # Clean up helper columns
    if "_idx" in df_aug.columns:
        df_aug = df_aug.drop(columns=["_idx"])

    log.info(
        f"Augmentation: {len(df)} originals × {n_per_sample} target → "
        f"{len(df_aug)} synthetic records generated "
        f"({len(df_aug)/len(df):.1f}× expansion)"
    )
    return df_aug


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    src = DATA_PROC_DIR / "D1_validated.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run 03_validate_physics.py first.")

    df_d1 = pd.read_csv(src)
    log.info(f"Loaded D1: {len(df_d1)} validated records.")

    # Mark originals
    df_d1["augmented"] = False
    df_d1["parent_idx"] = -1

    # Augment
    df_aug = physics_guided_augment(df_d1)

    # Combine original + synthetic
    df_all = pd.concat([df_d1, df_aug], ignore_index=True)

    # Re-validate the combined set to be safe
    valid_mask = df_all.apply(
        lambda r: _ordering_ok(r.to_dict()) and _bounds_ok(r.to_dict()), axis=1
    )
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        log.warning(f"Dropping {n_invalid} invalid augmented records post-hoc.")
    df_all = df_all[valid_mask].reset_index(drop=True)

    out = DATA_PROC_DIR / "D1_plus_augmented.csv"
    df_all.to_csv(out, index=False)

    print(f"\n── Augmentation Summary ──")
    print(f"  Original D1 records    : {len(df_d1)}")
    print(f"  Synthetic records      : {len(df_aug)}")
    print(f"  Combined (after filter): {len(df_all)}")
    print(f"  Expansion ratio        : {len(df_all)/len(df_d1):.2f}×")
    print(f"\n✅ Step 04 complete → {out}")


if __name__ == "__main__":
    main()
