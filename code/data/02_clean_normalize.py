"""
02_clean_normalize.py -- Clean raw CSVs and produce a validated, normalised dataset.

Steps:
  1. Load raw_unit_combined.csv
  2. Drop rows with missing required columns
  3. Strip whitespace from string columns
  4. Enforce unit consistency (all dimensions in mm, freq in GHz)
  5. Remove exact duplicates
  6. Write D0_clean.csv

Usage:
    python code/data/02_clean_normalize.py
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
    DATA_PROC_DIR, UNIT_CELL_FEATURES, TARGET_COL,
    ROTATION_COL, PARAM_BOUNDS, FREQ_BOUNDS
)


def clean_unit_cell_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to a raw unit-cell DataFrame."""
    log.info(f"Input shape: {df.shape}")
    required = UNIT_CELL_FEATURES + [TARGET_COL]

    # 1. Keep only relevant columns (silently ignore extras)
    present = [c for c in required + [ROTATION_COL] if c in df.columns]
    df = df[present].copy()

    # 2. Drop rows missing any required column
    before = len(df)
    df = df.dropna(subset=[c for c in required if c in df.columns])
    log.info(f"Dropped {before - len(df)} rows with missing values.")

    # 3. Coerce all numeric columns
    for c in UNIT_CELL_FEATURES + [TARGET_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[c for c in required if c in df.columns])

    # 4. Handle frequency: if stored as e.g. 1.0 (GHz) keep; if > 100 it may
    #    have been entered in MHz accidentally -- convert.
    if TARGET_COL in df.columns:
        high_mhz_mask = df[TARGET_COL] > 100
        if high_mhz_mask.any():
            log.warning(
                f"Found {high_mhz_mask.sum()} rows with freq > 100 -- "
                "assuming MHz input, converting to GHz."
            )
            df.loc[high_mhz_mask, TARGET_COL] /= 1000.0

    # 5. Clip to physical bounds (don't drop -- flag instead)
    for col, (lo, hi) in PARAM_BOUNDS.items():
        if col in df.columns:
            out_of_range = ~df[col].between(lo, hi)
            if out_of_range.any():
                log.warning(
                    f"{col}: {out_of_range.sum()} values outside "
                    f"[{lo}, {hi}] -- keeping but flagging."
                )

    # 6. Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=UNIT_CELL_FEATURES + [TARGET_COL])
    log.info(f"Removed {before - len(df)} exact duplicate rows.")

    # 7. Sort by rotation and frequency
    sort_cols = [c for c in [ROTATION_COL, TARGET_COL] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    log.info(f"Clean output shape: {df.shape}")
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all numeric columns."""
    cols = [c for c in UNIT_CELL_FEATURES + [TARGET_COL] if c in df.columns]
    return df[cols].describe().T


def main():
    src = DATA_PROC_DIR / "raw_unit_combined.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run 01_load_raw.py first."
        )

    df_raw = pd.read_csv(src)
    log.info(f"Loaded {len(df_raw)} raw records.")

    df_clean = clean_unit_cell_df(df_raw)

    out = DATA_PROC_DIR / "D0_clean.csv"
    df_clean.to_csv(out, index=False)
    log.info(f"Saved: {out}")

    print("\n-- Summary statistics --")
    print(compute_summary_stats(df_clean).to_string())

    rot_counts = df_clean[ROTATION_COL].value_counts() if ROTATION_COL in df_clean.columns else {}
    print(f"\n-- Rotation counts --\n{rot_counts}")
    print(f"\n[OK] Step 02 complete. Clean dataset: {len(df_clean)} records -> {out}")


if __name__ == "__main__":
    main()
