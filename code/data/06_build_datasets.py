"""
06_build_datasets.py — Assemble the three final training datasets.

Outputs:
  D1_validated.csv   — 111 CST-verified HF records (already made in step 03)
  D2_augmented.csv   — Original + physics-augmented records (~555 samples)
  D_lf_analytical.csv— 200 LHS candidates labelled by analytical LC oracle

Usage:
    python code/data/06_build_datasets.py
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
    DATA_PROC_DIR, LF_N_SAMPLES,
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
)
from analytical.lf_oracle import LFOracle


def build_d2_augmented() -> pd.DataFrame:
    """Combine D1 + augmented to form D2."""
    src = DATA_PROC_DIR / "D1_plus_augmented.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} — run 04_augment_physics.py first.")
    df = pd.read_csv(src)
    # Drop helper columns not needed for training
    drop_cols = [c for c in ["augmented", "parent_idx", "source"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    out = DATA_PROC_DIR / "D2_augmented.csv"
    df.to_csv(out, index=False)
    log.info(f"D2 saved: {len(df)} records → {out}")
    return df


def build_lf_dataset() -> pd.DataFrame:
    """
    Label the LHS pool with the analytical LC oracle to create LF training data.
    Only use candidates where the oracle returns a plausible freq (0.5–20 GHz).
    """
    pool_path = DATA_PROC_DIR / "D_lhs_pool.csv"
    if not pool_path.exists():
        raise FileNotFoundError(f"{pool_path} — run 05_lhs_pool.py first.")

    df_pool = pd.read_csv(pool_path)
    oracle  = LFOracle()

    records = []
    for _, row in df_pool.iterrows():
        geom = {c: row[c] for c in UNIT_CELL_FEATURES if c in row.index}
        freq = oracle.predict_freq(**geom)
        if freq is not None:
            rec = dict(geom)
            rec[TARGET_COL] = freq
            rec[ROTATION_COL] = int(row.get(ROTATION_COL, 0))
            rec["source"] = "lf_analytical"
            records.append(rec)

    df_lf = pd.DataFrame(records)
    # Subsample to LF_N_SAMPLES if more produced
    if len(df_lf) > LF_N_SAMPLES:
        df_lf = df_lf.sample(LF_N_SAMPLES, random_state=42).reset_index(drop=True)

    out = DATA_PROC_DIR / "D_lf_analytical.csv"
    df_lf.to_csv(out, index=False)
    log.info(f"LF dataset saved: {len(df_lf)} records → {out}")
    return df_lf


def print_dataset_summary(label: str, df: pd.DataFrame):
    print(f"\n── {label} ──")
    print(f"  Rows   : {len(df)}")
    if ROTATION_COL in df.columns:
        print(f"  Rot    : {df[ROTATION_COL].value_counts().to_dict()}")
    if TARGET_COL in df.columns:
        print(f"  Freq   : {df[TARGET_COL].min():.2f} – {df[TARGET_COL].max():.2f} GHz")


def main():
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Building D2 (augmented)…")
    df_d2 = build_d2_augmented()
    print_dataset_summary("D2_augmented", df_d2)

    log.info("Building D_lf_analytical via oracle…")
    df_lf = build_lf_dataset()
    print_dataset_summary("D_lf_analytical", df_lf)

    # Also confirm D1 exists
    d1_path = DATA_PROC_DIR / "D1_validated.csv"
    if d1_path.exists():
        df_d1 = pd.read_csv(d1_path)
        print_dataset_summary("D1_validated (HF)", df_d1)
    else:
        log.warning("D1_validated.csv not found — run step 03 first.")

    print("\n✅ Step 06 complete. All datasets ready in:", DATA_PROC_DIR)


if __name__ == "__main__":
    main()
