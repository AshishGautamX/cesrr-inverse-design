"""
03_validate_physics.py -- Audit the clean dataset for CeSRR structural constraints.

Checks:
  1. Ordering constraint: r1 > r2 > r3 > r4
  2. Dimensional bounds
  3. Frequency bounds

Produces:
  - D1_validated.csv    : physics-valid records only
  - validation_report.csv : per-row pass/fail flags (paper Figure 3 data)

Usage:
    python code/data/03_validate_physics.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))
from utils.config import (
    DATA_PROC_DIR, FIGURES_DIR,
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    ORDERING_COLS, PARAM_BOUNDS, FREQ_BOUNDS
)


# -----------------------------------------------------------------------------
# Per-record physics checks
# -----------------------------------------------------------------------------

def check_ordering_series(row: pd.Series) -> bool:
    """r1 > r2 > r3 > r4"""
    try:
        return (row["r1"] > row["r2"] > row["r3"] > row["r4"])
    except Exception:
        return False


def check_bounds_series(row: pd.Series) -> dict:
    """Return dict of {param: pass/fail} for each bounded parameter."""
    results = {}
    for col, (lo, hi) in PARAM_BOUNDS.items():
        if col in row.index:
            results[col] = bool(lo <= row[col] <= hi)
    return results


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append validation flag columns to df.

    New columns:
        valid_ordering  : bool -- r1>r2>r3>r4
        valid_bounds    : bool -- all params within PARAM_BOUNDS
        valid_freq      : bool -- freq within FREQ_BOUNDS
        valid_all       : bool -- all three checks pass
    """
    df = df.copy()

    df["valid_ordering"] = df.apply(check_ordering_series, axis=1)

    bounds_checks = df.apply(check_bounds_series, axis=1).apply(pd.Series)
    df["valid_bounds"] = bounds_checks.all(axis=1)

    df["valid_freq"] = df[TARGET_COL].between(*FREQ_BOUNDS)

    df["valid_all"] = df["valid_ordering"] & df["valid_bounds"] & df["valid_freq"]

    return df


def print_validation_report(df_val: pd.DataFrame):
    total = len(df_val)
    for col in ["valid_ordering", "valid_bounds", "valid_freq", "valid_all"]:
        n_pass = df_val[col].sum()
        pct = 100 * n_pass / total
        log.info(f"  {col:20s}: {n_pass}/{total} pass ({pct:.1f}%)")

    if ROTATION_COL in df_val.columns:
        for rot in sorted(df_val[ROTATION_COL].unique()):
            sub = df_val[df_val[ROTATION_COL] == rot]
            rate = sub["valid_all"].mean() * 100
            log.info(f"  Rotation {rot}deg: {rate:.1f}% fully valid")


def plot_violation_rates(df_val: pd.DataFrame, out_path: Path):
    """
    Bar chart of violation rates per check (paper Figure 3).
    """
    checks = {
        "Ordering\n(r1>r2>r3>r4)": "valid_ordering",
        "Bounds": "valid_bounds",
        "Freq range": "valid_freq",
        "All\ncombined": "valid_all",
    }
    violation_rates = [
        100 * (1 - df_val[col].mean()) for col in checks.values()
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(checks.keys()), violation_rates,
                  color=["#e07b54", "#5b8db8", "#6dbf67", "#9966cc"],
                  edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Violation rate (%)", fontsize=11)
    ax.set_title("CeSRR Dataset -- Physics Constraint Violation Rates", fontsize=12)
    ax.set_ylim(0, max(violation_rates) * 1.3 + 1)
    for bar, rate in zip(bars, violation_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Figure 3 saved: {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    src = DATA_PROC_DIR / "D0_clean.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run 02_clean_normalize.py first.")

    df = pd.read_csv(src)
    log.info(f"Loaded {len(df)} clean records.")

    df_val = validate_dataframe(df)

    # Save full report (with flags)
    report_path = DATA_PROC_DIR / "validation_report.csv"
    df_val.to_csv(report_path, index=False)
    log.info(f"Validation report saved: {report_path}")

    print("\n-- Validation Summary --")
    print_validation_report(df_val)

    # Save D1: only fully valid records
    df_d1 = df_val[df_val["valid_all"]].drop(
        columns=["valid_ordering", "valid_bounds", "valid_freq", "valid_all"]
    ).reset_index(drop=True)

    out = DATA_PROC_DIR / "D1_validated.csv"
    df_d1.to_csv(out, index=False)
    log.info(f"D1 (valid only): {len(df_d1)} records -> {out}")

    # Plot Figure 3
    plot_violation_rates(df_val, FIGURES_DIR / "fig03_violation_rates.png")

    print(f"\n[OK] Step 03 complete. {len(df_d1)} valid records -> {out}")


if __name__ == "__main__":
    main()
