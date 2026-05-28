"""
shap_rotation_compare.py -- Novel contribution: 0deg vs 180deg SHAP comparison.

Computes SHAP values separately for 0deg and 180deg CeSRR configurations and
compares feature importance rankings.

Paper claim: "r1 is the dominant frequency-control parameter at 0deg rotation
(SHAP rank #1), while groove height h gains significance at 180deg rotation
(rank shifts from #4 to #2) -- consistent with orientation-dependent electric
field concentration in the CeSRR topology."

This analysis is the first SHAP-based design rule extraction for CeSRR SWS.

Reference:
  Amini et al. (2025) Sci. Rep. 15:24029 (DOI: 10.1038/s41598-025-10156-1)
  -- first SHAP application to metasurface antenna design (different device class)
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import (
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    RESULTS_DIR, FIGURES_DIR
)
from utils.data_utils import load_processed
from interpretability.shap_global import compute_shap_rf, shap_feature_importance

log = logging.getLogger(__name__)


def compute_rotation_shap(
    df: pd.DataFrame,
    rf_model,
) -> tuple:
    """
    Compute SHAP values separately for 0deg and 180deg subsets.

    Parameters
    ----------
    df       : full validated dataset (both rotations)
    rf_model : fitted RIGIDInverse._rf (RandomForestRegressor)

    Returns
    -------
    (shap_0, shap_180, X_0, X_180) -- SHAP arrays and feature matrices
    """
    df_0   = df[df[ROTATION_COL] == 0].reset_index(drop=True)
    df_180 = df[df[ROTATION_COL] == 180].reset_index(drop=True)

    X_0   = df_0[UNIT_CELL_FEATURES].values
    X_180 = df_180[UNIT_CELL_FEATURES].values

    log.info("Computing SHAP for 0deg subset (%d samples)...", len(df_0))
    shap_0   = compute_shap_rf(rf_model, X_0)

    log.info("Computing SHAP for 180deg subset (%d samples)...", len(df_180))
    shap_180 = compute_shap_rf(rf_model, X_180)

    return shap_0, shap_180, X_0, X_180


def compare_importance(imp_0: dict, imp_180: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame of feature importance rankings
    for 0deg and 180deg configurations.

    Returns
    -------
    DataFrame with columns: Feature, Rank_0deg, SHAP_0deg, Rank_180deg, SHAP_180deg, Rank_Shift
    """
    rows = []
    for feat in UNIT_CELL_FEATURES:
        r0   = imp_0[feat]["rank"]
        r180 = imp_180[feat]["rank"]
        rows.append({
            "Feature":     feat,
            "Rank_0deg":   r0,
            "SHAP_0deg":   round(imp_0[feat]["mean_abs_shap"], 4),
            "Rank_180deg": r180,
            "SHAP_180deg": round(imp_180[feat]["mean_abs_shap"], 4),
            "Rank_Shift":  r0 - r180,   # positive = moved up in 180deg ranking
        })
    df = pd.DataFrame(rows).sort_values("Rank_0deg").reset_index(drop=True)
    return df


def plot_rotation_comparison(
    shap_0: np.ndarray,
    shap_180: np.ndarray,
    X_0: np.ndarray,
    X_180: np.ndarray,
    out_path: Path = None,
):
    """
    Side-by-side beeswarm plot: 0deg (left) vs 180deg (right).
    This is Figure 6 of the paper.
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or (FIGURES_DIR / "fig06_shap_rotation_compare.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sv, X, label in [
        (axes[0], shap_0,   X_0,   "0deg CeSRR"),
        (axes[1], shap_180, X_180, "180deg CeSRR"),
    ]:
        plt.sca(ax)
        shap.summary_plot(
            sv, X,
            feature_names=UNIT_CELL_FEATURES,
            show=False, plot_type="dot",
        )
        ax.set_title(f"SHAP -- {label}", fontsize=11)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "SHAP Feature Importance: 0deg vs 180deg CeSRR Configuration",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure 6 saved: %s", out_path)


def plot_rank_shift_bar(comparison_df: pd.DataFrame, out_path: Path = None):
    """
    Bar chart showing rank shift (0deg rank − 180deg rank) per feature.
    Features that increase in importance at 180deg have negative shift.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or (FIGURES_DIR / "fig06b_rank_shift.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = comparison_df.sort_values("Rank_Shift")
    colors = ["#e07b54" if v > 0 else "#5b8db8" for v in df["Rank_Shift"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["Feature"], df["Rank_Shift"], color=colors, edgecolor="white")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Rank shift (0deg rank − 180deg rank)\nNegative = more important at 180deg", fontsize=10)
    ax.set_title("CeSRR SHAP Importance Rank Shift (0deg -> 180deg)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Rank shift figure saved: %s", out_path)


def run_rotation_shap(save: bool = True) -> pd.DataFrame:
    """Full pipeline: fit RF, compute SHAP for both rotations, plot and save."""
    from models.baselines.rigid_inverse import RIGIDInverse
    from utils.data_utils import train_test_split_stratified

    df = load_processed("hf")
    model = RIGIDInverse()
    model.fit(df)  # fit on full HF dataset for SHAP (not split)

    shap_0, shap_180, X_0, X_180 = compute_rotation_shap(df, model._rf)

    imp_0   = shap_feature_importance(shap_0)
    imp_180 = shap_feature_importance(shap_180)
    comp_df = compare_importance(imp_0, imp_180)

    if save:
        out = RESULTS_DIR / "shap_rotation_comparison.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(out, index=False)
        log.info("Comparison table saved: %s", out)

        plot_rotation_comparison(shap_0, shap_180, X_0, X_180)
        plot_rank_shift_bar(comp_df)

    print("\n-- SHAP Rotation Comparison --")
    print(comp_df.to_string(index=False))
    return comp_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_rotation_shap()
