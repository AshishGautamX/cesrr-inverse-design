"""
shap_global.py -- Global SHAP analysis on the PI-cVAE forward surrogate.

Computes SHAP values using TreeExplainer (for Random Forest) or
KernelExplainer (for neural models) to identify which geometric parameters
most influence the predicted resonant frequency.

Paper contribution: first SHAP analysis of CeSRR geometry-frequency relations.

Reference:
  Lundberg & Lee (2017) NeurIPS 30:4768-4777. arXiv:1705.07874
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

log = logging.getLogger(__name__)


def compute_shap_rf(
    rf_model,
    X: np.ndarray,
    feature_names: list = None,
) -> np.ndarray:
    """
    Compute SHAP values for a RandomForestRegressor using TreeExplainer.

    Parameters
    ----------
    rf_model     : fitted sklearn RandomForestRegressor
    X            : (N, D) input features
    feature_names: list of feature names

    Returns
    -------
    shap_values : (N, D) SHAP values array
    """
    import shap
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    log.info("SHAP values computed for %d samples (TreeExplainer).", len(X))
    return shap_values


def compute_shap_kernel(
    predict_fn,
    X_background: np.ndarray,
    X_query: np.ndarray,
    feature_names: list = None,
    n_background: int = 50,
) -> np.ndarray:
    """
    Compute SHAP values using KernelExplainer (model-agnostic, slower).

    For neural network models (PI-cVAE surrogate).

    Parameters
    ----------
    predict_fn    : callable (N, D) -> (N,) -- forward prediction function
    X_background  : background dataset for SHAP (summarised by k-means)
    X_query       : samples to explain
    n_background  : number of background points (k-means summary)

    Returns
    -------
    shap_values : (N_query, D)
    """
    import shap
    background = shap.kmeans(X_background, n_background)
    explainer  = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_query, nsamples=100, silent=True)
    log.info("SHAP values computed for %d samples (KernelExplainer).", len(X_query))
    return shap_values


def shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: list = None,
) -> dict:
    """
    Compute mean |SHAP| per feature and rank them.

    Parameters
    ----------
    shap_values  : (N, D) SHAP values -- may have D > len(feature_names)
                   if rotation column was included in the RF input.
                   Only the first len(feature_names) columns are used.
    feature_names: list of feature names (default: UNIT_CELL_FEATURES)

    Returns
    -------
    dict {feature: {mean_abs_shap, rank}}, rank 1 = most important
    """
    feature_names = feature_names or UNIT_CELL_FEATURES
    n_feats   = len(feature_names)

    # Slice to geometry features only (exclude rotation column if present)
    sv = shap_values[:, :n_feats]
    mean_abs = np.abs(sv).mean(axis=0)   # (n_feats,)

    # Dense rank: sort descending -> assign 1, 2, 3...
    order = np.argsort(mean_abs)[::-1]   # indices sorted by importance
    ranks = np.empty(n_feats, dtype=int)
    ranks[order] = np.arange(1, n_feats + 1)

    return {
        feature_names[i]: {
            "mean_abs_shap": float(mean_abs[i]),
            "rank": int(ranks[i]),
        }
        for i in range(n_feats)
    }


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list = None,
    title: str = "SHAP Beeswarm",
    out_path: Path = None,
):
    """
    Generate a SHAP beeswarm plot (summary plot).
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_names = feature_names or UNIT_CELL_FEATURES
    out_path = out_path or (FIGURES_DIR / "fig06a_shap_beeswarm.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False, plot_type="dot",
    )
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP beeswarm saved: %s", out_path)


def run_global_shap(save: bool = True) -> dict:
    """
    Full SHAP pipeline using the RIGID RF model (TreeExplainer -- fast).

    Runs on the full validated HF dataset.
    """
    from models.baselines.rigid_inverse import RIGIDInverse
    from utils.data_utils import train_test_split_stratified

    df = load_processed("hf")
    df_tr, df_te = train_test_split_stratified(df, test_frac=0.2)

    model = RIGIDInverse()
    model.fit(df_tr)

    X_test = df_te[UNIT_CELL_FEATURES].values

    shap_vals = compute_shap_rf(model._rf, X_test)
    # shap_vals is (N, 10) -- 9 geometry + 1 rotation.
    # Feature importance is computed on geometry features only.
    importance = shap_feature_importance(shap_vals)  # slices [:, :9] internally

    if save:
        # Save importance as JSON
        out = RESULTS_DIR / "shap_global_importance.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(importance, f, indent=2)
        log.info("SHAP importance saved: %s", out)

        # Beeswarm plot -- use only geometry columns
        plot_shap_beeswarm(
            shap_vals[:, :len(UNIT_CELL_FEATURES)],
            X_test[:, :len(UNIT_CELL_FEATURES)],
            feature_names=UNIT_CELL_FEATURES,
            title="Global SHAP -- CeSRR Geometry Features",
            out_path=FIGURES_DIR / "fig06a_shap_global.png",
        )

    log.info("Feature importance ranking:")
    for feat, info in sorted(importance.items(), key=lambda x: x[1]["rank"]):
        log.info("  Rank %2d: %-4s  |SHAP|=%.4f", info["rank"], feat, info["mean_abs_shap"])

    return importance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    imp = run_global_shap()
    print(json.dumps(imp, indent=2))
