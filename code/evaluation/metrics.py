"""
metrics.py -- All evaluation metrics for the ablation study.

Four primary metrics (paper Table):
  1. Frequency prediction MAE (GHz) -- forward model accuracy
  2. Geometry feasibility rate (%) -- fraction satisfying r1>r2>r3>r4
  3. Design diversity (std mm) -- variety of generated geometries at fixed freq
  4. AL efficiency -- computed separately in al_metrics.py

Consistency: all metrics use the same functions regardless of which model
is being evaluated. This ensures fair comparison across the 9 ablation configs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import UNIT_CELL_FEATURES, TARGET_COL, ORDERING_COLS


# -----------------------------------------------------------------------------
# 1. Forward prediction metrics (regression)
# -----------------------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Compute MAE, RMSE, R2, and MaxError for frequency predictions.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays in GHz
    model_name     : label for the output dict

    Returns
    -------
    dict with keys: model, MAE_GHz, RMSE_GHz, R2, MaxError_GHz, N
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."

    mae  = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-12))
    maxe = float(np.max(np.abs(y_pred - y_true)))

    return {
        "model":         model_name,
        "N":             len(y_true),
        "MAE_GHz":       round(mae,  4),
        "RMSE_GHz":      round(rmse, 4),
        "R2":            round(r2,   4),
        "MaxError_GHz":  round(maxe, 4),
    }


# -----------------------------------------------------------------------------
# 2. Geometry feasibility rate
# -----------------------------------------------------------------------------

def geometry_feasibility_rate(X: np.ndarray) -> float:
    """
    Fraction of geometry samples satisfying r1 > r2 > r3 > r4.

    Parameters
    ----------
    X : (N, 9) array with columns in UNIT_CELL_FEATURES order
        [r1, r2, r3, r4, t, d1, d2, h, p]

    Returns
    -------
    float in [0, 1]
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # Columns 0,1,2,3 are r1,r2,r3,r4
    valid = (
        (X[:, 0] > X[:, 1]) &
        (X[:, 1] > X[:, 2]) &
        (X[:, 2] > X[:, 3])
    )
    return float(valid.mean())


# -----------------------------------------------------------------------------
# 3. Design diversity
# -----------------------------------------------------------------------------

def design_diversity(X: np.ndarray) -> dict:
    """
    Measure diversity of a set of generated geometry proposals.

    Metrics:
      - mean_std_mm  : average standard deviation across all 9 params (mm)
      - r1_std_mm    : std of outer radius specifically
      - freq_spread  : max-min range of r1 (proxy for freq spread)

    Parameters
    ----------
    X : (N, 9) geometry array in mm

    Returns
    -------
    dict of diversity metrics
    """
    X = np.asarray(X)
    std_per_dim = X.std(axis=0)
    return {
        "mean_std_mm": float(std_per_dim.mean()),
        "r1_std_mm":   float(std_per_dim[0]),
        "r4_std_mm":   float(std_per_dim[3]),
        "p_std_mm":    float(std_per_dim[8]),
        "r1_range_mm": float(X[:, 0].max() - X[:, 0].min()),
    }


def diversity_at_fixed_freq(
    sample_fn,          # callable(df_query) -> (N*n_samples, 9) geometry array
    df_query: pd.DataFrame,
    n_samples: int = 50,
) -> dict:
    """
    Measure geometric diversity among samples generated for a fixed target freq.

    Useful for Figure 7 of the paper.
    """
    X_samples = sample_fn(df_query)
    feas = geometry_feasibility_rate(X_samples)
    div  = design_diversity(X_samples)
    return {"feasibility_rate": feas, **div}


# -----------------------------------------------------------------------------
# 4. Ablation summary table (9 configs x 4 metrics)
# -----------------------------------------------------------------------------

ABLATION_CONFIG_NAMES = [
    "B1_GP_Matern",
    "B2_RIGID",
    "B3_Tandem",
    "B4_cVAE_noPhysics",
    "M1_MFGP",
    "M2_PIcVAE_soft",
    "M3_PIcVAE_hard_soft",
    "M4_M3_MF",
    "FULL_M4_AL",
]


def build_ablation_table(results: list) -> pd.DataFrame:
    """
    Build a summary DataFrame from a list of per-model result dicts.

    Column mapping:
      MAE_GHz        : forward prediction error (GP/MF-GP/RIGID only)
      Recon_MAE_GHz  : frequency error of generated geometries via GP oracle
                       (all inverse/generative models: Tandem, cVAE, PI-cVAE)
      Feasibility_%  : fraction of generated geometries satisfying r1>r2>r3>r4
      Diversity_std  : std of generated geometry parameters (mm)

    NOTE: Recon_MAE_GHz is the common metric that allows cross-model comparison.
    Forward models have MAE_GHz; generative models have Recon_MAE_GHz + Feasibility.
    """
    rows = []
    for r in results:
        name = r.get("model", "?")

        # Feasibility: check both keys
        feas = r.get("feasibility_rate", r.get("feasibility_rate_mcmc", np.nan))
        try:
            feas_pct = round(float(feas) * 100, 1) if feas is not None and not (isinstance(feas, float) and np.isnan(feas)) else np.nan
        except (TypeError, ValueError):
            feas_pct = np.nan

        # Diversity
        div = r.get("geometry_diversity_std", r.get("mean_std_mm", np.nan))

        # Reconstruction MAE: the COMMON metric across all inverse models
        recon_mae = r.get("reconstruction_mae_ghz", np.nan)

        rows.append({
            "Model":           name,
            "MAE_GHz":         r.get("MAE_GHz",  np.nan),   # forward surrogate models
            "RMSE_GHz":        r.get("RMSE_GHz", np.nan),
            "R2":              r.get("R2",        np.nan),
            "Recon_MAE_GHz":   recon_mae,                    # inverse / generative models
            "Feasibility_%":   feas_pct,
            "Diversity_std":   div,
            "Runtime_s":       r.get("runtime_s", np.nan),
        })
    df = pd.DataFrame(rows).set_index("Model")
    return df


