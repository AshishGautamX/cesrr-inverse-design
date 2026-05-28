"""
query_strategy.py -- Acquisition functions for active learning.

Strategies implemented:
  1. uncertainty_sampling : select points with highest MC-Dropout variance
  2. bald                 : Bayesian Active Learning by Disagreement
                            (maximise mutual information between predictions and model params)
  3. expected_improvement : BO-style EI over the GP posterior

The BALD acquisition is the primary strategy (highest information gain).
Uncertainty sampling is the fallback (simpler, nearly as effective).

Reference:
  Houlsby et al. (2011) Bayesian Active Learning for Classification and
  Preference Learning. arXiv:1112.5745.
"""

import sys
import logging
from pathlib import Path
from typing import Literal

import numpy as np

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import AL_N_QUERY_PER_ROUND, CVAE_MC_SAMPLES

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Uncertainty sampling
# -----------------------------------------------------------------------------

def uncertainty_sampling(
    uncertainties: np.ndarray,
    n_query: int = AL_N_QUERY_PER_ROUND,
    exclude_idx: set = None,
) -> np.ndarray:
    """
    Select indices of the n_query most uncertain unlabelled candidates.

    Parameters
    ----------
    uncertainties : (N,) array of uncertainty scores (higher = more uncertain)
    n_query       : number of points to select
    exclude_idx   : set of indices already labelled (skipped)

    Returns
    -------
    (n_query,) array of selected indices into the unlabelled pool
    """
    if exclude_idx:
        mask = np.ones(len(uncertainties), dtype=bool)
        for i in exclude_idx:
            mask[i] = False
        scores = np.where(mask, uncertainties, -np.inf)
    else:
        scores = uncertainties

    selected = np.argsort(scores)[::-1][:n_query]
    log.debug(
        "Uncertainty sampling: top-%d scores = %s",
        n_query, scores[selected].round(4)
    )
    return selected


# -----------------------------------------------------------------------------
# 2. BALD (Bayesian Active Learning by Disagreement)
# -----------------------------------------------------------------------------

def bald_score(
    mc_predictions: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute BALD acquisition scores from MC-Dropout samples.

    BALD(x) = H[ȳ] - E_{θ}[H[y|θ]]
             ≈ entropy of mean prediction - mean entropy of individual predictions

    For regression (continuous output), we approximate using:
      BALD ≈ Var[ȳ] - mean_sample_variance   (information gain proxy)

    This reduces to uncertainty sampling for Gaussian likelihoods.

    Parameters
    ----------
    mc_predictions : (n_mc, N, D) MC-Dropout predictions
                     (n_mc forward passes, N samples, D output dims)

    Returns
    -------
    (N,) BALD score (higher = more informative to label)
    """
    # Total variance (epistemic + aleatoric)
    total_var = mc_predictions.var(axis=0)        # (N, D)

    # Mean of per-sample variance (aleatoric only)
    # For regression without separate noise model, approximate as 0
    # BALD ≈ total variance (all epistemic in dropout regime)
    bald = total_var.sum(axis=-1)                 # (N,)
    return bald


def bald_acquisition(
    mc_predictions: np.ndarray,
    n_query: int = AL_N_QUERY_PER_ROUND,
    exclude_idx: set = None,
) -> np.ndarray:
    """
    Select n_query points maximising BALD score.

    Parameters
    ----------
    mc_predictions : (n_mc, N, D) array from MC-Dropout forward passes
    n_query        : number of points to select
    exclude_idx    : already-labelled indices to skip

    Returns
    -------
    (n_query,) selected indices
    """
    scores = bald_score(mc_predictions)
    return uncertainty_sampling(scores, n_query=n_query, exclude_idx=exclude_idx)


# -----------------------------------------------------------------------------
# 3. GP Expected Improvement (for GP-based AL)
# -----------------------------------------------------------------------------

def expected_improvement(
    mu: np.ndarray,
    std: np.ndarray,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected improvement acquisition function for GP-based models.

    EI(x) = E[max(f(x) - y_best, 0)]
           = (μ(x) - y_best - ξ)·Φ(Z) + σ(x)·φ(Z)
    where Z = (μ(x) - y_best - ξ) / σ(x)

    Used when GP_surrogate or MF_GP is the model in the AL loop.

    Parameters
    ----------
    mu    : (N,) GP predictive mean
    std   : (N,) GP predictive standard deviation
    y_best: current best observed frequency (scalar)
    xi    : exploration-exploitation trade-off (default 0.01)

    Returns
    -------
    (N,) EI values
    """
    from scipy.stats import norm

    std = np.clip(std, 1e-9, None)
    Z   = (mu - y_best - xi) / std
    ei  = (mu - y_best - xi) * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std < 1e-8] = 0.0
    return ei


def ei_acquisition(
    mu: np.ndarray,
    std: np.ndarray,
    y_best: float,
    n_query: int = AL_N_QUERY_PER_ROUND,
    exclude_idx: set = None,
) -> np.ndarray:
    """Select n_query points maximising Expected Improvement."""
    scores = expected_improvement(mu, std, y_best)
    return uncertainty_sampling(scores, n_query=n_query, exclude_idx=exclude_idx)


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------

def select_query_points(
    strategy: Literal["uncertainty", "bald", "ei"],
    n_query: int = AL_N_QUERY_PER_ROUND,
    exclude_idx: set = None,
    **kwargs,
) -> np.ndarray:
    """
    Dispatch to the appropriate acquisition function.

    kwargs for 'uncertainty': uncertainties
    kwargs for 'bald'       : mc_predictions
    kwargs for 'ei'         : mu, std, y_best
    """
    if strategy == "uncertainty":
        return uncertainty_sampling(
            kwargs["uncertainties"], n_query, exclude_idx)
    elif strategy == "bald":
        return bald_acquisition(
            kwargs["mc_predictions"], n_query, exclude_idx)
    elif strategy == "ei":
        return ei_acquisition(
            kwargs["mu"], kwargs["std"], kwargs["y_best"], n_query, exclude_idx)
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy!r}")
