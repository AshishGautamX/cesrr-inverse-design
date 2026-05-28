"""
mc_dropout.py -- MC-Dropout uncertainty estimation for PI-cVAE.

At training time, dropout is standard regularisation.
At inference time, keep dropout *active* and run N forward passes.
The variance across passes is the epistemic uncertainty estimate.

This uncertainty score is the acquisition function for the AL loop
(uncertainty sampling strategy).

Reference:
  Gal & Ghahramani (2016) Dropout as a Bayesian Approximation.
  ICML 2016. arXiv:1506.02142
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import CVAE_MC_SAMPLES, UNIT_CELL_FEATURES
from utils.data_utils import CeSRRScaler, build_condition

log = logging.getLogger(__name__)


def enable_mc_dropout(model: torch.nn.Module):
    """Set model to training mode (activates dropout) for MC sampling."""
    model.train()


def disable_mc_dropout(model: torch.nn.Module):
    """Set model back to eval mode."""
    model.eval()


def mc_dropout_uncertainty(
    model: torch.nn.Module,
    df,
    scaler: CeSRRScaler,
    n_mc: int = CVAE_MC_SAMPLES,
    device: torch.device = None,
) -> np.ndarray:
    """
    Compute MC-Dropout uncertainty for each sample in df.

    Runs n_mc stochastic forward passes through the model encoder with
    dropout active. Returns the sum of variance across latent dimensions
    as a scalar uncertainty score per sample.

    Parameters
    ----------
    model   : PICVAENet (must have model.base.encoder)
    df      : DataFrame with geometry and condition columns
    scaler  : fitted CeSRRScaler
    n_mc    : number of MC samples
    device  : torch device

    Returns
    -------
    uncertainty : (N,) array, higher = more uncertain
    """
    import pandas as pd

    if device is None:
        device = next(model.parameters()).device

    # Handle unlabelled pool candidates (freq_ghz = NaN)
    # Fill NaN with midpoint of the scaler's target range
    import pandas as pd
    df_safe = df.copy()
    if df_safe["freq_ghz"].isna().any():
        midpoint = float(scaler.tgt_scaler.inverse_transform([[0.5]])[0, 0])
        df_safe["freq_ghz"] = df_safe["freq_ghz"].fillna(midpoint)

    X = torch.tensor(
        scaler.transform_features(df_safe), dtype=torch.float32, device=device)
    C = torch.tensor(
        build_condition(df_safe, scaler), dtype=torch.float32, device=device)

    # Activate dropout
    enable_mc_dropout(model)

    mu_samples = []
    with torch.no_grad():
        for _ in range(n_mc):
            mu_z, _ = model.base.encoder(X, C)
            mu_samples.append(mu_z.cpu().numpy())

    disable_mc_dropout(model)

    mu_stack = np.stack(mu_samples, axis=0)        # (n_mc, N, latent_dim)
    variance = mu_stack.var(axis=0).sum(axis=-1)   # (N,)
    log.debug("MC-Dropout uncertainty: mean=%.4f, max=%.4f", variance.mean(), variance.max())
    return variance


def mc_dropout_geometry_samples(
    model: torch.nn.Module,
    df,
    scaler: CeSRRScaler,
    n_mc: int = CVAE_MC_SAMPLES,
    device: torch.device = None,
) -> np.ndarray:
    """
    Generate geometry samples via MC-Dropout (stochastic decoder).

    Returns
    -------
    (N x n_mc, 9) array of geometry proposals in mm.
    """
    if device is None:
        device = next(model.parameters()).device

    C = torch.tensor(
        build_condition(df, scaler), dtype=torch.float32, device=device)

    enable_mc_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(n_mc):
            x_s = model.sample(C, n=1).squeeze(1)   # (N, 9)
            samples.append(x_s.cpu().numpy())
    disable_mc_dropout(model)

    X_all = np.concatenate(samples, axis=0)          # (N*n_mc, 9)
    return scaler.inverse_transform_features(X_all)
