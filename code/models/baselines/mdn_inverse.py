"""
mdn_inverse.py -- Mixture Density Network baseline (B-MDN).

Maps (freq, rotation) -> parameters of a Gaussian Mixture Model over geometry.
Solves the one-to-many problem by explicitly modelling the multi-modal
distribution P(geometry | frequency, rotation).

Architecture:
  - Shared encoder: MLP([freq, rot]) -> hidden features
  - Three output heads per mixture component k:
      π_k  : mixing weights   (softmax)
      μ_k  : means            (linear)
      σ_k  : std deviations   (softplus for positivity)

Sampling: draw component k ~ Categorical(π), then sample x ~ N(μ_k, σ_k2)
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import (
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    MDN_N_GAUSSIANS, MDN_HIDDEN, TANDEM_LR,
    TANDEM_MAX_EPOCHS, TANDEM_PATIENCE, TANDEM_BATCH_SIZE,
    RESULTS_DIR, GLOBAL_SEED
)
from utils.data_utils import (CeSRRScaler, build_condition,
                              train_test_split_stratified, load_processed)
from utils.train_utils import (EarlyStopping, DEVICE, set_seed, save_checkpoint,
                               build_mlp)
from evaluation.metrics import geometry_feasibility_rate

log = logging.getLogger(__name__)


class MDNHead(nn.Module):
    """MDN output head: produces π, μ, σ for K mixture components."""

    def __init__(self, hidden_dim: int, out_dim: int, n_gaussians: int):
        super().__init__()
        self.K = n_gaussians
        self.D = out_dim
        self.pi_head    = nn.Linear(hidden_dim, n_gaussians)
        self.mu_head    = nn.Linear(hidden_dim, n_gaussians * out_dim)
        self.sigma_head = nn.Linear(hidden_dim, n_gaussians * out_dim)

    def forward(self, h: torch.Tensor):
        """
        Returns
        -------
        pi    : (B, K) mixing weights (sum-to-1 via softmax)
        mu    : (B, K, D) component means
        sigma : (B, K, D) component std deviations (positive via softplus)
        """
        B = h.size(0)
        pi    = F.softmax(self.pi_head(h), dim=-1)                    # (B, K)
        mu    = self.mu_head(h).view(B, self.K, self.D)               # (B, K, D)
        sigma = F.softplus(self.sigma_head(h)).view(B, self.K, self.D) + 1e-6
        return pi, mu, sigma


class MDNNet(nn.Module):
    """Full MDN: encoder MLP + MDN head."""

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 9,
        hidden: list = None,
        n_gaussians: int = MDN_N_GAUSSIANS,
    ):
        super().__init__()
        hidden = hidden or MDN_HIDDEN
        dims   = [in_dim] + hidden
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]),
                       nn.BatchNorm1d(dims[i+1]), nn.ReLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)
        self.head    = MDNHead(hidden[-1], out_dim, n_gaussians)

    def forward(self, c: torch.Tensor):
        h = self.encoder(c)
        return self.head(h)


def mdn_nll_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood loss for MDN.

    L = -log Σ_k π_k N(target | μ_k, σ_k2)

    Parameters
    ----------
    pi     : (B, K)
    mu     : (B, K, D)
    sigma  : (B, K, D)
    target : (B, D) ground truth geometry (scaled)
    """
    B, K, D = mu.shape
    target_exp = target.unsqueeze(1).expand_as(mu)   # (B, K, D)

    # Log-probability of target under each Gaussian component
    log_gauss = -0.5 * (
        ((target_exp - mu) / sigma) ** 2 +
        2 * torch.log(sigma) +
        np.log(2 * np.pi)
    ).sum(dim=-1)   # (B, K)

    # Mix: log Σ π_k exp(log_gauss_k)  = logsumexp over k
    log_pi     = torch.log(pi + 1e-8)             # (B, K)
    log_mix    = torch.logsumexp(log_pi + log_gauss, dim=-1)   # (B,)
    return -log_mix.mean()


class MDNInverseModel:
    """Scikit-learn-style wrapper for MDN inverse model."""

    def __init__(self, n_gaussians: int = MDN_N_GAUSSIANS, hidden: list = None):
        self.n_gaussians = n_gaussians
        self.hidden      = hidden or MDN_HIDDEN
        self.scaler: CeSRRScaler | None = None
        self._model: MDNNet | None = None

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> "MDNInverseModel":
        set_seed(GLOBAL_SEED)
        self.scaler = CeSRRScaler().fit(df_train)

        C_train = build_condition(df_train, self.scaler)
        X_train = self.scaler.transform_features(df_train)

        if df_val is None:
            n_val   = max(1, int(0.15 * len(df_train)))
            C_val, X_val     = C_train[-n_val:], X_train[-n_val:]
            C_train, X_train = C_train[:-n_val], X_train[:-n_val]
        else:
            C_val = build_condition(df_val, self.scaler)
            X_val = self.scaler.transform_features(df_val)

        self._model = MDNNet(
            in_dim=2, out_dim=9,
            hidden=self.hidden,
            n_gaussians=self.n_gaussians,
        ).to(DEVICE)
        opt   = torch.optim.Adam(self._model.parameters(), lr=TANDEM_LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)
        es    = EarlyStopping(patience=TANDEM_PATIENCE)

        from torch.utils.data import DataLoader, TensorDataset
        Ct  = torch.tensor(C_train, dtype=torch.float32, device=DEVICE)
        Xt  = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
        Cv  = torch.tensor(C_val,   dtype=torch.float32, device=DEVICE)
        Xv  = torch.tensor(X_val,   dtype=torch.float32, device=DEVICE)
        loader = DataLoader(TensorDataset(Ct, Xt), batch_size=TANDEM_BATCH_SIZE, shuffle=True)

        log.info("Training MDN (%d gaussians, %d samples)...", self.n_gaussians, len(C_train))
        for epoch in range(1, TANDEM_MAX_EPOCHS + 1):
            self._model.train()
            for cb, xb in loader:
                opt.zero_grad()
                pi, mu, sigma = self._model(cb)
                loss = mdn_nll_loss(pi, mu, sigma, xb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                opt.step()

            self._model.eval()
            with torch.no_grad():
                pi, mu, sigma = self._model(Cv)
                val_loss = mdn_nll_loss(pi, mu, sigma, Xv).item()
            sched.step(val_loss)
            if epoch % 50 == 0:
                log.info("  Epoch %d | val_NLL=%.4f", epoch, val_loss)
            if es(val_loss, self._model):
                break
        es.restore_best(self._model)
        return self

    def sample_geometry(self, df: pd.DataFrame, n_samples: int = 1) -> np.ndarray:
        """
        Sample n_samples geometry proposals for each row in df.

        Returns
        -------
        (N x n_samples, 9) array in mm scale.
        """
        C = build_condition(df, self.scaler)
        Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)
        self._model.eval()
        with torch.no_grad():
            pi, mu, sigma = self._model(Ct)   # (N, K, 9)

        pi_np = pi.cpu().numpy()
        mu_np = mu.cpu().numpy()
        sg_np = sigma.cpu().numpy()

        all_samples = []
        rng = np.random.default_rng(GLOBAL_SEED)
        for i in range(len(df)):
            for _ in range(n_samples):
                k = rng.choice(self.n_gaussians, p=pi_np[i])
                sample = mu_np[i, k] + sg_np[i, k] * rng.standard_normal(9)
                all_samples.append(sample)

        X_scaled = np.array(all_samples)
        return self.scaler.inverse_transform_features(X_scaled)

    def evaluate(self, df_test: pd.DataFrame, n_samples: int = 10) -> dict:
        X_samples = self.sample_geometry(df_test, n_samples=n_samples)
        feas = geometry_feasibility_rate(X_samples)
        return {
            "model": "MDN_Inverse",
            "feasibility_rate": feas,
            "n_samples_per_query": n_samples,
        }

    def save(self, path: Path = None):
        path = path or (RESULTS_DIR / "mdn_inverse.pt")
        save_checkpoint(self._model, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_processed("all")
    df_tr, df_te = train_test_split_stratified(df, test_frac=0.2)
    m = MDNInverseModel()
    m.fit(df_tr)
    print(json.dumps(m.evaluate(df_te), indent=2))
