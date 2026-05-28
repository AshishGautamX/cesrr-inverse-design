"""
train_utils.py -- Reusable PyTorch training utilities.
"""

import time
import logging
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        log.info("Using CPU.")
    return dev


DEVICE = get_device()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Stop training when val_loss stops improving. Saves best weights."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = deepcopy(model.state_dict())
        else:
            self.counter += 1
        if self.counter >= self.patience:
            log.info(f"Early stopping. Best val_loss={self.best_loss:.6f}")
            return True
        return False

    def restore_best(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def save_checkpoint(model: nn.Module, path: Path, extra: dict = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    log.info(f"Checkpoint saved: {path}")


def load_checkpoint(model: nn.Module, path: Path) -> dict:
    path = Path(path)
    payload = torch.load(path, map_location=DEVICE)
    model.load_state_dict(payload.pop("model_state"))
    return payload


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: list,
    dropout_p: float = 0.0,
) -> nn.Sequential:
    """Build a fully-connected MLP with BatchNorm and ReLU activations."""
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
        if dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def train_model(
    model: nn.Module,
    optimizer,
    loss_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 500,
    batch_size: int = 32,
    patience: int = 30,
    scheduler=None,
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> dict:
    """Generic supervised regression training loop."""
    model.to(device)

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    Xt, yt = _t(X_train), _t(y_train)
    Xv, yv = _t(X_val),   _t(y_val)
    loader  = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    es = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            if yb.ndim == 1:
                yb = yb.unsqueeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(Xt)

        model.eval()
        with torch.no_grad():
            yv_ = yv.unsqueeze(-1) if yv.ndim == 1 else yv
            vl = loss_fn(model(Xv), yv_).item()

        history["train_loss"].append(ep_loss)
        history["val_loss"].append(vl)
        if scheduler:
            scheduler.step(vl)
        if verbose and epoch % 50 == 0:
            log.info(f"Epoch {epoch:4d} | train={ep_loss:.5f} val={vl:.5f}")
        if es(vl, model):
            break

    es.restore_best(model)
    return history
