"""
loss_physics.py — Physics-informed loss components for the PI-cVAE.

Three additive penalty terms:
  L_ordering : penalises violations of r1 > r2 > r3 > r4
  L_bounds   : penalises outputs outside [PARAM_BOUNDS_lo, PARAM_BOUNDS_hi]
  L_smooth   : optional smoothness regulariser on geometry gradients

All terms are differentiable — PyTorch autograd flows through them cleanly.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import PARAM_BOUNDS, UNIT_CELL_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute bound tensors once (moved to function to avoid device issues)
# ─────────────────────────────────────────────────────────────────────────────

def _get_bound_tensors(device: torch.device):
    """Return (lo, hi) tensors on the correct device."""
    lo = torch.tensor(
        [PARAM_BOUNDS[c][0] for c in UNIT_CELL_FEATURES],
        dtype=torch.float32, device=device,
    )
    hi = torch.tensor(
        [PARAM_BOUNDS[c][1] for c in UNIT_CELL_FEATURES],
        dtype=torch.float32, device=device,
    )
    return lo, hi


def ordering_loss(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Soft ordering constraint loss.

    Penalises any violation of r1 > r2 > r3 > r4.

    x : (B, 9) tensor of decoded geometry in original (mm) scale.
        Columns correspond to UNIT_CELL_FEATURES = [r1, r2, r3, r4, t, d1, d2, h, p]

    Uses ReLU so that the gradient is non-zero only when the constraint
    is violated, and exactly zero when it is satisfied.

    L_order = Σ ReLU(r_{i+1} - r_i + eps)   for i in {r1>r2, r2>r3, r3>r4}
    """
    r1 = x[:, 0]
    r2 = x[:, 1]
    r3 = x[:, 2]
    r4 = x[:, 3]

    viol_12 = torch.relu(r2 - r1 + eps)   # r1 should be > r2
    viol_23 = torch.relu(r3 - r2 + eps)
    viol_34 = torch.relu(r4 - r3 + eps)

    return (viol_12 + viol_23 + viol_34).mean()


def bounds_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Soft bounds constraint loss.

    Penalises any parameter outside its physical range.

    L_bounds = Σ_j [ReLU(lo_j - x_j) + ReLU(x_j - hi_j)]
    """
    lo, hi = _get_bound_tensors(x.device)
    below = torch.relu(lo - x)   # violation below lower bound
    above = torch.relu(x - hi)   # violation above upper bound
    return (below + above).mean()


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Optional: penalise very large absolute parameter values (Tikhonov-style).
    This discourages the decoder from producing extreme geometries.

    L_smooth = mean(x²)   — equivalent to L2 regularisation on outputs
    """
    return (x ** 2).mean()


class PhysicsLoss(nn.Module):
    """
    Combined physics loss: L = λ_order * L_order + λ_bounds * L_bounds

    Configurable weights allow sweeping ablations:
      - Set λ_order=0, λ_bounds=0 → plain ELBO (B4 baseline)
      - Set λ_order>0, λ_bounds=0 → soft ordering only (M2)
      - Set λ_order>0, λ_bounds>0 → full physics loss (M3/FULL)
    """

    def __init__(
        self,
        lambda_order: float = 10.0,
        lambda_bounds: float = 5.0,
        lambda_smooth: float = 0.0,
        ordering_eps: float = 1e-4,
    ):
        super().__init__()
        self.lambda_order  = lambda_order
        self.lambda_bounds = lambda_bounds
        self.lambda_smooth = lambda_smooth
        self.ordering_eps  = ordering_eps

    def forward(
        self,
        x_decoded: torch.Tensor,
    ) -> tuple:
        """
        Parameters
        ----------
        x_decoded : (B, 9) decoded geometry in original scale

        Returns
        -------
        (total_phys_loss, loss_dict)
        """
        L_order  = ordering_loss(x_decoded, self.ordering_eps)
        L_bounds = bounds_loss(x_decoded)
        L_smooth = smoothness_loss(x_decoded)

        total = (
            self.lambda_order  * L_order  +
            self.lambda_bounds * L_bounds +
            self.lambda_smooth * L_smooth
        )

        loss_dict = {
            "L_order":  L_order.item(),
            "L_bounds": L_bounds.item(),
            "L_smooth": L_smooth.item(),
            "L_phys":   total.item(),
        }
        return total, loss_dict
