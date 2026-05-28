"""
loss_physics.py -- Physics-informed loss components for the PI-cVAE.

Two additive penalty terms (both operate on MinMax-SCALED [0,1] tensors):
  L_ordering : penalises violations of r1 > r2 > r3 > r4  (scale-invariant)
  L_bounds   : penalises outputs outside [0, 1]
  L_smooth   : optional L2 regulariser on outputs

NOTE ON SCALE:
  The cVAE decoder outputs are in [0, 1] MinMax-scaled space.
  - ordering_loss: scale-invariant (MinMax preserves relative ordering)
  - bounds_loss  : checks [0, 1] -- since MinMax maps [lo_mm, hi_mm] -> [0,1],
    any value outside [0,1] in scaled space is outside physical bounds in mm.
  Do NOT pass mm-scale tensors to these functions.

All terms are differentiable -- PyTorch autograd flows cleanly.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))


# -----------------------------------------------------------------------------
# Individual loss terms
# -----------------------------------------------------------------------------

def ordering_loss(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Soft ordering constraint loss.

    Penalises any violation of r1 > r2 > r3 > r4.

    x   : (B, 9) tensor -- [0,1] scaled. Column order = UNIT_CELL_FEATURES
          [r1, r2, r3, r4, t, d1, d2, h, p]
    eps : small margin to push toward strict inequality.

    L_order = mean(ReLU(r2-r1+eps) + ReLU(r3-r2+eps) + ReLU(r4-r3+eps))
    """
    r1 = x[:, 0]
    r2 = x[:, 1]
    r3 = x[:, 2]
    r4 = x[:, 3]

    viol_12 = torch.relu(r2 - r1 + eps)    # r1 should be > r2
    viol_23 = torch.relu(r3 - r2 + eps)
    viol_34 = torch.relu(r4 - r3 + eps)

    return (viol_12 + viol_23 + viol_34).mean()


def bounds_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Soft bounds constraint loss for [0, 1] MinMax-scaled inputs.

    Since CeSRRScaler.transform_features maps each param's physical range
    [lo_mm, hi_mm] -> [0, 1], a value outside [0, 1] means outside bounds.

    L_bounds = mean(ReLU(-x) + ReLU(x - 1))
    """
    below = torch.relu(-x)         # violation below 0
    above = torch.relu(x - 1.0)   # violation above 1
    return (below + above).mean()


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    """
    L2 regularisation on decoder outputs. Discourages extreme values.
    L_smooth = mean(x^2)
    """
    return (x ** 2).mean()


# -----------------------------------------------------------------------------
# Combined physics loss module
# -----------------------------------------------------------------------------

class PhysicsLoss(nn.Module):
    """
    Combined physics loss: L = lambda_order * L_order + lambda_bounds * L_bounds

    Configurable weights for ablation study:
      lambda_order=0, lambda_bounds=0  ->  plain ELBO (B4 baseline)
      lambda_order>0, lambda_bounds=0  ->  ordering only (M2)
      lambda_order>0, lambda_bounds>0  ->  full physics loss (M3/FULL)

    Input: x_decoded (B, 9) MinMax-scaled decoder output in [0, 1]
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

    def forward(self, x_decoded: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        x_decoded : (B, 9) MinMax-scaled decoded geometry tensor [0, 1]

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
