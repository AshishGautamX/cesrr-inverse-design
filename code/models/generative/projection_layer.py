"""
projection_layer.py -- Isotonic projection layer enforcing r1 > r2 > r3 > r4.

This implements a *hard* constraint (vs the soft penalty in loss_physics.py).
At inference time, the decoder output for (r1, r2, r3, r4) is projected onto
the isotonic cone so that the ordering is always satisfied -- even if the
decoder initially violates it.

Method: Isotonic regression (Pool Adjacent Violators algorithm) applied in
        *descending* order to the first 4 output dimensions.

Reference:
  Barlow, R. E., et al. (1972) Statistical Inference under Order Restrictions.
  (Standard PAV algorithm for isotonic regression)
"""

import torch
import torch.nn as nn
import numpy as np


def isotonic_descending(x: torch.Tensor) -> torch.Tensor:
    """
    Project a batch of 1-D vectors onto the descending isotonic cone.

    That is, find ŷ = argmin_{y: y₁>=y₂>=...>=yₙ} ||y - x||2

    Uses the Pool Adjacent Violators (PAV) algorithm.

    Parameters
    ----------
    x : (B, K) tensor  -- we want each row to be non-increasing.

    Returns
    -------
    (B, K) tensor with each row sorted to be non-increasing.
    """
    # PAV works on numpy; run it sample-by-sample
    # Note: isotonic_regression with 'decreasing' is equivalent to
    #       reversing, running ascending regression, then reversing back.
    x_np = x.detach().cpu().numpy()
    out  = np.zeros_like(x_np)

    for i in range(x_np.shape[0]):
        out[i] = _pav_descending(x_np[i])

    result = torch.tensor(out, dtype=x.dtype, device=x.device)
    return result


def _pav_descending(y: np.ndarray) -> np.ndarray:
    """PAV algorithm for isotonic regression in descending order."""
    # Flip to ascending, run PAV, flip back
    y_asc  = y[::-1].copy()
    pooled = _pav_ascending(y_asc)
    return pooled[::-1].copy()


def _pav_ascending(y: np.ndarray) -> np.ndarray:
    """PAV algorithm for isotonic regression in ascending order."""
    n = len(y)
    # Work with a list of (level, count) blocks
    blocks = [[y[i], 1] for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            # Merge blocks i and i+1
            new_level = (
                blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            ) / (blocks[i][1] + blocks[i + 1][1])
            new_count = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [new_level, new_count]
            del blocks[i + 1]
            # Re-check merged block against previous
            if i > 0:
                i -= 1
        else:
            i += 1

    # Expand blocks back to array
    result = np.empty(n)
    pos = 0
    for level, count in blocks:
        result[pos:pos + count] = level
        pos += count
    return result


class IsotonicProjectionLayer(nn.Module):
    """
    Hard projection of decoder output for (r1, r2, r3, r4) onto the
    descending isotonic cone (r1 >= r2 >= r3 >= r4).

    Wraps a base decoder and post-processes only the first 4 output dimensions.
    The remaining 5 dimensions (t, d1, d2, h, p) pass through unchanged.

    Usage
    -----
    >>> proj = IsotonicProjectionLayer()
    >>> x_constrained = proj(x_raw)   # x_raw shape: (B, 9)
    """

    def __init__(self, min_gap: float = 0.1):
        """
        Parameters
        ----------
        min_gap : minimum enforced gap (mm) between consecutive radii after
                  projection. Prevents degenerate r1 ≈ r2 ≈ r3 ≈ r4.
        """
        super().__init__()
        self.min_gap = min_gap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 9) tensor with columns [r1, r2, r3, r4, t, d1, d2, h, p]

        Returns
        -------
        (B, 9) tensor with r1 >= r2 >= r3 >= r4 guaranteed.
        """
        radii   = x[:, :4]         # (B, 4)
        rest    = x[:, 4:]         # (B, 5)

        # Project radii onto descending cone
        radii_proj = isotonic_descending(radii)

        # Enforce minimum gap between consecutive radii
        if self.min_gap > 0:
            radii_proj = self._enforce_min_gap(radii_proj)

        return torch.cat([radii_proj, rest], dim=1)

    def _enforce_min_gap(self, radii: torch.Tensor) -> torch.Tensor:
        """
        After isotonic projection, nudge values to ensure
        r1 - r2 >= min_gap, r2 - r3 >= min_gap, r3 - r4 >= min_gap.
        Works from inner to outer.
        """
        r = radii.clone()
        # Enforce from r4 upward
        for i in range(2, -1, -1):   # i = 2,1,0 -> gaps r3<r2, r2<r1
            gap = r[:, i] - r[:, i + 1]
            deficit = torch.relu(self.min_gap - gap)
            r[:, i] = r[:, i] + deficit
        return r
