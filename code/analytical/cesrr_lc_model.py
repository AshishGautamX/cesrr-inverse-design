"""
cesrr_lc_model.py -- Analytical LC equivalent-circuit model for circular CeSRR.

Based on:
  - Baena et al. (2005) IEEE TMTT 53(4):1451-1461 (DOI: 10.1109/TMTT.2005.845211)
  - Marqués et al. (2002) PRL 89:183901 (DOI: 10.1103/PhysRevLett.89.183901)

Physical picture
----------------
The complementary electric split-ring resonator (CeSRR) is modelled as a
parallel LC tank formed by:
  L_eff : effective inductance from the metallic ring structure
  C_eff : effective capacitance from the groove between rings

The resonant frequency is:
  f0 = 1 / (2π √(L_eff · C_eff))

This model is intentionally approximate (low-fidelity). It captures the
correct scaling laws and is used as the LF source in multi-fidelity GP.

All dimensions are in mm; output frequency is in GHz.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# Physical constants
MU0    = 4 * np.pi * 1e-7     # H/m
EPS0   = 8.854187817e-12      # F/m
C_LIGHT = 3e8                  # m/s


@dataclass
class CeSRRParams:
    """Container for a single CeSRR unit-cell geometry."""
    r1: float   # Outer radius of outer ring (mm)
    r2: float   # Inner radius of outer ring (mm)
    r3: float   # Outer radius of inner ring (mm)
    r4: float   # Inner radius of inner ring (mm)
    t:  float   # Conductor thickness (mm)
    d1: float   # Outer groove width, mm  (r1 - r2 nominally)
    d2: float   # Inner groove width, mm  (r3 - r4 nominally)
    h:  float   # Unit-cell height / period in axial direction (mm)
    p:  float   # Periodicity (mm)


def inductance_circular_loop(R_mm: float, w_mm: float) -> float:
    """
    Inductance of a thin circular conducting loop (Neumann formula approximation).

    Parameters
    ----------
    R_mm : mean ring radius (mm)
    w_mm : ring width = outer_radius - inner_radius (mm)

    Returns
    -------
    L in Henries
    """
    R = R_mm * 1e-3   # convert to m
    w = max(w_mm * 1e-3, 1e-9)   # avoid log(0)

    # Neumann approximation for a thin ring: L ≈ μ0 R [ln(8R/w) - 2]
    # Valid when R >> w (thin-ring limit)
    arg = 8 * R / w
    if arg <= 1:
        arg = 1 + 1e-9
    L = MU0 * R * (np.log(arg) - 2.0)
    return max(L, 1e-15)  # clamp to physical positive value


def capacitance_circular_groove(R_mean_mm: float, gap_mm: float, t_mm: float) -> float:
    """
    Effective gap capacitance of the CeSRR groove.

    Approximated as a parallel-plate capacitor rolled into a circle:
      C = ε0 · (circumference x t) / gap

    Parameters
    ----------
    R_mean_mm : mean radius of the groove (mm)
    gap_mm    : groove width (mm)
    t_mm      : conductor/slot thickness (mm)

    Returns
    -------
    C in Farads
    """
    R   = R_mean_mm * 1e-3
    g   = max(gap_mm * 1e-3, 1e-9)
    t   = max(t_mm  * 1e-3, 1e-9)
    circumference = 2 * np.pi * R
    C = EPS0 * (circumference * t) / g
    return max(C, 1e-18)


def cesrr_resonant_frequency(
    r1: float, r2: float, r3: float, r4: float,
    t: float, d1: float = None, d2: float = None,
    h: float = None, p: float = None,
) -> Optional[float]:
    """
    Compute the analytical resonant frequency estimate for a CeSRR unit cell.

    The CeSRR is modelled as two coupled circular LC resonators in series:
      - Outer ring: radius R_out = (r1+r2)/2, width w_out = r1-r2
      - Inner ring: radius R_in  = (r3+r4)/2, width w_in  = r3-r4

    Capacitance is formed in the groove between r2 and r3 (primary gap).

    Parameters
    ----------
    r1, r2, r3, r4 : radii in mm (r1 > r2 > r3 > r4)
    t              : conductor thickness in mm
    d1, d2         : groove widths (mm); if None, inferred from r1-r2, r3-r4
    h, p           : height and periodicity (not used in base LC model;
                     reserved for dispersion correction)

    Returns
    -------
    f0 in GHz, or None if inputs are unphysical.
    """
    # Validate ordering
    if not (r1 > r2 > r3 > r4 > 0):
        return None
    if t <= 0:
        return None

    # Groove widths (use explicit values if provided, else infer)
    g_outer = d1 if (d1 is not None and d1 > 0) else (r1 - r2)
    g_inner = d2 if (d2 is not None and d2 > 0) else (r3 - r4)

    # Mean radii
    R_out = (r1 + r2) / 2.0
    R_in  = (r3 + r4) / 2.0
    w_out = r1 - r2
    w_in  = r3 - r4

    # Inductances (series coupling of two ring loops)
    L_out = inductance_circular_loop(R_out, w_out)
    L_in  = inductance_circular_loop(R_in,  w_in)
    L_eff = L_out + L_in

    # Capacitance: primary gap is between r2 (inner edge of outer ring)
    # and r3 (outer edge of inner ring); use mean radius of this gap region
    R_gap = (r2 + r3) / 2.0
    gap   = r2 - r3  # radial gap between rings
    if gap <= 0:
        return None
    C_eff = capacitance_circular_groove(R_gap, gap, t)

    # Resonant frequency
    omega0 = 1.0 / np.sqrt(L_eff * C_eff)
    f0_hz  = omega0 / (2 * np.pi)
    f0_ghz = f0_hz / 1e9

    # Sanity check: clamp to physically plausible range
    if not (0.1 <= f0_ghz <= 100.0):
        return None

    return f0_ghz


def batch_predict(df) -> "np.ndarray":
    """
    Vectorised prediction for a DataFrame with columns r1..r4, t, [d1, d2, h, p].

    Returns array of predicted frequencies (GHz). NaN for invalid rows.
    """
    results = []
    for _, row in df.iterrows():
        f = cesrr_resonant_frequency(
            r1=row.get("r1"), r2=row.get("r2"),
            r3=row.get("r3"), r4=row.get("r4"),
            t=row.get("t"),
            d1=row.get("d1"), d2=row.get("d2"),
            h=row.get("h"), p=row.get("p"),
        )
        results.append(f if f is not None else np.nan)
    return np.array(results)


# -----------------------------------------------------------------------------
# Standalone test / demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Reference point from Wang et al. (2015): ~2.454 GHz at r1=20mm scale
    # (The paper uses a waveguide-scale structure; dimensions below are scaled
    # to match the unit-cell regime in our dataset.)
    test_cases = [
        dict(r1=42.8, r2=40.8, r3=34.8, r4=20.0, t=1.0, d1=2.0, d2=14.8),
        dict(r1=20.0, r2=18.0, r3=12.0, r4=5.0,  t=0.5, d1=2.0, d2=7.0),
        dict(r1=6.7,  r2=4.7,  r3=3.5,  r4=2.0,  t=0.5, d1=2.0, d2=1.5),
    ]

    print(f"{'r1':>6} {'r2':>6} {'r3':>6} {'r4':>6} | {'f0_GHz':>10}")
    print("-" * 50)
    for tc in test_cases:
        f0 = cesrr_resonant_frequency(**tc)
        print(
            f"{tc['r1']:6.1f} {tc['r2']:6.1f} {tc['r3']:6.1f} {tc['r4']:6.1f} "
            f"| {f0:10.3f} GHz" if f0 else "| INVALID"
        )
