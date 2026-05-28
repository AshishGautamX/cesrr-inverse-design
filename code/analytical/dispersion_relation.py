"""
dispersion_relation.py — Bloch-Floquet dispersion for CeSRR-loaded waveguide.

Computes the normalised phase advance β·p vs frequency curve for a periodic
CeSRR-loaded circular waveguide, based on the transmission-matrix (ABCD) method.

This gives:
  1. The dispersion diagram (β·p vs ω) — useful for bandwidth estimation
  2. The slow-wave factor (SWF) = c/vp = c·β/ω — design metric
  3. The group velocity vg = dω/dβ — identifies backward-wave regime

These quantities supplement the LC resonant frequency as additional
physics-informed features that can be fed into the MF-GP.

References:
  Pozar, D. (2012) Microwave Engineering, 4th ed. (Ch. 8, periodic structures)
  Wang et al. (2015) APL 107:153502 (DOI: 10.1063/1.4933106)
"""

import numpy as np
from typing import Optional


# Physical constants
C_LIGHT = 2.998e8   # m/s
MU0     = 4 * np.pi * 1e-7
EPS0    = 8.854e-12


def waveguide_cutoff_freq(R_wg_mm: float, mode: str = "TE11") -> float:
    """
    Cutoff frequency of a hollow circular waveguide.

    Parameters
    ----------
    R_wg_mm : inner radius of waveguide (mm)
    mode    : 'TE11' (dominant) or 'TM01'

    Returns
    -------
    fc in GHz
    """
    R = R_wg_mm * 1e-3
    # First zeros of Bessel functions: TE11 → 1.841, TM01 → 2.405
    chi = {"TE11": 1.8412, "TM01": 2.4048}.get(mode, 1.8412)
    fc = (C_LIGHT * chi) / (2 * np.pi * R)
    return fc / 1e9


def transfer_matrix_unit_cell(
    f_ghz: float,
    r1: float, r2: float, r3: float, r4: float,
    t: float, p: float,
) -> Optional[np.ndarray]:
    """
    Approximate ABCD transfer matrix for one CeSRR unit cell.

    Models the unit cell as a shunt LC resonator in a transmission line:
      [A B]   [1    0  ]   [1  Z0·Δl/λ]
      [C D] = [Y_sh 1  ] × [0      1   ]

    where Y_sh = 1/(jωL_eff - 1/(jωC_eff)) is the shunt admittance of the
    CeSRR resonator.

    Parameters
    ----------
    f_ghz : frequency in GHz
    r1..r4, t, p : CeSRR geometry (mm)

    Returns
    -------
    2×2 complex ABCD matrix, or None if unphysical.
    """
    from analytical.cesrr_lc_model import (
        inductance_circular_loop,
        capacitance_circular_groove,
    )

    if not (r1 > r2 > r3 > r4 > 0):
        return None

    omega = 2 * np.pi * f_ghz * 1e9

    # LC elements (from the analytical model)
    R_out = (r1 + r2) / 2
    R_in  = (r3 + r4) / 2
    L_eff = (inductance_circular_loop(R_out, r1 - r2) +
             inductance_circular_loop(R_in,  r3 - r4))

    R_gap = (r2 + r3) / 2
    gap   = r2 - r3
    if gap <= 0:
        return None
    C_eff = capacitance_circular_groove(R_gap, gap, t)

    # Shunt impedance of CeSRR resonator
    Z_res = 1j * omega * L_eff + 1.0 / (1j * omega * C_eff)

    # Characteristic impedance of the waveguide section (approximate)
    # For a circular WG with r_inner ≈ r4, use TEM approximation
    Z0 = 377.0  # Ohm (free-space, rough approximation)

    # Phase delay through the unit cell (electrical length)
    p_m     = p * 1e-3
    k       = omega / C_LIGHT
    phi     = k * p_m  # electrical length

    # ABCD of transmission line section
    T_line = np.array([
        [np.cos(phi),          1j * Z0 * np.sin(phi)],
        [1j * np.sin(phi) / Z0, np.cos(phi)         ]
    ], dtype=complex)

    # ABCD of shunt CeSRR
    T_shunt = np.array([
        [1.0,            0.0],
        [1.0 / Z_res, 1.0]
    ], dtype=complex)

    # Combined unit-cell ABCD = T_line × T_shunt
    T = T_line @ T_shunt
    return T


def compute_dispersion(
    f_ghz_range: np.ndarray,
    r1: float, r2: float, r3: float, r4: float,
    t: float, p: float,
) -> dict:
    """
    Compute the dispersion curve β·p vs frequency for a CeSRR SWS.

    Uses Bloch theorem:
      cos(β·p) = (A + D) / 2
    where A, D are the diagonal elements of the unit-cell ABCD matrix.

    Parameters
    ----------
    f_ghz_range : array of frequencies to sweep (GHz)
    r1..r4, t, p: CeSRR unit-cell geometry (mm)

    Returns
    -------
    dict with keys:
      'f_ghz'       : frequency array
      'beta_p'      : normalised phase advance β·p (radians), NaN where no solution
      'slow_wave_factor' : c·β/ω (dimensionless)
      'is_passband' : bool mask of frequency points in the passband
    """
    beta_p_arr = np.full(len(f_ghz_range), np.nan)
    swf_arr    = np.full(len(f_ghz_range), np.nan)

    for i, f in enumerate(f_ghz_range):
        T = transfer_matrix_unit_cell(f, r1, r2, r3, r4, t, p)
        if T is None:
            continue

        A, D = T[0, 0], T[1, 1]
        cos_bp = 0.5 * (A + D).real  # take real part only

        # Passband condition: |cos(β·p)| ≤ 1
        if abs(cos_bp) <= 1.0:
            bp = np.arccos(np.clip(cos_bp, -1.0, 1.0))
            beta_p_arr[i] = bp
            omega = 2 * np.pi * f * 1e9
            p_m   = p * 1e-3
            beta  = bp / p_m  if p_m > 0 else np.nan
            swf_arr[i] = C_LIGHT * beta / omega if omega > 0 else np.nan

    is_passband = ~np.isnan(beta_p_arr)

    return {
        "f_ghz":            f_ghz_range,
        "beta_p":           beta_p_arr,
        "slow_wave_factor": swf_arr,
        "is_passband":      is_passband,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    _code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
    sys.path.insert(0, str(_code_dir))

    # Example geometry from dataset neighbourhood
    geom = dict(r1=20.0, r2=18.0, r3=12.0, r4=5.0, t=1.0, p=30.0)
    f_range = np.linspace(0.5, 10.0, 200)
    disp = compute_dispersion(f_range, **geom)

    n_pass = disp["is_passband"].sum()
    if n_pass > 0:
        f_pass = disp["f_ghz"][disp["is_passband"]]
        swf    = disp["slow_wave_factor"][disp["is_passband"]]
        print(f"Passband: {f_pass[0]:.2f} – {f_pass[-1]:.2f} GHz")
        print(f"Mean SWF in passband: {np.nanmean(swf):.2f}")
    else:
        print("No passband found for these parameters.")

    fc = waveguide_cutoff_freq(geom["r1"])
    print(f"Empty waveguide TE11 cutoff (r={geom['r1']}mm): {fc:.2f} GHz")
