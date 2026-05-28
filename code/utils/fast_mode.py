"""
fast_mode.py -- Read CESRR_FAST_MODE env vars and patch config values.

Every training script calls apply_fast_mode() at the top if FAST_MODE is set.
This avoids touching config.py itself and keeps fast/full configs separate.

Usage (in each model script's __main__ block):
    from utils.fast_mode import apply_fast_mode
    apply_fast_mode()
"""

import os
import sys
from pathlib import Path

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))


def apply_fast_mode():
    """
    If CESRR_FAST_MODE=1 is set in the environment, override config module
    globals with reduced values so the full pipeline runs in ~30 min on Colab.

    Overrides:
        CVAE_MAX_EPOCHS   -> CESRR_MAX_EPOCHS   (default fast: 60)
        TANDEM_MAX_EPOCHS -> CESRR_MAX_EPOCHS
        GP_N_RESTARTS     -> CESRR_GP_RESTARTS  (default fast: 3)
        MF_GP_N_RESTARTS  -> CESRR_GP_RESTARTS
        AL_N_QUERY_ROUNDS -> CESRR_AL_ROUNDS    (default fast: 3)
        CVAE_PATIENCE     -> 10
        TANDEM_PATIENCE   -> 10
    """
    if os.environ.get("CESRR_FAST_MODE", "0") != "1":
        return  # nothing to do

    import utils.config as cfg

    max_epochs   = int(os.environ.get("CESRR_MAX_EPOCHS",  "60"))
    gp_restarts  = int(os.environ.get("CESRR_GP_RESTARTS", "3"))
    al_rounds    = int(os.environ.get("CESRR_AL_ROUNDS",   "3"))

    cfg.CVAE_MAX_EPOCHS    = max_epochs
    cfg.TANDEM_MAX_EPOCHS  = max_epochs
    cfg.MDN_N_GAUSSIANS    = min(cfg.MDN_N_GAUSSIANS, 3)   # fewer mixture components
    cfg.GP_N_RESTARTS      = gp_restarts
    cfg.MF_GP_N_RESTARTS   = gp_restarts
    cfg.AL_N_QUERY_ROUNDS  = al_rounds
    cfg.CVAE_PATIENCE      = min(cfg.CVAE_PATIENCE, 10)
    cfg.TANDEM_PATIENCE    = min(cfg.TANDEM_PATIENCE, 10)

    print(
        f"[FAST_MODE] epochs={max_epochs}, GP_restarts={gp_restarts}, "
        f"AL_rounds={al_rounds}"
    )
