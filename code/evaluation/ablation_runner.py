"""
ablation_runner.py -- Run all 9 ablation configurations and collect results.

Ablation matrix (from plan):
  B1  GP_Matern       -- primary surrogate baseline
  B2  RIGID           -- RF+MCMC, 2024 small-data SOTA
  B3  Tandem_MLP      -- standard inverse design method
  B4  cVAE_noPhysics  -- ablation: no physics constraints
  M1  MF_GP           -- novel: multi-fidelity GP
  M2  PIcVAE_soft     -- novel: soft ordering+bounds penalty only
  M3  PIcVAE_hard     -- novel: soft + hard isotonic projection
  M4  M3_MF           -- M3 trained on MF-augmented data
  FULL M4+AL          -- full system with active learning

Produces results/ablation_results.json and calls plot_all.py for Figure 8.
"""

import sys
import json
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import RESULTS_DIR
from utils.data_utils import (load_processed, load_combined_hf_lf,
                              train_test_split_stratified)
from evaluation.metrics import build_ablation_table, design_diversity

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Per-configuration runner functions
# -----------------------------------------------------------------------------

def run_b1_gp(df_tr, df_te) -> dict:
    from models.baselines.gp_surrogate import GPSurrogate
    m = GPSurrogate(); m.fit(df_tr)
    return m.evaluate(df_te)


def run_b2_rigid(df_tr, df_te) -> dict:
    from models.baselines.rigid_inverse import RIGIDInverse
    m = RIGIDInverse(); m.fit(df_tr)
    return m.evaluate(df_te)


def run_b3_tandem(df_tr, df_te) -> dict:
    from models.baselines.tandem_network import TandemNetwork
    m = TandemNetwork(); m.fit(df_tr)
    return m.evaluate(df_te)


def run_b4_cvae(df_tr, df_te) -> dict:
    from models.generative.cvae_base import CVAEBaseModel
    m = CVAEBaseModel(); m.fit(df_tr)
    return m.evaluate(df_te)


def run_m1_mfgp(df_hf_tr, df_te, df_lf) -> dict:
    from models.multifidelity.mf_gp import MultiFidelityGP
    m = MultiFidelityGP(); m.fit(df_lf, df_hf_tr)
    return m.evaluate(df_te)


def run_m2_picvae_soft(df_tr, df_te) -> dict:
    from models.generative.cvae_pi import PICVAEModel
    m = PICVAEModel(use_soft_physics=True, use_hard_projection=False)
    m.fit(df_tr)
    return m.evaluate(df_te)


def run_m3_picvae_hard(df_tr, df_te) -> dict:
    from models.generative.cvae_pi import PICVAEModel
    m = PICVAEModel(use_soft_physics=True, use_hard_projection=True)
    m.fit(df_tr)
    return m.evaluate(df_te)


def run_m4_mf(df_hf_tr, df_te, df_lf) -> dict:
    """M3 trained on HF + LF combined data."""
    from models.generative.cvae_pi import PICVAEModel
    import pandas as pd
    df_combined = pd.concat([df_hf_tr, df_lf], ignore_index=True)
    m = PICVAEModel(use_soft_physics=True, use_hard_projection=True)
    m.fit(df_combined)
    return m.evaluate(df_te)


def run_full(df_hf_tr, df_te, df_lf, df_lhs_pool) -> dict:
    """Full system: M4 + active learning loop."""
    from active_learning.al_loop import ActiveLearningLoop
    from models.generative.cvae_pi import PICVAEModel
    loop = ActiveLearningLoop(
        model_cls=PICVAEModel,
        model_kwargs={"use_soft_physics": True, "use_hard_projection": True},
    )
    tracker = loop.run(df_hf_tr, df_te, df_lhs_pool)
    # Return final round metrics
    last = tracker.records[-1]
    return {
        "model": "FULL_M4_AL",
        "feasibility_rate": last.feasibility_rate,
        "n_oracle_queries": last.n_oracle_queries,
        **last.extra,
    }


# -----------------------------------------------------------------------------
# Main ablation runner
# -----------------------------------------------------------------------------

def run_all_ablations(
    run_full_system: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run all ablation configs and return summary DataFrame.

    Parameters
    ----------
    run_full_system : if True, also run the full AL system (slow)
    save            : save results to RESULTS_DIR
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Load datasets
    df_hf          = load_processed("hf")
    df_aug         = load_processed("all")
    df_hf_tr, df_te = train_test_split_stratified(df_hf, test_frac=0.2)
    df_aug_tr, _    = train_test_split_stratified(df_aug, test_frac=0.2)
    df_hf_lf, df_lf = load_combined_hf_lf()
    df_lhs_pool     = load_processed("lf")

    all_results = []

    configs = [
        ("B1_GP_Matern",       lambda: run_b1_gp(df_hf_tr, df_te)),
        ("B2_RIGID",           lambda: run_b2_rigid(df_hf_tr, df_te)),
        ("B3_Tandem_MLP",      lambda: run_b3_tandem(df_aug_tr, df_te)),
        ("B4_cVAE_noPhysics",  lambda: run_b4_cvae(df_aug_tr, df_te)),
        ("M1_MF_GP",           lambda: run_m1_mfgp(df_hf_tr, df_te, df_lf)),
        ("M2_PIcVAE_soft",     lambda: run_m2_picvae_soft(df_aug_tr, df_te)),
        ("M3_PIcVAE_hard",     lambda: run_m3_picvae_hard(df_aug_tr, df_te)),
        ("M4_M3_MF",           lambda: run_m4_mf(df_hf_tr, df_te, df_lf)),
    ]

    if run_full_system:
        configs.append(
            ("FULL_M4_AL", lambda: run_full(df_hf_tr, df_te, df_lf, df_lhs_pool))
        )

    for name, runner in configs:
        log.info("Running config: %s...", name)
        t0 = time.time()
        try:
            result = runner()
            result["model"] = name
            result["runtime_s"] = round(time.time() - t0, 1)
            all_results.append(result)
            log.info("  Done in %.1fs | feas=%.3f",
                     result["runtime_s"],
                     result.get("feasibility_rate", float("nan")))
        except Exception as e:
            log.error("  FAILED for %s: %s", name, e)
            all_results.append({"model": name, "error": str(e)})

    df_summary = build_ablation_table(all_results)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # JSON (full details)
        out_json = RESULTS_DIR / "ablation_results.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info("Ablation results saved: %s", out_json)
        # CSV (summary table)
        out_csv = RESULTS_DIR / "ablation_summary.csv"
        df_summary.to_csv(out_csv)
        log.info("Ablation summary saved: %s", out_csv)

    print("\n-- Ablation Summary Table --")
    print(df_summary.to_string())
    return df_summary


if __name__ == "__main__":
    run_all_ablations(run_full_system=False, save=True)
