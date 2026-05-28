"""
colab_setup.py -- Colab/Drive environment setup and experiment dispatcher.

Run via colab_loader.ipynb Cell 3:
    !python code/utils/colab_setup.py --run=<experiment>

Experiments:
    data_prep       -- Steps 01-06: parse, clean, validate, augment, build
    baselines       -- B1, B2, B3, B4 baselines
    mf_gp           -- M1 multi-fidelity GP
    pi_cvae         -- M2, M3 PI-cVAE variants
    active_learning -- Full AL loop
    shap            -- Global SHAP + rotation comparison
    ablation        -- All 9 configs ablation study
    full_pipeline   -- Everything end-to-end
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def run_script(script_path: str):
    """Run a Python script as a subprocess and stream its output."""
    log.info("Running: %s", script_path)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        log.error("Script failed: %s (exit code %d)", script_path, result.returncode)
    return result.returncode == 0


def setup_drive_paths():
    """Create symlinks or copy data from Drive to local expected locations."""
    try:
        from google.colab import drive  # type: ignore
        import shutil, os
        # Assumes data was uploaded to Drive under cesrr_data/raw/
        drive_raw = Path("/content/drive/MyDrive/cesrr_data/raw")
        local_raw = _code_dir.parent / "data" / "raw"
        local_raw.mkdir(parents=True, exist_ok=True)
        for fname in [
            "Updated AI_ML Data.xlsx",
            "Updated AI_ML Data (1).xlsx",
            "Full_structure_dimensions.xlsx",
        ]:
            src = drive_raw / fname
            dst = local_raw / fname
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
                log.info("Copied from Drive: %s", fname)
            elif dst.exists():
                log.info("Already present: %s", fname)
            else:
                log.warning("Not found on Drive: %s", fname)
    except ImportError:
        log.info("Not running in Colab -- skipping Drive setup.")


DATA_SCRIPTS = [
    "code/data/01_load_raw.py",
    "code/data/02_clean_normalize.py",
    "code/data/03_validate_physics.py",
    "code/data/04_augment_physics.py",
    "code/data/05_lhs_pool.py",
    "code/data/06_build_datasets.py",
]

EXPERIMENT_MAP = {
    "data_prep": DATA_SCRIPTS,
    "baselines": [
        "code/models/baselines/gp_surrogate.py",
        "code/models/baselines/rigid_inverse.py",
        "code/models/baselines/tandem_network.py",
        "code/models/baselines/mlp_inverse.py",
        "code/models/baselines/mdn_inverse.py",
    ],
    "mf_gp": ["code/models/multifidelity/mf_gp.py"],
    "pi_cvae": [
        "code/models/generative/cvae_base.py",
        "code/models/generative/cvae_pi.py",
    ],
    "active_learning": ["code/active_learning/al_loop.py"],
    "shap": [
        "code/interpretability/shap_global.py",
        "code/interpretability/shap_rotation_compare.py",
    ],
    "ablation": ["code/evaluation/ablation_runner.py"],
    "full_pipeline": (
        DATA_SCRIPTS + [
            "code/models/baselines/gp_surrogate.py",
            "code/models/baselines/rigid_inverse.py",
            "code/models/baselines/tandem_network.py",
            "code/models/generative/cvae_base.py",
            "code/models/multifidelity/mf_gp.py",
            "code/models/generative/cvae_pi.py",
            "code/active_learning/al_loop.py",
            "code/interpretability/shap_global.py",
            "code/interpretability/shap_rotation_compare.py",
            "code/evaluation/ablation_runner.py",
            "code/evaluation/plot_all.py",
        ]
    ),
}


def main():
    parser = argparse.ArgumentParser(description="CeSRR experiment runner")
    parser.add_argument(
        "--run",
        choices=list(EXPERIMENT_MAP.keys()),
        default="data_prep",
        help="Experiment to run",
    )
    parser.add_argument(
        "--skip-drive", action="store_true",
        help="Skip Google Drive data copy (for local runs)",
    )
    args = parser.parse_args()

    if not args.skip_drive:
        setup_drive_paths()

    repo_root = _code_dir.parent
    scripts = EXPERIMENT_MAP[args.run]
    log.info("Experiment: %s  (%d scripts)", args.run, len(scripts))

    n_failed = 0
    for script in scripts:
        full_path = repo_root / script
        if not full_path.exists():
            log.warning("Script not found: %s", full_path)
            continue
        ok = run_script(str(full_path))
        if not ok:
            n_failed += 1

    if n_failed == 0:
        log.info("[OK] All scripts completed successfully.")
    else:
        log.error("[FAIL] %d script(s) failed.", n_failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
