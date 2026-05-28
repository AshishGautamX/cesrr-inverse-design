"""
colab_runner.py  —  Single-file Colab runner for CeSRR Inverse Design Framework
================================================================================

HOW TO USE
----------
Option A — Run the entire script at once in Colab:
    Open a Colab notebook, add ONE code cell:
        !python colab_runner.py

Option B — Run section by section (paste each section as a separate Colab cell).
    Each section is delimited by:
        # ══════════════════ SECTION N : Title ══════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA SETUP — Upload your data files to Google Drive BEFORE running.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create the following folder structure in YOUR Google Drive:

    My Drive/
    └── cesrr_data/
        └── raw/
            ├── Updated AI_ML Data.xlsx          ← 0° CeSRR unit-cell data  (56 rows)
            ├── Updated AI_ML Data (1).xlsx      ← 180° CeSRR unit-cell data (55 rows)
            └── Full_structure_dimensions.xlsx   ← full waveguide structure data (23 rows)

FILE FORMAT (all three are Excel .xlsx files):
    • Updated AI_ML Data.xlsx
        Long-format table. Each frequency entry spans ~5 rows:
        Row contains: Sl. No. | Frequency (e.g. "1 GHz") | Dimension Name | Value (mm)
        Dimension names: r1, r2, r3, r4, t (thickness), d1, d2, h, p
        Total: 56 unique frequency-geometry pairs (0° rotation).

    • Updated AI_ML Data (1).xlsx
        Same format as above but for 180° CeSRR rotation configuration.
        Total: 55 unique frequency-geometry pairs.

    • Full_structure_dimensions.xlsx
        Two side-by-side tables:
        Left  — Full structure waveguide dimensions (L, r, P, R, p_sep)
        Right — Corresponding unit-cell dimensions (r1..r4, t)
        Total: 23 entries, one per operating frequency.

DO NOT rename the files. The parser expects these exact filenames.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GITHUB — Set your repo URL below before running.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  edit these two lines before running
# ══════════════════════════════════════════════════════════════════════════════

GITHUB_REPO_URL = "https://github.com/AshishGautamX/cesrr-inverse-design.git"
# ↑ Replace with your actual GitHub repo URL, e.g.:
#   "https://github.com/ashishgx/cesrr-inverse-design.git"

DRIVE_DATA_FOLDER = "cesrr_data/raw"
# ↑ Path inside your Google Drive (relative to My Drive root).
#   Default: My Drive/cesrr_data/raw/
#   Change only if you put the data somewhere else.

EXPERIMENT = "full_pipeline"
# ↑ Choose what to run:
#   "data_prep"       → parse + clean + validate + augment only
#   "baselines"       → B1 GP, B2 RIGID, B3 Tandem, B4 cVAE
#   "mf_gp"           → M1 multi-fidelity GP
#   "pi_cvae"         → M2/M3 PI-cVAE
#   "active_learning" → AL loop
#   "shap"            → SHAP analysis + rotation comparison
#   "ablation"        → full 9-config ablation table
#   "full_pipeline"   → everything end-to-end

SAVE_RESULTS_TO_DRIVE = True
# If True, copies results/ and figures/ back to Drive after EACH script.

FAST_MODE = False
# If True, caps training epochs to ~50 so the full pipeline completes in
# ~30 min on Colab T4 GPU.  Set to False for full research-quality training.
# Effect: patches config.py values via environment variable before each script.

FAST_EPOCHS    = 60     # CVAE / Tandem / MDN epochs in FAST_MODE
FAST_GP_STARTS = 3      # GP random restarts in FAST_MODE (default: 10)
FAST_AL_ROUNDS = 3      # AL rounds in FAST_MODE (default: 10)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 : Mount Google Drive
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import shutil
import subprocess
from pathlib import Path


def section1_mount_drive() -> Path:
    """Mount Google Drive and return the path to the data folder."""
    print("\n" + "═" * 60)
    print("SECTION 1: Mounting Google Drive")
    print("═" * 60)

    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        drive_root = Path("/content/drive/MyDrive")
        print(f"✅ Drive mounted at: {drive_root}")
    except ImportError:
        # Not in Colab — use local fallback
        drive_root = Path.home() / "drive_mock"
        drive_root.mkdir(parents=True, exist_ok=True)
        print(f"⚠️  Not in Colab. Using local mock drive at: {drive_root}")

    data_path = drive_root / DRIVE_DATA_FOLDER
    if not data_path.exists():
        print(f"\n❌ ERROR: Data folder not found at: {data_path}")
        print("   Please create the following folder in your Google Drive:")
        print(f"   My Drive/{DRIVE_DATA_FOLDER}/")
        print("   and upload the three Excel files described at the top of this script.")
        sys.exit(1)

    # Verify all required data files exist
    required_files = [
        "Updated AI_ML Data.xlsx",
        "Updated AI_ML Data (1).xlsx",
        "Full_structure_dimensions.xlsx",
    ]
    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        print(f"\n❌ Missing files in {data_path}:")
        for f in missing:
            print(f"   • {f}")
        print("\nUploaded files found in that folder:")
        for f in data_path.iterdir():
            print(f"   • {f.name}")
        sys.exit(1)

    print(f"\n✅ All 3 data files found in: {data_path}")
    for f in required_files:
        size_kb = (data_path / f).stat().st_size / 1024
        print(f"   • {f} ({size_kb:.1f} KB)")

    return data_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 : Clone GitHub Repository
# ══════════════════════════════════════════════════════════════════════════════

def section2_clone_repo() -> Path:
    """Clone (or update) the GitHub repository into /content/."""
    print("\n" + "═" * 60)
    print("SECTION 2: Cloning GitHub Repository")
    print("═" * 60)

    if GITHUB_REPO_URL == "https://github.com/YOUR_USERNAME/cesrr-inverse-design.git":
        print("❌ ERROR: GITHUB_REPO_URL has not been set.")
        print("   Edit the CONFIGURATION section at the top of this script.")
        sys.exit(1)

    repo_name = GITHUB_REPO_URL.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = Path("/content") / repo_name

    if repo_path.exists():
        print(f"Repository already exists at {repo_path} — pulling latest changes…")
        result = subprocess.run(
            ["git", "-C", str(repo_path), "pull"],
            capture_output=True, text=True,
        )
        print(result.stdout or "Already up to date.")
    else:
        print(f"Cloning {GITHUB_REPO_URL}…")
        result = subprocess.run(
            ["git", "clone", GITHUB_REPO_URL, str(repo_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("❌ Git clone failed:")
            print(result.stderr)
            sys.exit(1)
        print(f"✅ Repository cloned to {repo_path}")

    # Add code directory to Python path
    code_dir = repo_path / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    print(f"✅ Added to Python path: {code_dir}")

    # Verify structure
    expected_dirs = ["code/data", "code/utils", "code/models", "code/analytical"]
    missing_dirs = [d for d in expected_dirs if not (repo_path / d).exists()]
    if missing_dirs:
        print(f"⚠️  Some expected directories missing: {missing_dirs}")
        print("   Make sure all code files are committed to GitHub.")
    else:
        print("✅ Repository structure verified.")

    return repo_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 : Install Dependencies
# ══════════════════════════════════════════════════════════════════════════════

def section3_install_deps(repo_path: Path):
    """
    Install Python dependencies -- Colab-safe strategy.

    ROOT CAUSE of ValueError: numpy.dtype size changed:
      requirements.txt had numpy<2.0. Colab ships numpy 2.x. pip downgrades
      it, breaking pre-compiled pandas/scipy C extensions.

    FIX: In Colab, only install packages NOT bundled with the runtime.
    numpy / pandas / scipy / torch / sklearn are already there -- leave them.
    Use --upgrade-strategy only-if-needed to prevent any downgrade.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: Installing Dependencies")
    print("=" * 60)

    # Packages Colab does NOT include -- safe to install fresh
    COLAB_MISSING = [
        "shap>=0.44",
        "pyDOE2>=1.3",
        "gpytorch>=1.11",
        "botorch>=0.9",
    ]

    # Detect Colab
    in_colab = False
    try:
        import google.colab  # type: ignore  # noqa: F401
        in_colab = True
    except ImportError:
        pass

    if in_colab:
        print("Colab detected -- installing only packages absent from Colab:")
        print("  (numpy/pandas/scipy/torch/sklearn already present -- skipping)")
        for pkg in COLAB_MISSING:
            print(f"  Installing {pkg} ...", end=" ", flush=True)
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg,
                 "--quiet", "--upgrade-strategy", "only-if-needed"],
                capture_output=True, text=True,
            )
            print("OK" if res.returncode == 0 else "WARN: " + res.stderr[-100:])
    else:
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            print(f"Local run -- installing from {req_file}")
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file),
                 "--quiet", "--upgrade-strategy", "only-if-needed"],
                capture_output=True, text=True,
            )
            print("OK" if res.returncode == 0 else "Issues: " + res.stderr[-300:])

    # Verify via SUBPROCESS (avoids contaminating this process with a
    # partially-loaded broken module -- that's how the original error escaped
    # the try/except ImportError: it raised ValueError, not ImportError).
    print("\nVerifying imports (subprocess check):")
    checks = [
        ("numpy",    "import numpy; print(numpy.__version__)"),
        ("pandas",   "import pandas; print(pandas.__version__)"),
        ("torch",    "import torch; print(torch.__version__)"),
        ("scipy",    "import scipy; print(scipy.__version__)"),
        ("sklearn",  "import sklearn; print(sklearn.__version__)"),
        ("shap",     "import shap; print(shap.__version__)"),
        ("openpyxl", "import openpyxl; print(openpyxl.__version__)"),
    ]
    all_ok = True
    for name, snippet in checks:
        res = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True,
        )
        if res.returncode == 0:
            print(f"  [OK]   {name} {res.stdout.strip()}")
        else:
            print(f"  [FAIL] {name}: {res.stderr.strip()[:100]}")
            all_ok = False

    if not all_ok:
        print("\nWARNING: Some packages failed -- experiment may error.")
        print("If numpy error appears, restart the Colab runtime and re-run.")




# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 : Copy Data from Drive to Repo
# ══════════════════════════════════════════════════════════════════════════════

def section4_copy_data(drive_data_path: Path, repo_path: Path):
    """
    Copy raw Excel files from Drive into the repo's data/raw/ directory.
    This avoids re-reading from Drive on every run (faster I/O in Colab).
    """
    print("\n" + "═" * 60)
    print("SECTION 4: Copying Data from Drive to Colab Runtime")
    print("═" * 60)

    local_raw = repo_path / "data" / "raw"
    local_raw.mkdir(parents=True, exist_ok=True)

    files = [
        "Updated AI_ML Data.xlsx",
        "Updated AI_ML Data (1).xlsx",
        "Full_structure_dimensions.xlsx",
    ]
    for fname in files:
        src = drive_data_path / fname
        dst = local_raw / fname
        if dst.exists():
            print(f"   Already present: {fname}")
        else:
            shutil.copy2(src, dst)
            print(f"   Copied: {fname}")

    # Also copy to repo root (alternative location the parsers check)
    for fname in files:
        src = drive_data_path / fname
        dst = repo_path / fname
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"\n✅ Data ready at: {local_raw}")

    # Create output directories
    for out_dir in ["data/processed", "results", "figures"]:
        (repo_path / out_dir).mkdir(parents=True, exist_ok=True)
    print("✅ Output directories created: data/processed/, results/, figures/")


# ──────────────────────────────────────────────────────────────────────────────
# Script lists (mirrors colab_setup.py EXPERIMENT_MAP -- kept here so
# section5 can run them directly without nested subprocess)
# ──────────────────────────────────────────────────────────────────────────────

_DATA_SCRIPTS = [
    "code/data/01_load_raw.py",
    "code/data/02_clean_normalize.py",
    "code/data/03_validate_physics.py",
    "code/data/04_augment_physics.py",
    "code/data/05_lhs_pool.py",
    "code/data/06_build_datasets.py",
]

_EXPERIMENT_MAP = {
    "data_prep":       _DATA_SCRIPTS,
    "baselines": [
        "code/models/baselines/gp_surrogate.py",
        "code/models/baselines/rigid_inverse.py",
        "code/models/baselines/tandem_network.py",
        "code/models/baselines/mlp_inverse.py",
        "code/models/baselines/mdn_inverse.py",
    ],
    "mf_gp":           ["code/models/multifidelity/mf_gp.py"],
    "pi_cvae": [
        "code/models/generative/cvae_base.py",
        "code/models/generative/cvae_pi.py",
    ],
    "active_learning": ["code/active_learning/al_loop.py"],
    "shap": [
        "code/interpretability/shap_global.py",
        "code/interpretability/shap_rotation_compare.py",
    ],
    "ablation":        ["code/evaluation/ablation_runner.py"],
}
_EXPERIMENT_MAP["full_pipeline"] = (
    _DATA_SCRIPTS + [
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
)


def _drive_checkpoint(repo_path: Path, drive_data_path: Path, label: str = ""):
    """
    Incrementally copy results/ and figures/ to Drive.
    Called after EACH script so no work is lost on Colab disconnect.
    """
    if not SAVE_RESULTS_TO_DRIVE or drive_data_path is None:
        return
    drive_out = drive_data_path.parent  # My Drive/cesrr_data/
    n_copied = 0
    for folder in ["results", "figures", "data/processed"]:
        src = repo_path / folder
        dst = drive_out / Path(folder).name
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if f.is_file():
                rel = f.relative_to(src)
                d = dst / rel
                d.parent.mkdir(parents=True, exist_ok=True)
                if not d.exists() or f.stat().st_mtime > d.stat().st_mtime:
                    shutil.copy2(f, d)
                    n_copied += 1
    if n_copied:
        print(f"  [Drive] Saved {n_copied} file(s) after '{label}'")


def _fast_mode_env():
    """
    Return an os.environ dict with FAST_MODE overrides.
    Scripts read these env vars to cap epochs without changing config.py.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if FAST_MODE:
        env["CESRR_FAST_MODE"]    = "1"
        env["CESRR_MAX_EPOCHS"]   = str(FAST_EPOCHS)
        env["CESRR_GP_RESTARTS"]  = str(FAST_GP_STARTS)
        env["CESRR_AL_ROUNDS"]    = str(FAST_AL_ROUNDS)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 : Run the Experiment Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def section5_run_experiment(repo_path: Path, drive_data_path: Path = None):
    """
    Run each pipeline script in sequence with REAL-TIME output streaming.

    KEY FIX: The original implementation called colab_setup.py as a subprocess,
    which then spawned MORE subprocesses. Jupyter never sees output from
    grandchild processes -- that is why the cell was silent for 44 minutes.

    NEW APPROACH:
      - Run each script directly as a Popen child (one level, not two)
      - Stream stdout+stderr line-by-line with flush=True
      - Save Drive checkpoint after every script
      - FAST_MODE caps epochs so pipeline completes in ~30 min on Colab T4
    """
    print("\n" + "=" * 60)
    print(f"SECTION 5: Running Experiment -- '{EXPERIMENT}'")
    if FAST_MODE:
        print(f"  FAST_MODE ON: epochs={FAST_EPOCHS}, GP_restarts={FAST_GP_STARTS}, AL_rounds={FAST_AL_ROUNDS}")
    print("=" * 60, flush=True)

    if EXPERIMENT not in _EXPERIMENT_MAP:
        print(f"Unknown experiment '{EXPERIMENT}'. Choose from: {list(_EXPERIMENT_MAP)}")
        return

    scripts = _EXPERIMENT_MAP[EXPERIMENT]
    env = _fast_mode_env()
    n_ok = n_fail = 0

    for i, script in enumerate(scripts, 1):
        full_path = repo_path / script
        script_name = Path(script).name
        print(f"\n[{i:02d}/{len(scripts):02d}] {script_name}", flush=True)
        print("-" * 55, flush=True)

        if not full_path.exists():
            print(f"  SKIP: {full_path} not found")
            continue

        # -u = unbuffered Python; env has PYTHONUNBUFFERED=1
        proc = subprocess.Popen(
            [sys.executable, "-u", str(full_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            env=env,
            bufsize=1,                  # line-buffered
            cwd=str(repo_path),
        )

        # Stream every line as it arrives
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()

        if proc.returncode == 0:
            print(f"  [OK] {script_name} done", flush=True)
            n_ok += 1
            _drive_checkpoint(repo_path, drive_data_path, script_name)
        else:
            print(f"  [FAIL] {script_name} exited {proc.returncode}", flush=True)
            n_fail += 1
            # Still checkpoint what we have
            _drive_checkpoint(repo_path, drive_data_path, f"{script_name} [partial]")

    print(f"\n{'=' * 55}")
    print(f"Done: {n_ok} succeeded, {n_fail} failed out of {len(scripts)} scripts.")
    if n_fail:
        print("Check the [FAIL] lines above for errors.")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 : Save Results Back to Drive (final full sync)
# ──────────────────────────────────────────────────────────────────────────────

def section6_save_results(repo_path: Path, drive_data_path: Path):
    """Final sync of ALL results/ figures/ data/processed/ to Drive."""
    if not SAVE_RESULTS_TO_DRIVE:
        print("\nSave to Drive: DISABLED")
        return

    print("\n" + "=" * 60)
    print("SECTION 6: Final Save to Google Drive")
    print("=" * 60)
    _drive_checkpoint(repo_path, drive_data_path, "final sync")
    drive_out = drive_data_path.parent
    print(f"All results saved to: My Drive/{drive_out.name}/")
    print("Download from the Drive UI or Files panel.")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 : Quick Results Preview
# ──────────────────────────────────────────────────────────────────────────────

def section7_preview_results(repo_path: Path):
    """Print key output files and show ablation summary if available."""
    print("\n" + "=" * 60)
    print("SECTION 7: Results Preview")
    print("=" * 60)

    # Generated figures
    fig_dir = repo_path / "figures"
    if fig_dir.exists():
        figs = list(fig_dir.glob("*.png"))
        if figs:
            print(f"\nGenerated figures ({len(figs)}):")
            for f in sorted(figs):
                print(f"   [fig] {f.name}")
        else:
            print("No figures generated yet.")

    # Ablation summary
    ablation_csv = repo_path / "results" / "ablation_summary.csv"
    if ablation_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(ablation_csv, index_col=0)
            print("\n-- Ablation Summary --")
            print(df.to_string())
        except Exception as e:
            print(f"Could not display ablation summary: {e}")

    # Processed dataset sizes
    proc_dir = repo_path / "data" / "processed"
    if proc_dir.exists():
        print("\n-- Processed datasets --")
        for csv in sorted(proc_dir.glob("*.csv")):
            try:
                import pandas as pd
                df = pd.read_csv(csv)
                print(f"   {csv.name:35s} {len(df):4d} rows")
            except Exception:
                print(f"   {csv.name}")

    print("\n[OK] Run complete.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("CeSRR Inverse Design -- Colab Runner")
    print("Physics-Informed, Multi-Fidelity, Data-Efficient")
    print("=" * 60)
    print(f"Experiment  : {EXPERIMENT}")
    print(f"GitHub URL  : {GITHUB_REPO_URL}")
    print(f"Drive path  : My Drive/{DRIVE_DATA_FOLDER}/")
    print(f"FAST_MODE   : {FAST_MODE}  (epochs={FAST_EPOCHS}, GP_restarts={FAST_GP_STARTS})")
    print(f"Drive save  : {SAVE_RESULTS_TO_DRIVE}  (checkpoint after each script)")
    print(flush=True)

    drive_data = section1_mount_drive()
    repo       = section2_clone_repo()

    section3_install_deps(repo)
    section4_copy_data(drive_data, repo)
    section5_run_experiment(repo, drive_data_path=drive_data)  # streams + checkpoints
    section6_save_results(repo, drive_data)                    # final full sync
    section7_preview_results(repo)

    print("\n" + "=" * 60)
    print("All done! Check My Drive/cesrr_data/ for results and figures.")
    print("=" * 60)
