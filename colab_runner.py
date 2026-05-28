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

GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/cesrr-inverse-design.git"
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
# ↑ If True, copies results/ and figures/ back to Drive after experiment.

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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 : Run the Experiment Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def section5_run_experiment(repo_path: Path):
    """Run the selected experiment using the colab_setup dispatcher."""
    print("\n" + "═" * 60)
    print(f"SECTION 5: Running Experiment — '{EXPERIMENT}'")
    print("═" * 60)

    runner = repo_path / "code" / "utils" / "colab_setup.py"
    if not runner.exists():
        print(f"❌ colab_setup.py not found at {runner}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(runner), f"--run={EXPERIMENT}", "--skip-drive"],
        cwd=str(repo_path),
    )

    if result.returncode == 0:
        print(f"\n✅ Experiment '{EXPERIMENT}' completed successfully.")
    else:
        print(f"\n❌ Experiment '{EXPERIMENT}' failed (exit code {result.returncode}).")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 : Save Results Back to Drive
# ══════════════════════════════════════════════════════════════════════════════

def section6_save_results(repo_path: Path, drive_data_path: Path):
    """Copy results/ and figures/ back to Google Drive for persistence."""
    if not SAVE_RESULTS_TO_DRIVE:
        print("\nSave to Drive: DISABLED (set SAVE_RESULTS_TO_DRIVE=True to enable)")
        return

    print("\n" + "═" * 60)
    print("SECTION 6: Saving Results to Google Drive")
    print("═" * 60)

    drive_out_root = drive_data_path.parent  # My Drive/cesrr_data/

    for folder in ["results", "figures"]:
        src = repo_path / folder
        dst = drive_out_root / folder
        if src.exists() and any(src.iterdir()):
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
            print(f"✅ Saved {folder}/ → Drive ({n_files} files)")
        else:
            print(f"   Skipping {folder}/ (empty or not found)")

    print(f"\n📁 Results saved to: My Drive/cesrr_data/")
    print("   You can download them from the Drive UI.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 : Quick Results Preview
# ══════════════════════════════════════════════════════════════════════════════

def section7_preview_results(repo_path: Path):
    """Print key output files and show ablation summary if available."""
    print("\n" + "═" * 60)
    print("SECTION 7: Results Preview")
    print("═" * 60)

    # List generated figures
    fig_dir = repo_path / "figures"
    if fig_dir.exists():
        figs = list(fig_dir.glob("*.png"))
        if figs:
            print(f"\nGenerated figures ({len(figs)}):")
            for f in sorted(figs):
                print(f"   📊 {f.name}")
        else:
            print("No figures generated yet.")

    # Show ablation summary table
    ablation_csv = repo_path / "results" / "ablation_summary.csv"
    if ablation_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(ablation_csv, index_col=0)
            print("\n── Ablation Summary ──")
            print(df.to_string())
        except Exception as e:
            print(f"Could not display ablation summary: {e}")

    # Show processed dataset sizes
    proc_dir = repo_path / "data" / "processed"
    if proc_dir.exists():
        print("\n── Processed datasets ──")
        for csv in sorted(proc_dir.glob("*.csv")):
            try:
                import pandas as pd
                df = pd.read_csv(csv)
                print(f"   {csv.name:35s} {len(df):4d} rows")
            except Exception:
                print(f"   {csv.name}")

    print("\n✅ Run complete.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  —  runs all sections in order
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     CeSRR Inverse Design — Colab Runner                     ║")
    print("║     Physics-Informed, Multi-Fidelity, Data-Efficient         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nExperiment : {EXPERIMENT}")
    print(f"GitHub URL : {GITHUB_REPO_URL}")
    print(f"Drive path : My Drive/{DRIVE_DATA_FOLDER}/")

    drive_data = section1_mount_drive()
    repo       = section2_clone_repo()

    section3_install_deps(repo)
    section4_copy_data(drive_data, repo)
    section5_run_experiment(repo)
    section6_save_results(repo, drive_data)
    section7_preview_results(repo)

    print("\n" + "═" * 60)
    print("All done! Check My Drive/cesrr_data/ for results and figures.")
    print("═" * 60)
