"""
config.py — Central configuration for the CeSRR inverse design framework.

All hyperparameters, paths, and constants live here. No magic numbers
anywhere else in the codebase.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Repository root (two levels up from this file: code/utils/ → code/ → root/)
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Create output dirs if needed
for d in [DATA_RAW_DIR, DATA_PROC_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Raw data filenames (expected in DATA_RAW_DIR or rajnishpaperdata root)
# ─────────────────────────────────────────────────────────────────────────────
RAW_FILES = {
    "unit_0deg":  "Updated AI_ML Data.xlsx",
    "unit_180deg": "Updated AI_ML Data (1).xlsx",
    "full_struct": "Full_structure_dimensions.xlsx",
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns (9 geometric parameters) and target
# ─────────────────────────────────────────────────────────────────────────────
UNIT_CELL_FEATURES = ["r1", "r2", "r3", "r4", "t", "d1", "d2", "h", "p"]
TARGET_COL = "freq_ghz"
ROTATION_COL = "rotation"   # 0 or 180

ALL_COLS = UNIT_CELL_FEATURES + [TARGET_COL, ROTATION_COL]

# Physical constraint: r1 > r2 > r3 > r4  (outer → inner radii ordering)
ORDERING_COLS = ["r1", "r2", "r3", "r4"]

# ─────────────────────────────────────────────────────────────────────────────
# Physical bounds (mm) — derived from dataset range + domain knowledge
# ─────────────────────────────────────────────────────────────────────────────
PARAM_BOUNDS = {
    "r1": (5.0,  50.0),
    "r2": (4.0,  48.0),
    "r3": (3.0,  40.0),
    "r4": (1.5,  25.0),
    "t":  (0.5,   3.0),
    "d1": (0.5,   8.0),
    "d2": (0.5,   5.0),
    "h":  (0.5,  15.0),
    "p":  (10.0, 60.0),
}

FREQ_BOUNDS = (0.5, 20.0)  # GHz — operating range

# ─────────────────────────────────────────────────────────────────────────────
# Data augmentation
# ─────────────────────────────────────────────────────────────────────────────
AUG_SIGMA_FRAC = 0.02          # ±2% Gaussian perturbation
AUG_N_PER_SAMPLE = 4           # target augmented copies per original record
AUG_OVERSAMPLE_FACTOR = 3      # generate 3× target, filter down
AUG_RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# LHS unlabelled pool for active learning
# ─────────────────────────────────────────────────────────────────────────────
LHS_N_CANDIDATES = 500
LHS_RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Low-fidelity analytical oracle (multi-fidelity)
# ─────────────────────────────────────────────────────────────────────────────
LF_N_SAMPLES = 200             # LHS samples evaluated through LC oracle

# ─────────────────────────────────────────────────────────────────────────────
# Model hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Gaussian Process
GP_MATERN_NU = 2.5
GP_N_RESTARTS = 10

# PI-cVAE
CVAE_LATENT_DIM = 8
CVAE_ENCODER_HIDDEN = [128, 64]
CVAE_DECODER_HIDDEN = [64, 128]
CVAE_BATCH_SIZE = 32
CVAE_MAX_EPOCHS = 500
CVAE_LR = 1e-3
CVAE_BETA_KL = 1.0             # KL weight in ELBO
CVAE_LAMBDA_ORDER = 10.0       # ordering constraint penalty weight
CVAE_LAMBDA_BOUNDS = 5.0       # bounds constraint penalty weight
CVAE_PATIENCE = 30             # early stopping patience
CVAE_VAL_FRAC = 0.15
CVAE_MC_SAMPLES = 50           # MC-Dropout forward passes

# Tandem network
TANDEM_HIDDEN = [128, 128]
TANDEM_LR = 1e-3
TANDEM_MAX_EPOCHS = 300
TANDEM_PATIENCE = 25
TANDEM_BATCH_SIZE = 32

# MDN (Mixture Density Network)
MDN_N_GAUSSIANS = 5
MDN_HIDDEN = [128, 64]

# MF-GP (multi-fidelity co-kriging)
MF_GP_N_RESTARTS = 10

# ─────────────────────────────────────────────────────────────────────────────
# Active learning
# ─────────────────────────────────────────────────────────────────────────────
AL_N_INITIAL = 20              # initial labelled points for AL
AL_N_QUERY_ROUNDS = 10         # number of AL query rounds
AL_N_QUERY_PER_ROUND = 5       # points queried per round
AL_ACQUISITION = "bald"        # "uncertainty" | "bald" | "ei"
AL_RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation / ablation
# ─────────────────────────────────────────────────────────────────────────────
CV_N_FOLDS = 5                 # cross-validation folds (LOO-CV if n<30)
DIVERSITY_N_SAMPLES = 50       # samples per target freq for diversity metric

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
