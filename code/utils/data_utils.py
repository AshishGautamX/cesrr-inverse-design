"""
data_utils.py — Shared data loading, scaling, and splitting utilities.

Used by every training script to ensure consistent preprocessing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import joblib
import logging

from utils.config import (
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    ORDERING_COLS, PARAM_BOUNDS, FREQ_BOUNDS,
    DATA_PROC_DIR, GLOBAL_SEED, CV_N_FOLDS
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_processed(split: str = "all") -> pd.DataFrame:
    """
    Load the processed, clean dataset from DATA_PROC_DIR.

    Parameters
    ----------
    split : str
        "all"   → full dataset (D2 augmented)
        "hf"    → high-fidelity only (original 111 CST records)
        "lf"    → low-fidelity analytical predictions
        "raw"   → raw validated records before augmentation (D1)

    Returns
    -------
    pd.DataFrame with columns: UNIT_CELL_FEATURES + [TARGET_COL, ROTATION_COL]
    """
    fname_map = {
        "all":  "D2_augmented.csv",
        "hf":   "D1_validated.csv",
        "lf":   "D_lf_analytical.csv",
        "raw":  "D1_validated.csv",
    }
    if split not in fname_map:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(fname_map)}")

    fpath = DATA_PROC_DIR / fname_map[split]
    if not fpath.exists():
        raise FileNotFoundError(
            f"Processed file not found: {fpath}\n"
            "Run: python code/data/06_build_datasets.py first."
        )
    df = pd.read_csv(fpath)
    log.info(f"Loaded {split} dataset: {len(df)} records from {fpath.name}")
    return df


def load_combined_hf_lf() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both HF (CST-verified) and LF (analytical) data for MF-GP training.

    Returns
    -------
    (df_hf, df_lf) — both DataFrames share the same column schema,
    with an additional 'fidelity' column ('hf' or 'lf').
    """
    df_hf = load_processed("hf").copy()
    df_lf = load_processed("lf").copy()
    df_hf["fidelity"] = "hf"
    df_lf["fidelity"] = "lf"
    return df_hf, df_lf


# ─────────────────────────────────────────────────────────────────────────────
# Physics constraint checkers
# ─────────────────────────────────────────────────────────────────────────────

def check_ordering(row: pd.Series) -> bool:
    """Return True if r1 > r2 > r3 > r4 (CeSRR structural constraint)."""
    return (row["r1"] > row["r2"] > row["r3"] > row["r4"])


def check_bounds(row: pd.Series) -> bool:
    """Return True if all parameters are within physical manufacturing bounds."""
    for col, (lo, hi) in PARAM_BOUNDS.items():
        if col in row.index:
            if not (lo <= row[col] <= hi):
                return False
    return True


def check_freq_bounds(freq: float) -> bool:
    """Return True if frequency is within the supported operating range."""
    return FREQ_BOUNDS[0] <= freq <= FREQ_BOUNDS[1]


def is_physically_valid(row: pd.Series) -> bool:
    """Combined physics validity check."""
    return (check_ordering(row) and
            check_bounds(row) and
            check_freq_bounds(row[TARGET_COL]))


def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to physics-valid rows only. Reports drop count."""
    mask = df.apply(is_physically_valid, axis=1)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        log.warning(f"Dropped {n_dropped}/{len(df)} records failing physics constraints.")
    return df[mask].reset_index(drop=True)


def feasibility_rate(df: pd.DataFrame) -> float:
    """Fraction of records satisfying r1>r2>r3>r4 constraint."""
    return df.apply(check_ordering, axis=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Scaling (fit only on training data, never on test/val)
# ─────────────────────────────────────────────────────────────────────────────

class CeSRRScaler:
    """
    MinMax scaler fitted separately for features and target.
    Persists to disk so the same scaler is used at inference.
    """

    def __init__(self, feature_range: tuple = (0.0, 1.0)):
        self.feature_range = feature_range
        self.feat_scaler = MinMaxScaler(feature_range=feature_range)
        self.tgt_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "CeSRRScaler":
        self.feat_scaler.fit(df[UNIT_CELL_FEATURES])
        self.tgt_scaler.fit(df[[TARGET_COL]])
        self._is_fitted = True
        return self

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        self._assert_fitted()
        return self.feat_scaler.transform(df[UNIT_CELL_FEATURES])

    def transform_target(self, df: pd.DataFrame) -> np.ndarray:
        self._assert_fitted()
        return self.tgt_scaler.transform(df[[TARGET_COL]]).ravel()

    def inverse_transform_features(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self.feat_scaler.inverse_transform(X)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self.tgt_scaler.inverse_transform(
            y.reshape(-1, 1)
        ).ravel()

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info(f"Scaler saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "CeSRRScaler":
        obj = joblib.load(path)
        log.info(f"Scaler loaded from {path}")
        return obj

    def _assert_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform.")


# ─────────────────────────────────────────────────────────────────────────────
# Train / validation / test splitting
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split_stratified(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    seed: int = GLOBAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split preserving rotation balance (0° and 180° proportions maintained).
    """
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for rot in df[ROTATION_COL].unique():
        sub = df[df[ROTATION_COL] == rot].sample(frac=1, random_state=seed)
        n_test = max(1, int(len(sub) * test_frac))
        test_parts.append(sub.iloc[:n_test])
        train_parts.append(sub.iloc[n_test:])
    df_train = pd.concat(train_parts).reset_index(drop=True)
    df_test  = pd.concat(test_parts).reset_index(drop=True)
    log.info(f"Split: train={len(df_train)}, test={len(df_test)}")
    return df_train, df_test


def get_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = CV_N_FOLDS,
    seed: int = GLOBAL_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return list of (train_idx, val_idx) arrays for k-fold CV.
    Falls back to LOO-CV if n_splits >= len(df).
    """
    if n_splits >= len(df):
        log.warning("Using Leave-One-Out CV (dataset smaller than n_splits).")
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        return list(loo.split(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(df))


# ─────────────────────────────────────────────────────────────────────────────
# One-hot encoding for rotation condition
# ─────────────────────────────────────────────────────────────────────────────

def encode_rotation(df: pd.DataFrame) -> np.ndarray:
    """
    Return rotation as a single binary column: 0 → 0.0, 180 → 1.0.
    Used as the conditioning variable in cVAE.
    """
    return (df[ROTATION_COL] == 180).astype(float).values.reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: build condition vector for cVAE (freq + rotation)
# ─────────────────────────────────────────────────────────────────────────────

def build_condition(df: pd.DataFrame, scaler: CeSRRScaler) -> np.ndarray:
    """
    Build 2-D condition vector [freq_scaled, rotation_binary] for PI-cVAE.
    Shape: (N, 2)
    """
    freq_scaled = scaler.transform_target(df).reshape(-1, 1)
    rot_enc = encode_rotation(df)
    return np.hstack([freq_scaled, rot_enc])
