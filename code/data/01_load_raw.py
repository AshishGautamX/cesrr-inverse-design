"""
01_load_raw.py -- Parse all 3 XLSX files into clean wide-format CSVs.

Layout of 'Updated AI_ML Data.xlsx' and 'Updated AI_ML Data (1).xlsx':
  Each design point spans ~9 consecutive rows.
  The FIRST row of each block contains: Sl.No. | Frequency | r1_label | r1_value
  Subsequent rows contain the remaining dimension pairs (label, value).

Exact dimension labels in the file:
  'Outer radius of outer circumference (r1)'  -> r1
  'Inner radius of outer circumference (r2)'  -> r2
  'Outer radius of inner circumference (r3)'  -> r3
  'Inner radius of inner circumference (r4)'  -> r4
  'Thickness (t)'                             -> t
  'Groove width (d1)'                         -> d1
  'Metal bridge  (d2)'                        -> d2
  'Groove height (h)'                         -> h
  'Periodicity (p)'                           -> p

Usage:
    python code/data/01_load_raw.py
"""

import re
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import DATA_PROC_DIR, ROOT

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Dimension label -> column name (exact substring matching, case-insensitive)
# -----------------------------------------------------------------------------

_DIM_MAP = [
    ("outer radius of outer",  "r1"),
    ("inner radius of outer",  "r2"),
    ("outer radius of inner",  "r3"),
    ("inner radius of inner",  "r4"),
    ("thickness",              "t"),
    ("groove width",           "d1"),
    ("metal bridge",           "d2"),
    ("groove height",          "h"),
    ("periodicity",            "p"),
]


def _match_dim(text) -> str | None:
    """Return column name for a dimension label cell, or None."""
    if not text or not isinstance(text, str):
        return None
    low = text.lower().strip()
    for pattern, col in _DIM_MAP:
        if pattern in low:
            return col
    return None


def _parse_num(val) -> float | None:
    """Extract first float from a cell (string like '42.8 mm' or numeric)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if not (isinstance(val, float) and (val != val)) else None
    m = re.search(r"[-+]?\d*\.?\d+", str(val))
    return float(m.group()) if m else None


def _parse_freq(val) -> float | None:
    """Parse a frequency cell -> GHz. Handles '1 GHz', '1.2GHz', '2.454'."""
    if val is None:
        return None
    txt = str(val).strip()
    # Must contain a digit
    if not any(c.isdigit() for c in txt):
        return None
    num = _parse_num(txt)
    if num is None:
        return None
    # Values like '1', '2.454', '12' are already in GHz in this dataset
    return float(num)


def _is_freq_cell(val) -> bool:
    """True if a cell looks like a frequency entry (e.g. '1 GHz', '1.2GHz')."""
    if val is None:
        return False
    txt = str(val).strip().upper()
    return "GHZ" in txt or txt.replace(".", "").replace(" ", "").isdigit()


# -----------------------------------------------------------------------------
# Main unit-cell parser
# -----------------------------------------------------------------------------

def parse_unit_cell_xlsx(fpath: Path, rotation: int) -> pd.DataFrame:
    """
    Parse 'Updated AI_ML Data.xlsx' or 'Updated AI_ML Data (1).xlsx'.

    Strategy
    --------
    Scan every row. Whenever we find a cell matching a dimension label
    immediately followed by a numeric value in the same row, record it.
    A new record starts when we see an integer Sl.No. AND a frequency cell
    in the same row.

    Key fix: r1 appears on the SAME row as Sl.No. + freq, so we must parse
    the dimension pair from that row even as we start the new record.
    """
    import openpyxl
    wb = openpyxl.load_workbook(fpath, data_only=True)
    ws = wb.active

    records = []
    current: dict = {}
    current_freq: float | None = None

    for row in ws.iter_rows(values_only=True):
        row_vals = list(row)

        # -- Try to extract ALL dimension pairs from this row ---------------
        # Scan every adjacent (cell, cell+1) pair for (label, value)
        dims_in_row: dict = {}
        for i in range(len(row_vals) - 1):
            col = _match_dim(row_vals[i])
            if col:
                val = _parse_num(row_vals[i + 1])
                if val is not None:
                    dims_in_row[col] = val

        # -- Detect start of a new record -----------------------------------
        # Look for an integer serial number AND a freq cell in this row
        freq_found = None
        sl_found   = None
        for v in row_vals:
            if isinstance(v, str) and _is_freq_cell(v):
                f = _parse_freq(v)
                if f is not None and 0.5 <= f <= 25:
                    freq_found = f
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                iv = int(v) if float(v) == int(v) else None
                if iv is not None and 1 <= iv <= 500:
                    sl_found = iv

        if sl_found is not None and freq_found is not None:
            # Save previous record
            if current and current_freq is not None:
                rec = dict(current)
                rec["freq_ghz"] = current_freq
                rec["rotation"] = rotation
                records.append(rec)
            # Start new record -- include any dims found on this same row (r1)
            current      = dict(dims_in_row)
            current_freq = freq_found
        else:
            # Continuation row -- add any dims found
            for col, val in dims_in_row.items():
                if col not in current:
                    current[col] = val

    # Save last record
    if current and current_freq is not None:
        rec = dict(current)
        rec["freq_ghz"] = current_freq
        rec["rotation"] = rotation
        records.append(rec)

    df = pd.DataFrame(records)
    log.info("Parsed %d records from %s (rotation=%ddeg)", len(df), fpath.name, rotation)

    # Report any missing columns
    expected = ["r1","r2","r3","r4","t","d1","d2","h","p"]
    for col in expected:
        if col not in df.columns:
            log.warning("  Column '%s' not found in %s", col, fpath.name)
        else:
            n_missing = df[col].isna().sum()
            if n_missing:
                log.warning("  Column '%s': %d missing values", col, n_missing)
    return df


# -----------------------------------------------------------------------------
# Full-structure XLSX parser
# -----------------------------------------------------------------------------

_FULL_DIM_MAP = [
    ("waveguide couplers length", "L"),
    ("inner radius of waveguide",  "r_wg"),
    ("length of the circular waveguide", "P_wg"),
    ("inner radius of cesrr",      "R_uc"),
    ("separation between two",     "p_sep"),
]

_FULL_UC_MAP = [
    ("outer radius of outer", "r1"),
    ("inner radius of outer", "r2"),
    ("outer radius of inner", "r3"),
    ("inner radius of inner", "r4"),
    ("thickness",             "t"),
]


def parse_full_structure_xlsx(fpath: Path) -> pd.DataFrame:
    """
    Parse 'Full_structure_dimensions.xlsx' -- two side-by-side tables per page.
    Left  table: full waveguide dimensions (L, r_wg, P_wg, R_uc, p_sep)
    Right table: unit-cell dimensions      (r1..r4, t)
    Each record starts on the row that contains a new Sl.No. + Freq.
    """
    import openpyxl
    wb = openpyxl.load_workbook(fpath, data_only=True)
    ws = wb.active

    records = []
    current: dict = {}
    current_freq: float | None = None

    for row in ws.iter_rows(values_only=True):
        row_vals = list(row)

        # Extract all dimension pairs
        dims_in_row: dict = {}
        for i in range(len(row_vals) - 1):
            label = row_vals[i]
            if not isinstance(label, str):
                continue
            low = label.lower().strip()
            for pattern, col in (_FULL_DIM_MAP + _FULL_UC_MAP):
                if pattern in low:
                    val = _parse_num(row_vals[i + 1])
                    if val is not None:
                        dims_in_row[col] = val

        # Detect new record
        freq_found, sl_found = None, None
        for v in row_vals:
            if isinstance(v, str) and _is_freq_cell(v):
                f = _parse_freq(v)
                if f and 0.5 <= f <= 25:
                    freq_found = f
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                iv = int(v) if float(v) == int(v) else None
                if iv and 1 <= iv <= 200:
                    sl_found = iv

        if sl_found is not None and freq_found is not None:
            if current and current_freq is not None:
                rec = dict(current)
                rec["freq_ghz"] = current_freq
                records.append(rec)
            current      = dict(dims_in_row)
            current_freq = freq_found
        else:
            for col, val in dims_in_row.items():
                if col not in current:
                    current[col] = val

    if current and current_freq is not None:
        rec = dict(current)
        rec["freq_ghz"] = current_freq
        records.append(rec)

    df = pd.DataFrame(records)
    log.info("Parsed %d full-structure records from %s", len(df), fpath.name)
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    def _find(fname):
        for d in [ROOT / "data" / "raw", ROOT]:
            p = d / fname
            if p.exists():
                return p
        raise FileNotFoundError(f"Cannot find raw file: {fname}")

    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)

    df_0 = parse_unit_cell_xlsx(_find("Updated AI_ML Data.xlsx"), rotation=0)
    df_0.to_csv(DATA_PROC_DIR / "raw_unit_0deg.csv", index=False)
    log.info("Saved: raw_unit_0deg.csv (%d rows)", len(df_0))

    df_180 = parse_unit_cell_xlsx(_find("Updated AI_ML Data (1).xlsx"), rotation=180)
    df_180.to_csv(DATA_PROC_DIR / "raw_unit_180deg.csv", index=False)
    log.info("Saved: raw_unit_180deg.csv (%d rows)", len(df_180))

    df_full = parse_full_structure_xlsx(_find("Full_structure_dimensions.xlsx"))
    df_full.to_csv(DATA_PROC_DIR / "raw_full_structure.csv", index=False)
    log.info("Saved: raw_full_structure.csv (%d rows)", len(df_full))

    df_combined = pd.concat([df_0, df_180], ignore_index=True)
    df_combined.to_csv(DATA_PROC_DIR / "raw_unit_combined.csv", index=False)
    log.info("Saved: raw_unit_combined.csv (%d rows)", len(df_combined))

    print("Step 01 complete. Files written to:", DATA_PROC_DIR)


if __name__ == "__main__":
    main()
