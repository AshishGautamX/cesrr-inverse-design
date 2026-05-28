"""
al_metrics.py -- Tracker and plotter for AL efficiency curves (paper Figure 5).

Tracks per-round metrics and produces:
  - MAE / feasibility rate vs. #oracle_queries
  - Comparison: AL vs. random sampling (control)
"""

import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import FIGURES_DIR, RESULTS_DIR

log = logging.getLogger(__name__)


@dataclass
class ALRoundRecord:
    round_idx: int
    n_oracle_queries: int
    n_labelled: int
    feasibility_rate: float
    extra: dict = field(default_factory=dict)


class ALMetricsTracker:
    """Collects per-round AL metrics and provides summary/export."""

    def __init__(self, name: str = "AL"):
        self.name    = name
        self.records: List[ALRoundRecord] = []

    def record(
        self,
        round_idx: int,
        n_oracle_queries: int,
        n_labelled: int,
        feasibility_rate: float,
        extra: dict = None,
    ):
        rec = ALRoundRecord(
            round_idx=round_idx,
            n_oracle_queries=n_oracle_queries,
            n_labelled=n_labelled,
            feasibility_rate=feasibility_rate,
            extra=extra or {},
        )
        self.records.append(rec)

    def summary(self) -> str:
        lines = [f"\n-- {self.name} AL Efficiency --"]
        lines.append(f"{'Round':>6} {'Queries':>8} {'Labelled':>9} {'Feasibility':>12}")
        for r in self.records:
            lines.append(
                f"{r.round_idx:6d} {r.n_oracle_queries:8d} "
                f"{r.n_labelled:9d} {r.feasibility_rate:12.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "records": [asdict(r) for r in self.records],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ALMetricsTracker":
        t = cls(name=d["name"])
        for rec in d["records"]:
            t.records.append(ALRoundRecord(**rec))
        return t

    def queries_array(self) -> np.ndarray:
        return np.array([r.n_oracle_queries for r in self.records])

    def feasibility_array(self) -> np.ndarray:
        return np.array([r.feasibility_rate for r in self.records])


def plot_al_curve(
    trackers: list,
    out_path: Path = None,
    title: str = "Active Learning Efficiency (Figure 5)",
):
    """
    Plot feasibility rate vs. #oracle_queries for one or more AL strategies.

    Parameters
    ----------
    trackers : list of ALMetricsTracker instances (one per strategy)
    out_path : save path for the figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or (FIGURES_DIR / "fig05_al_efficiency.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#e07b54", "#5b8db8", "#6dbf67", "#9966cc", "#f0c040"]

    for tracker, color in zip(trackers, colors):
        q = tracker.queries_array()
        f = tracker.feasibility_array() * 100
        ax.plot(q, f, "o-", color=color, linewidth=2,
                markersize=5, label=tracker.name)

    ax.set_xlabel("Cumulative oracle queries", fontsize=11)
    ax.set_ylabel("Geometry feasibility rate (%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Figure 5 saved: %s", out_path)


def compare_al_vs_random(
    al_tracker: ALMetricsTracker,
    random_tracker: ALMetricsTracker,
    out_path: Path = None,
):
    """Compare BALD AL vs. random sampling side-by-side."""
    al_tracker.name    = "BALD (AL)"
    random_tracker.name = "Random sampling"
    plot_al_curve([al_tracker, random_tracker], out_path=out_path)
