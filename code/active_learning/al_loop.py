"""
al_loop.py -- Active Learning orchestration loop.

Algorithm:
  1. Start with D_init (small labelled subset from D1)
  2. Fit PI-cVAE on D_init
  3. Score unlabelled pool via MC-Dropout BALD acquisition
  4. Query top-k points -> label via analytical LF oracle (cheap)
  5. Add to labelled set; repeat for N_ROUNDS
  6. Log MAE vs. #oracle_queries curve (paper Figure 5)

The oracle is the LF analytical model (not CST). This demonstrates how
AL can reduce the expensive CST simulation budget in practice.
"""

import sys
import json
import logging
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

_code_dir = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_dir))

from utils.config import (
    UNIT_CELL_FEATURES, TARGET_COL, ROTATION_COL,
    AL_N_INITIAL, AL_N_QUERY_ROUNDS, AL_N_QUERY_PER_ROUND,
    AL_ACQUISITION, AL_RANDOM_SEED, RESULTS_DIR
)
from utils.data_utils import load_processed, train_test_split_stratified
from analytical.lf_oracle import LFOracle
from models.generative.cvae_pi import PICVAEModel
from models.uncertainty.mc_dropout import mc_dropout_uncertainty
from active_learning.query_strategy import select_query_points
from active_learning.al_metrics import ALMetricsTracker
from evaluation.metrics import geometry_feasibility_rate

log = logging.getLogger(__name__)


class ActiveLearningLoop:
    """
    AL loop orchestrator for the PI-cVAE model.

    Parameters
    ----------
    model_cls      : model class with fit() and get_uncertainty() methods
    model_kwargs   : kwargs forwarded to model_cls constructor
    oracle         : LFOracle instance for cheap labelling
    strategy       : acquisition function ('bald' | 'uncertainty' | 'ei')
    n_initial      : size of initial labelled set
    n_rounds       : number of AL rounds
    n_query        : queries per round
    """

    def __init__(
        self,
        model_cls=None,
        model_kwargs: dict = None,
        oracle: LFOracle = None,
        strategy: str = AL_ACQUISITION,
        n_initial: int = AL_N_INITIAL,
        n_rounds: int = AL_N_QUERY_ROUNDS,
        n_query: int = AL_N_QUERY_PER_ROUND,
        seed: int = AL_RANDOM_SEED,
    ):
        self.model_cls    = model_cls or PICVAEModel
        self.model_kwargs = model_kwargs or {
            "use_soft_physics": True,
            "use_hard_projection": True,
        }
        self.oracle    = oracle or LFOracle()
        self.strategy  = strategy
        self.n_initial = n_initial
        self.n_rounds  = n_rounds
        self.n_query   = n_query
        self.seed      = seed
        self.tracker   = ALMetricsTracker()

    def _initial_labelled_set(self, df_pool: pd.DataFrame) -> tuple:
        """
        Split pool into initial labelled subset + remaining unlabelled pool.
        Initial set is sampled uniformly (random seed fixed).
        """
        rng   = np.random.default_rng(self.seed)
        idx   = rng.choice(len(df_pool), size=self.n_initial, replace=False)
        mask  = np.zeros(len(df_pool), dtype=bool)
        mask[idx] = True
        return df_pool[mask].reset_index(drop=True), df_pool[~mask].reset_index(drop=True)

    def run(
        self,
        df_hf: pd.DataFrame,
        df_test: pd.DataFrame,
        df_lhs_pool: pd.DataFrame,
    ) -> ALMetricsTracker:
        """
        Execute the full AL loop.

        Parameters
        ----------
        df_hf       : HF labelled dataset (D1_validated)
        df_test     : held-out test set (never used for training)
        df_lhs_pool : unlabelled LHS candidates

        Returns
        -------
        ALMetricsTracker with per-round metrics
        """
        # Initialise labelled set from HF data
        df_labelled, df_unlabelled = self._initial_labelled_set(df_hf)
        n_oracle_queries = 0
        labelled_idx = set()

        log.info(
            "AL loop start | initial_labelled=%d, pool=%d, rounds=%d, k=%d",
            len(df_labelled), len(df_lhs_pool), self.n_rounds, self.n_query,
        )

        for round_i in range(self.n_rounds + 1):  # +1 for round 0 (baseline)
            # -- Train model on current labelled set -----------------------
            model = self.model_cls(**self.model_kwargs)
            model.fit(df_labelled)

            # -- Evaluate on test set --------------------------------------
            eval_result = model.evaluate(df_test, n_samples=10)
            feas = eval_result.get("feasibility_rate", np.nan)

            # Compute forward MAE via GP on labelled set (proxy)
            # (PI-cVAE is a generative model; we compute reconstruction MAE)
            X_recon = model.sample_geometry(df_test.head(20), n_samples=1)
            feas_test = geometry_feasibility_rate(X_recon)

            self.tracker.record(
                round_idx=round_i,
                n_oracle_queries=n_oracle_queries,
                n_labelled=len(df_labelled),
                feasibility_rate=feas,
                extra=eval_result,
            )
            log.info(
                "Round %2d | labelled=%3d | queries=%3d | feasibility=%.3f",
                round_i, len(df_labelled), n_oracle_queries, feas,
            )

            if round_i == self.n_rounds:
                break  # final eval done, stop

            # -- Score unlabelled pool ------------------------------------
            uncertainties = mc_dropout_uncertainty(
                model._model, df_lhs_pool, model.scaler
            )

            selected = select_query_points(
                strategy=self.strategy,
                n_query=self.n_query,
                exclude_idx=labelled_idx,
                uncertainties=uncertainties,
            )

            # -- Label selected points via oracle -------------------------
            new_records = []
            for idx in selected:
                if idx in labelled_idx:
                    continue
                row = df_lhs_pool.iloc[idx]
                geom = {c: row[c] for c in UNIT_CELL_FEATURES if c in row.index}
                freq = self.oracle.predict_freq(**geom)
                if freq is None:
                    continue
                rec = dict(geom)
                rec[TARGET_COL]   = freq
                rec[ROTATION_COL] = int(row.get(ROTATION_COL, 0))
                new_records.append(rec)
                labelled_idx.add(idx)
                n_oracle_queries += 1

            if new_records:
                df_new = pd.DataFrame(new_records)
                df_labelled = pd.concat(
                    [df_labelled, df_new], ignore_index=True
                )
                log.info("  Added %d oracle-labelled points.", len(new_records))

        log.info("AL loop complete. Total oracle queries: %d", n_oracle_queries)
        return self.tracker


def run_al_experiment(save: bool = True) -> ALMetricsTracker:
    """
    End-to-end convenience function to run the AL experiment.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df_hf = load_processed("hf")
    df_hf_tr, df_hf_te = train_test_split_stratified(df_hf, test_frac=0.2)
    df_pool = load_processed("lf")  # LHS candidates with LF labels

    loop    = ActiveLearningLoop()
    tracker = loop.run(df_hf_tr, df_hf_te, df_pool)

    if save:
        out = RESULTS_DIR / "al_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(tracker.to_dict(), f, indent=2)
        log.info("AL results saved: %s", out)

    return tracker


if __name__ == "__main__":
    tracker = run_al_experiment()
    print(tracker.summary())
