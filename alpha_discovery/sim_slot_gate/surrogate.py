"""surrogate.py — OBJECT B: submission-component surrogate (§7).

Predicts the components BRAIN's submission test checks, from one alpha's structure
phi(X), as calibrated quantiles.  The headline requirement is CALIBRATION not
accuracy (§7.4): the gate trusts predictive *intervals*, never point estimates,
so a weak-but-calibrated model is the goal and the acceptance test is quantile
coverage.

Heads (quantile gradient-boosting at levels 0.16/0.50/0.84, sigma ~ q84-q50):
    subsharpe  <- sub-universe Sharpe   (tightest must-pass gate; label-sparse)
    sharpe     <- main-universe Sharpe  (feeds fitness)
    ret        <- returns               (feeds fitness)
    turnover   <- turnover              (near-deterministic in structure; reliable kill)

Fitness is RECONSTRUCTED, not regressed (§7.3):
    fitness = sharpe * sqrt(|ret| / max(turnover, 0.125))
with the optimistic edge pushed through the monotone formula.

Learner: sklearn HistGradientBoostingRegressor(loss="quantile") — one model per
(target, quantile).  Histogram boosting handles the ~50-dim phi and tens of
thousands of rows quickly and gives proper conditional-quantile estimates.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import db_adapter
from .db_adapter import TURNOVER_FLOOR
from .features import default_featurizer

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(_HERE, "alpha_discovery/sim_slot_gate", "surrogate.pkl")

QUANTILES = (0.16, 0.50, 0.84)


def _make_gbr(q: float, n: int) -> HistGradientBoostingRegressor:
    """Quantile regressor; regularised harder when labels are sparse (subsharpe)."""
    if n < 1000:
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, learning_rate=0.05, max_iter=120,
            max_depth=2, min_samples_leaf=max(8, n // 25), l2_regularization=1.0,
            early_stopping=False, random_state=0,
        )
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, learning_rate=0.08, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=0.5,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=0,
    )


class Head:
    """A target's quantile regressors at QUANTILES."""

    def __init__(self):
        self.models: dict[float, HistGradientBoostingRegressor] = {}

    def fit(self, X, y):
        for q in QUANTILES:
            self.models[q] = _make_gbr(q, len(y)).fit(X, y)
        # enforce monotone non-crossing quantiles at predict time (handled in predict_q)
        return self

    def predict_q(self, X, q):
        return self.models[q].predict(X)


class Surrogate:
    TARGETS = ("subsharpe", "sharpe", "ret", "turnover")

    def __init__(self):
        self.heads: dict[str, Head] = {}
        self.featurizer = default_featurizer()
        self.coverage: dict = {}

    # ---- training -------------------------------------------------------- #
    def _design_matrix(self, records):
        X, ys = [], {t: [] for t in self.TARGETS}
        for rec in records:
            phi = self.featurizer.featurize(rec.expr)
            if phi is None:
                continue
            X.append(phi)
            ys["sharpe"].append(rec.sharpe)
            ys["ret"].append(rec.ret)
            ys["turnover"].append(rec.turnover)
            ys["subsharpe"].append(rec.sub_sharpe)
        return np.array(X), {t: np.array(v, dtype=object) for t, v in ys.items()}

    def fit(self, records=None, holdout_frac=0.3, seed=0, verbose=True):
        records = records or db_adapter.load_sims()
        X, ys = self._design_matrix(records)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        n_hold = int(len(X) * holdout_frac)
        hold, train = perm[:n_hold], perm[n_hold:]

        report = {}
        for t in self.TARGETS:
            yt = ys[t]
            mask = np.array([v is not None for v in yt])
            tr = train[mask[train]]
            ho = hold[mask[hold]]
            if len(tr) < 60:
                report[t] = {"trained": False, "n": int(len(tr))}
                if verbose:
                    print(f"  head {t:10s} SKIPPED (only {len(tr)} labels)")
                continue
            yv = yt.astype(float)
            self.heads[t] = Head().fit(X[tr], yv[tr])
            cov = {}
            for q in QUANTILES:
                pred = self.heads[t].predict_q(X[ho], q)
                cov[q] = float(np.mean(yv[ho] <= pred)) if len(ho) else float("nan")
            # turnover-at-cap precision (the head expected to be genuinely accurate)
            extra = {}
            if t == "turnover" and len(ho):
                pred50 = self.heads[t].predict_q(X[ho], 0.50)
                flagged = pred50 > db_adapter.SUBMIT_MAX_TURNOVER
                if flagged.sum():
                    extra["precision_at_cap"] = float(
                        np.mean(yv[ho][flagged] > db_adapter.SUBMIT_MAX_TURNOVER))
            report[t] = {
                "trained": True, "n_train": int(len(tr)), "n_hold": int(len(ho)),
                "coverage": cov, **extra,
            }
            if verbose:
                print(f"  head {t:10s} n_train={len(tr):6d} n_hold={len(ho):5d} "
                      f"coverage@(.16/.50/.84)={[round(cov[q],3) for q in QUANTILES]}"
                      + (f" prec@cap={extra['precision_at_cap']:.2f}" if extra else ""))
        self.coverage = report
        return report

    # ---- inference ------------------------------------------------------- #
    def q(self, target: str, level: float, phi: np.ndarray) -> float:
        if target not in self.heads:
            return float("nan")
        return float(self.heads[target].predict_q(phi.reshape(1, -1), level)[0])

    def sigma(self, target: str, phi: np.ndarray) -> float:
        if target not in self.heads:
            return float("inf")  # untrained -> infinite uncertainty -> simulate
        X = phi.reshape(1, -1)
        hi = float(self.heads[target].predict_q(X, 0.84)[0])
        mid = float(self.heads[target].predict_q(X, 0.50)[0])
        return abs(hi - mid)

    def fitness_upper(self, phi: np.ndarray, beta: float) -> float:
        """Optimistic fitness: push each input's optimistic edge through §7.3 formula."""
        sh_hi = self.q("sharpe", 0.50, phi) + beta * self.sigma("sharpe", phi)
        rt_hi = self.q("ret", 0.50, phi) + beta * self.sigma("ret", phi)
        to_lo = self.q("turnover", 0.50, phi) - beta * self.sigma("turnover", phi)
        if not np.isfinite([sh_hi, rt_hi, to_lo]).all():
            return float("inf")
        return sh_hi * np.sqrt(abs(rt_hi) / max(to_lo, TURNOVER_FLOOR))

    # ---- persistence ----------------------------------------------------- #
    def save(self, path: str = MODEL_PATH):
        with open(path, "wb") as fh:
            pickle.dump({"heads": self.heads, "coverage": self.coverage}, fh)
        with open(path.replace(".pkl", "_coverage.json"), "w") as fh:
            json.dump(self.coverage, fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "Surrogate":
        s = cls()
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        s.heads = d["heads"]
        s.coverage = d.get("coverage", {})
        return s
