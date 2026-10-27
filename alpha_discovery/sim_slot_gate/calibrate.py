"""calibrate.py — Object A calibration: psi (corr_model -> realized corr) + sigma_corr.

Pure numpy (the runtime env has numpy only — no scipy/sklearn).  We fit the
isotonic map psi on |corr_model| -> |realized corr| because the redundancy term
uses |psi(corr_model)| (a negated twin is still a duplicate, §3.6).

Calibration data (Discovery "Case B-plus"):
  * self-corr triples  ~3.5k  (X, Y_argmax, max_self_correlation)   high-corr biased
  * pool PnL anchor    ~0.5k  (exact pairwise corr(pnl) among pool)  full range

Go/no-go (§9): held-out Spearman of |corr_model| vs |realized corr| must be a
clear positive monotone relationship, else the structural metric is uninformative
for redundancy here and we say so loudly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from . import db_adapter
from .aggregate import CorrModel
from .signature import build_signature

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAMS_PATH = os.path.join(_HERE, "alpha_discovery/sim_slot_gate", "metric_params.json")


# --------------------------------------------------------------------------- #
# Isotonic map (sklearn) + Spearman (scipy)
# --------------------------------------------------------------------------- #
class Isotonic:
    """Monotone non-decreasing map, fit by sklearn IsotonicRegression."""

    def __init__(self, xk: np.ndarray, yk: np.ndarray):
        self.xk = np.asarray(xk, float)
        self.yk = np.asarray(yk, float)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "Isotonic":
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        ir.fit(np.asarray(x, float), np.asarray(y, float))
        return cls(ir.X_thresholds_, ir.y_thresholds_)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.xk, self.yk, left=self.yk[0], right=self.yk[-1])

    def to_dict(self) -> dict:
        return {"xk": self.xk.tolist(), "yk": self.yk.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Isotonic":
        return cls(np.array(d["xk"]), np.array(d["yk"]))

    @classmethod
    def identity(cls) -> "Isotonic":
        g = np.linspace(0.0, 1.0, 11)
        return cls(g, g)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    rho = spearmanr(a, b).statistic
    return float(rho)


# --------------------------------------------------------------------------- #
# Calibration driver
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    psi: Isotonic
    sigma_corr: float
    report: dict = field(default_factory=dict)


def _corr_model_for_expr_pairs(pairs, model: CorrModel, max_pairs: int | None = None):
    """pairs: list[(x_expr, y_expr, realized)] -> (xcm, yreal) arrays (skips unparseable)."""
    sig_cache: dict[str, object] = {}

    def sig(e):
        s = sig_cache.get(e, 0)
        if s == 0:
            s = build_signature(e)
            sig_cache[e] = s
        return s

    xcm, yreal = [], []
    n = 0
    for x, y, r in pairs:
        if max_pairs and n >= max_pairs:
            break
        sx, sy = sig(x), sig(y)
        if sx is None or sy is None:
            continue
        cm = model.corr_model_sig(sx, sy)
        xcm.append(abs(cm))
        yreal.append(abs(float(r)))
        n += 1
    return np.array(xcm), np.array(yreal)


def calibrate(model: CorrModel | None = None, holdout_frac: float = 0.3, seed: int = 0,
              max_self_pairs: int | None = None) -> CalibrationResult:
    model = model or CorrModel()
    rng = np.random.default_rng(seed)

    # --- self-corr triples (expr pairs) --------------------------------------
    self_pairs = db_adapter.load_self_corr_pairs()
    xs_self, ys_self = _corr_model_for_expr_pairs(self_pairs, model, max_self_pairs)

    # --- pool PnL anchor (id pairs -> expr pairs) ----------------------------
    id2expr = db_adapter.build_id_to_expr()
    pnl_pairs_ids = db_adapter.pool_pnl_corr_pairs()
    pnl_expr_pairs = [
        (id2expr[a], id2expr[b], r)
        for a, b, r in pnl_pairs_ids
        if a in id2expr and b in id2expr
    ]
    xs_pnl, ys_pnl = _corr_model_for_expr_pairs(pnl_expr_pairs, model)

    # --- combine, split, fit -------------------------------------------------
    xs = np.concatenate([xs_self, xs_pnl]) if len(xs_pnl) else xs_self
    ys = np.concatenate([ys_self, ys_pnl]) if len(ys_pnl) else ys_self
    n = len(xs)
    perm = rng.permutation(n)
    n_hold = int(n * holdout_frac)
    hold, train = perm[:n_hold], perm[n_hold:]

    psi = Isotonic.fit(xs[train], ys[train])

    # held-out diagnostics
    sp_raw = spearman(xs[hold], ys[hold])
    yhat = psi.predict(xs[hold])
    resid = ys[hold] - yhat
    sigma_corr = float(np.std(resid)) if len(resid) else 0.2
    mse_psi = float(np.mean(resid ** 2)) if len(resid) else float("nan")
    mse_id = float(np.mean((ys[hold] - xs[hold]) ** 2)) if len(hold) else float("nan")
    pear = float(np.corrcoef(xs[hold], ys[hold])[0, 1]) if len(hold) > 2 else float("nan")

    # Regime-conditional realized corr: the DECISION-relevant view.  The gate
    # only kills on HIGH corr_model, so what matters is "conditional on
    # corr_model >= t, is realized corr reliably elevated?" (not the global
    # Spearman, which is dragged down by a factor-driven PnL-corr floor on the
    # structurally-dissimilar majority).
    regime = {}
    for t in (0.0, 0.3, 0.5, 0.7, 0.85):
        msk = xs >= t
        if msk.sum() >= 3:
            regime[f">={t}"] = {
                "n": int(msk.sum()),
                "realized_mean": float(ys[msk].mean()),
                "realized_median": float(np.median(ys[msk])),
            }

    report = {
        "n_self_pairs": int(len(xs_self)),
        "n_pnl_pairs": int(len(xs_pnl)),
        "n_total": int(n),
        "n_holdout": int(n_hold),
        "spearman_holdout": sp_raw,
        "pearson_holdout": pear,
        "mse_psi": mse_psi,
        "mse_identity": mse_id,
        "psi_improves_over_identity": bool(mse_psi < mse_id),
        "sigma_corr": sigma_corr,
        "self_corr_realized_mean": float(np.mean(ys_self)) if len(ys_self) else None,
        "pnl_corr_realized_mean": float(np.mean(ys_pnl)) if len(ys_pnl) else None,
        "regime_conditional_realized": regime,
    }
    return CalibrationResult(psi=psi, sigma_corr=sigma_corr, report=report)


def save_params(result: CalibrationResult, path: str = PARAMS_PATH) -> None:
    payload = {
        "version": 1,
        "psi": result.psi.to_dict(),
        "sigma_corr": result.sigma_corr,
        "report": result.report,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_params(path: str = PARAMS_PATH):
    if not os.path.exists(path):
        return Isotonic.identity(), 0.2
    with open(path) as fh:
        d = json.load(fh)
    return Isotonic.from_dict(d["psi"]), float(d.get("sigma_corr", 0.2))
