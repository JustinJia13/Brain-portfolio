"""verify.py — reproducible verification harness for VERIFICATION_PLAN.md.

Runs the assumption tests (ids match ASSUMPTIONS.md), all on ALREADY-RUN sims
(zero new sim budget), writing one JSON per test into test_results/ and updating
CALIBRATION_REPORT.md. Predictive tests use a TIME split (train older, test
newer) — this is what makes the verdict honest and stresses transfer/decay.

Stage 0 finding baked in: the 303 sub-universe-Sharpe labels are all in March, so
sub-Sharpe cannot be time-split; the feasibility gate (T-B1) and coverage (T-B5)
run on `is_sharpe` (abundant, full span), and sub-Sharpe is flagged transfer-
UNVERIFIED.

    PY=/home/.../envs/worldquant/bin/python
    $PY -m alpha_discovery/sim_slot_gate.verify build_cache      # featurize corpus once
    $PY -m alpha_discovery/sim_slot_gate.verify b1               # the HARD GATE
    $PY -m alpha_discovery/sim_slot_gate.verify all              # every test + report
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from . import db_adapter as db
from .features import default_featurizer

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_HERE, "test_results")
CACHE = os.path.join(RESULTS, "_phi_cache.npz")
CACHE_EXPRS = os.path.join(RESULTS, "_phi_cache_exprs.json")
QLEVELS = (0.16, 0.50, 0.84)


# --------------------------------------------------------------------------- #
def _stamp(test_id: str, payload: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    payload = {"test": test_id, "utc": datetime.utcnow().isoformat(), **payload}
    path = os.path.join(RESULTS, f"{test_id}_{int(time.time())}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"  -> {path}")


def build_cache() -> None:
    """Featurize every sim with a timestamp + is_sharpe; cache phi/labels/exprs."""
    fz = default_featurizer()
    cols = {k: [] for k in ("sharpe", "ret", "turn", "fit", "subsh")}
    phis, tord, exprs = [], [], []
    for d in db.iter_master():
        e, t = d.get("expression"), d.get("logged_at")
        if not isinstance(e, str) or not t or d.get("is_sharpe") is None:
            continue
        phi = fz.featurize(e)
        if phi is None:
            continue
        try:
            ts = datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        phis.append(phi); tord.append(ts); exprs.append(e)
        cols["sharpe"].append(d.get("is_sharpe")); cols["ret"].append(d.get("is_returns"))
        cols["turn"].append(d.get("is_turnover")); cols["fit"].append(d.get("is_fitness"))
        cols["subsh"].append(d.get("check_low_sub_universe_sharpe_value"))
    a = lambda v: np.array([np.nan if x is None else x for x in v], float)
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(CACHE, X=np.array(phis), t=np.array(tord),
                        **{k: a(v) for k, v in cols.items()})
    json.dump(exprs, open(CACHE_EXPRS, "w"))
    print(f"cached {len(phis)} sims")


def _load():
    d = np.load(CACHE)
    return d, d["t"]


def _split(t, frac=0.7):
    s = t[np.argsort(t)[int(len(t) * frac)]]
    return t <= s, t > s, s


def _fit_q(X, y, q, mask):
    m = mask & np.isfinite(y)
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=250, max_leaf_nodes=31,
        min_samples_leaf=50, learning_rate=0.08, early_stopping=True,
        random_state=0).fit(X[m], y[m])


# --------------------------------------------------------------------------- #
def b1():
    """HARD GATE: does structure predict FUTURE quality? rank-IC + decile lift + baselines."""
    d, t = _load(); X = d["X"]; sh, turn, fit = d["sharpe"], d["turn"], d["fit"]
    passes = (sh >= 1.25) & (fit >= 1.0) & (turn <= 0.7)
    tr, te, _ = _split(t)
    m = _fit_q(X, sh, 0.5, tr); pred = m.predict(X[te & np.isfinite(sh)])
    yv = sh[te & np.isfinite(sh)]
    ic = float(spearmanr(pred, yv).statistic)
    teix = np.where(te & np.isfinite(sh))[0]
    base = float(passes[teix].mean())
    top = teix[np.argsort(pred)[-len(pred) // 10:]]
    lift = float(passes[top].mean() / base) if base else float("nan")
    ic_turn = float(spearmanr(turn[te & np.isfinite(sh)], yv).statistic)
    verdict = "PASS" if (ic > 0.1 and ic > abs(ic_turn) and lift > 1) else "FAIL"
    out = {"assumption": "B1", "target": "is_sharpe", "rank_IC": ic,
           "turnover_baseline_IC": ic_turn, "decile_lift": lift,
           "base_pass": base, "verdict": verdict,
           "criterion": "rank-IC>~0.1 (default, confirm) & >baseline & lift>1"}
    _stamp("B1_feasibility", out); print(f"B1 {verdict}: IC={ic:.3f} lift={lift:.2f}")
    return out


def b5():
    """Out-of-time quantile coverage per head (the real Object-B acceptance test)."""
    d, t = _load(); X = d["X"]; tr, te, _ = _split(t)
    levels = [0.1, 0.25, 0.5, 0.75, 0.84, 0.9]
    out = {"assumption": "B5", "levels": levels, "coverage": {}}
    for name in ("sharpe", "ret", "turn"):
        y = d[name]; tem = te & np.isfinite(y)
        cov = [float((y[tem] <= _fit_q(X, y, q, tr).predict(X[tem])).mean()) for q in levels]
        out["coverage"][name] = cov
        out[f"{name}_mean_abs_err"] = float(np.mean(np.abs(np.array(cov) - levels)))
    _stamp("B5_coverage", out); print("B5 coverage:", out["coverage"]); return out


def g1():
    """Turnover reliable-kill: out-of-time IC + precision at the optimistic cap edge."""
    d, t = _load(); X = d["X"]; turn = d["turn"]; tr, te, _ = _split(t)
    tem = te & np.isfinite(turn)
    p50 = _fit_q(X, turn, 0.5, tr).predict(X[tem]); p16 = _fit_q(X, turn, 0.16, tr).predict(X[tem])
    ic = float(spearmanr(p50, turn[tem]).statistic)
    flagged = p16 > db.SUBMIT_MAX_TURNOVER
    prec = float((turn[tem][flagged] > db.SUBMIT_MAX_TURNOVER).mean()) if flagged.sum() else float("nan")
    out = {"assumption": "G1", "turnover_IC": ic, "precision_at_cap": prec,
           "n_flagged": int(flagged.sum())}
    _stamp("G1_reliable_kills", out); print(f"G1: IC={ic:.3f} prec@cap={prec:.3f}"); return out


def b2():
    """Feature-group ablations: held-out IC delta per group (sufficiency)."""
    d, t = _load(); X = d["X"]; sh = d["sharpe"]; tr, te, _ = _split(t)
    from .features import N_SCALAR, N_OP_HASH, N_PATH_HASH, N_CAT
    groups = {"scalars": (0, N_SCALAR), "op_hash": (N_SCALAR, N_SCALAR + N_OP_HASH),
              "path_ngrams": (N_SCALAR + N_OP_HASH, N_SCALAR + N_OP_HASH + N_PATH_HASH),
              "field_cat": (N_SCALAR + N_OP_HASH + N_PATH_HASH, X.shape[1])}
    tem = te & np.isfinite(sh)
    full = float(spearmanr(_fit_q(X, sh, 0.5, tr).predict(X[tem]), sh[tem]).statistic)
    out = {"assumption": "B2", "full_IC": full, "drop_IC": {}}
    for g, (a, b) in groups.items():
        Xd = X.copy(); Xd[:, a:b] = 0.0
        ic = float(spearmanr(_fit_q(Xd, sh, 0.5, tr).predict(Xd[tem]), sh[tem]).statistic)
        out["drop_IC"][g] = {"IC_without": ic, "delta": full - ic}
    _stamp("B2_ablations", out); print("B2 full IC", round(full, 3), out["drop_IC"]); return out


_TESTS = {"b1": b1, "b5": b5, "g1": g1, "b2": b2}


def main(argv):
    if not argv or argv[0] == "all":
        if not os.path.exists(CACHE):
            build_cache()
        for fn in _TESTS.values():
            fn()
        return
    cmd = argv[0]
    if cmd == "build_cache":
        build_cache()
    elif cmd in _TESTS:
        _TESTS[cmd]()
    else:
        print(f"unknown: {cmd}; choices: build_cache, all, {list(_TESTS)}")


if __name__ == "__main__":
    main(sys.argv[1:])
