"""backtest.py — budget backtest (§9): slots saved vs good alphas killed.

Replays historical sims past the gate and measures the only metric that
ultimately matters: how many sim slots the gate would have saved, and at what
cost in good alphas wrongly killed, swept over beta / thresholds.

Leakage control: the surrogate was trained on a fixed 70/30 split (seed=0).  We
evaluate ONLY on the held-out 30% so the quality predictions are out-of-sample.
The coverage index (no labels) and pool (submitted book) carry no label leakage.

Efficiency: per-candidate components (phi, head quantiles, c_max, n_eff, truth)
are precomputed ONCE; each (beta, tau) point in the sweep is then cheap vectorised
arithmetic that mirrors gate.Gate.decide exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import db_adapter
from .db_adapter import TURNOVER_FLOOR
from .features import default_featurizer
from .gate import GateParams
from .metric import AlphaMetric


@dataclass
class _Components:
    # truth
    passes: np.ndarray         # bool: submission_passes
    # head medians/sigmas
    sub_q50: np.ndarray; sub_sig: np.ndarray
    sh_q50: np.ndarray;  sh_sig: np.ndarray
    rt_q50: np.ndarray;  rt_sig: np.ndarray
    to_q50: np.ndarray;  to_sig: np.ndarray
    # object A / coverage
    c_max: np.ndarray
    n_eff: np.ndarray


def precompute(am: AlphaMetric, n_eval: int = 6000, seed: int = 0,
               holdout_frac: float = 0.3, verbose: bool = True) -> _Components:
    from .signature import build_signature
    sims = db_adapter.load_sims()
    fz = default_featurizer()
    # reproduce the surrogate's split, then sample held-out for tractability
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sims))
    hold = perm[:int(len(sims) * holdout_frac)]
    rng2 = np.random.default_rng(seed + 1)
    if n_eval and len(hold) > n_eval:
        hold = hold[rng2.permutation(len(hold))[:n_eval]]

    # featurize + truth + per-candidate Object-A terms (single pass)
    phis, passes, c_max, n_eff = [], [], [], []
    for i in hold:
        r = sims[i]
        p = db_adapter.submission_passes(r)
        if p is None:
            continue
        phi = fz.featurize(r.expr)
        if phi is None:
            continue
        sig = build_signature(r.expr)
        phis.append(phi)
        passes.append(bool(p))
        c_max.append(am.redundancy.c_max(sig))
        n_eff.append(am.coverage.n_eff(phi) if am.coverage else 1e9)
    X = np.array(phis)
    if verbose:
        print(f"  featurized {len(X)} held-out candidates; batch-predicting heads...")

    # BATCH the surrogate predicts (the expensive part) over the whole matrix
    S = am.surrogate

    def head_q(t, q):
        return (S.heads[t].predict_q(X, q) if t in S.heads
                else np.full(len(X), np.nan))

    def head_pair(t):
        q50 = head_q(t, 0.50)
        sig = np.abs(head_q(t, 0.84) - q50)
        return q50, sig

    sub_q50, sub_sig = head_pair("subsharpe")
    sh_q50, sh_sig = head_pair("sharpe")
    rt_q50, rt_sig = head_pair("ret")
    to_q50, to_sig = head_pair("turnover")

    return _Components(
        passes=np.array(passes, dtype=bool),
        sub_q50=sub_q50, sub_sig=sub_sig, sh_q50=sh_q50, sh_sig=sh_sig,
        rt_q50=rt_q50, rt_sig=rt_sig, to_q50=to_q50, to_sig=to_sig,
        c_max=np.array(c_max), n_eff=np.array(n_eff),
    )


def evaluate(C: _Components, p: GateParams, *, disable_quality: bool = False,
             disable_coverage: bool = False) -> dict:
    """Vectorised replica of gate.decide -> aggregate slot/good metrics."""
    n = len(C.passes)
    sim = np.ones(n, dtype=bool)          # default simulate
    reason = np.array(["quality_pass"] * n, dtype=object)

    # (1) reliable kills (use beta_kill, decoupled from the quality UCB)
    to_lcb = C.to_q50 - p.beta_kill * C.to_sig
    kill_to = to_lcb > p.tau_to
    redund = C.c_max - p.beta_kill * p.sigma_corr > p.tau_corr
    # (3) coverage abstention (simulate) -- applied before quality, after kills
    # NB keep masks bool: `~python_bool` is int(-1) and would poison dtype -> the
    # later sim[mask]=False would become integer fancy-indexing, not masking.
    cover = (C.n_eff < p.n_min) if not disable_coverage else np.zeros(n, dtype=bool)
    # (4) quality kill
    sub_ucb = C.sub_q50 + p.beta * C.sub_sig
    fit_hi = (C.sh_q50 + p.beta * C.sh_sig) * np.sqrt(
        np.abs(C.rt_q50 + p.beta * C.rt_sig) / np.maximum(C.to_q50 - p.beta * C.to_sig, TURNOVER_FLOOR))
    # nan-safe: nan UCB means untrained head -> treat as pass (simulate)
    sub_ok = ~np.isfinite(sub_ucb) | (sub_ucb >= p.tau_sub)
    fit_ok = ~np.isfinite(fit_hi) | (fit_hi >= p.tau_fit)
    kill_quality = ~(sub_ok & fit_ok) if not disable_quality else np.zeros(n, dtype=bool)

    # resolution order matches gate.decide: kills -> coverage(sim) -> quality
    sim[:] = True
    killed = kill_to | redund
    sim[killed] = False
    # coverage forces simulate only where not already killed
    forced_sim = cover & ~killed
    # quality kill only where not killed and not coverage-forced
    q_zone = ~killed & ~forced_sim
    sim[q_zone & kill_quality] = False

    simulated = sim
    n_sim = simulated.sum()
    slots_saved = 1.0 - n_sim / n
    good = C.passes
    good_sim = (good & simulated).sum()
    good_total = good.sum()
    good_recall = good_sim / good_total if good_total else float("nan")
    good_killed_reliable = (good & (kill_to | redund)).sum()
    good_killed_quality = (good & q_zone & kill_quality).sum()

    # ISOLATE the quality gate (Object B): only candidates that actually reach the
    # quality conjunction (not reliably killed, not coverage-forced).  This is the
    # honest test of whether B enriches -- separate from redundancy dedup, which
    # deliberately removes good-but-duplicate alphas.
    qz = q_zone
    qz_n = int(qz.sum())
    qz_good = int((good & qz).sum())
    qz_sim = qz & ~kill_quality
    qz_kept = int(qz_sim.sum())
    qz_good_kept = int((good & qz_sim).sum())
    qz_base = qz_good / qz_n if qz_n else float("nan")
    qz_prec = qz_good_kept / qz_kept if qz_kept else float("nan")
    qz_recall = qz_good_kept / qz_good if qz_good else float("nan")
    qz_saved = 1.0 - qz_kept / qz_n if qz_n else float("nan")

    return {
        "n": int(n),
        "base_rate_good": float(good_total / n),
        "slots_saved": float(slots_saved),
        "good_recall": float(good_recall),
        "good_total": int(good_total),
        "good_killed_quality": int(good_killed_quality),
        "good_killed_reliable": int(good_killed_reliable),
        "kill_turnover": int(kill_to.sum()),
        "kill_redundant": int(redund.sum()),
        "coverage_forced_sim": int(forced_sim.sum()),
        "precision_kept": float(good_sim / max(n_sim, 1)),
        # quality-gate-isolated (the Object-B enrichment test)
        "qz_n": qz_n,
        "qz_base_good": float(qz_base),
        "qz_slots_saved": float(qz_saved),
        "qz_good_recall": float(qz_recall),
        "qz_precision_kept": float(qz_prec),
        "qz_enrichment": float(qz_prec / qz_base) if qz_base else float("nan"),
    }


def sweep(C: _Components, betas=(0.5, 1.0, 1.5, 2.0, 3.0), base: GateParams | None = None):
    base = base or GateParams()
    rows = []
    for b in betas:
        p = GateParams(**{**base.__dict__, "beta": b, "eps": 0.0})
        r = evaluate(C, p)
        r["beta"] = b
        rows.append(r)
    return rows


def robustness_floor(C: _Components, base: GateParams | None = None) -> dict:
    """With B's quality signal disabled, the gate must still kill via turnover +
    redundancy and otherwise simulate (good_recall ~ 1.0)."""
    base = base or GateParams()
    p = GateParams(**{**base.__dict__, "eps": 0.0})
    return evaluate(C, p, disable_quality=True)
