"""cutover.py — live shadow -> authoritative trigger (SHADOW_CUTOVER_TRIGGER.md).

Cutover conditions 1 & 2 (post-prune frontier; challenger beats the inert q-table)
are MET offline. This module is condition 3: prove it LIVE, then flip, then guard.

Three live conditions, each with power (achieved CI half-width vs target):
  1. quality-zone recall   >= 0.88 (backtest 0.91 - margin)   [PRIMARY, outcome]
  2. calibration coverage  mean|err| <= 0.05                  [HOLD on fail]
  3. novelty (dist. shift) novel-rate < ~10%                  [HOLD/retrain on fail]

Everything is recomputed from the shadow log's RAW per-head quantiles, so beta is a
post-hoc knob. Realized outcomes are joined from master.jsonl (the q-table is inert,
so every shadow candidate is simulated and lands there).

Ship-mode state machine (cutover_state.json):
  shadow   -> log only; q-table authoritative (effectively simulate-all)
  reliable -> dedup c_max + turnover hard-kill are LIVE (adopt-now); quality shadow
  full     -> quality surrogate authoritative (post-cutover)
Auto-rollback (full -> reliable) on a rolling breach (§5).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import numpy as np

from . import db_adapter as db
from .gate import GateParams

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(_HERE, "alpha_discovery/sim_slot_gate", "cutover_state.json")
SHADOW_LOG = os.path.join(_HERE, "artifacts", "gate_shadow.jsonl")
RESULTS = os.path.join(_HERE, "test_results")

IN_ZONE_REASONS = {"quality_pass_simulate", "kill_quality"}
LIVE_SKIP_REASONS = {
    "shadow": set(),
    "reliable": {"kill_turnover", "kill_redundant"},
    "full": {"kill_turnover", "kill_redundant", "kill_quality"},
}

# pass thresholds (SHADOW_CUTOVER_TRIGGER.md §2); defaults — confirm with user
RECALL_FLOOR = 0.88
COVERAGE_MAX_ERR = 0.05
NOVEL_RATE_MAX = 0.10
ROLLBACK = {"recall": 0.85, "cov_err": 0.07, "novel": 0.20}


# --------------------------------------------------------------------------- #
# Ship-mode state
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass
    return {"mode": "shadow", "beta": 2.5, "updated": None, "note": "default"}


def save_state(mode: str, beta: float, note: str = "") -> dict:
    st = {"mode": mode, "beta": float(beta), "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note}
    with open(STATE_PATH, "w") as fh:
        json.dump(st, fh, indent=2)
    return st


def should_live_skip(decision: dict, mode: str) -> bool:
    """Whether the live pipeline should actually skip, given the ship mode."""
    return decision.get("reason") in LIVE_SKIP_REASONS.get(mode, set())


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #
def _half_width(p_hat: float, n: int) -> float:
    if n <= 0:
        return float("inf")
    return 1.96 * math.sqrt(max(p_hat * (1 - p_hat), 1e-9) / n)


def _powered(p_hat: float, n: int, target_h: float, min_n: int = 30) -> bool:
    """A condition is powered only when the CI is tight enough AND there is a real
    sample. Clamp p_hat away from {0,1}: with n=1 the Wald variance p(1-p) collapses
    to 0, which would FALSELY declare 1 observation 'powered'. The clamp + min_n
    guard is the fix for that degeneracy."""
    if n < min_n or math.isnan(p_hat):
        return False
    pc = min(max(p_hat, 0.1), 0.9)
    return _half_width(pc, n) <= target_h


# --------------------------------------------------------------------------- #
# Join shadow decisions <-> realized outcomes
# --------------------------------------------------------------------------- #
def _load_realized() -> dict[str, dict]:
    """expression -> realized {sharpe, ret, turn, fit, subsh, passed}."""
    out: dict[str, dict] = {}
    for r in db.load_sims():
        if r.sharpe is None or r.turnover is None or r.fitness is None:
            continue
        passed = db.submission_passes(r)
        out[r.expr.strip()] = {"sharpe": r.sharpe, "ret": r.ret, "turn": r.turnover,
                               "fit": r.fitness, "subsh": r.sub_sharpe, "passed": bool(passed)}
    return out


def _load_shadow(log_path: str) -> list[dict]:
    rows = []
    if not os.path.exists(log_path):
        return rows
    with open(log_path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _keep_in_zone(raw: dict, beta: float, p: GateParams) -> bool:
    """Recompute the quality keep/kill from raw quantiles at an arbitrary beta."""
    sub = raw.get("subsharpe"); sh = raw.get("sharpe"); rt = raw.get("ret"); to = raw.get("turnover")
    def ucb(q3):
        return q3[1] + beta * (q3[2] - q3[1]) if q3 and np.isfinite(q3[1]) else float("nan")
    sub_ucb = ucb(sub)
    if sh and rt and to and np.isfinite([sh[1], rt[1], to[1]]).all():
        fit_hi = (sh[1] + beta * (sh[2] - sh[1])) * math.sqrt(
            abs(rt[1] + beta * (rt[2] - rt[1])) / max(to[1] - beta * (to[2] - to[1]), db.TURNOVER_FLOOR))
    else:
        fit_hi = float("nan")
    sub_ok = (not np.isfinite(sub_ucb)) or (sub_ucb >= p.tau_sub)
    fit_ok = (not np.isfinite(fit_hi)) or (fit_hi >= p.tau_fit)
    return sub_ok and fit_ok


# --------------------------------------------------------------------------- #
# The monitor
# --------------------------------------------------------------------------- #
def monitor(beta: float | None = None, log_path: str = SHADOW_LOG, p: GateParams | None = None,
            update_state: bool = False) -> dict:
    p = p or GateParams()
    beta = beta if beta is not None else load_state().get("beta", 2.5)
    rows = _load_shadow(log_path)
    realized = _load_realized()

    joined = []
    for r in rows:
        e = (r.get("expression") or "").strip()
        rz = realized.get(e)
        if rz is None:
            continue
        dec = r.get("decision", {})
        joined.append({"reason": dec.get("reason"), "raw": dec.get("raw", {}),
                       "n_eff": dec.get("n_eff"), "c_max": dec.get("c_max"), **rz})

    rep = {"beta": beta, "n_shadow": len(rows), "n_joined": len(joined)}
    if not joined:
        rep["recommendation"] = "KEEP-ACCUMULATING (0 joined live labels)"
        _persist(rep)
        return rep

    inzone = [j for j in joined if j["reason"] in IN_ZONE_REASONS]
    inzone_good = [j for j in inzone if j["passed"]]

    # ---- Condition 1: quality-zone recall ----------------------------------
    if inzone_good:
        kept = [_keep_in_zone(j["raw"], beta, p) for j in inzone_good]
        recall = float(np.mean(kept))
        n1 = len(inzone_good); h1 = _half_width(recall, n1)
    else:
        recall, n1, h1 = float("nan"), 0, float("inf")
    # slots saved over the whole live set at this beta (incl reliable kills + coverage)
    def _skip(j):
        if j["reason"] in ("kill_turnover", "kill_redundant"):
            return True
        if j["reason"] == "coverage_simulate" or j["reason"] == "eps_explore":
            return False
        if j["reason"] in IN_ZONE_REASONS:
            return not _keep_in_zone(j["raw"], beta, p)
        return False
    slots_saved = float(np.mean([_skip(j) for j in joined]))
    cond1_pass = (not math.isnan(recall)) and recall >= RECALL_FLOOR
    cond1_powered = _powered(recall, n1, 0.04)

    # ---- Condition 2: calibration coverage (q16/q50/q84 from raw) -----------
    levels = {"q16": (0, 0.16), "q50": (1, 0.50), "q84": (2, 0.84)}
    head_key = {"sharpe": "sharpe", "ret": "ret", "turn": "turnover", "subsh": "subsharpe"}
    cov_errs = []
    cov_detail = {}
    for real_field, raw_head in head_key.items():
        vals = [(j["raw"].get(raw_head), j[real_field]) for j in joined
                if j["raw"].get(raw_head) and j.get(real_field) is not None and np.isfinite(j.get(real_field, np.nan))]
        if len(vals) < 30:
            continue
        per = {}
        for name, (idx, q) in levels.items():
            emp = float(np.mean([1.0 if real <= q3[idx] else 0.0 for q3, real in vals]))
            per[name] = round(emp, 3); cov_errs.append(abs(emp - q))
        cov_detail[real_field] = {"n": len(vals), **per}
    cov_err = float(np.mean(cov_errs)) if cov_errs else float("nan")
    n2 = len(joined); h2 = _half_width(0.9, n2)
    cond2_pass = (not math.isnan(cov_err)) and cov_err <= COVERAGE_MAX_ERR
    cond2_powered = (not math.isnan(cov_err)) and h2 <= 0.03

    # ---- Condition 3: novelty / distribution shift -------------------------
    neffs = [j["n_eff"] for j in joined if j["n_eff"] is not None]
    novel_rate = float(np.mean([n < p.n_min for n in neffs])) if neffs else float("nan")
    novel = [j for j in joined if j["n_eff"] is not None and j["n_eff"] < p.n_min]
    novel_err = None
    if len(novel) >= 30:
        e = [abs(j["raw"]["sharpe"][1] - j["sharpe"]) for j in novel
             if j["raw"].get("sharpe") and np.isfinite(j.get("sharpe", np.nan))]
        novel_err = float(np.mean(e)) if e else None
    cond3_pass = (not math.isnan(novel_rate)) and novel_rate < NOVEL_RATE_MAX
    # need enough total n_eff observations to trust the rate estimate
    cond3_powered = len(neffs) >= 200

    n_needed = int(0.91 * 0.09 * (1.96 / 0.04) ** 2)
    rep.update({
        "base_good_live": float(np.mean([j["passed"] for j in joined])),
        "n_inzone": len(inzone),
        "n_inzone_good": n1,
        "cond1_recall": {"value": recall, "n_inzone_good": n1, "half_width": h1,
                         "target_h": 0.04, "powered": cond1_powered, "pass": cond1_pass,
                         "floor": RECALL_FLOOR, "n_needed_h04": n_needed},
        "slots_saved": slots_saved,
        "cond2_coverage": {"mean_abs_err": cov_err, "by_head": cov_detail, "n": n2,
                           "half_width": h2, "powered": cond2_powered, "pass": cond2_pass,
                           "max_err": COVERAGE_MAX_ERR},
        "cond3_novelty": {"novel_rate": novel_rate, "n_novel": len(novel),
                          "novel_pred_err": novel_err, "powered": cond3_powered,
                          "pass": cond3_pass, "max_rate": NOVEL_RATE_MAX},
    })

    powered = cond1_powered and cond2_powered and cond3_powered
    rep["recommendation"] = _recommend(
        cond1_pass, cond1_powered, cond2_pass, cond2_powered,
        cond3_pass, cond3_powered, n1, n_needed, recall, beta)
    if update_state and powered and cond1_pass and cond2_pass and cond3_pass:
        save_state("full", beta, note=f"cutover: recall={recall:.3f} cov_err={cov_err:.3f}")
        rep["state_updated"] = "full"
    _persist(rep)
    return rep


def _recommend(c1_pass, c1_powered, c2_pass, c2_powered, c3_pass, c3_powered,
               n1, n_needed, recall, beta):
    # A leading-indicator FAIL is only a HOLD when it is POWERED (confirmed);
    # an underpowered fail just means keep accumulating.
    if c2_powered and not c2_pass:
        return "HOLD — calibration drift (Cond2 FAIL, powered): retrain/widen quantiles. Do NOT cut over."
    if c3_powered and not c3_pass:
        return "HOLD — high novelty (Cond3 FAIL, powered): corpus no longer covers live; retrain."
    if not (c1_powered and c2_powered and c3_powered):
        return (f"KEEP-ACCUMULATING — underpowered: in-zone-good {n1}/{n_needed} for "
                f"Cond1(h=0.04); Cond2 powered={c2_powered}; Cond3 powered={c3_powered}.")
    if not c1_powered:
        need = int(0.91 * 0.09 * (1.96 / 0.04) ** 2)
        return f"KEEP-ACCUMULATING — Cond1 underpowered ({n1}/{need} in-zone-good for h=0.04)."
    if not c1_pass:
        return f"Cond1 recall {recall:.3f} < floor — raise beta and re-derive; if still short, revert quality axis to reliable-kills-only."
    return f"RECOMMEND CUTOVER at beta={beta} (all three conditions PASS & powered)."


def _persist(rep: dict):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"shadow_cutover_{int(time.time())}.json")
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    rep["_artifact"] = path


# --------------------------------------------------------------------------- #
# §5 rolling rollback (post-cutover kill-switch)
# --------------------------------------------------------------------------- #
def rolling_rollback(window_n: int = 500, log_path: str = SHADOW_LOG, p: GateParams | None = None) -> dict:
    p = p or GateParams()
    st = load_state()
    if st.get("mode") != "full":
        return {"mode": st.get("mode"), "action": "none (not authoritative)"}
    rep = monitor(beta=st.get("beta", 2.5), log_path=log_path, p=p)
    c1 = rep.get("cond1_recall", {}).get("value", 1.0)
    c2 = rep.get("cond2_coverage", {}).get("mean_abs_err", 0.0)
    c3 = rep.get("cond3_novelty", {}).get("novel_rate", 0.0)
    breach = (c1 < ROLLBACK["recall"]) or (c2 > ROLLBACK["cov_err"]) or (c3 > ROLLBACK["novel"])
    if breach:
        save_state("reliable", st.get("beta", 2.5),
                   note=f"AUTO-REVERT recall={c1} cov_err={c2} novel={c3}")
        return {"action": "AUTO-REVERT to reliable-kills-only", "recall": c1, "cov_err": c2, "novel": c3}
    return {"action": "none (healthy)", "recall": c1, "cov_err": c2, "novel": c3}


def _print(rep: dict):
    print(f"\n=== CUTOVER MONITOR (beta={rep['beta']}) ===")
    print(f"shadow rows={rep['n_shadow']}  joined w/ realized={rep['n_joined']}")
    if rep["n_joined"] == 0:
        print(rep["recommendation"]); return
    print(f"live base-rate good={rep.get('base_good_live', float('nan')):.3f}  "
          f"in-zone={rep.get('n_inzone')}  in-zone-good={rep.get('n_inzone_good')}")
    c1, c2, c3 = rep["cond1_recall"], rep["cond2_coverage"], rep["cond3_novelty"]
    pw = lambda b: "powered" if b else "UNDERPOWERED"
    ps = lambda b: "PASS" if b else "FAIL"
    print(f"  Cond1 quality-zone recall = {c1['value']:.3f} (floor {c1['floor']}) "
          f"n_inzone_good={c1['n_inzone_good']} h={c1['half_width']:.3f}/{c1['target_h']} "
          f"[{pw(c1['powered'])}, {ps(c1['pass'])}]   slots_saved={rep['slots_saved']:.3f}")
    print(f"  Cond2 coverage mean|err| = {c2['mean_abs_err']:.3f} (max {c2['max_err']}) "
          f"n={c2['n']} [{pw(c2['powered'])}, {ps(c2['pass'])}]  by_head={c2['by_head']}")
    print(f"  Cond3 novel_rate = {c3['novel_rate']:.3f} (max {c3['max_rate']}) "
          f"n_novel={c3['n_novel']} pred_err={c3['novel_pred_err']} [{pw(c3['powered'])}, {ps(c3['pass'])}]")
    print(f"\n  >>> {rep['recommendation']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        print(rolling_rollback())
    else:
        b = float(sys.argv[sys.argv.index("--beta") + 1]) if "--beta" in sys.argv else None
        _print(monitor(beta=b, update_state=("--apply" in sys.argv)))
