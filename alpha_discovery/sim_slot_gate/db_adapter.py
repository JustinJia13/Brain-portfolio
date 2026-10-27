"""db_adapter.py — load the historical sim store + calibration data.

ADAPT layer: everything that knows the on-disk schema lives here so the rest of
alpha_discovery/sim_slot_gate is storage-agnostic.  Resolved by the Discovery report:

  * sim store        artifacts/sim/master.jsonl  (~85k rows, summary stats only)
  * id -> expression artifacts/sim/master.jsonl + records/brain*/alphas_part*.jsonl
  * self-corr labels max_self_correlation (+ matched alpha id)   ~3.5k rows
  * pool PnL         records/brain/os_part0001.jsonl  (cumulative; 31 submitted)
  * pool definition  stage == "OS" / date_submitted set  (the 31)

There are NO per-sim PnL vectors (only the 31 pool alphas have PnL), so Object A
calibration is "Case B-plus": psi is fit from realized self-correlation triples,
anchored by exact pairwise PnL-correlation among the pool.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASTER = os.path.join(_HERE, "artifacts/sim/master.jsonl")
ALPHAS_PARTS = sorted(glob.glob(os.path.join(_HERE, "records/*/alphas_part*.jsonl")))
OS_PARTS = sorted(glob.glob(os.path.join(_HERE, "records/*/os_part*.jsonl")))

# Confirmed numerically in Discovery (mean abs err 0.0025 = rounding):
#   fitness = sharpe * sqrt(|ret| / max(turnover, 0.125))
TURNOVER_FLOOR = 0.125
SUBMIT_MIN_SHARPE = 1.25
SUBMIT_MIN_FITNESS = 1.0
SUBMIT_MAX_TURNOVER = 0.7


@dataclass
class SimRecord:
    expr: str
    alpha_id: str | None
    sharpe: float | None
    ret: float | None
    turnover: float | None
    fitness: float | None
    sub_sharpe: float | None
    self_corr: float | None
    grade: str | None
    fields: tuple[str, ...]
    logged_at: str | None


def _f(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def iter_master(path: str = MASTER):
    """Yield raw dict records from the sim master log."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_subsharpe_from_details() -> dict[str, float]:
    """alpha_id -> sub-universe Sharpe, mined from raw detail files (Task 3).

    The master log only surfaced 303 values, but ~78k are on disk in the
    LOW_SUB_UNIVERSE_SHARPE check of each *_raw_alpha_detail.json. Cached to
    alpha_discovery/sim_slot_gate/_subsharpe_from_details.json (rebuild by re-scanning details).
    """
    cache = os.path.join(_HERE, "alpha_discovery/sim_slot_gate", "_subsharpe_from_details.json")
    if os.path.exists(cache):
        try:
            with open(cache) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


_SUBSHARPE_DETAILS = _load_subsharpe_from_details()


def load_sims(path: str = MASTER) -> list[SimRecord]:
    out: list[SimRecord] = []
    for d in iter_master(path):
        expr = d.get("expression")
        if not isinstance(expr, str) or not expr.strip():
            continue
        aid = d.get("alpha_id")
        sub = d.get("check_low_sub_universe_sharpe_value")
        if sub is None and aid in _SUBSHARPE_DETAILS:
            sub = _SUBSHARPE_DETAILS[aid]
        out.append(
            SimRecord(
                expr=expr,
                alpha_id=aid,
                sharpe=_f(d.get("is_sharpe")),
                ret=_f(d.get("is_returns")),
                turnover=_f(d.get("is_turnover")),
                fitness=_f(d.get("is_fitness")),
                sub_sharpe=_f(sub),
                self_corr=_f(d.get("max_self_correlation")),
                grade=d.get("grade"),
                fields=tuple(d.get("fields") or []),
                logged_at=d.get("logged_at"),
            )
        )
    return out


def build_id_to_expr(path: str = MASTER) -> dict[str, str]:
    """alpha_id -> expression, from master + records/*/alphas_part*.jsonl."""
    mapping: dict[str, str] = {}
    for d in iter_master(path):
        aid, expr = d.get("alpha_id"), d.get("expression")
        if aid and isinstance(expr, str) and expr.strip():
            mapping.setdefault(aid, expr)
    for part in ALPHAS_PARTS:
        try:
            with open(part) as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    aid, expr = d.get("alpha_id"), d.get("expression")
                    if aid and isinstance(expr, str) and expr.strip():
                        mapping.setdefault(aid, expr)
        except OSError:
            continue
    return mapping


def load_self_corr_pairs(path: str = MASTER) -> list[tuple[str, str, float]]:
    """(x_expr, y_expr, realized_max_self_corr) triples for psi calibration.

    Y is the most-correlated pool member (the argmax), looked up by alpha id.
    These are high-corr biased (they are maxima); the pool PnL anchor supplies the
    low-corr range.
    """
    id2expr = build_id_to_expr(path)
    pairs: list[tuple[str, str, float]] = []
    for d in iter_master(path):
        c = _f(d.get("max_self_correlation"))
        yid = d.get("max_self_correlation_alpha_id")
        x = d.get("expression")
        if c is None or not yid or not isinstance(x, str):
            continue
        y = id2expr.get(yid)
        if not y or y == x:
            continue
        pairs.append((x, y, c))
    return pairs


# --------------------------------------------------------------------------- #
# Pool PnL (exact corr anchor)
# --------------------------------------------------------------------------- #
def load_pool_pnl() -> dict[str, dict[str, float]]:
    """alpha_id -> {date: cumulative_pnl} for the submitted pool."""
    out: dict[str, dict[str, float]] = {}
    for part in OS_PARTS:
        try:
            with open(part) as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    aid, pnl = d.get("alpha_id"), d.get("pnl")
                    if aid and isinstance(pnl, dict) and pnl:
                        out[aid] = {k: float(v) for k, v in pnl.items() if v is not None}
        except OSError:
            continue
    return out


def _daily_returns(cum: dict[str, float]) -> dict[str, float]:
    """First-difference a cumulative PnL series into daily PnL."""
    dates = sorted(cum)
    out: dict[str, float] = {}
    prev = None
    for dt in dates:
        v = cum[dt]
        if prev is not None:
            out[dt] = v - prev
        prev = v
    return out


def pool_pnl_corr_pairs(min_overlap: int = 250) -> list[tuple[str, str, float]]:
    """(x_id, y_id, realized_pnl_corr) over all pool pairs (exact, full range)."""
    pnl = load_pool_pnl()
    daily = {aid: _daily_returns(c) for aid, c in pnl.items()}
    ids = sorted(daily)
    out: list[tuple[str, str, float]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = daily[ids[i]], daily[ids[j]]
            shared = sorted(set(a) & set(b))
            if len(shared) < min_overlap:
                continue
            va = np.array([a[d] for d in shared])
            vb = np.array([b[d] for d in shared])
            if va.std() < 1e-9 or vb.std() < 1e-9:
                continue
            r = float(np.corrcoef(va, vb)[0, 1])
            out.append((ids[i], ids[j], r))
    return out


def load_pool_exprs() -> list[str]:
    """Expressions of the submitted pool (the 31 alphas with stored PnL)."""
    id2expr = build_id_to_expr()
    out: list[str] = []
    for aid in load_pool_pnl():
        e = id2expr.get(aid)
        if e:
            out.append(e)
    return out


def submission_passes(rec: SimRecord) -> bool | None:
    """Did this sim clear the (Sharpe, fitness, turnover) submission thresholds?
    None if any required field is missing."""
    if rec.sharpe is None or rec.fitness is None or rec.turnover is None:
        return None
    return (
        rec.sharpe >= SUBMIT_MIN_SHARPE
        and rec.fitness >= SUBMIT_MIN_FITNESS
        and rec.turnover <= SUBMIT_MAX_TURNOVER
    )
