"""Local (no-brain-account) score-addition estimate.

The official competition score-addition (brain_score_delta) is only available
from the brain account. But its two ingredients — diversification (self
correlation) and marginal pool contribution (delta_pool_sharpe) — are both
computable from PnL, which is account-independent (the same expression+settings
produces the same daily PnL on any account). So a brain2-simulated alpha can be
scored against the brain portfolio entirely on disk, with no brain sim.

This reuses admission.py's math (pnl_to_daily_returns, equal_weight_returns,
annualized_sharpe, pearson_corr) so the local estimate matches the pipeline's
fallback signal exactly. Brain re-sim remains the source of truth for the
official number; this gate just kills the ~99% that can't add score before any
scarce brain sim is spent.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from core.portfolio.admission import (
    _aligned_values,
    _portfolio_delay,
    _same_delay_pnls,
    annualized_sharpe,
    equal_weight_returns,
    pearson_corr,
    pnl_to_daily_returns,
)

MIN_SHARED_DAYS = 100
CORR_LIMIT = 0.70


def load_brain_portfolio(records_dir: str = "records/brain") -> dict[str, dict[str, Any]]:
    """Build portfolio_alpha_data {alpha_id: {pnl, delay, sharpe}} from disk.

    Joins os_part*.jsonl (daily PnL of submitted alphas) with alphas_part*.jsonl
    (settings/delay, is_sharpe). No brain session required.
    """
    root = Path(records_dir)
    meta: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(str(root / "alphas_part*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = r.get("alpha_id")
                if aid:
                    meta[aid] = {
                        "delay": (r.get("settings") or {}).get("delay", 1),
                        "sharpe": r.get("is_sharpe"),
                    }

    portfolio: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(str(root / "os_part*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = r.get("alpha_id")
                pnl = r.get("pnl")
                if not aid or not isinstance(pnl, dict) or not pnl:
                    continue
                m = meta.get(aid, {})
                portfolio[aid] = {"pnl": pnl, "delay": m.get("delay", 1), "sharpe": m.get("sharpe")}
    return portfolio


def score_locally(
    candidate_pnl: dict[str, float],
    delay: int,
    portfolio: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Local score-addition estimate for one candidate.

    Returns max_self_correlation (+ matched alpha id/sharpe) and
    delta_pool_sharpe (marginal Sharpe contribution to the same-delay
    equal-weight pool). Mirrors admission.py's fallback computation.
    """
    out: dict[str, Any] = {
        "max_self_correlation": None,
        "max_self_correlation_alpha_id": None,
        "max_self_correlation_alpha_sharpe": None,
        "delta_pool_sharpe": None,
        "shared_days": 0,
    }
    if not candidate_pnl:
        return out
    cand_ret = pnl_to_daily_returns(candidate_pnl)
    same_delay = _same_delay_pnls(portfolio, delay)
    if not same_delay:
        return out

    # max self-correlation vs each same-delay portfolio alpha
    best_corr = None
    for aid, pnl in same_delay.items():
        other_ret = pnl_to_daily_returns(pnl)
        a, b, n = _aligned_values(cand_ret, other_ret)
        if n < 20:
            continue
        c = pearson_corr(a, b)
        if c is None:
            continue
        if best_corr is None or c > best_corr:
            best_corr = c
            out["max_self_correlation"] = round(c, 4)
            out["max_self_correlation_alpha_id"] = aid
            out["max_self_correlation_alpha_sharpe"] = portfolio[aid].get("sharpe")

    # marginal pool-Sharpe contribution
    pool_before = equal_weight_returns(same_delay)
    pool_after = equal_weight_returns({**same_delay, "__candidate__": candidate_pnl})
    before_vals, after_vals, shared = _aligned_values(pool_before, pool_after)
    out["shared_days"] = shared
    if shared >= MIN_SHARED_DAYS:
        before, after = annualized_sharpe(before_vals), annualized_sharpe(after_vals)
        if before is not None and after is not None:
            out["delta_pool_sharpe"] = round(after - before, 4)
    return out


def passes_local_gate(metrics: dict[str, Any], *, corr_limit: float = CORR_LIMIT) -> bool:
    """Diversifying AND additive: corr below limit and positive pool contribution."""
    corr = metrics.get("max_self_correlation")
    delta = metrics.get("delta_pool_sharpe")
    if corr is not None and corr >= corr_limit:
        return False
    if delta is None:           # couldn't estimate (too little overlap) — don't block
        return True
    return delta > 0.0


__all__ = ["CORR_LIMIT", "load_brain_portfolio", "passes_local_gate", "score_locally"]
