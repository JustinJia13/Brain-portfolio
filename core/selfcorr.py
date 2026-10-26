"""Self-correlation, portfolio PnL and sim-record construction.

These are platform concerns shared by every lane. They lived in
alpha_discovery/grammar_method/run_pipeline.py, which meant the enhancement engine and the
portfolio gate had to import from a discovery method. Lifted out so the
dependency graph stays acyclic."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from core.brain.api import (
    get_datafields, sign_in, fetch_yearly_stats_with_retry,
    fetch_pnl_series, fetch_pnl_series_with_retry, fetch_competition_alpha_ids,
)
from core.robustness import attach_robustness, normalize_yearly_stats


def get_alpha_detail(session, alpha_id: str) -> dict[str, Any]:
    try:
        return session.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}").json()
    except Exception:
        return {}


def _fetch_pnl_with_backoff(
    session,
    alpha_id: str,
    *,
    timeout: float = 20.0,
    max_attempts: int | None = None,
) -> dict[str, float]:
    """Fetch PnL retrying at 1s intervals until data is returned.

    Retries indefinitely (max_attempts=None) on both rate limits (None) and
    async delay (empty {}).  Pass max_attempts=N to cap for portfolio loading.
    """
    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        result = fetch_pnl_series(session, alpha_id, timeout=timeout)
        if result:
            return result
        logging.warning("[PnL] not ready / rate limited for %s — waiting 1s (attempt %d)", alpha_id, attempt)
        print(f"[PnL] not ready / rate limited — waiting 1s (attempt {attempt}) ...")
        time.sleep(1)
    logging.warning("[PnL] exhausted retries for %s", alpha_id)
    return {}


def _portfolio_self_corr_info_template() -> dict[str, Any]:
    return {
        "max_self_correlation": None,
        "max_self_correlation_alpha_id": None,
        "max_self_correlation_alpha_sharpe": None,
        "max_self_correlation_alpha_fitness": None,
    }


def _extract_is_perf(alpha_detail: dict[str, Any]) -> tuple[float | None, float | None]:
    is_stats = alpha_detail.get("is") or {}
    raw_sharpe = is_stats.get("sharpe")
    raw_fitness = is_stats.get("fitness")
    sharpe = float(raw_sharpe) if raw_sharpe is not None else None
    fitness = float(raw_fitness) if raw_fitness is not None else None
    return sharpe, fitness


def load_portfolio_alpha_data(
    session,
    competition_id: str = "IQC2026S1",
    *,
    max_pnl_attempts: int | None = 10,
) -> dict[str, dict[str, Any]]:
    """Fetch PnL plus IS performance metadata for all competition alphas.

    Returns {alpha_id: {"pnl": {...}, "is_sharpe": float|None, "is_fitness": float|None}}.
    Alphas with no PnL are skipped.
    """
    alpha_ids = fetch_competition_alpha_ids(session, competition_id)
    print(f"[portfolio] Fetching PnL for {len(alpha_ids)} competition alphas ({competition_id}) ...")
    portfolio: dict[str, dict[str, Any]] = {}
    for aid in alpha_ids:
        pnl = _fetch_pnl_with_backoff(session, aid, max_attempts=max_pnl_attempts)
        if not pnl:
            continue
        detail = get_alpha_detail(session, aid)
        is_sharpe, is_fitness = _extract_is_perf(detail)
        settings = detail.get("settings") or {}
        portfolio[aid] = {
            "pnl": pnl,
            "is_sharpe": is_sharpe,
            "is_fitness": is_fitness,
            "delay": settings.get("delay"),
            "settings": settings,
            "expression": (detail.get("regular") or {}).get("code"),
            "status": detail.get("status"),
            "stage": detail.get("stage"),
        }
    print(f"[portfolio] Loaded PnL for {len(portfolio)}/{len(alpha_ids)} alphas.")
    return portfolio


def _pnl_to_daily_returns(pnl: dict[str, float]) -> dict[str, float]:
    """Convert cumulative PnL (levels) to daily returns (first differences).

    The Brain /recordsets/pnl endpoint returns cumulative PnL in dollar terms.
    Correlation must be computed on daily changes, not levels, to avoid
    spurious correlation from the unit root.
    """
    keys = sorted(pnl)
    if len(keys) < 2:
        return {}
    return {keys[i]: pnl[keys[i]] - pnl[keys[i - 1]] for i in range(1, len(keys))}


def compute_max_self_corr(
    candidate_pnl: dict[str, float],
    portfolio_pnls: dict[str, dict[str, float]],
) -> float | None:
    """Compatibility wrapper returning only the signed max self-correlation."""
    portfolio_alpha_data = {
        aid: {"pnl": pnl, "is_sharpe": None, "is_fitness": None}
        for aid, pnl in portfolio_pnls.items()
    }
    return compute_max_self_corr_info(candidate_pnl, portfolio_alpha_data)["max_self_correlation"]


def compute_max_self_corr_info(
    candidate_pnl: dict[str, float],
    portfolio_alpha_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return signed max self-correlation plus matched portfolio benchmark fields.

    Inputs are cumulative PnL series ({date: cumulative_pnl}); they are
    first-differenced to daily returns before correlation is computed.

    Returns the SIGNED maximum (not abs): a negatively-correlated alpha
    passes the self-correlation check because it adds diversification.
    Brain uses signed max for the self-correlation gate.

    Aligns on shared dates. If candidate_pnl is empty or no portfolio alpha
    shares enough dates (min 20) with the candidate, all returned fields are
    None.
    """
    info = _portfolio_self_corr_info_template()
    if not candidate_pnl or not portfolio_alpha_data:
        return info
    try:
        import numpy as np
    except ImportError:
        return info

    candidate_returns = _pnl_to_daily_returns(candidate_pnl)
    if not candidate_returns:
        return info

    candidate_dates = set(candidate_returns)
    max_corr: float | None = None
    matched_alpha_id: str | None = None
    matched_alpha_sharpe: float | None = None
    matched_alpha_fitness: float | None = None

    for alpha_id, alpha_data in portfolio_alpha_data.items():
        port_pnl = alpha_data.get("pnl") or {}
        port_returns = _pnl_to_daily_returns(port_pnl)
        shared = sorted(candidate_dates & set(port_returns))
        if len(shared) < 20:
            continue
        a = np.array([candidate_returns[d] for d in shared], dtype=float)
        b = np.array([port_returns[d] for d in shared], dtype=float)
        # Skip if either series is constant (std=0)
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if max_corr is None or corr > max_corr:
            max_corr = corr
            matched_alpha_id = alpha_id
            raw_sharpe = alpha_data.get("is_sharpe")
            raw_fitness = alpha_data.get("is_fitness")
            matched_alpha_sharpe = float(raw_sharpe) if raw_sharpe is not None else None
            matched_alpha_fitness = float(raw_fitness) if raw_fitness is not None else None

    info["max_self_correlation"] = max_corr
    info["max_self_correlation_alpha_id"] = matched_alpha_id
    info["max_self_correlation_alpha_sharpe"] = matched_alpha_sharpe
    info["max_self_correlation_alpha_fitness"] = matched_alpha_fitness
    return info


def _parse_is_checks(is_checks: list[dict[str, Any]]) -> dict[str, bool]:
    """Return {check_name: passed} for each entry in the is.checks list."""
    return {c["name"]: c.get("result", "FAIL") == "PASS" for c in (is_checks or [])}


def _parse_check_values(is_checks: list[dict[str, Any]]) -> dict[str, float | None]:
    """Return {check_name: value} for checks that carry a numeric value field."""
    return {
        c["name"]: float(c["value"]) if c.get("value") is not None else None
        for c in (is_checks or [])
    }


def build_sim_record(
    candidate: dict[str, Any],
    settings: dict[str, Any],
    progress_url: str | None,
    final_progress: dict[str, Any],
    alpha_id: str | None,
    status: str,
    alpha_detail: dict[str, Any],
    yearly_stats: dict[int, dict] | None = None,
) -> dict[str, Any]:
    final_progress = final_progress or {}
    alpha_detail = alpha_detail or {}
    is_stats = alpha_detail.get("is") or {}
    os_stats = alpha_detail.get("os") or {}
    raw_checks = is_stats.get("checks") or []
    checks = _parse_is_checks(raw_checks)
    check_values = _parse_check_values(raw_checks)
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate.get("candidate_id"),
        "family_id": candidate.get("family_id"),
        "intention_id": candidate.get("intention_id"),
        "intention_label": candidate.get("intention_label"),
        "search_lane": candidate.get("search_lane"),
        "category": candidate.get("category"),
        "dataset_id": candidate.get("dataset_id"),
        "expression": candidate.get("expression"),
        "fields": candidate.get("fields"),
        "field_types": candidate.get("field_types"),
        "cluster_ids": candidate.get("cluster_ids"),
        "group": candidate.get("group"),
        "params": candidate.get("params"),
        "horizon_bucket": candidate.get("horizon_bucket"),
        "template_id": candidate.get("template_id"),
        "tree_depth": candidate.get("tree_depth"),
        "priority_score": candidate.get("priority_score"),
        "status": status,
        "progress_url": progress_url,
        "alpha_id": alpha_id,
        "settings": settings,
        "raw_progress_result": final_progress,
        "is_sharpe": is_stats.get("sharpe"),
        "is_fitness": is_stats.get("fitness"),
        "is_turnover": is_stats.get("turnover"),
        "is_returns": is_stats.get("returns"),
        "is_drawdown": is_stats.get("drawdown"),
        "is_margin": is_stats.get("margin"),
        "is_ic": is_stats.get("ic"),
        "is_concentrated_weight_ok": checks.get("CONCENTRATED_WEIGHT", True),
        "is_sub_universe_sharpe_ok": checks.get("LOW_SUB_UNIVERSE_SHARPE", True),
        "max_self_correlation": is_stats.get("selfCorrelation"),
        "max_self_correlation_alpha_id": None,
        "max_self_correlation_alpha_sharpe": None,
        "max_self_correlation_alpha_fitness": None,
        "max_concentration": check_values.get("CONCENTRATED_WEIGHT"),
        "os_sharpe": os_stats.get("sharpe"),
        "os_fitness": os_stats.get("fitness"),
        "grade": alpha_detail.get("grade"),
        "checks": is_stats.get("checks"),
        "date_created": alpha_detail.get("dateCreated"),
        "yearly_stats": yearly_stats or {},
        "oos_sharpe_2023": (normalize_yearly_stats(yearly_stats).get(2023) or {}).get("sharpe"),
        "oos_fitness_2023": (normalize_yearly_stats(yearly_stats).get(2023) or {}).get("fitness"),
        "raw_alpha_detail": alpha_detail,
        "candidate_spec": candidate,
    }
    return attach_robustness(record)
