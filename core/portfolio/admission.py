"""Portfolio-aware admission scoring for workshop candidates.

This module is read-only against Brain. It uses the portal's official
before/after competition score projection as the primary admission signal.
If that endpoint does not return a usable score after the configured retries,
it falls back to the local same-delay equal-weight Sharpe delta. The global
self-correlation rule still applies against all submitted alphas when PnL
context is available.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.brain.api import fetch_pnl_series_with_retry
from core.robustness import record_robustness_score
from core.quality import SELF_CORR_LIMIT, SELF_CORR_SHARPE_MULTIPLIER
from core.selfcorr import compute_max_self_corr_info, get_alpha_detail


BRAIN_BASE = "https://api.worldquantbrain.com"

PROMOTE = "PROMOTE"
WATCH = "WATCH"
REJECT = "REJECT"


@dataclass(frozen=True)
class AdmissionConfig:
    min_brain_score_delta: float = 0.0
    min_fallback_delta_sharpe: float = 0.0
    corr_limit: float = SELF_CORR_LIMIT
    corr_sharpe_multiplier: float = SELF_CORR_SHARPE_MULTIPLIER
    min_shared_days: int = 20


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sleep_from_retry_after(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return max(float(value), 0.0)
    except ValueError:
        return default


def fetch_brain_performance_comparison(
    session,
    alpha_id: str,
    competition_id: str = "IQC2026S1",
    *,
    max_attempts: int = 5,
    sleep_seconds: float = 2.0,
    timeout: float = 20.0,
) -> dict[str, Any] | None:
    """Fetch Brain's official before/after competition score projection.

    This is the endpoint used by the portal's Performance Comparison panel.
    Brain may return an empty 200 while the projection is being prepared, so
    callers should poll briefly before treating the score as unavailable.
    """
    if not alpha_id:
        return None

    url = f"{BRAIN_BASE}/competitions/{competition_id}/alphas/{alpha_id}/before-and-after-performance"
    for attempt in range(1, max_attempts + 1):
        resp = session.get(url, timeout=timeout)
        if resp.status_code in (404, 410):
            return None
        if resp.status_code == 429:
            if attempt < max_attempts:
                time.sleep(_sleep_from_retry_after(resp.headers.get("Retry-After"), sleep_seconds))
                continue
            return None
        resp.raise_for_status()

        if resp.text.strip():
            data = resp.json()
            if isinstance(data, dict) and data:
                score = data.get("score") or {}
                if _float_or_none(score.get("before")) is not None and _float_or_none(score.get("after")) is not None:
                    return data
            if attempt < max_attempts:
                time.sleep(_sleep_from_retry_after(resp.headers.get("Retry-After"), sleep_seconds))
                continue
            return data if isinstance(data, dict) and data else None

        if attempt < max_attempts:
            time.sleep(_sleep_from_retry_after(resp.headers.get("Retry-After"), sleep_seconds))

    return None


def _apply_brain_performance(metrics: dict[str, Any], performance: dict[str, Any] | None) -> None:
    if not performance:
        return

    score = performance.get("score") or {}
    score_before = _float_or_none(score.get("before"))
    score_after = _float_or_none(score.get("after"))
    competition = performance.get("competition") or {}

    metrics.update(
        {
            "brain_performance_available": score_before is not None and score_after is not None,
            "brain_performance_competition": competition.get("id"),
            "brain_score_before": score_before,
            "brain_score_after": score_after,
            "brain_score_delta": (
                score_after - score_before
                if score_before is not None and score_after is not None
                else None
            ),
        }
    )


def pnl_to_daily_returns(pnl: dict[str, float]) -> dict[str, float]:
    """Convert cumulative PnL levels to daily first differences."""
    keys = sorted(pnl)
    if len(keys) < 2:
        return {}
    return {keys[i]: pnl[keys[i]] - pnl[keys[i - 1]] for i in range(1, len(keys))}


def annualized_sharpe(values: list[float]) -> float | None:
    """Annualized Sharpe of daily returns, using population std for stability."""
    if len(values) < 20:
        return None
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(var)
    if std < 1e-12:
        return None
    return math.sqrt(252.0) * mean / std


def pearson_corr(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 20:
        return None
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    da = [x - mean_a for x in a]
    db = [x - mean_b for x in b]
    var_a = sum(x * x for x in da)
    var_b = sum(x * x for x in db)
    if var_a < 1e-12 or var_b < 1e-12:
        return None
    return sum(x * y for x, y in zip(da, db)) / math.sqrt(var_a * var_b)


def equal_weight_returns(pnls: dict[str, dict[str, float]]) -> dict[str, float]:
    """Return equal-weight daily returns on dates shared by every input alpha."""
    returns = {
        alpha_id: pnl_to_daily_returns(pnl)
        for alpha_id, pnl in pnls.items()
        if pnl
    }
    if not returns:
        return {}
    shared_dates = set.intersection(*(set(series) for series in returns.values()))
    if not shared_dates:
        return {}
    return {
        date: sum(series[date] for series in returns.values()) / len(returns)
        for date in sorted(shared_dates)
    }


def _record_delay(record: dict) -> int:
    settings = record.get("settings") or record.get("settings_override") or {}
    try:
        return int(settings.get("delay", 1))
    except (TypeError, ValueError):
        return 1


def _portfolio_delay(alpha_data: dict) -> int:
    if alpha_data.get("delay") is not None:
        try:
            return int(alpha_data["delay"])
        except (TypeError, ValueError):
            pass
    settings = alpha_data.get("settings") or {}
    try:
        return int(settings.get("delay", 1))
    except (TypeError, ValueError):
        return 1


def _same_delay_pnls(
    portfolio_alpha_data: dict[str, dict[str, Any]],
    delay: int,
) -> dict[str, dict[str, float]]:
    return {
        alpha_id: data["pnl"]
        for alpha_id, data in portfolio_alpha_data.items()
        if _portfolio_delay(data) == delay
        and isinstance(data.get("pnl"), dict)
        and data.get("pnl")
    }


def _aligned_values(a: dict[str, float], b: dict[str, float]) -> tuple[list[float], list[float], int]:
    shared = sorted(set(a) & set(b))
    return [a[d] for d in shared], [b[d] for d in shared], len(shared)


def _candidate_alpha_id(record: dict) -> str:
    return str(record.get("alpha_id") or record.get("candidate_id") or record.get("expression") or "candidate")


def _classify(metrics: dict[str, Any], config: AdmissionConfig) -> tuple[str, str]:
    if metrics.get("corr_gate_evaluated") and not metrics.get("corr_gate_pass"):
        return REJECT, "HIGH_GLOBAL_CORR"

    brain_delta = metrics.get("brain_score_delta")
    if brain_delta is not None:
        if brain_delta <= 0.0:
            return REJECT, "NON_POSITIVE_BRAIN_SCORE_DELTA"
        if brain_delta < config.min_brain_score_delta:
            return WATCH, "LOW_BRAIN_SCORE_DELTA"
        return PROMOTE, "POSITIVE_BRAIN_SCORE_DELTA"

    fallback_delta = metrics.get("delta_pool_sharpe")
    if fallback_delta is not None:
        if fallback_delta <= 0.0:
            return REJECT, "NON_POSITIVE_FALLBACK_DELTA_SHARPE"
        if fallback_delta < config.min_fallback_delta_sharpe:
            return WATCH, "LOW_FALLBACK_DELTA_SHARPE"
        return PROMOTE, "POSITIVE_FALLBACK_DELTA_SHARPE"

    if not metrics.get("portfolio_available"):
        return WATCH, "BRAIN_SCORE_UNAVAILABLE_AND_PORTFOLIO_PNL_UNAVAILABLE"
    if not metrics.get("candidate_pnl_available"):
        return WATCH, "BRAIN_SCORE_UNAVAILABLE_AND_PENDING_PNL"
    return WATCH, "BRAIN_SCORE_AND_FALLBACK_DELTA_UNAVAILABLE"


def score_record_admission(
    record: dict,
    candidate_pnl: dict[str, float] | None,
    portfolio_alpha_data: dict[str, dict[str, Any]],
    *,
    config: AdmissionConfig | None = None,
    brain_performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return admission metrics and PROMOTE/WATCH/REJECT decision for one record."""
    cfg = config or AdmissionConfig()
    delay = _record_delay(record)
    candidate_pnl = candidate_pnl or {}
    candidate_available = bool(candidate_pnl)
    portfolio_available = bool(portfolio_alpha_data)

    metrics: dict[str, Any] = {
        "decision": WATCH,
        "reason": "UNSCORED",
        "delay": delay,
        "candidate_pnl_available": candidate_available,
        "portfolio_available": portfolio_available,
        "signal": None,
        "fallback_delta_used": False,
        "same_delay_pool_size": 0,
        "global_pool_size": len(portfolio_alpha_data),
        "global_max_corr": None,
        "max_corr_alpha_id": None,
        "max_corr_alpha_sharpe": None,
        "max_corr_alpha_fitness": None,
        "corr_gate_evaluated": False,
        "corr_gate_pass": False,
        "brain_performance_available": False,
        "brain_performance_competition": None,
        "brain_score_before": None,
        "brain_score_after": None,
        "brain_score_delta": None,
        "pool_sharpe_before": None,
        "pool_sharpe_after": None,
        "delta_pool_sharpe": None,
        "residual_sharpe": None,
        "shared_days": 0,
    }

    _apply_brain_performance(
        metrics,
        brain_performance
        or record.get("brain_performance_comparison")
        or record.get("admission_brain_performance_comparison"),
    )

    def _apply_corr_gate(
        global_corr: float | None,
        matched_alpha_id: str | None,
        matched_sharpe: float | None,
        matched_fitness: float | None,
    ) -> None:
        metrics.update(
            {
                "global_max_corr": global_corr,
                "max_corr_alpha_id": matched_alpha_id,
                "max_corr_alpha_sharpe": matched_sharpe,
                "max_corr_alpha_fitness": matched_fitness,
                "corr_gate_evaluated": True,
                "corr_gate_pass": False,
            }
        )
        if global_corr is None or global_corr < cfg.corr_limit:
            metrics["corr_gate_pass"] = True
        elif matched_sharpe is not None and matched_sharpe > 0:
            metrics["corr_gate_pass"] = (record.get("is_sharpe") or 0.0) + 1e-12 >= cfg.corr_sharpe_multiplier * float(matched_sharpe)

    if candidate_available and portfolio_available:
        corr_info = compute_max_self_corr_info(candidate_pnl, portfolio_alpha_data)
        _apply_corr_gate(
            corr_info.get("max_self_correlation"),
            corr_info.get("max_self_correlation_alpha_id"),
            corr_info.get("max_self_correlation_alpha_sharpe"),
            corr_info.get("max_self_correlation_alpha_fitness"),
        )

    record_corr = record.get("max_self_correlation")
    if record_corr is not None and metrics.get("global_max_corr") is None:
        _apply_corr_gate(
            float(record_corr),
            record.get("max_self_correlation_alpha_id"),
            record.get("max_self_correlation_alpha_sharpe"),
            record.get("max_self_correlation_alpha_fitness"),
        )

    if metrics.get("brain_score_delta") is None and candidate_available and portfolio_available:
        same_delay = _same_delay_pnls(portfolio_alpha_data, delay)
        metrics["same_delay_pool_size"] = len(same_delay)
        candidate_key = _candidate_alpha_id(record)
        while candidate_key in same_delay:
            candidate_key = f"{candidate_key}_candidate"

        candidate_returns = pnl_to_daily_returns(candidate_pnl)
        pool_returns = equal_weight_returns(same_delay)

        if same_delay:
            with_candidate = equal_weight_returns({**same_delay, candidate_key: candidate_pnl})
            before_vals, after_vals, shared_days = _aligned_values(pool_returns, with_candidate)
            metrics["shared_days"] = shared_days
            before = annualized_sharpe(before_vals)
            after = annualized_sharpe(after_vals)
            pool_vals, candidate_vals, residual_days = _aligned_values(pool_returns, candidate_returns)
            if residual_days >= cfg.min_shared_days:
                mean_pool = sum(pool_vals) / len(pool_vals)
                mean_candidate = sum(candidate_vals) / len(candidate_vals)
                centered_pool = [value - mean_pool for value in pool_vals]
                centered_candidate = [value - mean_candidate for value in candidate_vals]
                pool_var = sum(value * value for value in centered_pool)
                beta = (
                    sum(a * b for a, b in zip(centered_candidate, centered_pool)) / pool_var
                    if pool_var >= 1e-12
                    else 0.0
                )
                residual = [
                    candidate_value - beta * pool_value
                    for candidate_value, pool_value in zip(candidate_vals, pool_vals)
                ]
                metrics["residual_sharpe"] = annualized_sharpe(residual)
        else:
            before = 0.0
            after = annualized_sharpe(list(candidate_returns.values()))
            metrics["shared_days"] = len(candidate_returns)
            metrics["residual_sharpe"] = after

        metrics["pool_sharpe_before"] = before
        metrics["pool_sharpe_after"] = after
        if before is not None and after is not None and metrics["shared_days"] >= cfg.min_shared_days:
            metrics["delta_pool_sharpe"] = after - before

    decision, reason = _classify(metrics, cfg)
    if metrics.get("brain_score_delta") is not None:
        metrics["signal"] = "brain_score_delta"
    elif metrics.get("delta_pool_sharpe") is not None:
        metrics["signal"] = "fallback_delta_pool_sharpe"
        metrics["fallback_delta_used"] = True
    metrics["decision"] = decision
    metrics["reason"] = reason
    return metrics


def annotate_records_with_admission(
    records: list[dict],
    portfolio_alpha_data: dict[str, dict[str, Any]],
    *,
    config: AdmissionConfig | None = None,
    candidate_pnls: dict[str, dict[str, float]] | None = None,
    fetch_candidate_pnl: Callable[[str], dict[str, float]] | None = None,
    performance_comparisons: dict[str, dict[str, Any]] | None = None,
    fetch_performance_comparison: Callable[[str], dict[str, Any] | None] | None = None,
) -> list[dict]:
    """Return copies of records with flattened admission_* fields."""
    annotated: list[dict] = []
    pnls = candidate_pnls or {}
    for record in records:
        alpha_id = str(record.get("alpha_id") or "")
        candidate_pnl = pnls.get(alpha_id, {}) if alpha_id else {}
        if not candidate_pnl and fetch_candidate_pnl is not None and alpha_id:
            candidate_pnl = fetch_candidate_pnl(alpha_id)
        brain_performance = (performance_comparisons or {}).get(alpha_id) if alpha_id else None
        if brain_performance is None and fetch_performance_comparison is not None and alpha_id:
            brain_performance = fetch_performance_comparison(alpha_id)

        admission = score_record_admission(
            record,
            candidate_pnl,
            portfolio_alpha_data,
            config=config,
            brain_performance=brain_performance,
        )
        enriched = dict(record)
        enriched["admission"] = admission
        for key, value in admission.items():
            enriched[f"admission_{key}"] = value
        annotated.append(enriched)
    return annotated


def load_submitted_portfolio_data(
    session,
    competition_id: str = "IQC2026S1",
    *,
    max_pnl_attempts: int | None = 3,
    sleep_seconds: float = 0.5,
    timeout: float = 20.0,
) -> dict[str, dict[str, Any]]:
    """Download submitted competition alpha metadata and PnL."""
    url = f"{BRAIN_BASE}/competitions/{competition_id}/alphas"
    resp = session.get(url, params={"limit": 100, "offset": 0}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    results = list(payload.get("results") or [])
    total = int(payload.get("count") or len(results))
    offset = 100
    while offset < total:
        page = session.get(url, params={"limit": 100, "offset": offset}, timeout=30)
        page.raise_for_status()
        results.extend(page.json().get("results") or [])
        offset += 100

    portfolio: dict[str, dict[str, Any]] = {}
    for alpha in results:
        alpha_id = str(alpha.get("id") or "")
        if not alpha_id:
            continue
        pnl = fetch_pnl_series_with_retry(
            session,
            alpha_id,
            max_attempts=max_pnl_attempts,
            sleep_seconds=sleep_seconds,
            timeout=timeout,
        )
        if not pnl:
            continue
        detail = get_alpha_detail(session, alpha_id)
        settings = detail.get("settings") or alpha.get("settings") or {}
        is_stats = detail.get("is") or alpha.get("is") or {}
        regular = detail.get("regular") or alpha.get("regular") or {}
        portfolio[alpha_id] = {
            "pnl": pnl,
            "delay": settings.get("delay"),
            "settings": settings,
            "is_sharpe": is_stats.get("sharpe"),
            "is_fitness": is_stats.get("fitness"),
            "expression": regular.get("code"),
            "status": detail.get("status") or alpha.get("status"),
            "stage": detail.get("stage") or alpha.get("stage"),
        }
        if sleep_seconds > 0:
            time.sleep(min(sleep_seconds, 0.5))
    return portfolio


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _normalise_expr(expr: str) -> str:
    return " ".join((expr or "").split())


def _build_log_lookup(paths: list[Path]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for path in paths:
        for record in _load_jsonl(path):
            expr = _normalise_expr(record.get("expression", ""))
            if not expr:
                continue
            existing = lookup.get(expr)
            if existing is None or (record.get("is_fitness") or -999.0) > (existing.get("is_fitness") or -999.0):
                lookup[expr] = record
    return lookup


def _merge_candidate_detail(candidates: list[dict], log_lookup: dict[str, dict]) -> list[dict]:
    merged: list[dict] = []
    for candidate in candidates:
        expr = _normalise_expr(candidate.get("expression", ""))
        source = log_lookup.get(expr, {})
        record = dict(source)
        record.update({k: v for k, v in candidate.items() if v is not None})
        if not record.get("alpha_id") and source.get("alpha_id"):
            record["alpha_id"] = source["alpha_id"]
        if "settings" not in record and source.get("settings"):
            record["settings"] = source["settings"]
        merged.append(record)
    return merged


def _admission_sort_delta(record: dict) -> float:
    parsed = _float_or_none(record.get("admission_brain_score_delta"))
    if parsed is None:
        parsed = _float_or_none(record.get("admission_delta_pool_sharpe"))
    return parsed if parsed is not None else -999.0


def _rank_key(record: dict) -> tuple[float, float, float]:
    decision_rank = {PROMOTE: 2.0, WATCH: 1.0, REJECT: 0.0}
    rob = record_robustness_score(record)
    return (
        decision_rank.get(record.get("admission_decision"), -1.0),
        _admission_sort_delta(record),
        rob if rob is not None else 0.0,
    )


def write_ranked_outputs(records: list[dict], out_jsonl: Path, out_txt: Path) -> None:
    ranked = sorted(records, key=_rank_key, reverse=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for record in ranked:
            f.write(json.dumps(record) + "\n")

    def fmt(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.3f}"

    lines = [
        "Portfolio admission ranking\n",
        f"Records: {len(ranked)}\n\n",
        f"{'#':>4} {'decision':<8} {'reason':<38} {'id':<9} {'d':>1} {'fit':>6} {'sh':>6} {'corr':>7} {'d_score':>8} {'d_fallback':>10} {'signal':<27} expression\n",
        "-" * 166 + "\n",
    ]
    for i, record in enumerate(ranked, 1):
        expr = _normalise_expr(record.get("expression", ""))[:72]
        lines.append(
            f"{i:>4} {record.get('admission_decision',''):<8} {record.get('admission_reason',''):<38} "
            f"{str(record.get('alpha_id') or ''):<9} {str(record.get('admission_delay')):>1} "
            f"{fmt(record.get('is_fitness')):>6} {fmt(record.get('is_sharpe')):>6} "
            f"{fmt(record.get('admission_global_max_corr')):>7} "
            f"{fmt(record.get('admission_brain_score_delta')):>8} "
            f"{fmt(record.get('admission_delta_pool_sharpe')):>10} "
            f"{str(record.get('admission_signal') or ''):<27} {expr}\n"
        )
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank workshop candidates by official Brain score contribution.")
    parser.add_argument("--competition-id", default="IQC2026S1")
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument(
        "--candidates",
        default="artifacts/workshop/workshop_admission.jsonl",
        help="JSONL candidates to rescore; defaults to the workshop admission audit.",
    )
    parser.add_argument(
        "--candidate-logs",
        nargs="*",
        default=["artifacts/workshop/workshop_sim.jsonl", "artifacts/sim/master.jsonl"],
    )
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--min-brain-score-delta", type=float, default=0.0)
    parser.add_argument("--min-fallback-delta-sharpe", type=float, default=0.0)
    parser.add_argument("--skip-brain-performance-comparison", action="store_true")
    parser.add_argument("--performance-attempts", type=int, default=5)
    parser.add_argument("--performance-wait", type=float, default=2.0)
    parser.add_argument("--max-pnl-attempts", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--out-jsonl", default="artifacts/portfolio_admission/ranked.jsonl")
    parser.add_argument("--out-txt", default="artifacts/portfolio_admission/ranked.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from core.brain.api import sign_in

    session = sign_in(args.credentials)
    portfolio = load_submitted_portfolio_data(
        session,
        args.competition_id,
        max_pnl_attempts=args.max_pnl_attempts,
        sleep_seconds=args.sleep,
    )
    candidates = _load_jsonl(Path(args.candidates))[: args.top]
    lookup = _build_log_lookup([Path(p) for p in args.candidate_logs])
    records = _merge_candidate_detail(candidates, lookup)

    def fetch_candidate(alpha_id: str) -> dict[str, float]:
        return fetch_pnl_series_with_retry(
            session,
            alpha_id,
            max_attempts=args.max_pnl_attempts,
            sleep_seconds=args.sleep,
            timeout=20,
        )

    def fetch_performance(alpha_id: str) -> dict[str, Any] | None:
        if args.skip_brain_performance_comparison:
            return None
        return fetch_brain_performance_comparison(
            session,
            alpha_id,
            args.competition_id,
            max_attempts=args.performance_attempts,
            sleep_seconds=args.performance_wait,
            timeout=20,
        )

    annotated = annotate_records_with_admission(
        records,
        portfolio,
        config=AdmissionConfig(
            min_brain_score_delta=args.min_brain_score_delta,
            min_fallback_delta_sharpe=args.min_fallback_delta_sharpe,
        ),
        fetch_candidate_pnl=fetch_candidate,
        fetch_performance_comparison=fetch_performance,
    )
    write_ranked_outputs(annotated, Path(args.out_jsonl), Path(args.out_txt))

    counts: dict[str, int] = {}
    for record in annotated:
        counts[record.get("admission_decision", WATCH)] = counts.get(record.get("admission_decision", WATCH), 0) + 1
    print(f"Portfolio alphas loaded: {len(portfolio)}")
    print(f"Candidates scored:       {len(annotated)}")
    print(f"Decisions:               {counts}")
    print(f"Wrote:                   {args.out_jsonl}")
    print(f"Wrote:                   {args.out_txt}")


if __name__ == "__main__":
    main()
