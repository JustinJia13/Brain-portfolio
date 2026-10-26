"""Shared workshop seed scoring and admission-context helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from wq_alpha.workshop.core import route_record


RouteFn = Callable[[dict[str, Any]], str | None]


def score_seed_records(
    seeds: list[dict[str, Any]],
    args: Any,
    session: Any,
    *,
    settings: dict[str, Any],
    seed_resim_log: Path,
    prescore_fn: Callable[..., list[dict[str, Any]]] | None = None,
    reroute_fn: Callable[[list[dict[str, Any]], Any], list[dict[str, Any]]] | None = None,
    route_fn: Callable[..., str | None] | None = None,
    submit_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Score semantic/workshop seeds, optionally resimulating score-missing seeds."""
    prescore = prescore_fn or prescore_seeds
    resolved_route_fn = route_fn or route_record

    if getattr(args, "resimulate_missing_score_seeds", True):
        from wq_alpha.workshop.seed_resim import (
            prescore_uncached_score_seeds,
            prime_cached_seed_resims_before_scoring,
            resimulate_missing_score_seeds,
        )

        scored = prime_cached_seed_resims_before_scoring(
            seeds,
            args,
            session,
            seed_resim_log,
            route_fn=resolved_route_fn,
        )
        scored = prescore_uncached_score_seeds(
            scored,
            args,
            session,
            prescore_fn=prescore,
            route_fn=resolved_route_fn,
        )
        scored = resimulate_missing_score_seeds(
            scored,
            args,
            session,
            settings=settings,
            seed_resim_log=seed_resim_log,
            prescore_fn=prescore,
            route_fn=resolved_route_fn,
            submit_fn=submit_fn,
        )
    else:
        scored = prescore(
            seeds,
            session,
            competition_id=args.competition_id,
            pnl_sleep=args.pnl_sleep,
        )
    if reroute_fn is not None:
        return reroute_fn(scored, args)
    return reroute_after_seed_scoring(scored, args, route_fn=resolved_route_fn)


def prescore_seeds(
    selected: list[dict[str, Any]],
    session: Any,
    *,
    competition_id: str = "IQC2026S1",
    pnl_sleep: float = 1.0,
    max_attempts: int = 5,
) -> list[dict[str, Any]]:
    """Enrich selected records with official Brain before/after score deltas."""
    from wq_alpha.portfolio.admission import fetch_brain_performance_comparison

    enriched: list[dict[str, Any]] = []
    for i, record in enumerate(selected):
        alpha_id = record.get("alpha_id")
        if not alpha_id:
            enriched.append(record)
            continue
        try:
            updated = dict(record)
            performance = fetch_brain_performance_comparison(
                session,
                str(alpha_id),
                competition_id,
                max_attempts=max_attempts,
                sleep_seconds=pnl_sleep,
                timeout=20.0,
            )
            score = (performance or {}).get("score") or {}
            before = _score_float(score.get("before"), default=None)
            after = _score_float(score.get("after"), default=None)
            delta = after - before if before is not None and after is not None else None
            if delta is None:
                decision = "WATCH"
                reason = "BRAIN_SCORE_UNAVAILABLE"
                signal = None
            elif delta > 0.0:
                decision = "PROMOTE"
                reason = "POSITIVE_BRAIN_SCORE_DELTA"
                signal = "brain_score_delta"
            else:
                decision = "REJECT"
                reason = "NON_POSITIVE_BRAIN_SCORE_DELTA"
                signal = "brain_score_delta"
            admission = {
                "decision": decision,
                "reason": reason,
                "delay": _record_delay_for_admission(updated),
                "signal": signal,
                "fallback_delta_used": False,
                "brain_performance_available": delta is not None,
                "brain_performance_competition": (performance or {}).get("competition", {}).get("id"),
                "brain_score_before": before,
                "brain_score_after": after,
                "brain_score_delta": delta,
                "delta_pool_sharpe": None,
            }
            updated["admission"] = admission
            for key, value in admission.items():
                updated[f"admission_{key}"] = value
            score_delta = updated.get("admission_brain_score_delta")
            score_str = f"{score_delta:+.0f}" if score_delta is not None else "N/A"
            print(
                f"  [{i + 1}/{len(selected)}] {alpha_id}  "
                f"brain_score_delta={score_str}  "
                f"signal={updated.get('admission_signal') or 'N/A'}  "
                f"{record.get('expression', '')[:60]}"
            )
            enriched.append(updated)
        except Exception as exc:
            print(f"  [score-seeds] {alpha_id}: {exc}")
            enriched.append(record)
    return enriched


def reroute_after_seed_scoring(
    selected: list[dict[str, Any]],
    args: Any,
    *,
    route_fn: Callable[..., str | None] | None = None,
) -> list[dict[str, Any]]:
    """Apply strict source gates after official Brain score fetch."""
    resolved_route_fn = route_fn or route_record
    kept: list[dict[str, Any]] = []
    dropped_non_positive = 0
    dropped_route = 0
    bypass_self_corr = bool(getattr(args, "bypass_correlation_check", False))
    for record in selected:
        enriched = dict(record)
        structural_route = _selection_route(
            enriched,
            bypass_self_corr=bypass_self_corr,
            ignore_admission_scores=True,
            route_fn=resolved_route_fn,
        )
        if structural_route == "negate":
            enriched["_route"] = structural_route
            kept.append(enriched)
            continue
        if structural_route not in {"repair", "enhance"}:
            dropped_route += 1
            continue
        if not has_positive_fetched_brain_score(enriched):
            dropped_non_positive += 1
            continue
        route = resolved_route_fn(enriched, bypass_self_corr=bypass_self_corr)
        if route not in {"repair", "enhance"}:
            dropped_route += 1
            continue
        enriched["_route"] = route
        kept.append(enriched)
    if dropped_non_positive:
        print(f"  Dropped {dropped_non_positive} repair/enhance seed(s) without positive fetched Brain score delta.")
    if dropped_route:
        print(f"  Dropped {dropped_route} seed(s) after rerouting.")
    return kept


def has_positive_fetched_brain_score(record: dict[str, Any]) -> bool:
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    value = record.get("admission_brain_score_delta")
    if value is None:
        value = record.get("brain_score_delta")
    if value is None:
        value = admission.get("brain_score_delta")
    parsed = _score_float(value, default=None)
    return parsed is not None and parsed > 0.0


def with_portfolio_prompt_feedback(
    record: dict[str, Any],
    session: Any,
    portfolio_alpha_data: dict[str, Any] | None,
    *,
    competition_id: str = "IQC2026S1",
    pnl_sleep: float = 1.0,
    max_attempts: int = 3,
    performance_max_attempts: int | None = None,
    performance_sleep_seconds: float | None = None,
) -> dict[str, Any]:
    """Return a copy with admission_* fields for LLM prompt context."""
    if not session or not portfolio_alpha_data or record.get("admission_decision"):
        return record
    alpha_id = record.get("alpha_id")
    if not alpha_id:
        return record
    try:
        from wq_alpha.brain.api import fetch_pnl_series_with_retry
        from wq_alpha.portfolio.admission import fetch_brain_performance_comparison, score_record_admission

        candidate_pnl = fetch_pnl_series_with_retry(
            session,
            str(alpha_id),
            max_attempts=max_attempts,
            sleep_seconds=pnl_sleep,
            timeout=20.0,
        )
        enriched = dict(record)
        performance_kwargs: dict[str, Any] = {}
        if performance_max_attempts is not None:
            performance_kwargs["max_attempts"] = performance_max_attempts
        if performance_sleep_seconds is not None:
            performance_kwargs["sleep_seconds"] = performance_sleep_seconds
        performance = fetch_brain_performance_comparison(
            session,
            str(alpha_id),
            competition_id,
            **performance_kwargs,
        )
        admission = score_record_admission(
            enriched,
            candidate_pnl,
            portfolio_alpha_data,
            brain_performance=performance,
        )
        enriched["admission"] = admission
        for key, value in admission.items():
            enriched[f"admission_{key}"] = value
        return enriched
    except Exception as exc:
        print(f"  [admit] prompt feedback unavailable for {alpha_id}: {exc}")
        return record


def _selection_route(
    record: dict[str, Any],
    *,
    bypass_self_corr: bool = False,
    ignore_admission_scores: bool = False,
    route_fn: Callable[..., str | None] = route_record,
) -> str | None:
    if ignore_admission_scores:
        record = _without_admission_scores(record)
    return route_fn(record, bypass_self_corr=bypass_self_corr)


def _without_admission_scores(record: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(record)
    for key in (
        "admission_brain_score_delta",
        "brain_score_delta",
        "admission_delta_pool_sharpe",
    ):
        stripped.pop(key, None)
    admission = stripped.get("admission")
    if isinstance(admission, dict):
        admission = dict(admission)
        admission.pop("brain_score_delta", None)
        admission.pop("delta_pool_sharpe", None)
        stripped["admission"] = admission
    return stripped


def _record_delay_for_admission(record: dict[str, Any]) -> int:
    settings = record.get("settings") or record.get("settings_override") or {}
    try:
        return int(settings.get("delay", 1))
    except (AttributeError, TypeError, ValueError):
        return 1


def _score_float(value: Any, default: float | None = -999.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "has_positive_fetched_brain_score",
    "prescore_seeds",
    "reroute_after_seed_scoring",
    "score_seed_records",
    "with_portfolio_prompt_feedback",
]
