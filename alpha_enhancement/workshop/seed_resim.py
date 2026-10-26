"""Seed resimulation helpers for score-missing semantic/workshop sources."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.brain.submit import DEFAULT_SETTINGS, _regime_key
from alpha_enhancement.workshop.core import _make_candidate, route_record
from alpha_enhancement.workshop.records import load_records
from alpha_enhancement.workshop.robustness_card import record_score_addition
from alpha_enhancement.workshop.simulation import submit_candidate_batch


def prime_cached_seed_resims_before_scoring(
    seeds: list[dict[str, Any]],
    args: Any,
    session: Any,
    seed_resim_log: Path,
    *,
    route_fn: Callable[..., str | None] = route_record,
) -> list[dict[str, Any]]:
    """Replace score-missing repair/enhance seeds with cached resims before source scoring."""
    candidate_by_seed_index: dict[int, dict[str, Any]] = {}
    wanted_keys: set[tuple[str, str]] = set()
    for idx, seed in enumerate(seeds):
        if not score_missing_repair_enhance_seed(seed, args, route_fn=route_fn):
            continue
        expr = str(seed.get("expression") or "")
        if not expr:
            continue
        candidate = make_seed_resim_candidate(seed)
        candidate_by_seed_index[idx] = candidate
        wanted_keys.add(seed_resim_lookup_key(candidate))

    if not wanted_keys:
        return seeds

    cached_by_key = seed_resim_records_by_key(load_seed_resim_cache(seed_resim_log, wanted_keys))
    if not cached_by_key:
        return seeds

    cached_records = backfill_seed_resim_yearly_stats(
        list(cached_by_key.values()),
        session,
        args,
        seed_resim_log,
    )
    cached_by_key = seed_resim_records_by_key(cached_records)
    out = list(seeds)
    replaced = 0
    for idx, candidate in candidate_by_seed_index.items():
        replacement = cached_by_key.get(seed_resim_lookup_key(candidate))
        if replacement is None:
            continue
        seed = seeds[idx]
        replacement = dict(replacement)
        replacement["_seed_resim_cache_hit"] = True
        replacement["_resimulated_from_alpha_id"] = seed.get("alpha_id")
        replacement["_resimulated_from_expression"] = seed.get("expression")
        out[idx] = replacement
        replaced += 1

    if replaced:
        print(
            f"  Using {replaced} cached seed-resim record(s) before source score fetch; "
            "only uncached seeds will be checked/resubmitted."
        )
    return out


def prescore_uncached_score_seeds(
    seeds: list[dict[str, Any]],
    args: Any,
    session: Any,
    *,
    prescore_fn: Callable[..., list[dict[str, Any]]],
    route_fn: Callable[..., str | None] = route_record,
) -> list[dict[str, Any]]:
    """Prescore score-missing source seeds that were not replaced by a cached resim."""
    indices: list[int] = []
    prescore_inputs: list[dict[str, Any]] = []
    for idx, seed in enumerate(seeds):
        if seed.get("_seed_resim_cache_hit"):
            continue
        if not score_missing_repair_enhance_seed(seed, args, route_fn=route_fn):
            continue
        indices.append(idx)
        prescore_inputs.append(seed)

    if not prescore_inputs:
        return seeds

    print(f"  Scoring {len(prescore_inputs)} uncached score-missing source seed(s).")
    scored_subset = prescore_fn(
        prescore_inputs,
        session,
        competition_id=args.competition_id,
        pnl_sleep=args.pnl_sleep,
    )
    out = list(seeds)
    for idx, record in zip(indices, scored_subset, strict=False):
        out[idx] = record
    return out


def resimulate_missing_score_seeds(
    seeds: list[dict[str, Any]],
    args: Any,
    session: Any,
    *,
    settings: dict[str, Any],
    seed_resim_log: Path,
    prescore_fn: Callable[..., list[dict[str, Any]]] | None = None,
    route_fn: Callable[..., str | None] = route_record,
    submit_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Resimulate repair/enhance seeds whose official score is unavailable on this account."""
    from alpha_enhancement.workshop.scoring import prescore_seeds

    resolved_prescore = prescore_fn or prescore_seeds
    resolved_submit = submit_fn or submit_candidate_batch

    missing: list[dict[str, Any]] = []
    passthrough: list[dict[str, Any]] = []
    bypass_self_corr = bool(getattr(args, "bypass_correlation_check", False))
    for seed in seeds:
        if brain_score_delta(seed) is not None:
            passthrough.append(seed)
            continue
        route = route_fn(seed, bypass_self_corr=bypass_self_corr)
        if route in {"repair", "enhance"}:
            missing.append(seed)
        else:
            passthrough.append(seed)
    if not missing:
        return seeds

    print(
        f"  Resimulating {len(missing)} score-unavailable repair/enhance seed(s) "
        "on the active Brain account before source gating."
    )
    candidates: list[dict[str, Any]] = []
    for seed in missing:
        expr = str(seed.get("expression") or "")
        if not expr:
            passthrough.append(seed)
            continue
        candidates.append(make_seed_resim_candidate(seed))

    if not candidates:
        return seeds

    candidate_keys = [seed_resim_lookup_key(candidate) for candidate in candidates]
    cached_by_key = seed_resim_records_by_key(
        load_seed_resim_cache(seed_resim_log, set(candidate_keys))
    )
    if cached_by_key:
        cached_records = backfill_seed_resim_yearly_stats(
            list(cached_by_key.values()),
            session,
            args,
            seed_resim_log,
        )
        cached_by_key = seed_resim_records_by_key(cached_records)
        print(
            f"  Reusing {len(cached_by_key)} cached seed-resim record(s) "
            f"from {seed_resim_log}."
        )

    pending_candidates = [
        candidate
        for candidate, key in zip(candidates, candidate_keys, strict=False)
        if key not in cached_by_key
    ]
    resim_records: list[dict[str, Any]] = []
    if pending_candidates:
        resim_records = resolved_submit(
            pending_candidates,
            session,
            seed_resim_log,
            settings,
            args.credentials,
            args.batch_size,
            portfolio_alpha_data=None,
            pnl_sleep=args.pnl_sleep,
            yearly_stats_attempts=args.yearly_stats_attempts,
            yearly_stats_sleep=args.yearly_stats_sleep,
        )

    all_resim_records = dedupe_seed_resim_records(list(cached_by_key.values()) + resim_records)
    if not all_resim_records:
        return list(passthrough) + missing

    scored_resims = resolved_prescore(
        all_resim_records,
        session,
        competition_id=args.competition_id,
        pnl_sleep=args.pnl_sleep,
    )
    by_resim_key = seed_resim_records_by_key(scored_resims)
    out = list(passthrough)
    for seed in missing:
        replacement = by_resim_key.get(seed_resim_lookup_key(seed))
        if replacement is None:
            out.append(seed)
            continue
        replacement = dict(replacement)
        replacement["_resimulated_from_alpha_id"] = seed.get("_resimulated_from_alpha_id") or seed.get("alpha_id")
        replacement["_resimulated_from_expression"] = seed.get("_resimulated_from_expression") or seed.get("expression")
        out.append(replacement)
    return out


def score_missing_repair_enhance_seed(
    record: dict[str, Any],
    args: Any,
    *,
    route_fn: Callable[..., str | None] = route_record,
) -> bool:
    if brain_score_delta(record) is not None:
        return False
    route = route_fn(record, bypass_self_corr=bool(getattr(args, "bypass_correlation_check", False)))
    return route in {"repair", "enhance"}


def make_seed_resim_candidate(seed: dict[str, Any]) -> dict[str, Any]:
    expr = str(seed.get("expression") or "")
    params = {
        "original_alpha_id": seed.get("_resimulated_from_alpha_id") or seed.get("alpha_id"),
        "original_expression": seed.get("_resimulated_from_expression") or expr,
        "root_expression": root_expression(seed),
        "seed_resim_reason": "missing_brain_score_delta_on_active_account",
    }
    candidate = _make_candidate(expr, seed, "semantic_seed_resim", params)
    candidate["skip_workshop"] = True
    return candidate


def seed_resim_lookup_key(record: dict[str, Any]) -> tuple[str, str]:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    expression = str(params.get("original_expression") or record.get("expression") or "")
    return (expression, record_regime_key(record))


def load_seed_resim_cache(
    seed_resim_log: Path,
    wanted_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if not wanted_keys:
        return []
    records: list[dict[str, Any]] = []
    for record in load_records([str(seed_resim_log)]):
        if record.get("status") != "done" or not record.get("expression"):
            continue
        if seed_resim_lookup_key(record) in wanted_keys:
            records.append(record)
    return records


def seed_resim_records_by_key(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "done" or not record.get("expression"):
            continue
        key = seed_resim_lookup_key(record)
        if not key[0]:
            continue
        existing = best.get(key)
        if existing is None or seed_resim_rank_key(record) > seed_resim_rank_key(existing):
            best[key] = record
    return best


def dedupe_seed_resim_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(seed_resim_records_by_key(records).values())


def seed_resim_rank_key(record: dict[str, Any]) -> tuple[int, int, tuple[float, float, float, float], str]:
    return (
        1 if brain_score_delta(record) is not None else 0,
        1 if record.get("yearly_stats") else 0,
        seed_rank_key(record),
        str(record.get("logged_at") or ""),
    )


def backfill_seed_resim_yearly_stats(
    records: list[dict[str, Any]],
    session: Any,
    args: Any,
    seed_resim_log: Path,
) -> list[dict[str, Any]]:
    attempts = int(getattr(args, "yearly_stats_attempts", 1) or 0)
    if attempts <= 0:
        return records
    missing = [
        record for record in records
        if record.get("alpha_id") and not record.get("yearly_stats")
    ]
    if not missing:
        return records

    from core.brain.api import fetch_yearly_stats_with_retry

    print(f"  Backfilling yearly stats for {len(missing)} cached seed-resim record(s).")
    updated_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in missing:
        alpha_id = str(record.get("alpha_id"))
        yearly = fetch_yearly_stats_with_retry(
            session,
            alpha_id,
            max_attempts=1,
            sleep_seconds=0.0,
        )
        if not yearly:
            print(f"  [yearly] {alpha_id} yearly-stats unavailable on cache refresh")
            continue
        updated = dict(record)
        updated["yearly_stats"] = yearly
        year_2023 = yearly.get(2023) or yearly.get("2023") or {}
        updated["oos_sharpe_2023"] = year_2023.get("sharpe")
        updated["oos_fitness_2023"] = year_2023.get("fitness")
        updated["yearly_stats_backfilled_at"] = datetime.now(timezone.utc).isoformat()
        updated_by_key[seed_resim_lookup_key(updated)] = updated
        print(f"  [yearly] {alpha_id} yearly-stats backfilled")
    if updated_by_key:
        rewrite_seed_resim_cache(seed_resim_log, updated_by_key)
    return [updated_by_key.get(seed_resim_lookup_key(record), record) for record in records]


def rewrite_seed_resim_cache(seed_resim_log: Path, updated_by_key: dict[tuple[str, str], dict[str, Any]]) -> None:
    records = load_records([str(seed_resim_log)])
    written_updates: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        key = seed_resim_lookup_key(record)
        if key in updated_by_key:
            if key in written_updates:
                continue
            out.append(updated_by_key[key])
            written_updates.add(key)
        else:
            out.append(record)
    for key, record in updated_by_key.items():
        if key not in written_updates:
            out.append(record)
    seed_resim_log.parent.mkdir(parents=True, exist_ok=True)
    with seed_resim_log.open("w") as handle:
        for record in out:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def brain_score_delta(record: dict[str, Any]) -> float | None:
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    for value in (
        record.get("admission_brain_score_delta"),
        record.get("brain_score_delta"),
        admission.get("brain_score_delta"),
    ):
        if value is None:
            continue
        return float_or_none(value)
    return None


def seed_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    brain, fallback = record_score_addition(record)
    score = brain if brain is not None else fallback
    passing_bonus = 2.0 if _standalone_passes(record) else 0.0
    return (
        score if score is not None else -1e9,
        passing_bonus + abs(float_or_none(record.get("is_fitness")) or 0.0),
        abs(float_or_none(record.get("is_sharpe")) or 0.0),
        -(float_or_none(record.get("is_turnover")) or 9.0),
    )


def root_expression(record: dict[str, Any]) -> str:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    return str(params.get("root_expression") or params.get("original_expression") or record.get("expression") or "")


def record_regime_key(record: dict[str, Any]) -> str:
    settings = record.get("settings")
    if isinstance(settings, dict):
        return _regime_key({**DEFAULT_SETTINGS, **settings})
    override = record.get("settings_override")
    if isinstance(override, dict):
        return _regime_key({**DEFAULT_SETTINGS, **override})
    return str(record.get("regime") or _regime_key(DEFAULT_SETTINGS))


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _standalone_passes(record: dict[str, Any]) -> bool:
    from alpha_enhancement.workshop.core import is_passing

    return (
        record.get("status") == "done"
        and (float_or_none(record.get("is_sharpe")) or -999.0) >= 1.25
        and (float_or_none(record.get("is_fitness")) or -999.0) >= 1.0
        and is_passing(record)
    )


__all__ = [
    "backfill_seed_resim_yearly_stats",
    "brain_score_delta",
    "dedupe_seed_resim_records",
    "load_seed_resim_cache",
    "make_seed_resim_candidate",
    "prescore_uncached_score_seeds",
    "prime_cached_seed_resims_before_scoring",
    "record_regime_key",
    "resimulate_missing_score_seeds",
    "score_missing_repair_enhance_seed",
    "seed_resim_lookup_key",
    "seed_resim_records_by_key",
]
