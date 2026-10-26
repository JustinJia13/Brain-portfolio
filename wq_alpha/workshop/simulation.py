"""Shared workshop candidate simulation helpers.

This module is the public home for submitting already-built workshop
candidates.  Older modules may still expose private compatibility wrappers, but
new code should import :func:`submit_candidate_batch` from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wq_alpha.workshop.core import regime_from_settings


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _record_pair(record: dict[str, Any], default_settings: dict[str, Any]) -> tuple[str, str] | None:
    expr = str(record.get("expression") or "")
    if not expr:
        return None
    settings = record.get("settings")
    if not isinstance(settings, dict):
        regime = record.get("regime") or regime_from_settings(default_settings)
        return expr, str(regime)
    return expr, regime_from_settings(settings)


def _candidate_pair(candidate: dict[str, Any], settings: dict[str, Any]) -> tuple[str, str]:
    candidate_settings = {**settings, **candidate.get("settings_override", {})}
    return candidate.get("expression", ""), regime_from_settings(candidate_settings)


def _has_sim_data(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "done"
        and bool(record.get("alpha_id"))
        and (record.get("is_sharpe") is not None or record.get("is_fitness") is not None)
    )


def _year_2023(yearly: dict[Any, Any]) -> dict[str, Any]:
    row = yearly.get(2023) or yearly.get("2023") or {}
    return row if isinstance(row, dict) else {}


def _apply_detail(record: dict[str, Any], detail: dict[str, Any]) -> bool:
    if not detail:
        return False
    is_ = detail.get("is", {}) or {}
    os_ = detail.get("os", {}) or {}
    checks = is_.get("checks") or []
    by_name = {check["name"]: check.get("result") == "PASS" for check in checks if "name" in check}
    updates = {
        "raw_alpha_detail": detail,
        "is_sharpe": is_.get("sharpe"),
        "is_fitness": is_.get("fitness"),
        "is_turnover": is_.get("turnover"),
        "is_returns": is_.get("returns"),
        "is_drawdown": is_.get("drawdown"),
        "os_sharpe": os_.get("sharpe"),
        "grade": detail.get("grade"),
        "checks": checks,
        "is_concentrated_weight_ok": by_name.get("CONCENTRATED_WEIGHT", True),
        "is_sub_universe_sharpe_ok": by_name.get("LOW_SUB_UNIVERSE_SHARPE", True),
    }
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        if record.get(key) != value:
            record[key] = value
            changed = True
    return changed


def _apply_yearly(record: dict[str, Any], yearly: dict[Any, Any]) -> bool:
    if not yearly:
        return False
    year_2023 = _year_2023(yearly)
    changed = record.get("yearly_stats") != yearly
    record["yearly_stats"] = yearly
    record["oos_sharpe_2023"] = year_2023.get("sharpe")
    record["oos_fitness_2023"] = year_2023.get("fitness")
    return changed


def submit_candidate_batch(
    candidates: list[dict[str, Any]],
    session: Any,
    sim_log: Path,
    settings: dict[str, Any],
    credentials: str,
    batch_size: int,
    *,
    portfolio_alpha_data: dict[str, Any] | None = None,
    pnl_sleep: float = 1.0,
    yearly_stats_attempts: int = 1,
    yearly_stats_sleep: float = 30.0,
) -> list[dict[str, Any]]:
    """Submit workshop candidates and append normalized records to ``sim_log``.

    This is the shared workshop submission boundary used by alpha, batch,
    semantic, and targeted workshop flows.  It preserves the existing record
    shape and Brain polling/admission behavior while avoiding CLI-wrapper
    imports from library code.
    """

    from wq_alpha.brain.submit import build_record, fetch_detail, poll_one, submit_one
    from wq_alpha.brain.api import fetch_yearly_stats_with_retry

    records = _load_jsonl(sim_log)
    by_pair: dict[tuple[str, str], int] = {}
    for idx, record in enumerate(records):
        pair = _record_pair(record, settings)
        if pair:
            by_pair[pair] = idx

    pending: list[tuple[dict[str, Any], int | None]] = []
    rewrite_needed = False
    for candidate in candidates:
        pair = _candidate_pair(candidate, settings)
        existing_idx = by_pair.get(pair)
        if existing_idx is None:
            pending.append((candidate, None))
            continue
        existing = records[existing_idx]
        alpha_id = existing.get("alpha_id")
        if alpha_id and not _has_sim_data(existing):
            detail = fetch_detail(session, alpha_id)
            if _apply_detail(existing, detail):
                print(f"  [detail] {alpha_id} sim detail backfilled")
                rewrite_needed = True
        if not _has_sim_data(existing):
            pending.append((candidate, existing_idx))
            continue
        if alpha_id and not existing.get("yearly_stats"):
            yearly = fetch_yearly_stats_with_retry(
                session,
                alpha_id,
                max_attempts=yearly_stats_attempts,
                sleep_seconds=yearly_stats_sleep,
            )
            if _apply_yearly(existing, yearly):
                print(f"  [yearly] {alpha_id} yearly-stats backfilled")
                rewrite_needed = True

    if rewrite_needed:
        _write_jsonl(sim_log, records)
    if not pending:
        return []

    new_records: list[dict[str, Any]] = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        in_flight: list[tuple[dict[str, Any], str, dict[str, Any], int | None]] = []

        for candidate, replace_idx in batch:
            expr = candidate["expression"]
            candidate_settings = {**settings, **candidate.get("settings_override", {})}
            url = submit_one(session, expr, candidate_settings, credentials=credentials)
            if url:
                in_flight.append((candidate, url, candidate_settings, replace_idx))
                print(f"  [submit] {expr[:72]}")
            else:
                record = build_record(candidate, candidate_settings, None, {}, None, "submit_failed", {})
                if replace_idx is None:
                    _append_jsonl(sim_log, record)
                    records.append(record)
                else:
                    records[replace_idx] = record
                    rewrite_needed = True

        for candidate, url, candidate_settings, replace_idx in in_flight:
            progress = poll_one(session, url, credentials=credentials)
            alpha_id = progress.get("alpha")
            if alpha_id:
                detail = fetch_detail(session, alpha_id)
                yearly = fetch_yearly_stats_with_retry(
                    session,
                    alpha_id,
                    max_attempts=yearly_stats_attempts,
                    sleep_seconds=yearly_stats_sleep,
                )
                if yearly:
                    print(f"  [yearly] {alpha_id} yearly-stats fetched")
                elif yearly_stats_attempts > 1:
                    print(
                        f"  [yearly] {alpha_id} yearly-stats unavailable "
                        f"after {yearly_stats_attempts} attempts"
                    )
                status = "done"
            else:
                detail = {}
                yearly = {}
                status = "no_alpha_id"

            record = build_record(candidate, candidate_settings, url, progress, alpha_id, status, detail, yearly)

            if portfolio_alpha_data and alpha_id:
                from alpha_grammar.run_pipeline import compute_max_self_corr_info
                from wq_alpha.brain.api import fetch_pnl_series_with_retry
                from wq_alpha.portfolio.admission import (
                    fetch_brain_performance_comparison,
                    score_record_admission,
                )

                print(f"  [corr] fetching PnL for {alpha_id} ...")
                candidate_pnl = fetch_pnl_series_with_retry(
                    session,
                    alpha_id,
                    max_attempts=8,
                    sleep_seconds=pnl_sleep,
                )
                corr_info = compute_max_self_corr_info(candidate_pnl, portfolio_alpha_data)
                if corr_info.get("max_self_correlation") is not None:
                    record.update(corr_info)
                    corr = corr_info["max_self_correlation"]
                    print(f"  [corr] {alpha_id}  corr={corr:.3f}")
                else:
                    print(f"  [corr] {alpha_id}  corr=N/A")
                performance = fetch_brain_performance_comparison(session, alpha_id)
                admission = score_record_admission(
                    record,
                    candidate_pnl,
                    portfolio_alpha_data,
                    brain_performance=performance,
                )
                record["admission"] = admission
                for key, value in admission.items():
                    record[f"admission_{key}"] = value
                score_delta = admission.get("brain_score_delta")
                score_tag = "N/A" if score_delta is None else f"{float(score_delta):+.0f}"
                print(
                    f"  [admit] {alpha_id}  {admission.get('decision')} "
                    f"{admission.get('reason')}  brain_score_delta={score_tag}"
                )

            if replace_idx is None:
                _append_jsonl(sim_log, record)
                records.append(record)
            else:
                records[replace_idx] = record
                rewrite_needed = True
            new_records.append(record)

            sharpe = record.get("is_sharpe")
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
            corr = record.get("max_self_correlation")
            corr_tag = f" corr={corr:.3f}" if corr is not None else ""
            print(
                f"  [result]  sharpe={sharpe_str} grade={record.get('grade')}{corr_tag}  "
                f"{candidate.get('expression', '')[:60]}"
            )

    if rewrite_needed:
        _write_jsonl(sim_log, records)

    return new_records


__all__ = ["submit_candidate_batch"]
