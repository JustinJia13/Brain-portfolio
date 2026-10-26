"""Portfolio admission scoring helpers for economic theory simulations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core.selfcorr import get_alpha_detail
from core.brain.api import fetch_pnl_series_with_retry, sign_in
from core.portfolio.admission import (
    AdmissionConfig,
    annotate_records_with_admission,
    fetch_brain_performance_comparison,
    load_submitted_portfolio_data as load_competition_portfolio_data,
)


def score_sim_log(
    sim_log: str | Path,
    *,
    credentials: str = "brain.txt",
    competition_id: str = "IQC2026S1",
    alpha_ids: list[str] | None = None,
    out_path: str | Path | None = None,
    top_per_theory: int = 3,
    min_sharpe: float = 1.0,
    min_brain_score_delta: float = 0.0,
    min_fallback_delta_sharpe: float = 0.0,
    performance_attempts: int = 5,
    performance_wait_seconds: float = 2.0,
    max_pnl_attempts: int = 3,
    sleep_seconds: float = 0.5,
) -> Path:
    """Score useful simulation rows and write a sidecar JSONL with admission fields."""
    sim_path = Path(sim_log)
    output = Path(out_path) if out_path else sim_path.with_name(f"{sim_path.stem}_scored{sim_path.suffix}")
    rows = _read_jsonl(sim_path)
    session = sign_in(credentials)
    portfolio = load_submitted_portfolio_data(
        session,
        competition_id=competition_id,
        alpha_ids=alpha_ids or [],
        max_pnl_attempts=max_pnl_attempts,
        sleep_seconds=sleep_seconds,
    )

    def fetch_candidate(alpha_id: str) -> dict[str, float]:
        return fetch_pnl_series_with_retry(
            session,
            alpha_id,
            max_attempts=max_pnl_attempts,
            sleep_seconds=sleep_seconds,
            timeout=20,
        )

    def fetch_performance(alpha_id: str) -> dict[str, Any] | None:
        return fetch_brain_performance_comparison(
            session,
            alpha_id,
            competition_id,
            max_attempts=performance_attempts,
            sleep_seconds=performance_wait_seconds,
            timeout=20,
        )

    scored = score_records_for_admission(
        rows,
        portfolio,
        top_per_theory=top_per_theory,
        min_sharpe=min_sharpe,
        config=AdmissionConfig(
            min_brain_score_delta=min_brain_score_delta,
            min_fallback_delta_sharpe=min_fallback_delta_sharpe,
        ),
        fetch_candidate_pnl=fetch_candidate,
        fetch_performance_comparison=fetch_performance,
    )
    _write_jsonl(output, scored)
    return output


def load_submitted_portfolio_data(
    session: Any,
    *,
    competition_id: str = "IQC2026S1",
    alpha_ids: list[str] | None = None,
    max_pnl_attempts: int | None = 3,
    sleep_seconds: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Load submitted portfolio data, supporting explicit submitted alpha IDs."""
    explicit = [str(alpha_id).strip() for alpha_id in (alpha_ids or []) if str(alpha_id).strip()]
    if not explicit:
        return load_competition_portfolio_data(
            session,
            competition_id,
            max_pnl_attempts=max_pnl_attempts,
            sleep_seconds=sleep_seconds,
        )

    portfolio: dict[str, dict[str, Any]] = {}
    for alpha_id in explicit:
        pnl = fetch_pnl_series_with_retry(
            session,
            alpha_id,
            max_attempts=max_pnl_attempts,
            sleep_seconds=sleep_seconds,
            timeout=20,
        )
        if not pnl:
            continue
        detail = get_alpha_detail(session, alpha_id)
        settings = detail.get("settings") or {}
        is_stats = detail.get("is") or {}
        regular = detail.get("regular") or {}
        portfolio[alpha_id] = {
            "pnl": pnl,
            "delay": settings.get("delay"),
            "settings": settings,
            "is_sharpe": is_stats.get("sharpe"),
            "is_fitness": is_stats.get("fitness"),
            "expression": regular.get("code") if isinstance(regular, dict) else regular,
            "status": detail.get("status"),
            "stage": detail.get("stage"),
        }
        if sleep_seconds > 0:
            time.sleep(min(sleep_seconds, 0.5))
    return portfolio


def score_records_for_admission(
    records: list[dict[str, Any]],
    portfolio_alpha_data: dict[str, dict[str, Any]],
    *,
    top_per_theory: int = 3,
    min_sharpe: float = 1.0,
    config: AdmissionConfig | None = None,
    candidate_pnls: dict[str, dict[str, float]] | None = None,
    fetch_candidate_pnl: Any | None = None,
    fetch_performance_comparison: Any | None = None,
) -> list[dict[str, Any]]:
    """Return all records, with admission fields added to selected scoreable rows."""
    selected = _select_scoreable_records(
        records,
        top_per_theory=top_per_theory,
        min_sharpe=min_sharpe,
    )
    annotated = annotate_records_with_admission(
        selected,
        portfolio_alpha_data,
        config=config,
        candidate_pnls=candidate_pnls,
        fetch_candidate_pnl=fetch_candidate_pnl,
        fetch_performance_comparison=fetch_performance_comparison,
    )
    annotated_by_key = {_record_key(row): row for row in annotated}
    out: list[dict[str, Any]] = []
    for row in records:
        out.append(annotated_by_key.get(_record_key(row), row))
    return out


def _select_scoreable_records(
    records: list[dict[str, Any]],
    *,
    top_per_theory: int,
    min_sharpe: float,
) -> list[dict[str, Any]]:
    by_theory: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for row in records:
        if not row.get("alpha_id") or row.get("status") != "done":
            continue
        if _standalone_passing(row) or _float(row.get("is_sharpe")) >= min_sharpe:
            selected[_record_key(row)] = row
        theory_id = str(row.get("theory_id") or row.get("economic_source") or "")
        if theory_id:
            by_theory.setdefault(theory_id, []).append(row)

    for rows in by_theory.values():
        ranked = sorted(
            [r for r in rows if r.get("alpha_id") and r.get("status") == "done"],
            key=lambda r: _float(r.get("is_sharpe")),
            reverse=True,
        )
        for row in ranked[: max(0, int(top_per_theory))]:
            selected[_record_key(row)] = row
    return list(selected.values())


def _standalone_passing(record: dict[str, Any]) -> bool:
    checks = record.get("checks") or []
    return bool(checks) and not any(
        c.get("result") == "FAIL"
        for c in checks
        if isinstance(c, dict) and c.get("name") != "SELF_CORRELATION"
    )


def _record_key(record: dict[str, Any]) -> str:
    return "|".join(
        str(record.get(key) or "")
        for key in ("alpha_id", "candidate_id", "theory_id", "expression")
    )


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
