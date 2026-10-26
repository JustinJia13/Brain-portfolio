"""Minimal Brain submission runner for economic theory candidate plans.

This module intentionally avoids the legacy template-sweep controls in
``alpha_search.submit.run``. Economic probes should be evaluated as complete
theory experiments, not stopped by shared template ids.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wq_alpha.brain.api import fetch_yearly_stats, sign_in

BRAIN_BASE = "https://api.worldquantbrain.com"
DEFAULT_SETTINGS: dict[str, Any] = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.01,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _regime_key(settings: dict[str, Any]) -> str:
    from wq_alpha.brain.submit import _regime_key as _canonical_regime_key

    return _canonical_regime_key(settings)


def _completed_pairs(sim_log: Path, default_settings: dict[str, Any]) -> set[tuple[str, str]]:
    if not sim_log.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in sim_log.open():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        expr = row.get("expression")
        if not expr:
            continue
        settings = row.get("settings")
        if isinstance(settings, dict):
            regime = _regime_key(settings)
        else:
            regime = row.get("regime") or _regime_key(default_settings)
        done.add((expr, regime))
    return done


def _submit_one(session: Any, expression: str, settings: dict[str, Any], *, max_retries: int = 5) -> str | None:
    payload = {"type": "REGULAR", "settings": settings, "regular": expression}
    for attempt in range(max_retries):
        try:
            response = session.post(f"{BRAIN_BASE}/simulations", json=payload)
            if "Location" in response.headers:
                return response.headers["Location"]
            logging.warning(
                "Brain submit attempt %d/%d failed: status=%s body=%s",
                attempt + 1,
                max_retries,
                response.status_code,
                response.text[:300],
            )
        except Exception as exc:
            logging.warning("Brain submit attempt %d/%d exception: %s", attempt + 1, max_retries, exc)
        time.sleep(30)
    return None


def _poll_one(session: Any, progress_url: str, *, sleep_seconds: float = 30.0) -> dict[str, Any]:
    while True:
        try:
            response = session.get(progress_url)
            retry_after = float(response.headers.get("Retry-After", 0))
            if retry_after == 0:
                return response.json()
            time.sleep(max(retry_after, sleep_seconds))
        except Exception as exc:
            logging.warning("Brain poll exception: %s", exc)
            time.sleep(sleep_seconds)


def _fetch_detail(session: Any, alpha_id: str) -> dict[str, Any]:
    try:
        return session.get(f"{BRAIN_BASE}/alphas/{alpha_id}").json()
    except Exception:
        return {}


def _build_record(
    candidate: dict[str, Any],
    settings: dict[str, Any],
    progress_url: str | None,
    progress: dict[str, Any],
    alpha_id: str | None,
    status: str,
    detail: dict[str, Any],
    yearly_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    is_ = detail.get("is", {}) or {}
    os_ = detail.get("os", {}) or {}
    checks = is_.get("checks") or []
    by_name = {c["name"]: c.get("result") == "PASS" for c in checks if isinstance(c, dict) and c.get("name")}
    record = dict(candidate)
    record.update(
        {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "progress_url": progress_url,
            "alpha_id": alpha_id,
            "settings": settings,
            "regime": _regime_key(settings),
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
            "yearly_stats": yearly_stats or {},
            "oos_sharpe_2023": (yearly_stats or {}).get(2023, {}).get("sharpe"),
            "oos_fitness_2023": (yearly_stats or {}).get(2023, {}).get("fitness"),
            "raw_alpha_detail": detail,
        }
    )
    return record


def submit_plan(
    plan_path: str | Path,
    sim_log: str | Path,
    *,
    batch_size: int = 3,
    credentials: str = "brain.txt",
    settings: dict[str, Any] | None = None,
    max_candidates: int = 0,
) -> list[dict[str, Any]]:
    """Submit a candidate plan without template abandonment or workshop logic."""
    plan_path = Path(plan_path)
    sim_log = Path(sim_log)
    default_settings = settings or dict(DEFAULT_SETTINGS)
    candidates = _read_jsonl(plan_path)
    done = _completed_pairs(sim_log, default_settings)
    pending = [
        candidate
        for candidate in candidates
        if (
            candidate.get("expression", ""),
            _regime_key({**default_settings, **(candidate.get("settings_override") or {})}),
        )
        not in done
    ]
    if max_candidates > 0:
        pending = pending[:max_candidates]

    print(f"Economic plan: {len(candidates)} total | {len(done)} done | {len(pending)} pending")
    session = sign_in(credentials)
    new_records: list[dict[str, Any]] = []

    n_batches = (len(pending) + batch_size - 1) // batch_size if pending else 0
    for batch_index, start in enumerate(range(0, len(pending), batch_size), 1):
        batch = pending[start : start + batch_size]
        print(f"\n=== Economic batch {batch_index}/{n_batches} | Logged this run: {len(new_records)} ===")
        in_flight: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        for candidate in batch:
            expression = str(candidate["expression"])
            candidate_settings = {**default_settings, **(candidate.get("settings_override") or {})}
            progress_url = _submit_one(session, expression, candidate_settings)
            label = candidate.get("theory_id") or candidate.get("candidate_id") or "economic"
            if progress_url:
                in_flight.append((candidate, progress_url, candidate_settings))
                print(f"  Submitted [{label}]: {expression[:72]}")
            else:
                record = _build_record(candidate, candidate_settings, None, {}, None, "submit_failed", {}, {})
                _append_record(sim_log, record)
                new_records.append(record)
                print(f"  FAILED [{label}]: {expression[:72]}")

        for candidate, progress_url, candidate_settings in in_flight:
            progress = _poll_one(session, progress_url)
            alpha_id = progress.get("alpha")
            if alpha_id:
                detail = _fetch_detail(session, alpha_id)
                yearly = fetch_yearly_stats(session, alpha_id)
                status = "done"
            else:
                detail = {}
                yearly = {}
                status = "no_alpha_id"
            record = _build_record(candidate, candidate_settings, progress_url, progress, alpha_id, status, detail, yearly)
            _append_record(sim_log, record)
            new_records.append(record)
            sharpe = record.get("is_sharpe")
            fitness = record.get("is_fitness")
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
            fitness_str = f"{fitness:.3f}" if fitness is not None else "N/A"
            print(
                f"  Result: fitness={fitness_str}  sharpe={sharpe_str}  "
                f"[{candidate.get('theory_id', '?')}]  {candidate.get('expression', '')[:50]}"
            )

    print(f"\nDone. Logged {len(new_records)} new economic records to {sim_log}.")
    return new_records
