"""State, ledger, and promotion-harvest helpers for continuous economic search."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schemas import normalize_expression
from .theory_store import TheoryStore, candidate_index_key, row_delay


LEDGER_COLUMNS = [
    "timestamp",
    "cycle_id",
    "delay",
    "alpha_id",
    "theory_id",
    "theory_title",
    "stage",
    "candidate_id",
    "expression",
    "is_sharpe",
    "is_fitness",
    "is_turnover",
    "checks_summary",
    "admission_brain_score_delta",
    "admission_delta_pool_sharpe",
    "admission_signal",
    "admission_global_max_corr",
    "admission_pool_sharpe_before",
    "admission_pool_sharpe_after",
    "admission_reason",
    "sim_log_path",
    "candidate_path",
    "decision_path",
    "ledger_status",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def cycle_id_for_now(*, target_delay: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"delay{target_delay}_{stamp}"


def read_run_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return {}


def write_run_state(path: str | Path, state: dict[str, Any]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_name(f"{state_path.name}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(state_path)


def append_event(path: str | Path, event_type: str, payload: dict[str, Any] | None = None) -> None:
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": utc_timestamp(),
        "event_type": event_type,
        "payload": payload or {},
    }
    with event_path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


class ContinuousRunLock:
    """Small lock-file guard to keep one continuous runner per state file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def __enter__(self) -> "ContinuousRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self._fd = os.open(str(self.path), flags)
        except FileExistsError as exc:
            raise RuntimeError(f"continuous runner lock already exists: {self.path}") from exc
        payload = f"pid={os.getpid()} started_at={utc_timestamp()}\n"
        os.write(self._fd, payload.encode("utf-8"))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def parse_alpha_id_text(text: str) -> list[str]:
    raw = text.replace(",", " ").split()
    return merge_alpha_ids(raw)


def read_active_alpha_ids(path: str | Path) -> list[str]:
    active_path = Path(path)
    if not active_path.exists():
        return []
    return parse_alpha_id_text(active_path.read_text())


def write_active_alpha_ids(path: str | Path, alpha_ids: Iterable[str]) -> None:
    active_path = Path(path)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text("\n".join(merge_alpha_ids(alpha_ids)) + "\n")


def append_active_alpha_ids(path: str | Path, alpha_ids: Iterable[str]) -> list[str]:
    merged = merge_alpha_ids(read_active_alpha_ids(path), alpha_ids)
    write_active_alpha_ids(path, merged)
    return merged


def merge_alpha_ids(*groups: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for raw in group:
            alpha_id = str(raw).strip()
            if not alpha_id or alpha_id in seen:
                continue
            seen.add(alpha_id)
            merged.append(alpha_id)
    return merged


def harvest_promotions(
    *,
    store: TheoryStore,
    target_delay: int,
    cycle_id: str,
    timestamp: str,
    sim_logs: Iterable[str | Path] = (),
) -> list[dict[str, Any]]:
    """Collect admission-promoted rows from scored logs and theory result files."""
    candidate_index = store.candidate_index()
    rows: list[dict[str, Any]] = []
    for sim_log in sim_logs:
        sim_path = Path(sim_log)
        for row in _read_jsonl(sim_path):
            enriched = dict(row)
            enriched["_sim_log_path"] = str(sim_path)
            rows.append(enriched)

    for theory in store.list_theories():
        for row in store.all_results(theory.theory_id, target_delay=target_delay):
            rows.append(dict(row))

    promotions: list[dict[str, Any]] = []
    seen_alpha_ids: set[str] = set()
    for row in rows:
        if not _is_promoted_row(row, target_delay=target_delay):
            continue
        alpha_id = str(row.get("alpha_id") or "").strip()
        if not alpha_id or alpha_id in seen_alpha_ids:
            continue
        seen_alpha_ids.add(alpha_id)
        promotions.append(
            _ledger_row(
                row,
                store=store,
                candidate_index=candidate_index,
                target_delay=target_delay,
                cycle_id=cycle_id,
                timestamp=timestamp,
            )
        )
    return promotions


def append_promotions_to_ledger(
    ledger_path: str | Path,
    promotions: Iterable[dict[str, Any]],
    *,
    mirror_xlsx: bool = True,
) -> list[dict[str, Any]]:
    """Append only new alpha IDs to the CSV ledger and optionally mirror to XLSX."""
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _ledger_alpha_ids(path)
    new_rows: list[dict[str, Any]] = []
    for row in promotions:
        alpha_id = str(row.get("alpha_id") or "").strip()
        if not alpha_id or alpha_id in existing:
            continue
        existing.add(alpha_id)
        new_rows.append({column: row.get(column, "") for column in LEDGER_COLUMNS})

    if not new_rows:
        if mirror_xlsx:
            mirror_ledger_to_xlsx_if_available(path)
        return []

    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if needs_header:
            writer.writeheader()
        for row in new_rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})

    if mirror_xlsx:
        mirror_ledger_to_xlsx_if_available(path)
    return new_rows


def mirror_ledger_to_xlsx_if_available(ledger_path: str | Path) -> Path | None:
    """Create an XLSX mirror when openpyxl is installed; otherwise do nothing."""
    if importlib.util.find_spec("openpyxl") is None:
        return None
    import openpyxl  # type: ignore[import-not-found]

    csv_path = Path(ledger_path)
    if not csv_path.exists():
        return None
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "promotions"
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            worksheet.append(row)
    out = csv_path.with_suffix(".xlsx")
    workbook.save(out)
    return out


def _ledger_row(
    row: dict[str, Any],
    *,
    store: TheoryStore,
    candidate_index: dict[str, dict[str, Any]],
    target_delay: int,
    cycle_id: str,
    timestamp: str,
) -> dict[str, Any]:
    expr = normalize_expression(str(row.get("expression") or ""))
    meta = candidate_index.get(candidate_index_key(expr, target_delay), {}) if expr else {}
    if not meta and expr:
        meta = candidate_index.get(expr, {})
    theory_id = str(row.get("theory_id") or meta.get("theory_id") or "")
    theory_title = str(row.get("theory_title") or meta.get("theory_title") or "")
    if theory_id and not theory_title:
        try:
            theory_title = store.load_theory(theory_id).title
        except Exception:
            theory_title = ""
    return {
        "timestamp": timestamp,
        "cycle_id": cycle_id,
        "delay": target_delay,
        "alpha_id": row.get("alpha_id"),
        "theory_id": theory_id,
        "theory_title": theory_title,
        "stage": row.get("experiment_stage") or meta.get("experiment_stage") or row.get("stage") or "",
        "candidate_id": row.get("candidate_id") or meta.get("candidate_id") or "",
        "expression": expr,
        "is_sharpe": row.get("is_sharpe"),
        "is_fitness": row.get("is_fitness"),
        "is_turnover": row.get("is_turnover"),
        "checks_summary": _checks_summary(row.get("checks")),
        "admission_brain_score_delta": row.get("admission_brain_score_delta"),
        "admission_delta_pool_sharpe": row.get("admission_delta_pool_sharpe"),
        "admission_signal": row.get("admission_signal"),
        "admission_global_max_corr": _first_present(
            row,
            "admission_global_max_corr",
            "admission_max_global_corr",
            "admission_self_corr",
        ),
        "admission_pool_sharpe_before": _first_present(
            row,
            "admission_pool_sharpe_before",
            "admission_same_delay_pool_sharpe_before",
            "pool_sharpe_before",
        ),
        "admission_pool_sharpe_after": _first_present(
            row,
            "admission_pool_sharpe_after",
            "admission_same_delay_pool_sharpe_after",
            "pool_sharpe_after",
        ),
        "admission_reason": row.get("admission_reason") or row.get("admission_decision_reason") or "",
        "sim_log_path": row.get("_sim_log_path") or "",
        "candidate_path": row.get("_candidate_path") or meta.get("_candidate_path") or "",
        "decision_path": str(store.decision_path(theory_id, target_delay)) if theory_id else "",
        "ledger_status": "admission_promoted",
    }


def _is_promoted_row(row: dict[str, Any], *, target_delay: int) -> bool:
    alpha_id = str(row.get("alpha_id") or "").strip()
    if not alpha_id:
        return False
    if str(row.get("status") or "").lower() != "done":
        return False
    if row_delay(row) != int(target_delay):
        return False
    if str(row.get("admission_decision") or "").upper() != "PROMOTE":
        return False
    return _standalone_passing(row)


def _standalone_passing(row: dict[str, Any]) -> bool:
    checks = row.get("checks") or []
    return bool(checks) and not any(
        check.get("result") == "FAIL"
        for check in checks
        if isinstance(check, dict) and check.get("name") != "SELF_CORRELATION"
    )


def _checks_summary(checks: Any) -> str:
    if not isinstance(checks, list):
        return ""
    parts: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "")
        result = str(check.get("result") or "")
        if name or result:
            parts.append(f"{name}:{result}")
    return ";".join(parts)


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _ledger_alpha_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        return {
            str(row.get("alpha_id") or "").strip()
            for row in csv.DictReader(f)
            if str(row.get("alpha_id") or "").strip()
        }


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
