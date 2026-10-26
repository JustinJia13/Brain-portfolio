"""Filesystem-backed per-theory storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import CandidatePlanRecord, TheoryEvent, TheorySpec, normalize_expression, utc_now


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_THEORY_DIR = DEFAULT_ARTIFACTS_DIR / "theories"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


class TheoryStore:
    def __init__(self, theory_dir: Path = DEFAULT_THEORY_DIR) -> None:
        self.theory_dir = Path(theory_dir)
        self.theory_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, theory_id: str) -> Path:
        return self.theory_dir / theory_id

    def theory_path(self, theory_id: str) -> Path:
        return self.path_for(theory_id) / "theory.json"

    def history_path(self, theory_id: str) -> Path:
        return self.path_for(theory_id) / "history.jsonl"

    def memory_path(self, theory_id: str) -> Path:
        return self.path_for(theory_id) / "memory.md"

    def memory_path_for_delay(self, theory_id: str, target_delay: int | None = None) -> Path:
        if target_delay is None:
            return self.memory_path(theory_id)
        return self.path_for(theory_id) / f"memory_delay{int(target_delay)}.md"

    def decision_path(self, theory_id: str, target_delay: int | None = None) -> Path:
        if target_delay is None:
            return self.path_for(theory_id) / "decision.json"
        return self.path_for(theory_id) / f"decision_delay{int(target_delay)}.json"

    def save_theory(self, theory: TheorySpec, *, event_type: str | None = None, payload: dict[str, Any] | None = None) -> None:
        tdir = self.path_for(theory.theory_id)
        tdir.mkdir(parents=True, exist_ok=True)
        data = theory.to_dict()
        data["updated_at"] = utc_now()
        self.theory_path(theory.theory_id).write_text(json.dumps(data, indent=2, sort_keys=True))
        if event_type:
            self.append_event(theory.theory_id, event_type, payload or {})

    def load_theory(self, theory_id: str) -> TheorySpec:
        data = _read_json(self.theory_path(theory_id), {})
        if not data:
            raise FileNotFoundError(f"Theory not found: {theory_id}")
        return TheorySpec.from_dict(data)

    def list_theories(self, statuses: set[str] | None = None) -> list[TheorySpec]:
        theories: list[TheorySpec] = []
        for path in sorted(self.theory_dir.glob("*/theory.json")):
            try:
                theory = TheorySpec.from_dict(json.loads(path.read_text()))
            except Exception:
                continue
            if statuses is None or theory.status in statuses:
                theories.append(theory)
        return theories

    def update_status(self, theory_id: str, status: str, reason: str = "") -> TheorySpec:
        theory = self.load_theory(theory_id).with_status(status)
        self.save_theory(theory, event_type="status_updated", payload={"status": status, "reason": reason})
        return theory

    def append_event(self, theory_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        event = TheoryEvent(theory_id=theory_id, event_type=event_type, payload=payload or {})
        path = self.history_path(theory_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    def events(self, theory_id: str) -> list[dict[str, Any]]:
        return _read_jsonl(self.history_path(theory_id))

    def write_memory(self, theory_id: str, text: str, *, target_delay: int | None = None) -> None:
        path = self.memory_path_for_delay(theory_id, target_delay)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n")

    def write_decision(
        self,
        theory_id: str,
        decision: dict[str, Any],
        *,
        target_delay: int | None = None,
    ) -> None:
        path = self.decision_path(theory_id, target_delay)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision, indent=2, sort_keys=True))
        payload = dict(decision)
        if target_delay is not None:
            payload["target_delay"] = int(target_delay)
        self.append_event(theory_id, "decision_updated", payload)

    def read_decision(self, theory_id: str, *, target_delay: int | None = None) -> dict[str, Any]:
        return _read_json(self.decision_path(theory_id, target_delay), {})

    def next_candidates_path(self, theory_id: str, stage: str, *, target_delay: int | None = None) -> Path:
        tdir = self.path_for(theory_id)
        tdir.mkdir(parents=True, exist_ok=True)
        suffix = f"_delay{int(target_delay)}" if target_delay is not None else ""
        existing = sorted(tdir.glob(f"candidates_{stage}_*{suffix}.jsonl"))
        return tdir / f"candidates_{stage}_{len(existing) + 1:03d}{suffix}.jsonl"

    def result_path_for_candidates(self, candidates_path: Path) -> Path:
        name = candidates_path.name.replace("candidates_", "results_", 1)
        return candidates_path.with_name(name)

    def write_candidates(
        self,
        theory_id: str,
        stage: str,
        candidates: list[CandidatePlanRecord],
        *,
        target_delay: int | None = None,
    ) -> Path:
        path = self.next_candidates_path(theory_id, stage, target_delay=target_delay)
        _write_jsonl(path, [c.to_dict() for c in candidates])
        self.append_event(
            theory_id,
            "generated",
            {
                "stage": stage,
                "path": str(path),
                "n_candidates": len(candidates),
                "target_delay": target_delay,
            },
        )
        return path

    def candidate_index(self) -> dict[str, dict[str, Any]]:
        """Return normalized expression -> candidate metadata across all theories."""
        index: dict[str, dict[str, Any]] = {}
        for path in sorted(self.theory_dir.glob("*/candidates_*.jsonl")):
            for row in _read_jsonl(path):
                expr = normalize_expression(str(row.get("expression") or ""))
                if expr:
                    meta = {**row, "_candidate_path": str(path)}
                    index[expr] = meta
                    delay = row_delay(row)
                    if delay is not None:
                        index[candidate_index_key(expr, delay)] = meta
        return index

    def theory_candidate_expressions(
        self,
        theory_id: str,
        *,
        target_delay: int | None = None,
        exclude_delay: int | None = None,
    ) -> set[str]:
        exprs: set[str] = set()
        for path in sorted(self.path_for(theory_id).glob("candidates_*.jsonl")):
            for row in _read_jsonl(path):
                delay = row_delay(row)
                if target_delay is not None and delay != int(target_delay):
                    continue
                if exclude_delay is not None and delay == int(exclude_delay):
                    continue
                expr = normalize_expression(str(row.get("expression") or ""))
                if expr:
                    exprs.add(expr)
        return exprs

    def write_ingested_results(self, grouped_results: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for theory_id, rows in grouped_results.items():
            existing_paths = sorted(self.path_for(theory_id).glob("results_ingest_*.jsonl"))
            path = self.path_for(theory_id) / f"results_ingest_{len(existing_paths) + 1:03d}.jsonl"
            _write_jsonl(path, rows)
            counts[theory_id] = len(rows)
            self.append_event(
                theory_id,
                "ingested",
                {"path": str(path), "n_results": len(rows)},
            )
        return counts

    def write_results_for_candidate_path(self, candidate_path: str | Path, rows: list[dict[str, Any]]) -> Path:
        candidate_path = Path(candidate_path)
        path = self.result_path_for_candidates(candidate_path)
        _write_jsonl(path, rows)
        theory_id = str(rows[0].get("theory_id") or candidate_path.parent.name) if rows else candidate_path.parent.name
        self.append_event(
            theory_id,
            "ingested",
            {"path": str(path), "n_results": len(rows), "candidate_path": str(candidate_path)},
        )
        return path

    def all_results(self, theory_id: str, *, target_delay: int | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pattern in ("results_*.jsonl",):
            for path in sorted(self.path_for(theory_id).glob(pattern)):
                rows.extend(_read_jsonl(path))
        if target_delay is not None:
            rows = [row for row in rows if row_delay(row) == int(target_delay)]
        return rows


def row_delay(row: dict[str, Any]) -> int | None:
    for source in (
        {"delay": row.get("target_delay")},
        row.get("settings") or {},
        row.get("settings_override") or {},
        {"delay": row.get("admission_delay")},
    ):
        try:
            value = source.get("delay")
        except AttributeError:
            continue
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def candidate_index_key(expression: str, target_delay: int) -> str:
    return f"delay{int(target_delay)}\t{normalize_expression(expression)}"
