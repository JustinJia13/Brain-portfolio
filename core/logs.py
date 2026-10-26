"""Shared sim-log utilities: session log creation and master merge.

Each pipeline run creates a timestamped session log under artifacts/sim/sessions/.
At end of run, call merge_session_to_master() to absorb it into master.jsonl.
The session log can then be deleted or kept for debugging.

Layout:
    artifacts/sim/
    ├── master.jsonl                      ← permanent deduped history
    └── sessions/
        ├── grammar_20250401_120000.jsonl ← one per grammar run
        ├── workshop_20250401_130000.jsonl
        └── search_20250401_140000.jsonl

Runner text logs are written to:
    logs/YYYY-MM-DD/{source}.log
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
MASTER = ROOT / "artifacts" / "sim" / "master.jsonl"
SESSIONS_DIR = ROOT / "artifacts" / "sim" / "sessions"
DETAILS_DIR = ROOT / "artifacts" / "sim" / "details"
LOGS_DIR = ROOT / "logs"


def run_log_path(source: str) -> Path:
    """Return the log file path for a named source under logs/YYYY-MM-DD/.

    Example: run_log_path("grammar") → logs/2026-04-24/grammar.log
    Creates the date directory if it doesn't exist.
    Multiple runs on the same day append to the same file.
    """
    date_dir = LOGS_DIR / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir / f"{source}.log"


def session_log_path(pipeline: str) -> Path:
    """Return a timestamped session log path for the given pipeline name.

    Example: session_log_path("grammar") →
             artifacts/sim/sessions/grammar_20250401_120000.jsonl
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SESSIONS_DIR / f"{pipeline}_{ts}.jsonl"


def merge_session_to_master(
    session_path: str | Path,
    *,
    delete_after: bool = False,
) -> tuple[int, int]:
    """Merge a session log into master.jsonl.

    Dedup key: normalized expression plus effective simulation regime.
    On conflict: prefer status=="done", then higher abs(is_sharpe).

    Returns (n_new, n_updated) — records added or updated in master.
    """
    session_path = Path(session_path)
    if not session_path.exists():
        return 0, 0

    # Load existing master into memory
    master_records: dict[str, dict] = {}
    if MASTER.exists():
        with MASTER.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = sim_record_key(rec)
                    if key:
                        rec = compact_master_record(rec)
                        existing = master_records.get(key)
                        if existing is None or _record_priority(rec) > _record_priority(existing):
                            master_records[key] = rec
                except json.JSONDecodeError:
                    continue

    # Merge session records
    n_new = n_updated = 0
    with session_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = sim_record_key(rec)
            if not key:
                continue
            rec = compact_master_record(rec)
            if key not in master_records:
                master_records[key] = rec
                n_new += 1
            else:
                existing = master_records[key]
                if _record_priority(rec) > _record_priority(existing):
                    master_records[key] = rec
                    n_updated += 1

    # Rewrite master (done records first, sorted by logged_at)
    MASTER.parent.mkdir(parents=True, exist_ok=True)
    done = sorted(
        [r for r in master_records.values() if r.get("status") == "done"],
        key=lambda r: r.get("logged_at", ""),
    )
    other = [r for r in master_records.values() if r.get("status") != "done"]
    with MASTER.open("w") as f:
        for rec in done + other:
            f.write(json.dumps(rec) + "\n")

    if delete_after:
        session_path.unlink()

    return n_new, n_updated


def _record_priority(rec: dict) -> tuple:
    is_done = int(rec.get("status") == "done")
    sharpe = abs(float(rec.get("is_sharpe") or 0.0))
    return (is_done, sharpe)


def normalize_expression(expression: object) -> str:
    """Collapse whitespace in an expression for stable log identity."""
    return " ".join(str(expression or "").split())


def sim_record_regime(rec: dict) -> str:
    """Return a stable regime string including settings axes that affect results."""
    settings = rec.get("settings")
    if not isinstance(settings, dict):
        settings = rec.get("settings_override")
    if not isinstance(settings, dict):
        settings = {}

    parts = [
        f"delay={settings.get('delay', '')}",
        f"neutralization={settings.get('neutralization', '')}",
        f"universe={settings.get('universe', '')}",
        f"decay={settings.get('decay', '')}",
        f"truncation={settings.get('truncation', '')}",
    ]

    legacy_regime = rec.get("regime")
    if legacy_regime and not any(part.split("=", 1)[1] for part in parts):
        return str(legacy_regime)
    return ";".join(parts)


def sim_record_key(rec: dict) -> str:
    """Stable master-log key for one simulation row.

    Expression is required because it is the only key shared by every lane.
    The regime suffix prevents same-expression runs under different settings
    from replacing each other in the permanent master history.
    """
    expression = normalize_expression(rec.get("expression"))
    if not expression:
        return ""
    return f"{expression}\t{sim_record_regime(rec)}"


def compact_master_record(rec: dict) -> dict:
    """Return a copy suitable for the permanent master log.

    Full Brain responses are useful for debugging, but storing them inline in
    every permanent row makes master.jsonl grow quickly. Keep top-level metrics
    and checks in master, and spill the raw blobs into stable per-alpha files.
    """
    out = dict(rec)
    for field in ("raw_alpha_detail", "raw_progress_result"):
        blob = out.pop(field, None)
        if blob:
            ref = _write_detail_blob(out, field, blob)
            if ref:
                out[f"{field}_path"] = ref
    return out


def _write_detail_blob(rec: dict, field: str, blob: object) -> str:
    alpha_id = str(rec.get("alpha_id") or "").strip()
    candidate_id = str(rec.get("candidate_id") or "").strip()
    if alpha_id:
        stem = alpha_id
    elif candidate_id:
        stem = candidate_id
    else:
        import hashlib

        stem = hashlib.sha1(sim_record_key(rec).encode()).hexdigest()[:16]
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    path = DETAILS_DIR / f"{stem}_{field}.json"
    if not path.exists():
        with path.open("w") as f:
            json.dump(blob, f, sort_keys=True, default=str)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def master_record_count() -> dict[str, int]:
    """Quick stats on master.jsonl without loading all records."""
    counts: dict[str, int] = {"total": 0, "done": 0, "failed": 0}
    if not MASTER.exists():
        return counts
    with MASTER.open() as f:
        for line in f:
            if not line.strip():
                continue
            counts["total"] += 1
            try:
                r = json.loads(line)
                if r.get("status") == "done":
                    counts["done"] += 1
                else:
                    counts["failed"] += 1
            except json.JSONDecodeError:
                pass
    return counts


__all__ = [
    "MASTER", "SESSIONS_DIR", "DETAILS_DIR", "LOGS_DIR",
    "run_log_path", "session_log_path", "merge_session_to_master", "master_record_count",
    "normalize_expression", "sim_record_regime", "sim_record_key", "compact_master_record",
]
