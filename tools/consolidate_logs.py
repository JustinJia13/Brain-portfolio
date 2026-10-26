"""One-time consolidation of all scattered sim logs into artifacts/sim/master.jsonl.

Dedup key: normalized expression plus effective simulation regime.
On conflict: prefer status=="done", then higher abs(is_sharpe).
Old logs are moved to artifacts/sim/archive/ (not deleted).

Usage:
    cd /home/justinjj_/world_quant
    python tools/consolidate_logs.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from wq_alpha.logs import compact_master_record, sim_record_key


ROOT = Path(__file__).parent.parent
MASTER = ROOT / "artifacts" / "sim" / "master.jsonl"
ARCHIVE = ROOT / "artifacts" / "sim" / "archive"

SOURCE_LOGS: list[Path] = [
    ROOT / "alpha_grammar" / "artifacts" / "all_sim.jsonl",
    ROOT / "alpha_grammar" / "artifacts" / "all_batch1.jsonl",
    ROOT / "alpha_grammar" / "artifacts" / "all_batch2.jsonl",
    ROOT / "alpha_grammar" / "artifacts" / "all_batch3.jsonl",
    ROOT / "artifacts" / "workshop" / "workshop_sim.jsonl",
    ROOT / "alpha_search" / "artifacts" / "all_sim.jsonl",
    ROOT / "alpha_search" / "artifacts" / "model16_sim.jsonl",
    ROOT / "artifacts" / "manual_sim.jsonl",
]


def _record_priority(rec: dict) -> tuple:
    """Higher = better. Used to pick winner among dupes."""
    is_done = int(rec.get("status") == "done")
    sharpe = abs(float(rec.get("is_sharpe") or 0.0))
    return (is_done, sharpe)


def load_all(sources: list[Path]) -> dict[str, dict]:
    """Load all records, dedup by expression+regime keeping the best record."""
    best: dict[str, dict] = {}
    stats = {"total": 0, "skipped_no_expr": 0, "dupes_resolved": 0}
    for path in sources:
        if not path.exists():
            print(f"  [skip] {path} (not found)")
            continue
        n = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                stats["total"] += 1
                key = sim_record_key(rec)
                if not key:
                    stats["skipped_no_expr"] += 1
                    continue
                rec = compact_master_record(rec)
                if key not in best:
                    best[key] = rec
                else:
                    stats["dupes_resolved"] += 1
                    if _record_priority(rec) > _record_priority(best[key]):
                        best[key] = rec
        print(f"  read {n:>6}  {path}")
    print(
        f"\n  Total read:          {stats['total']}")
    print(f"  Skipped (no expr):   {stats['skipped_no_expr']}")
    print(f"  Dupes resolved:      {stats['dupes_resolved']}")
    print(f"  Unique expressions:  {len(best)}")
    return best


def write_master(records: dict[str, dict], path: Path, dry_run: bool) -> None:
    done = [r for r in records.values() if r.get("status") == "done"]
    other = [r for r in records.values() if r.get("status") != "done"]
    # Write done records first (sorted by logged_at if available), then others
    sorted_done = sorted(done, key=lambda r: r.get("logged_at", ""))
    ordered = sorted_done + other
    if dry_run:
        print(f"\n[dry-run] Would write {len(ordered)} records to {path}")
        print(f"  done: {len(done)}  other: {len(other)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in ordered:
            f.write(json.dumps(rec) + "\n")
    print(f"\nWrote {len(ordered)} records to {path}")
    print(f"  done: {len(done)}  other: {len(other)}")


def archive_sources(sources: list[Path], archive_dir: Path, dry_run: bool) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in sources:
        if not path.exists():
            continue
        dest = archive_dir / path.name
        # Avoid collision if same filename from different dirs
        if dest.exists():
            stem = path.parent.name + "_" + path.name
            dest = archive_dir / stem
        if dry_run:
            print(f"[dry-run] Would move {path} → {dest}")
        else:
            shutil.move(str(path), str(dest))
            print(f"  archived {path.name} → {dest}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Show what would happen, write nothing")
    args = p.parse_args()

    print("=== Reading source logs ===")
    records = load_all(SOURCE_LOGS)

    print("\n=== Writing master log ===")
    write_master(records, MASTER, dry_run=args.dry_run)

    print("\n=== Archiving source logs ===")
    archive_sources(SOURCE_LOGS, ARCHIVE, dry_run=args.dry_run)

    print("\nDone.")
    if args.dry_run:
        print("(dry-run — nothing was written or moved)")


if __name__ == "__main__":
    main()
