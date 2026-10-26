"""Consolidate all sim logs and runner logs into alpha_search/artifacts/.

Steps performed:
  1. Load all *_sim.jsonl from every artifact directory.
     Filter: status == 'done' AND is_sharpe is not None.
     Dedup:  by expression+regime; keep higher is_sharpe on collision.
     Write:  alpha_search/artifacts/all_sim.jsonl  (overwrites existing).

  2. Concatenate all runner logs in chronological order.
     Write:  alpha_search/artifacts/runner.log  (new file).

  3. Move non-sim batch .jsonl files from outer artifacts/ → alpha_search/artifacts/.

Run from project root:
    python alpha_search/consolidate_logs.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from wq_alpha.logs import sim_record_key

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Step 1 — sim log consolidation
# ---------------------------------------------------------------------------

SIM_SOURCES = [
    "artifacts/all_sim.jsonl",
    "artifacts/model16_sim.jsonl",
    "artifacts/cw_repair_sim.jsonl",
    "artifacts/repair_sim.jsonl",
    "alpha_grammar/artifacts/analyst4_sim.jsonl",
    "alpha_grammar/artifacts/fundamental2_sim.jsonl",
    "alpha_grammar/artifacts/fundamental6_sim.jsonl",
    "alpha_grammar/artifacts/model16_sim.jsonl",
    "alpha_grammar/artifacts/pv1_sim.jsonl",
    "alpha_multi/artifacts/news12-analyst4_sim.jsonl",
    "alpha_multi/artifacts/pv1-fundamental6-pv13_sim.jsonl",
    "alpha_search/artifacts/all_sim.jsonl",
    "alpha_search/artifacts/weight_sweep_sim.jsonl",
    "alpha_search/artifacts/hybrid_fundamental2_sim.jsonl",
    "alpha_search/artifacts/hybrid_fundamental6_sim.jsonl",
    "alpha_search/artifacts/hybrid_pv1_sim.jsonl",
    "simulation_log.jsonl",
]

SIM_OUTPUT = ROOT / "alpha_search/artifacts/all_sim.jsonl"


def consolidate_sims() -> None:
    total_loaded = 0
    best: dict[str, dict] = {}  # expression+regime -> best record

    for rel in SIM_SOURCES:
        path = ROOT / rel
        if not path.exists():
            print(f"  [skip] {rel}")
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
                total_loaded += 1

                # filter
                if rec.get("status") != "done":
                    continue
                if rec.get("is_sharpe") is None:
                    continue

                key = sim_record_key(rec)
                if not key:
                    continue

                # dedup — keep higher sharpe
                existing = best.get(key)
                if existing is None or (rec.get("is_sharpe") or 0) > (existing.get("is_sharpe") or 0):
                    best[key] = rec

        print(f"  {n:>5} records  ← {rel}")

    records = list(best.values())
    records.sort(key=lambda r: r.get("is_sharpe") or 0, reverse=True)

    with SIM_OUTPUT.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    import collections
    grades = collections.Counter(r.get("grade") for r in records)

    print(f"\nLoaded {total_loaded} records from {len(SIM_SOURCES)} sources")
    print(f"After filter + dedup: {len(records)}")
    print(f"Wrote {SIM_OUTPUT}")
    print(f"Grade breakdown: {dict(grades)}")


# ---------------------------------------------------------------------------
# Step 2 — runner log consolidation
# ---------------------------------------------------------------------------

LOG_SOURCES = [
    "search_runner.log",
    "alpha_grammar/artifacts/fundamental6_runner.log",
    "alpha_grammar/artifacts/fundamental2_runner.log",
    "alpha_grammar/artifacts/analyst4_runner.log",
    "alpha_grammar/artifacts/pv1_runner.log",
    "alpha_grammar/artifacts/model16_runner.log",
    "alpha_multi/artifacts/pv1-fundamental6-pv13_runner.log",
    "alpha_search/artifacts/hybrid_pv1_runner.log",
    "alpha_search/artifacts/hybrid_fundamental6_runner.log",
    "alpha_search/artifacts/hybrid_fundamental2_runner.log",
]

LOG_OUTPUT = ROOT / "alpha_search/artifacts/runner.log"

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _first_timestamp(path: Path) -> str:
    """Return the first timestamp found in the file, or '' if none."""
    try:
        with path.open(errors="replace") as f:
            for line in f:
                m = _TS_RE.search(line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return ""


def consolidate_logs() -> None:
    entries: list[tuple[str, Path]] = []
    for rel in LOG_SOURCES:
        path = ROOT / rel
        if not path.exists():
            print(f"  [skip] {rel}")
            continue
        ts = _first_timestamp(path)
        entries.append((ts, path))
        print(f"  {ts or '(no ts)'}  ← {rel}")

    entries.sort(key=lambda x: x[0])  # chronological by first timestamp

    with LOG_OUTPUT.open("w") as out:
        for ts, path in entries:
            out.write(f"\n{'='*80}\n")
            out.write(f"# SOURCE: {path.relative_to(ROOT)}  (first ts: {ts or 'unknown'})\n")
            out.write(f"{'='*80}\n\n")
            out.write(path.read_text(errors="replace"))

    total_lines = sum(p.read_text(errors="replace").count("\n") for _, p in entries)
    print(f"\nConcatenated {len(entries)} log files ({total_lines} lines)")
    print(f"Wrote {LOG_OUTPUT}")


# ---------------------------------------------------------------------------
# Step 3 — move batch files from outer artifacts/ → alpha_search/artifacts/
# ---------------------------------------------------------------------------

BATCH_SOURCES = [
    "artifacts/analyst4_v2_batch.jsonl",
    "artifacts/analyst4_v3_batch.jsonl",
    "artifacts/analyst4_v4_batch.jsonl",
    "artifacts/analyst4_cd_followup_build.jsonl",
    "artifacts/analyst4_cd_followup_only_batch.jsonl",
    "artifacts/analyst4_v2_only_batch.jsonl",
    "artifacts/analyst4_v3_sweep_only_batch.jsonl",
    "artifacts/model16_v2_batch.jsonl",
    "artifacts/model51_batch.jsonl",
    "artifacts/news12_v2_batch.jsonl",
    "artifacts/pv1_batch.jsonl",
    "artifacts/f6_v2_batch.jsonl",
    "artifacts/socialmedia12_batch.jsonl",
    "artifacts/cw_repair_batch.jsonl",
    "artifacts/repair_batch.jsonl",
]

BATCH_DEST = ROOT / "alpha_search/artifacts"


def move_batches() -> None:
    moved = 0
    for rel in BATCH_SOURCES:
        src = ROOT / rel
        if not src.exists():
            print(f"  [skip] {rel}")
            continue
        dst = BATCH_DEST / src.name
        if dst.exists():
            print(f"  [exists, skip] {src.name}")
            continue
        shutil.move(str(src), str(dst))
        print(f"  moved  {rel}  →  alpha_search/artifacts/{src.name}")
        moved += 1
    print(f"\nMoved {moved} batch files")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("STEP 1 — Consolidate sim logs")
    print("=" * 60)
    consolidate_sims()

    print()
    print("=" * 60)
    print("STEP 2 — Consolidate runner logs")
    print("=" * 60)
    consolidate_logs()

    print()
    print("=" * 60)
    print("STEP 3 — Move batch files")
    print("=" * 60)
    move_batches()

    print()
    print("Done. Verify alpha_search/artifacts/all_sim.jsonl, then run Step 4 deletions.")


if __name__ == "__main__":
    main()
