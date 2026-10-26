"""Post-hoc yearly stats backfill for passing sim log entries.

Brain computes the per-year backtest breakdown asynchronously — it typically
becomes available 5-15 minutes after simulation completes. The submission loop
uses a short 3-minute retry window, so most records end up with empty
yearly_stats. This script fills that gap by fetching yearly stats for records
that passed Brain's checks but have no yearly_stats data.

Usage (run 10+ minutes after a grammar/search batch completes):

    python -m tools.backfill_yearly_stats \\
        --sim-log alpha_discovery/grammar_method/artifacts/all_sim.jsonl

Reads the sim log in-place, updates matching records, writes back.
A .bak backup is created before any writes.

Flags:
    --sim-log        Path to sim log to backfill (required)
    --credentials    Path to brain.txt (default: brain.txt)
    --min-sharpe     Min IS Sharpe to attempt fetch (default: 1.0)
    --min-fitness    Min IS Fitness to attempt fetch (default: 0.8)
    --max-records    Max records to fetch in one run (default: 100)
    --dry-run        Print which records would be fetched, don't write
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill yearly_stats for passing sim records.")
    p.add_argument("--sim-log", required=True, help="Path to sim log JSONL")
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--min-sharpe", type=float, default=1.0)
    p.add_argument("--min-fitness", type=float, default=0.8)
    p.add_argument("--max-records", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _needs_backfill(record: dict, min_sharpe: float, min_fitness: float) -> bool:
    if record.get("status") != "done":
        return False
    if not record.get("alpha_id"):
        return False
    sharpe = record.get("is_sharpe") or 0.0
    fitness = record.get("is_fitness") or 0.0
    if sharpe < min_sharpe or fitness < min_fitness:
        return False
    yearly = record.get("yearly_stats") or {}
    # Already filled
    if yearly and any(str(y) in yearly or y in yearly for y in (2022, 2023)):
        return False
    return True


def _extract_year_stats(yearly_stats: dict) -> tuple[float | None, float | None]:
    """Return (oos_sharpe_2023, oos_fitness_2023) from yearly_stats dict.

    yearly_stats keys may be int or str after JSON round-trip.
    """
    row = yearly_stats.get(2023) or yearly_stats.get("2023") or {}
    return row.get("sharpe"), row.get("fitness")


def main() -> None:
    args = parse_args()
    sim_path = Path(args.sim_log)
    if not sim_path.exists():
        raise FileNotFoundError(sim_path)

    # Load all records
    records: list[dict] = []
    with sim_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({})  # preserve line count

    # Find candidates
    to_fill = [
        i for i, r in enumerate(records)
        if _needs_backfill(r, args.min_sharpe, args.min_fitness)
    ]
    print(f"Records in log:           {len(records)}")
    print(f"Candidates for backfill:  {len(to_fill)}")

    if args.dry_run:
        for i in to_fill[:20]:
            r = records[i]
            print(f"  alpha_id={r.get('alpha_id')} sharpe={r.get('is_sharpe'):.3f} "
                  f"fitness={r.get('is_fitness'):.3f}  {r.get('expression', '')[:60]}")
        if len(to_fill) > 20:
            print(f"  ... and {len(to_fill) - 20} more")
        return

    if not to_fill:
        print("Nothing to backfill.")
        return

    # Limit
    batch = to_fill[:args.max_records]
    print(f"Fetching yearly stats for {len(batch)} records...")

    from core.brain.api import sign_in, fetch_yearly_stats
    session = sign_in(args.credentials)

    filled = 0
    for i in batch:
        r = records[i]
        alpha_id = r["alpha_id"]
        stats = fetch_yearly_stats(session, alpha_id)
        if not stats:
            # Brain not ready yet — try once more after a short wait
            time.sleep(10)
            stats = fetch_yearly_stats(session, alpha_id)
        if stats:
            # Normalize to string keys for consistent JSON storage
            stats = {str(k): v for k, v in stats.items()}
            r["yearly_stats"] = stats
            sharpe_2023, fitness_2023 = _extract_year_stats(stats)
            r["oos_sharpe_2023"] = sharpe_2023
            r["oos_fitness_2023"] = fitness_2023
            filled += 1
            print(f"  Filled: alpha_id={alpha_id} 2023_sharpe={sharpe_2023} 2023_fitness={fitness_2023}")
        else:
            print(f"  Not ready: alpha_id={alpha_id} (skip)")

    print(f"\nFilled {filled}/{len(batch)} records.")

    if filled == 0:
        print("No updates to write.")
        return

    # Backup then write
    bak_path = sim_path.with_suffix(".jsonl.bak")
    shutil.copy2(sim_path, bak_path)
    print(f"Backup written to: {bak_path}")

    with sim_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Updated sim log written to: {sim_path}")


if __name__ == "__main__":
    main()
