"""Backfill max_self_correlation for historical sim records using PnL-based computation.

Brain only populates is.selfCorrelation for submitted competition alphas.
For all other records (grammar, sweep, etc.) we compute it ourselves:
  1. Load PnL series for all current competition portfolio alphas once.
  2. Load IS Sharpe/Fitness for those portfolio alphas once.
  3. For each sim record, fetch the alpha's PnL from Brain.
  4. Compute signed max Pearson correlation (first-differenced daily returns).
  5. Write max_self_correlation plus matched portfolio benchmark metadata.

Saves incrementally every --save-every records so the run is resumable.

Usage:
    python -m tools.backfill_self_correlation [--lane grammar] [--max-records 500]

Flags:
    --sim-log        Path to sim log JSONL (default: artifacts/sim/master.jsonl)
    --credentials    Path to brain.txt (default: brain.txt)
    --competition-id Competition ID to fetch portfolio from (default: IQC2026S1)
    --lane           Filter by search_lane (default: all lanes)
    --max-records    Max records to process in one run (default: 500)
    --sleep          Seconds between PnL fetches (default: 2.0)
    --save-every     Write checkpoint every N processed records (default: 100)
    --portfolio-max-attempts
                     Max attempts per portfolio alpha PnL fetch
                     (default: retry forever)
    --record-max-attempts
                     Max attempts per candidate alpha PnL fetch
                     (default: retry forever)
    --recompute      Also recompute records that already have max_self_correlation
    --dry-run        Print which records would be fetched, don't write
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from core.qtable import is_degenerate_sim_record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill max_self_correlation via PnL.")
    p.add_argument("--sim-log", default="artifacts/sim/master.jsonl")
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--competition-id", default="IQC2026S1")
    p.add_argument("--lane", default=None)
    p.add_argument("--max-records", type=int, default=500)
    p.add_argument("--sleep", type=float, default=2.0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--timeout", type=float, default=8.0,
                   help="HTTP timeout for PnL fetches in seconds (default: 8)")
    p.add_argument(
        "--portfolio-max-attempts",
        type=int,
        default=None,
        help="Max attempts per portfolio alpha fetch (default: retry forever)",
    )
    p.add_argument(
        "--record-max-attempts",
        type=int,
        default=None,
        help="Max attempts per candidate alpha fetch (default: retry forever)",
    )
    p.add_argument("--recompute", action="store_true",
                   help="Recompute even records that already have max_self_correlation")
    p.add_argument("--since-days", type=int, default=0,
                   help="Only backfill records logged within the last N days (0 = all)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _missing_corr_benchmark_metadata(record: dict) -> bool:
    required_keys = (
        "max_self_correlation_alpha_id",
        "max_self_correlation_alpha_sharpe",
        "max_self_correlation_alpha_fitness",
    )
    return any(record.get(key) is None for key in required_keys)


def _needs_backfill(record: dict, lane: str | None, recompute: bool, since_days: int) -> bool:
    if record.get("status") != "done":
        return False
    if not record.get("alpha_id"):
        return False
    if is_degenerate_sim_record(record):
        return False
    if lane is not None and record.get("search_lane") != lane:
        return False
    has_corr = record.get("max_self_correlation") is not None
    if not recompute and has_corr and not _missing_corr_benchmark_metadata(record):
        return False
    if since_days > 0:
        logged_at = record.get("logged_at") or ""
        if logged_at:
            from datetime import datetime, timezone, timedelta
            try:
                ts = datetime.fromisoformat(logged_at.replace("Z", "+00:00"))
                cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
                if ts < cutoff:
                    return False
            except ValueError:
                pass
    return True


def _save_log(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(path)


def _attempt_budget_label(max_attempts: int | None) -> str:
    return "forever" if max_attempts is None else str(max_attempts)


def main() -> None:
    args = parse_args()
    sim_path = Path(args.sim_log)
    if not sim_path.exists():
        raise FileNotFoundError(sim_path)

    records: list[dict] = []
    with sim_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({})

    skipped_degenerate = sum(1 for r in records if is_degenerate_sim_record(r))
    to_fill = [i for i, r in enumerate(records)
               if _needs_backfill(r, args.lane, args.recompute, args.since_days)]

    print(f"Records in log:           {len(records)}")
    print(f"Candidates for backfill:  {len(to_fill)}")
    if skipped_degenerate:
        print(f"Degenerate rows skipped:  {skipped_degenerate}")
    if args.lane:
        print(f"Lane filter:              {args.lane}")
    if args.recompute:
        print("Mode: recompute (overwrite existing values)")

    if args.dry_run:
        for i in to_fill[:20]:
            r = records[i]
            print(f"  alpha_id={r.get('alpha_id')}  lane={r.get('search_lane')}  "
                  f"sharpe={r.get('is_sharpe')}  existing_corr={r.get('max_self_correlation')}  "
                  f"benchmark_metadata_missing={_missing_corr_benchmark_metadata(r)}  "
                  f"{r.get('expression','')[:60]}")
        if len(to_fill) > 20:
            print(f"  ... and {len(to_fill) - 20} more")
        return

    if not to_fill:
        print("Nothing to backfill.")
        return

    from core.brain.api import (
        sign_in, fetch_pnl_series_with_retry,
    )
    from core.selfcorr import load_portfolio_alpha_data, compute_max_self_corr_info

    session = sign_in(args.credentials)

    print(
        "Retry policy: "
        f"portfolio={_attempt_budget_label(args.portfolio_max_attempts)} "
        f"record={_attempt_budget_label(args.record_max_attempts)}"
    )

    # Load portfolio PnL plus benchmark metadata once.
    print(f"\nLoading portfolio PnL + benchmark metadata for {args.competition_id} ...")
    portfolio_alpha_data = load_portfolio_alpha_data(
        session,
        competition_id=args.competition_id,
        max_pnl_attempts=args.portfolio_max_attempts,
    )
    print(f"Portfolio: {len(portfolio_alpha_data)} alphas with PnL + metadata")

    if not portfolio_alpha_data:
        print("No portfolio PnL loaded — cannot compute correlations. Exiting.")
        return

    batch = to_fill[:args.max_records]
    print(f"\nProcessing {len(batch)} records (sleep={args.sleep}s between fetches) ...\n")

    # Make a backup before first write
    bak_path = sim_path.with_suffix(".jsonl.bak")
    shutil.copy2(sim_path, bak_path)
    print(f"Backup: {bak_path}")

    updated = skipped_no_pnl = rate_limit_waits = errors = 0

    for idx, i in enumerate(batch):
        r = records[i]
        alpha_id = r["alpha_id"]

        # fetch_pnl_series_with_retry handles both:
        #   - async delay (empty {} right after simulation, needs polling)
        #   - rate limits (None return, retries with sleep)
        pnl = fetch_pnl_series_with_retry(
            session,
            alpha_id,
            max_attempts=args.record_max_attempts,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
        )

        if not pnl:
            skipped_no_pnl += 1
            print(f"  [{idx+1}/{len(batch)}] no_pnl  {alpha_id}  "
                  f"lane={r.get('search_lane')}  sharpe={r.get('is_sharpe')}")
            time.sleep(args.sleep)
            continue

        corr_info = compute_max_self_corr_info(pnl, portfolio_alpha_data)
        corr = corr_info["max_self_correlation"]
        if corr is None:
            skipped_no_pnl += 1
            print(f"  [{idx+1}/{len(batch)}] no_corr  {alpha_id}  (corr=None, likely constant PnL)")
            time.sleep(args.sleep)
            continue

        old_val = r.get("max_self_correlation")
        changed = False
        for key, new_value in corr_info.items():
            if key == "max_self_correlation" and old_val is not None and not args.recompute:
                continue
            if r.get(key) != new_value:
                r[key] = new_value
                changed = True
        if not changed:
            print(f"  [{idx+1}/{len(batch)}] unchanged  {alpha_id}  corr={corr:.4f}")
            time.sleep(args.sleep)
            continue
        updated += 1

        flag = " *** HIGH ***" if corr >= 0.70 else ""
        old_s = f"  (was {old_val:.3f})" if old_val is not None else ""
        print(f"  [{idx+1}/{len(batch)}] updated  {alpha_id}  "
              f"corr={corr:.4f}{flag}{old_s}  {r.get('expression','')[:60]}")

        if updated % args.save_every == 0:
            _save_log(sim_path, records)
            print(f"  → Checkpoint saved ({updated} updated so far)")

        time.sleep(args.sleep)

    print(f"\nUpdated: {updated}  |  No PnL / skipped: {skipped_no_pnl}")

    if updated == 0:
        print("No updates to write.")
        return

    _save_log(sim_path, records)
    print(f"Updated: {sim_path}")
    print(f"\nNext step: rebuild Q-table from updated log:")
    print(f"  python -m alpha_discovery.grammar_method.run_pipeline --fields-csv all_fields.csv --rebuild-qtable --rounds 0")


if __name__ == "__main__":
    main()
