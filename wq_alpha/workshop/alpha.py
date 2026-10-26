"""Unified alpha fix/enhancement workshop.

Pipeline:
  1. Load candidates from a sim log (--sim-log-input) or from explicit
     expressions (--expr / --expr-file) looked up in existing sim logs.
  2. Route each record: negate / repair / enhance.
  3. Run three sequential stages:
       Stage 1 — Negate:  submit sign-flipped expressions.
       Stage 2 — Repair:  LLM repair for original repair seeds + any negate
                          results that upgraded to repair candidates.
       Stage 3 — Enhance: LLM enhance for original enhance seeds + any repair
                          results that upgraded to enhance candidates.
  4. Each stage result feeds the next; candidate count stays ~N × 5 per stage.
  5. All simulation results update the grammar Q-table.

Routing rules:
  - Fitness <= -1.5                           →  negate (sign-flip, no LLM)
  - Repairable near-miss + concentration <40%
    + positive/unknown Brain score delta      →  repair (LLM)
  - Passing + corr-eligible
    + positive/unknown Brain score delta      →  enhance (LLM)
  Records with no correlation data are treated as corr-eligible.

Usage:
    # Workshop everything from a sim log
    python tools/alpha_workshop.py --sim-log-input artifacts/sim/master.jsonl --submit

    # Dry-run routing from a sim log
    python tools/alpha_workshop.py --sim-log-input artifacts/grammar_sim.jsonl --dry-run

    # Workshop specific expressions (looked up in all known logs)
    python tools/alpha_workshop.py --expr-file expressions.txt --submit

    # Single expression
    python tools/alpha_workshop.py --expr "rank(fnd6_acdo)" --submit
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wq_alpha.workshop.core import (
    generate_candidates,
    route_record,
    failing_checks,
    record_regime,
    record_workshop_key,
)
from wq_alpha.workshop.scoring import with_portfolio_prompt_feedback
from wq_alpha.workshop.simulation import submit_candidate_batch

DEFAULT_SIM_GLOBS = [
    str(ROOT / "*" / "artifacts" / "*sim*.jsonl"),
    str(ROOT / "artifacts" / "*sim*.jsonl"),
]
DEFAULT_OUT_DIR = ROOT / "artifacts" / "workshop"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _find_records(expr: str, records: list[dict]) -> list[dict]:
    """Return best record per regime for a given expression."""
    matches = [
        r for r in records
        if r.get("expression", "").strip() == expr.strip()
        and r.get("status") == "done"
    ]
    if not matches:
        return []
    by_regime: dict[str, dict] = {}
    for record in matches:
        regime = record_regime(record)
        existing = by_regime.get(regime)
        if existing is None or (record.get("is_sharpe") or 0.0) > (existing.get("is_sharpe") or 0.0):
            by_regime[regime] = record
    return sorted(by_regime.values(), key=record_regime)


def _dedup_records(records: list[dict]) -> list[dict]:
    """Deduplicate done records by (expression, regime), keeping best sharpe."""
    by_key: dict[str, dict] = {}
    for r in records:
        if r.get("status") != "done":
            continue
        key = record_workshop_key(r)
        existing = by_key.get(key)
        if existing is None or (r.get("is_sharpe") or 0.0) > (existing.get("is_sharpe") or 0.0):
            by_key[key] = r
    return list(by_key.values())


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _update_qtable(records: list[dict]) -> None:
    """Incrementally update the grammar Q-table from a list of sim records."""
    try:
        from alpha_grammar.qtable import update_qtable_from_records
        n = update_qtable_from_records(records)
        if n:
            print(f"  [qtable] updated {n} record(s)")
    except Exception as e:
        print(f"  [qtable] update skipped: {e}")


# ---------------------------------------------------------------------------
# Simulate unknown expressions
# ---------------------------------------------------------------------------

def _simulate_expressions(
    expressions: list[str],
    session,
    sim_log: Path,
    settings: dict,
    credentials: str,
    *,
    yearly_stats_attempts: int = 1,
    yearly_stats_sleep: float = 30.0,
) -> list[dict]:
    """Submit expressions that have no sim result yet. Returns new records."""
    from wq_alpha.brain.submit import submit_one, poll_one, fetch_detail, build_record
    from wq_alpha.brain.api import fetch_yearly_stats_with_retry
    import hashlib

    new_records: list[dict] = []
    for expr in expressions:
        print(f"  [simulate] {expr[:80]}")
        url = submit_one(session, expr, settings, credentials=credentials)
        if not url:
            print(f"  [simulate] FAILED to submit: {expr[:60]}")
            continue
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
                print(f"  [yearly] {alpha_id} yearly-stats unavailable after {yearly_stats_attempts} attempts")
            status = "done"
        else:
            detail = {}
            yearly = {}
            status = "no_alpha_id"

        cid = hashlib.sha1(expr.encode()).hexdigest()[:16]
        candidate = {"candidate_id": cid, "expression": expr, "search_lane": "manual"}
        record = build_record(candidate, settings, url, progress, alpha_id, status, detail, yearly)
        _append_jsonl(sim_log, record)
        new_records.append(record)

        sharpe = record.get("is_sharpe")
        sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
        print(f"  [simulate] sharpe={sharpe_str} grade={record.get('grade')}  {expr[:60]}")

    return new_records


# ---------------------------------------------------------------------------
# Submit candidates
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def _write_leaderboard(
    all_records: list[dict],
    sim_log: Path,
    out_dir: Path,
) -> Path:
    """Write a ranked leaderboard of all passing alphas across the workshop history.

    Calls find_passing_examples for every unique (dataset_id, regime) pair found
    in the combined record pool, deduplicates, sorts by Sharpe descending, and
    writes a human-readable text file + a JSONL file.

    Returns the path to the text leaderboard.
    """
    from wq_alpha.workshop.core import find_passing_examples

    # Merge workshop sim log records into the pool (they may not be in all_records yet)
    workshop_records = _load_records([str(sim_log)])
    combined_by_key: dict[str, dict] = {
        record_workshop_key(r): r
        for r in all_records + workshop_records
        if r.get("expression", "")
    }
    combined = list(combined_by_key.values())

    # Collect passing records across every unique (dataset, regime)
    seen_pairs: set[tuple[str, str]] = set()
    leaderboard_by_key: dict[str, dict] = {}

    for r in combined:
        dataset = r.get("dataset_id") or ""
        regime = record_regime(r)
        pair = (dataset, regime)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        # Pass a dummy target with the right dataset/regime; n=len(combined) = no cap
        dummy = {"dataset_id": dataset, "regime": regime}
        for p in find_passing_examples(combined, dummy, n=len(combined)):
            leaderboard_by_key[record_workshop_key(p)] = p

    leaderboard = sorted(
        leaderboard_by_key.values(),
        key=lambda r: r.get("is_sharpe") or 0.0,
        reverse=True,
    )

    if not leaderboard:
        print("  [leaderboard] No passing alphas found.")
        return out_dir / "leaderboard.txt"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── JSONL ─────────────────────────────────────────────────────────────
    jsonl_path = out_dir / "leaderboard.jsonl"
    with jsonl_path.open("w") as f:
        for r in leaderboard:
            f.write(json.dumps(r) + "\n")

    # ── Text table ────────────────────────────────────────────────────────
    txt_path = out_dir / "leaderboard.txt"
    header = (
        f"{'Rank':>4}  {'Sharpe':>6}  {'Fitness':>7}  "
        f"{'OOS-23':>6}  {'Corr':>6}  {'Lane':<8}  {'Dataset':<12}  Expression"
    )
    sep = "─" * len(header)

    from datetime import datetime
    lines = [
        f"Workshop Leaderboard — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source: {sim_log}  |  {len(leaderboard)} passing alphas  |  {len(combined)} total records",
        "",
        header,
        sep,
    ]
    for rank, r in enumerate(leaderboard, 1):
        sharpe  = r.get("is_sharpe") or 0.0
        fitness = r.get("is_fitness") or 0.0
        oos23   = r.get("oos_sharpe_2023")
        corr    = r.get("max_self_correlation")
        lane    = (r.get("search_lane") or "")[:8]
        dataset = (r.get("dataset_id") or "")[:12]
        expr    = r.get("expression", "")

        oos23_s = f"{oos23:>6.3f}" if oos23 is not None else "   N/A"
        corr_s  = f"{corr:>6.3f}" if corr is not None else "   N/A"

        lines.append(
            f"{rank:>4}  {sharpe:>6.3f}  {fitness:>7.3f}  "
            f"{oos23_s}  {corr_s}  {lane:<8}  {dataset:<12}  {expr}"
        )

    with txt_path.open("w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [leaderboard] {len(leaderboard)} passing alphas → {txt_path}")
    return txt_path


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _run_stage(
    label: str,
    pool: list[dict],
    all_records: list[dict],
    history,
    known_exprs: set[str],
    session,
    sim_log: Path,
    settings: dict,
    credentials: str,
    batch_size: int,
    api_key: str | None,
    *,
    skip_negate: bool,
    portfolio_alpha_data: dict | None = None,
    competition_id: str = "IQC2026S1",
    pnl_sleep: float = 1.0,
    yearly_stats_attempts: int = 1,
    yearly_stats_sleep: float = 30.0,
) -> list[dict]:
    """Generate candidates for a single stage, submit, and update state.

    Returns new sim records from this stage.
    """
    if not pool:
        print(f"\n{label}: no candidates — skipping.")
        return []

    print(f"\n{'='*55}")
    print(f"{label}  ({len(pool)} input records)")
    print(f"{'='*55}")

    pool = [
        with_portfolio_prompt_feedback(
            r,
            session,
            portfolio_alpha_data,
            competition_id=competition_id,
            pnl_sleep=pnl_sleep,
        )
        for r in pool
    ]
    candidates = generate_candidates(
        pool, all_records,
        history=history,
        api_key=api_key,
        known_exprs=known_exprs,
        verbose=True,
        skip_negate=skip_negate,
    )
    if not candidates:
        print("  No candidates generated.")
        return []

    print(f"\nSubmitting {len(candidates)} candidates...")
    new_records = submit_candidate_batch(
        candidates, session, sim_log, settings, credentials, batch_size,
        portfolio_alpha_data=portfolio_alpha_data,
        pnl_sleep=pnl_sleep,
        yearly_stats_attempts=yearly_stats_attempts,
        yearly_stats_sleep=yearly_stats_sleep,
    )
    all_records.extend(new_records)
    _update_qtable(new_records)
    return new_records


# ---------------------------------------------------------------------------
# Routing display
# ---------------------------------------------------------------------------

def _print_routing(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Print only actionable records and return (negate_q, repair_q, enhance_q)."""
    negate_q, repair_q, enhance_q = [], [], []
    n_skip = 0
    for r in records:
        action = route_record(r)
        if action is None:
            n_skip += 1
            continue
        sharpe = r.get("is_sharpe", 0.0) or 0.0
        fitness = r.get("is_fitness", 0.0) or 0.0
        regime = record_regime(r)
        expr = r.get("expression", "")
        tag = {
            "negate":  f"NEGATE   regime={regime}  sharpe={sharpe:.3f}",
            "repair":  f"REPAIR   regime={regime}  sharpe={sharpe:.3f}  failing={failing_checks(r)}",
            "enhance": f"ENHANCE  regime={regime}  sharpe={sharpe:.3f}  fitness={fitness:.3f}",
        }[action]
        print(f"  {tag}  {expr[:65]}")
        if action == "negate":
            negate_q.append(r)
        elif action == "repair":
            repair_q.append(r)
        elif action == "enhance":
            enhance_q.append(r)
    if n_skip:
        print(f"  ({n_skip} records skipped — outside routing thresholds)")
    return negate_q, repair_q, enhance_q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument(
        "--sim-log-input", metavar="PATH",
        help="Sim log to scan for workshop candidates (routes all done records).",
    )
    input_grp.add_argument("--expr", nargs="+", help="One or more expressions.")
    input_grp.add_argument("--expr-file", help="Text file, one expression per line (# = comment).")

    parser.add_argument(
        "--submit", action="store_true",
        help="Submit candidates to Brain automatically (requires Brain credentials).",
    )
    parser.add_argument(
        "--sim-log", default="",
        help="Sim log to write results to (default: artifacts/workshop/workshop_sim.jsonl).",
    )
    parser.add_argument(
        "--lookup-logs", nargs="*",
        help="Additional sim logs for history context and known_exprs dedup "
             "(default: all pipeline logs). When using --sim-log-input, the input "
             "log is always included automatically.",
    )
    parser.add_argument("--api-key", default=None, help="Anthropic API key.")
    parser.add_argument("--credentials", default="brain.txt", help="Brain credentials file.")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument(
        "--competition-id", default="IQC2026S1",
        help="Competition ID for loading portfolio PnL (default: IQC2026S1).",
    )
    parser.add_argument(
        "--skip-portfolio-pnl", action="store_true",
        help="Skip loading portfolio PnL (disables inline self-correlation and LLM portfolio-impact feedback).",
    )
    parser.add_argument(
        "--pnl-sleep", type=float, default=1.0, metavar="SECONDS",
        help="Sleep between PnL fetch retries (default: 1.0).",
    )
    parser.add_argument(
        "--yearly-stats-attempts",
        type=int,
        default=1,
        help="Retry yearly-stats fetch this many times after each completed workshop simulation.",
    )
    parser.add_argument(
        "--yearly-stats-sleep",
        type=float,
        default=30.0,
        help="Seconds to wait between yearly-stats retry attempts.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for plan artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Show routing only, no LLM or submission.")
    args = parser.parse_args()

    # ── Resolve output paths ──────────────────────────────────────────────
    sim_log = Path(args.sim_log) if args.sim_log else DEFAULT_OUT_DIR / "workshop_sim.jsonl"
    out_dir = Path(args.out_dir)

    # ── Load all_records (history context + known_exprs) ─────────────────
    lookup_paths: list[str] = []
    if args.lookup_logs:
        for pattern in args.lookup_logs:
            expanded = glob.glob(pattern)
            lookup_paths.extend(expanded if expanded else [pattern])
    else:
        for pattern in DEFAULT_SIM_GLOBS:
            lookup_paths.extend(glob.glob(pattern))

    # When using --sim-log-input, always include it in the history context
    if args.sim_log_input and args.sim_log_input not in lookup_paths:
        lookup_paths.append(args.sim_log_input)

    print(f"Loading {len(lookup_paths)} sim log(s) for context...")
    all_records = _load_records(lookup_paths)
    known_exprs: set[str] = {record_workshop_key(r) for r in all_records if r.get("expression", "")}
    print(f"  {len(all_records)} records loaded ({len(known_exprs)} unique)")

    # ── Build the candidate pool ──────────────────────────────────────────
    not_found: list[str] = []

    if args.sim_log_input:
        # Scan sim log directly — route all done records
        input_records = _dedup_records(_load_records([args.sim_log_input]))
        print(f"\nRouting {len(input_records)} records from {args.sim_log_input}:")
        negate_q, repair_q, enhance_q = _print_routing(input_records)
        found_records = negate_q + repair_q + enhance_q
    else:
        # Look up explicit expressions in the loaded sim logs
        if args.expr:
            expressions = args.expr
        else:
            lines = Path(args.expr_file).read_text().splitlines()
            expressions = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        print(f"\nLooking up {len(expressions)} expression(s):")

        found_records = []
        for expr in expressions:
            records = _find_records(expr, all_records)
            if not records:
                print(f"  NOT FOUND  {expr[:80]}")
                not_found.append(expr)
                continue
            for record in records:
                action = route_record(record)
                sharpe = record.get("is_sharpe", 0.0) or 0.0
                fitness = record.get("is_fitness", 0.0) or 0.0
                regime = record_regime(record)
                tag = {
                    "negate":  f"NEGATE   regime={regime}  sharpe={sharpe:.3f}",
                    "repair":  f"REPAIR   regime={regime}  sharpe={sharpe:.3f}  failing={failing_checks(record)}",
                    "enhance": f"ENHANCE  regime={regime}  sharpe={sharpe:.3f}  fitness={fitness:.3f}",
                }.get(action or "", f"SKIP     regime={regime}  sharpe={sharpe:.3f}")
                print(f"  {tag}  {expr[:65]}")
                if action:
                    found_records.append(record)

        negate_q = [r for r in found_records if route_record(r) == "negate"]
        repair_q  = [r for r in found_records if route_record(r) == "repair"]
        enhance_q = [r for r in found_records if route_record(r) == "enhance"]

    print(
        f"\nRouting: {len(negate_q)} negate | {len(repair_q)} repair | "
        f"{len(enhance_q)} enhance | {len(not_found)} not found"
    )

    if args.dry_run:
        from wq_alpha.workshop.repair import _sign_flip
        if negate_q:
            print("\n[DRY RUN] Negations:")
            for r in negate_q:
                orig = r.get("expression", "")
                print(f"  {orig[:55]}  →  {_sign_flip(orig)[:55]}")
        print(
            "\n[DRY RUN] Would run 3 sequential stages: negate → repair → enhance."
            if args.submit else "\n[DRY RUN] No submission."
        )
        return

    if not args.submit:
        # Write plan files only (manual mode — no Brain session needed)
        candidates = generate_candidates(
            found_records, all_records,
            api_key=args.api_key,
            known_exprs=known_exprs,
            verbose=True,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        lane_buckets: dict[str, list[dict]] = {"negate": [], "repair": [], "enhance": []}
        for c in candidates:
            lane = c.get("search_lane", "")
            if lane in lane_buckets:
                lane_buckets[lane].append(c)
        for lane, items in lane_buckets.items():
            if not items:
                continue
            p = out_dir / f"workshop_{lane}.jsonl"
            with open(p, "w") as f:
                for c in items:
                    f.write(json.dumps(c) + "\n")
            print(f"\nWritten {len(items)} {lane} candidates → {p}")
            print(f"  Submit: python -m alpha_search.submit --plan {p} --sim-log {sim_log}")
        print(f"\nDone. {sum(len(v) for v in lane_buckets.values())} total candidates.")
        _write_leaderboard(all_records, sim_log, out_dir)
        return

    # ── Full auto mode: three sequential stages ───────────────────────────
    from wq_alpha.brain.api import sign_in
    from wq_alpha.brain.submit import DEFAULT_SETTINGS
    from wq_alpha.workshop.history import WorkshopHistory

    session = sign_in(args.credentials)
    settings = dict(DEFAULT_SETTINGS)

    portfolio_alpha_data: dict | None = None
    if not args.skip_portfolio_pnl:
        from alpha_grammar.run_pipeline import load_portfolio_alpha_data
        portfolio_alpha_data = load_portfolio_alpha_data(
            session, competition_id=args.competition_id,
        )

    # Simulate not-found expressions before staging
    if not_found:
        print(f"\nSimulating {len(not_found)} not-found expression(s)...")
        new_sim = _simulate_expressions(
            not_found,
            session,
            sim_log,
            settings,
            args.credentials,
            yearly_stats_attempts=args.yearly_stats_attempts,
            yearly_stats_sleep=args.yearly_stats_sleep,
        )
        all_records.extend(new_sim)
        known_exprs.update(record_workshop_key(r) for r in new_sim if r.get("expression", ""))
        for r in new_sim:
            action = route_record(r)
            if action == "negate":
                negate_q.append(r)
            elif action == "repair":
                repair_q.append(r)
            elif action == "enhance":
                enhance_q.append(r)
            if action:
                found_records.append(r)

    if not found_records:
        print("Nothing to workshop.")
        return

    history = WorkshopHistory([record_workshop_key(r) for r in found_records])

    stage_kwargs = dict(
        all_records=all_records,
        history=history,
        known_exprs=known_exprs,
        session=session,
        sim_log=sim_log,
        settings=settings,
        credentials=args.credentials,
        batch_size=args.batch_size,
        api_key=args.api_key,
        portfolio_alpha_data=portfolio_alpha_data,
        competition_id=args.competition_id,
        pnl_sleep=args.pnl_sleep,
        yearly_stats_attempts=args.yearly_stats_attempts,
        yearly_stats_sleep=args.yearly_stats_sleep,
    )

    # Stage 1: Negate
    stage1_results = _run_stage(
        "STAGE 1 — NEGATE", negate_q, skip_negate=False, **stage_kwargs,
    )
    history.update(stage1_results, iteration=0)

    # Stage 2: Repair — original repair seeds + negate results that upgraded
    repair_pool = repair_q + [r for r in stage1_results if route_record(r) == "repair"]
    stage2_results = _run_stage(
        "STAGE 2 — REPAIR", repair_pool, skip_negate=True, **stage_kwargs,
    )
    history.update(stage2_results, iteration=1)

    # Stage 3: Enhance — original enhance seeds + repair results that upgraded
    enhance_pool = enhance_q + [r for r in stage1_results + stage2_results if route_record(r) == "enhance"]
    stage3_results = _run_stage(
        "STAGE 3 — ENHANCE", enhance_pool, skip_negate=True, **stage_kwargs,
    )
    history.update(stage3_results, iteration=2)

    # Summary
    all_workshop_records = _load_records([str(sim_log)])
    n_pass = sum(
        1 for r in all_workshop_records
        if (r.get("is_sharpe") or 0) >= 1.25 and (r.get("is_fitness") or 0) >= 1.0
    )
    total_submitted = len(stage1_results) + len(stage2_results) + len(stage3_results)
    print(f"\n{'='*55}")
    print(f"Workshop complete. Sim log: {sim_log}")
    print(
        f"Submitted this run: {total_submitted} "
        f"({len(stage1_results)} negate | {len(stage2_results)} repair | {len(stage3_results)} enhance)"
    )
    print(f"Total in log: {len(all_workshop_records)} | Passing (sharpe≥1.25, fitness≥1.0): {n_pass}")
    if sim_log.exists():
        from wq_alpha.logs import MASTER, merge_session_to_master

        n_new, n_updated = merge_session_to_master(sim_log)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    _write_leaderboard(all_records, sim_log, out_dir)


if __name__ == "__main__":
    main()
