"""Targeted workshop runner for specific alphas.

This is the small, explicit-alpha path for cases where a user wants to bypass
batch autoworkshop source selection and send chosen seeds directly to repair or
enhance. Generation, submission, inline feedback, admission, and reporting are
delegated to :mod:`alpha_enhancement.workshop.batch`.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logs import MASTER, merge_session_to_master
from alpha_enhancement.workshop.batch import (
    WORKSHOP_DIR,
    _load_records,
    _promote_to_templates,
    _workshop_alpha_iterative,
    _write_per_alpha_results,
)
from alpha_enhancement.workshop.core import (
    _looks_like_fastexpr,
    collect_trial_history,
    failing_checks,
    find_passing_examples,
    is_passing,
    record_regime,
    record_workshop_key,
    workshop_expr_key,
)
from alpha_enhancement.workshop.history import WorkshopHistory

DEFAULT_SIM_LOG = WORKSHOP_DIR / "workshop_sim.jsonl"

_DEFAULT_LOOKUP_GLOBS = [
    "artifacts/sim/master.jsonl",
    "artifacts/sim/sessions/*.jsonl",
    "artifacts/workshop/workshop_sim.jsonl",
    "alpha_discovery/grammar_method/artifacts/all_sim.jsonl",
    "alpha_discovery/grammar_method/artifacts/sessions/*.jsonl",
    "alpha_multi/artifacts/*.jsonl",
]


def _expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(str(ROOT / pattern)) if not Path(pattern).is_absolute() else glob.glob(pattern)
        if not matches:
            matches = [str(ROOT / pattern)] if not Path(pattern).is_absolute() else [pattern]
        for match in matches:
            path = str(Path(match))
            key = str(Path(path).resolve()) if Path(path).exists() else path
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def _default_lookup_paths() -> list[str]:
    return [p for p in _expand_paths(_DEFAULT_LOOKUP_GLOBS) if Path(p).exists()]


def _parse_stat(raw: str | None, idx: int) -> tuple[float | None, float | None]:
    if not raw:
        return None, None
    parts = raw.split(",")
    try:
        sharpe = float(parts[0]) if parts and parts[0] else None
        fitness = float(parts[1]) if len(parts) > 1 and parts[1] else None
        return sharpe, fitness
    except ValueError as exc:
        raise SystemExit(
            f"--seed-stats[{idx}]: expected 'sharpe,fitness[,turnover%]', got '{raw}'"
        ) from exc


def _format_float(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _fetch_seed_from_api(session, alpha_id: str) -> dict | None:
    """Fetch one Brain alpha and convert it to a sim-record-like dict."""
    from core.selfcorr import build_sim_record, get_alpha_detail
    from core.brain.api import fetch_yearly_stats_with_retry

    print(f"  [api] {alpha_id}: fetching detail...")
    detail = get_alpha_detail(session, alpha_id)
    if not detail or not detail.get("id"):
        print(f"  [api] {alpha_id}: not found or inaccessible")
        return None

    regular = detail.get("regular")
    if isinstance(regular, dict):
        expression = regular.get("code") or ""
    elif isinstance(regular, str):
        expression = regular
    else:
        expression = detail.get("expression") or ""
    if not expression:
        print(f"  [api] {alpha_id}: no expression in detail")
        return None

    settings = dict(detail.get("settings") or {})
    candidate = {
        "alpha_id": alpha_id,
        "expression": expression,
        "settings": settings,
        "dataset_id": detail.get("dataset") or "",
        "fields": [],
        "cluster_ids": [],
        "params": {},
    }
    yearly = fetch_yearly_stats_with_retry(session, alpha_id, max_attempts=6, sleep_seconds=20.0)
    record = build_sim_record(candidate, settings, None, {}, alpha_id, "done", detail, yearly)
    record["alpha_id"] = alpha_id
    record.setdefault("params", {})
    return record


def _load_seed_records(
    ids: list[str],
    exprs: list[str],
    stats_list: list[str | None],
    all_records: list[dict],
    *,
    credentials: str,
) -> list[dict]:
    seeds: list[dict] = []
    missing_ids: list[str] = []
    missing_exprs: list[str] = []

    for alpha_id in ids or []:
        matches = [
            r for r in all_records
            if r.get("alpha_id") == alpha_id and r.get("status") == "done"
        ]
        if not matches:
            missing_ids.append(alpha_id)
            continue
        record = max(matches, key=lambda r: r.get("is_sharpe") or 0.0)
        print(
            f"  FOUND id={alpha_id}  sharpe={_format_float(record.get('is_sharpe'))}  "
            f"fitness={_format_float(record.get('is_fitness'))}  "
            f"{record.get('expression', '')[:72]}"
        )
        seeds.append(dict(record))

    for idx, expr in enumerate(exprs or []):
        raw_stat = stats_list[idx] if stats_list and idx < len(stats_list) else None
        matches = [
            r for r in all_records
            if r.get("expression", "").strip() == expr.strip()
            and r.get("status") == "done"
        ]
        if matches:
            record = max(matches, key=lambda r: r.get("is_sharpe") or 0.0)
            print(
                f"  FOUND expr  sharpe={_format_float(record.get('is_sharpe'))}  "
                f"fitness={_format_float(record.get('is_fitness'))}  {expr[:72]}"
            )
            seeds.append(dict(record))
            continue
        if raw_stat:
            sharpe, fitness = _parse_stat(raw_stat, idx)
            record = {
                "expression": expr,
                "status": "done",
                "is_sharpe": sharpe,
                "is_fitness": fitness,
                "dataset_id": "",
                "fields": [],
                "cluster_ids": [],
                "params": {},
                "_synthetic": True,
            }
            print(
                f"  SYNTHETIC  sharpe={_format_float(sharpe)}  "
                f"fitness={_format_float(fitness)}  {expr[:72]}"
            )
            seeds.append(record)
            continue
        missing_exprs.append(expr)

    if missing_ids:
        from core.brain.api import sign_in

        session = sign_in(credentials)
        for alpha_id in missing_ids:
            record = _fetch_seed_from_api(session, alpha_id)
            if record:
                print(
                    f"  FOUND api id={alpha_id}  sharpe={_format_float(record.get('is_sharpe'))}  "
                    f"fitness={_format_float(record.get('is_fitness'))}  "
                    f"{record.get('expression', '')[:72]}"
                )
                seeds.append(record)
            else:
                missing_exprs.append(f"alpha_id={alpha_id}")

    if missing_exprs:
        raise SystemExit(
            "Seeds not found. For expressions missing from logs, add --seed-stats "
            "'sharpe,fitness[,turnover%]'.\n  "
            + "\n  ".join(missing_exprs)
        )

    return seeds


def _make_context_records(exprs: list[str], stats_list: list[str | None]) -> list[dict]:
    records: list[dict] = []
    for idx, expr in enumerate(exprs or []):
        raw_stat = stats_list[idx] if stats_list and idx < len(stats_list) else None
        sharpe, fitness = _parse_stat(raw_stat, idx) if raw_stat else (None, None)
        records.append({
            "expression": expr,
            "status": "done",
            "is_sharpe": sharpe,
            "is_fitness": fitness,
            "dataset_id": "",
            "fields": [],
            "cluster_ids": [],
        })
    return records


def _lane_for_record(record: dict, requested: str) -> str:
    if requested in {"repair", "enhance"}:
        return requested
    return "repair" if failing_checks(record) else "enhance"


def _with_targeted_context(record: dict, context_records: list[dict]) -> dict:
    enriched = dict(record)
    params = dict(enriched.get("params") or {})
    params.setdefault("root_expression", enriched.get("expression", ""))
    params.setdefault("original_expression", enriched.get("expression", ""))
    enriched["params"] = params
    if context_records:
        enriched["context_records"] = list(enriched.get("context_records") or []) + context_records
    return enriched


def _resolve_api_key(value: str | None) -> str:
    api_key = value or os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        return api_key.strip()
    key_file = ROOT / "openaiapi.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    return ""


def _print_seed_table(routed: list[tuple[dict, str]]) -> None:
    print(f"\n{'ID':<14} {'lane':<8} {'regime':<26} {'sharpe':>7} {'fitness':>7}  failing")
    print("-" * 88)
    for record, lane in routed:
        alpha_id = record.get("alpha_id") or "-"
        failing = ",".join(failing_checks(record)) or "-"
        print(
            f"{alpha_id:<14} {lane:<8} {record_regime(record):<26} "
            f"{_format_float(record.get('is_sharpe')):>7} "
            f"{_format_float(record.get('is_fitness')):>7}  {failing}"
        )


def _dry_run_seed_generation(
    record: dict,
    lane: str,
    agent,
    *,
    known_exprs: set[str],
    all_records: list[dict],
    history: WorkshopHistory,
) -> int:
    original = record.get("expression", "")
    regime = record_regime(record)
    orig_from_params = (record.get("params") or {}).get("original_expression") or original
    root = history.root_for(
        workshop_expr_key(original, regime),
        workshop_expr_key(orig_from_params, regime),
    )
    history_ctx = (
        history.context_for_repair(root) if lane == "repair"
        else history.context_for_enhance(root)
    )
    log_trials = collect_trial_history(record, all_records) if lane == "repair" else []
    combined_trials = history_ctx + [
        trial for trial in log_trials
        if trial.get("expression") not in {h.get("expression") for h in history_ctx}
    ]
    context_records = list(record.get("context_records") or [])
    passing_examples = find_passing_examples(all_records, record)

    print(f"\n{'=' * 60}")
    print(f"[{lane} dry-run] {record.get('alpha_id') or '-'}  {original[:80]}")
    print(f"{'=' * 60}")
    if lane == "repair":
        result = agent.repair(
            record,
            passing_examples=passing_examples,
            trial_history=combined_trials or None,
            context_records=context_records or None,
        )
    else:
        result = agent.enhance(
            record,
            passing_examples=passing_examples,
            context_records=(history_ctx + context_records) or None,
        )

    accepted = 0
    skipped: list[tuple[str, str]] = []
    seen_this = {original}
    for expr in result.repair_expressions[:6]:
        expr_key = workshop_expr_key(expr, regime)
        if not expr:
            skipped.append(("empty", expr))
            continue
        if expr in seen_this:
            skipped.append(("duplicate in response", expr))
            continue
        if expr_key in known_exprs:
            skipped.append(("already known/simulated", expr))
            continue
        if not _looks_like_fastexpr(expr):
            skipped.append(("not FASTEXPR-looking", expr))
            continue
        seen_this.add(expr)
        accepted += 1
        print(f"  [{lane}] would submit → {expr}")

    if not accepted:
        raw = (result.raw_response or "").strip()
        if not raw:
            print(f"  [{lane}] no visible LLM output after retry.")
        elif not result.repair_expressions:
            print(f"  [{lane}] LLM output did not parse into FASTEXPR lines.")
        if skipped:
            for reason, expr in skipped[:6]:
                print(f"    skip {reason}: {expr[:100]}")
        if raw:
            print(f"  raw preview: {raw[:500]}")
    return accepted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    seed_group = parser.add_argument_group("Seeds")
    seed_group.add_argument("--seed-id", nargs="+", default=[], metavar="ALPHA_ID")
    seed_group.add_argument("--seed-expr", nargs="+", default=[], metavar="EXPR")
    seed_group.add_argument(
        "--seed-stats", nargs="*", default=[], metavar="SHARPE,FITNESS[,TURN%]",
        help="Stats for seed expressions not found in logs.",
    )

    context_group = parser.add_argument_group("Context-only signals")
    context_group.add_argument("--context-expr", nargs="*", default=[], metavar="EXPR")
    context_group.add_argument(
        "--context-stats", nargs="*", default=[], metavar="SHARPE,FITNESS",
    )

    parser.add_argument(
        "--lane", choices=["enhance", "repair", "auto"], default="enhance",
        help="Force all seeds into a lane, or use pass/fail auto routing (default: enhance).",
    )
    parser.add_argument(
        "--rounds", "--iterations", dest="rounds", type=int, default=3,
        help="LLM rounds per seed (default: 3).",
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Rows per seed in per-alpha report and promoted-seed output (default: 5).",
    )
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--llm-max-tokens", type=int, default=4096,
        help="Visible+reasoning token budget for each LLM response (default: 4096).",
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument(
        "--sim-log", "--out-sim-log", dest="sim_log", default=str(DEFAULT_SIM_LOG),
        help=f"Output sim log JSONL (default: {DEFAULT_SIM_LOG}).",
    )
    parser.add_argument(
        "--lookup-logs", nargs="*", default=None,
        help="Extra sim logs/globs to search for seeds.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--competition-id", default="IQC2026S1",
        help="Competition ID for Brain performance comparison and portfolio PnL.",
    )
    parser.add_argument(
        "--skip-portfolio-pnl", action="store_true",
        help="Skip submitted-portfolio PnL loading. Final admission fallback will be unavailable.",
    )
    parser.add_argument(
        "--skip-admission-gate", action="store_true",
        help="Promote standalone-passing variants without admission scoring.",
    )
    parser.add_argument(
        "--admission-min-brain-score-delta", type=float, default=0.0,
        help="Minimum official Brain score lift for final promotion (default 0).",
    )
    parser.add_argument(
        "--admission-min-fallback-delta-sharpe", type=float, default=0.0,
        help="Minimum local fallback pool Sharpe delta when Brain score is unavailable (default 0).",
    )
    parser.add_argument(
        "--admission-pnl-attempts", type=int, default=3,
        help="Max PnL fetch attempts during final admission scoring (default 3).",
    )
    parser.add_argument(
        "--pnl-sleep", type=float, default=1.0,
        help="Sleep between Brain score/PnL fetch retries (default 1.0).",
    )
    parser.add_argument(
        "--bypass-correlation-check", action="store_true",
        help="Accepted for CLI compatibility; targeted runs already bypass source routing.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="Deprecated compatibility option; targeted workshop submission uses shared batch helper.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.seed_id and not args.seed_expr:
        parser.error("Specify at least one of --seed-id or --seed-expr.")

    lookup_paths = _expand_paths(args.lookup_logs) if args.lookup_logs else _default_lookup_paths()
    print(f"Loading sim logs ({len(lookup_paths)} files)...")
    all_records = _load_records(lookup_paths)
    print(f"  {len(all_records)} records loaded")

    print("\nSeeds:")
    seed_records = _load_seed_records(
        args.seed_id,
        args.seed_expr,
        args.seed_stats,
        all_records,
        credentials=args.credentials,
    )
    context_records = _make_context_records(args.context_expr, args.context_stats)
    if context_records:
        print(f"\nContext-only signals: {len(context_records)}")

    routed = [
        (_with_targeted_context(seed, context_records), _lane_for_record(seed, args.lane))
        for seed in seed_records
    ]
    _print_seed_table(routed)

    api_key = _resolve_api_key(args.api_key)
    if not api_key:
        raise SystemExit("No OPENAI_API_KEY found. Set env var, pass --api-key, or place it in openaiapi.txt.")

    from alpha_enhancement.agents.repair import AlphaRepairAgent

    agent = AlphaRepairAgent(api_key=api_key, max_tokens=args.llm_max_tokens)
    history = WorkshopHistory([record_workshop_key(seed) for seed, _ in routed])
    known_exprs: set[str] = {
        record_workshop_key(record)
        for record in all_records + [seed for seed, _ in routed]
        if record.get("expression")
    }
    all_records = list(all_records) + [seed for seed, _ in routed]

    out_sim_log = Path(args.sim_log)
    if args.dry_run:
        print("\nDry-run: generating first-round LLM candidates only; no Brain submission.")
        total = 0
        for seed, lane in routed:
            total += _dry_run_seed_generation(
                seed,
                lane,
                agent,
                known_exprs=known_exprs,
                all_records=all_records,
                history=history,
            )
        print(f"\nDry-run complete: {total} new candidate(s) would be submitted.")
        print(f"Full run would use {args.rounds} round(s) per seed into {out_sim_log}")
        return

    from core.selfcorr import load_portfolio_alpha_data
    from core.brain.api import sign_in
    from core.brain.submit import DEFAULT_SETTINGS

    session = sign_in(args.credentials)
    portfolio_alpha_data: dict | None = None
    if not args.skip_portfolio_pnl:
        print(f"\nLoading submitted portfolio PnL for {args.competition_id}...")
        portfolio_alpha_data = load_portfolio_alpha_data(
            session,
            competition_id=args.competition_id,
        )
        print(f"  {len(portfolio_alpha_data)} portfolio alphas loaded")

    settings = dict(DEFAULT_SETTINGS)

    workshop_args = SimpleNamespace(
        credentials=args.credentials,
        batch_size=args.batch_size,
        pnl_sleep=args.pnl_sleep,
        admission_pnl_attempts=args.admission_pnl_attempts,
        competition_id=args.competition_id,
    )

    run_records: list[dict] = []
    for seed, lane in routed:
        print(f"\n{'=' * 60}")
        print(f"[{lane}] {seed.get('alpha_id') or '-'}  {seed.get('expression', '')[:80]}")
        print(f"{'=' * 60}")
        results = _workshop_alpha_iterative(
            seed,
            lane,
            agent,
            session,
            out_sim_log,
            settings,
            workshop_args,
            n_rounds=max(1, args.rounds),
            known_exprs=known_exprs,
            all_records=all_records,
            history=history,
            portfolio_alpha_data=portfolio_alpha_data,
        )
        run_records.extend(results)

    per_alpha_path = WORKSHOP_DIR / "focused_enhance_per_alpha_results.txt"
    _write_per_alpha_results(
        run_records,
        per_alpha_path,
        top_n=args.top_n,
        seed_records=[seed for seed, _lane in routed],
    )

    n_pass = sum(
        1 for record in run_records
        if record.get("status") == "done"
        and is_passing(record)
        and (record.get("is_sharpe") or 0.0) >= 1.25
        and (record.get("is_fitness") or 0.0) >= 1.0
    )
    print(f"\nDone. {len(run_records)} variants simulated, {n_pass} standalone passing.")

    if run_records:
        print("\nRunning final admission/template promotion for this targeted run...")
        n_added = _promote_to_templates(
            out_sim_log,
            records=run_records,
            session=session,
            portfolio_alpha_data=portfolio_alpha_data,
            competition_id=args.competition_id,
            admission_gate=not args.skip_admission_gate,
            min_brain_score_delta=args.admission_min_brain_score_delta,
            min_fallback_delta_sharpe=args.admission_min_fallback_delta_sharpe,
            pnl_sleep=args.pnl_sleep,
            admission_pnl_attempts=args.admission_pnl_attempts,
            per_seed_top_n=args.top_n,
        )
        print(f"Templates promoted: {n_added}")

    print(f"Sim log: {out_sim_log}")
    if out_sim_log.exists():
        n_new, n_updated = merge_session_to_master(out_sim_log)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    print(f"Per-alpha summary: {per_alpha_path}")


if __name__ == "__main__":
    main()
