"""Workshop specific alphas by Brain alpha ID.

Fetches stats for each ID from Brain, determines route (enhance if all checks
pass, repair otherwise), and runs the iterative LLM workshop.

No qualification bar — user-specified alphas bypass the fitness/sharpe/pool
gates used in batch_workshop. The intent is to workshop any alpha you choose.

Usage:
    python tools/workshop_by_id.py \\
        --alpha-ids abc123 def456 ghi789 \\
        --credentials brain.txt

    # Cap rounds per alpha (default 2)
    python tools/workshop_by_id.py --alpha-ids abc123 --rounds 3

    # Skip portfolio PnL loading (no corr feedback)
    python tools/workshop_by_id.py --alpha-ids abc123 --skip-portfolio-pnl

    # Dry-run: fetch stats and show route, don't call LLM or submit
    python tools/workshop_by_id.py --alpha-ids abc123 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_grammar.run_pipeline import (
    get_alpha_detail,
    build_sim_record,
    load_portfolio_alpha_data,
    compute_max_self_corr_info,
)
from wq_alpha.brain.api import fetch_pnl_series_with_retry, fetch_yearly_stats_with_retry, sign_in
from wq_alpha.logs import MASTER, merge_session_to_master
from wq_alpha.portfolio.admission import fetch_brain_performance_comparison, score_record_admission
from wq_alpha.workshop.core import (
    is_passing,
    failing_checks,
    record_regime,
    record_workshop_key,
)
from wq_alpha.workshop.batch import (
    _workshop_alpha_iterative,
    _write_per_alpha_results,
    WORKSHOP_DIR,
)

DEFAULT_SIM_LOG = ROOT / "artifacts" / "workshop" / "by_id_sim.jsonl"


# ---------------------------------------------------------------------------
# Record construction from Brain alpha ID
# ---------------------------------------------------------------------------

def _fetch_record(
    session,
    alpha_id: str,
    portfolio_alpha_data: dict | None,
    *,
    competition_id: str = "IQC2026S1",
) -> dict | None:
    """Fetch Brain detail for alpha_id and return a workshop-compatible sim record."""
    print(f"  [{alpha_id}] fetching detail...")
    detail = get_alpha_detail(session, alpha_id)
    if not detail or not detail.get("id"):
        print(f"  [{alpha_id}] not found or inaccessible — skipping")
        return None

    regular = detail.get("regular") or {}
    expression = regular.get("code") or detail.get("expression") or ""
    if not expression:
        print(f"  [{alpha_id}] no expression in detail — skipping")
        return None

    settings = dict(detail.get("settings") or {})
    yearly = fetch_yearly_stats_with_retry(session, alpha_id, max_attempts=6, sleep_seconds=20.0)

    # Minimal candidate dict — only fields needed for workshop routing/prompts
    candidate: dict = {
        "expression": expression,
        "alpha_id": alpha_id,
        "settings": settings,
        "dataset_id": detail.get("dataset") or "",
        "fields": [],
        "cluster_ids": [],
        "params": {},
    }

    record = build_sim_record(candidate, settings, None, {}, alpha_id, "done", detail, yearly)
    performance = fetch_brain_performance_comparison(session, alpha_id, competition_id)
    if performance:
        admission = score_record_admission(record, {}, {}, brain_performance=performance)
        record["admission"] = admission
        for key, value in admission.items():
            record[f"admission_{key}"] = value

    # Inline self-correlation + admission scoring when portfolio is loaded
    if portfolio_alpha_data:
        print(f"  [{alpha_id}] fetching PnL for corr...")
        pnl = fetch_pnl_series_with_retry(session, alpha_id, max_attempts=8, sleep_seconds=1.0)
        if pnl:
            corr_info = compute_max_self_corr_info(pnl, portfolio_alpha_data)
            if corr_info.get("max_self_correlation") is not None:
                record.update(corr_info)
            admission = score_record_admission(
                record,
                pnl,
                portfolio_alpha_data,
                brain_performance=performance,
            )
            record["admission"] = admission
            for key, value in admission.items():
                record[f"admission_{key}"] = value
            score_delta = admission.get("brain_score_delta")
            fallback_delta = admission.get("delta_pool_sharpe")
            corr  = record.get("max_self_correlation")
            score_s = f"{score_delta:+.0f}" if score_delta is not None else "N/A"
            fallback_s = f"{fallback_delta:+.3f}" if fallback_delta is not None else "N/A"
            corr_s  = f"{corr:.3f}"   if corr  is not None else "N/A"
            print(
                f"  [{alpha_id}] corr={corr_s}  brain_score_delta={score_s}  "
                f"fallback_delta={fallback_s}  signal={admission.get('signal') or 'N/A'}  "
                f"{admission.get('decision')} {admission.get('reason')}"
            )
        else:
            print(f"  [{alpha_id}] PnL unavailable — corr skipped")

    return record


# ---------------------------------------------------------------------------
# Routing: no qualification gates, pure pass/fail on checks
# ---------------------------------------------------------------------------

def _route_by_id(record: dict) -> str:
    """Route without any fitness/sharpe/score gates.

    enhance — all submission checks pass (SELF_CORRELATION excluded)
    repair  — any check fails (regardless of whether it's in REPAIRABLE_CHECKS)
    """
    if is_passing(record):
        return "enhance"
    return "repair"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--alpha-ids", nargs="+", required=True, metavar="ID",
        help="One or more Brain alpha IDs to workshop.",
    )
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (default: openaiapi.txt)")
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="LLM rounds per alpha (default 2).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=3,
        help="Simulation batch size (default 3).",
    )
    parser.add_argument(
        "--pnl-sleep", type=float, default=1.0,
        help="Sleep between PnL fetch retries (default 1.0s).",
    )
    parser.add_argument(
        "--admission-pnl-attempts", type=int, default=3,
        help="Max PnL fetch attempts for admission scoring in LLM context (default 3).",
    )
    parser.add_argument(
        "--out-sim-log", default=str(DEFAULT_SIM_LOG),
        help=f"Output sim log JSONL (default: {DEFAULT_SIM_LOG}).",
    )
    parser.add_argument(
        "--competition-id", default="IQC2026S1",
        help="Competition ID for portfolio PnL loading (default: IQC2026S1).",
    )
    parser.add_argument(
        "--skip-portfolio-pnl", action="store_true",
        help="Skip loading portfolio PnL. Disables corr/delta feedback in LLM context.",
    )
    parser.add_argument(
        "--bypass-correlation-check", action="store_true",
        help="Bypass self-correlation gate in LLM candidate generation.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch stats and show route for each ID, then exit without calling LLM.",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or ""
    if not api_key:
        key_file = ROOT / "openaiapi.txt"
        if key_file.exists():
            api_key = key_file.read_text().strip()

    out_sim_log = Path(args.out_sim_log)
    out_sim_log.parent.mkdir(parents=True, exist_ok=True)

    # Sign in
    session = sign_in(args.credentials)

    # Load portfolio data
    portfolio_alpha_data: dict | None = None
    if not args.skip_portfolio_pnl:
        print("Loading submitted portfolio PnL...")
        portfolio_alpha_data = load_portfolio_alpha_data(
            session, competition_id=args.competition_id,
        )

    # Fetch records for all requested IDs
    print(f"\nFetching stats for {len(args.alpha_ids)} alpha(s)...")
    records: list[dict] = []
    for alpha_id in args.alpha_ids:
        rec = _fetch_record(
            session,
            alpha_id,
            portfolio_alpha_data,
            competition_id=args.competition_id,
        )
        if rec:
            records.append(rec)

    if not records:
        print("No valid records fetched — nothing to do.")
        return

    # Route and display
    print(f"\n{'─'*70}")
    print(f"{'ID':<14} {'route':<8} {'sharpe':>7} {'fitness':>7} {'dScore':>8} {'failing checks'}")
    print(f"{'─'*70}")
    routed: list[tuple[dict, str]] = []
    for rec in records:
        route = _route_by_id(rec)
        alpha_id = rec.get("alpha_id") or "?"
        sharpe  = rec.get("is_sharpe")
        fitness = rec.get("is_fitness")
        score_delta = rec.get("admission_brain_score_delta")
        failing = ", ".join(failing_checks(rec)) or "—"
        sharpe_s  = f"{sharpe:.3f}"  if sharpe  is not None else " N/A"
        fitness_s = f"{fitness:.3f}" if fitness is not None else " N/A"
        score_s = f"{score_delta:+.0f}" if score_delta is not None else "N/A"
        print(f"{alpha_id:<14} {route:<8} {sharpe_s:>7} {fitness_s:>7} {score_s:>8}  {failing}")
        routed.append((rec, route))
    print()

    if args.dry_run:
        print("Dry-run — stopping before LLM calls.")
        return

    if not api_key:
        print("No OpenAI API key found — cannot run LLM workshop.")
        return

    # Build shared state
    from wq_alpha.agents.repair import AlphaRepairAgent
    agent = AlphaRepairAgent(api_key=api_key)

    # Load all existing sim records for passing-examples context
    from wq_alpha.workshop.batch import _load_records, _resolve_input_paths
    try:
        ctx_paths = list((ROOT / "artifacts" / "sim").glob("*.jsonl"))
        all_records = _load_records(ctx_paths) if ctx_paths else []
    except Exception:
        all_records = []

    known_exprs: set[str] = {record_workshop_key(r) for r in all_records if r.get("expression")}

    # Namespace object expected by _workshop_alpha_iterative
    import types
    fake_args = types.SimpleNamespace(
        bypass_correlation_check=args.bypass_correlation_check,
        pnl_sleep=args.pnl_sleep,
        admission_pnl_attempts=args.admission_pnl_attempts,
        batch_size=args.batch_size,
        credentials=args.credentials,
        api_key=api_key,
    )

    settings: dict = {}  # will be overridden per alpha from record["settings"]
    run_records: list[dict] = []

    for rec, route in routed:
        alpha_id = rec.get("alpha_id") or "?"
        alpha_settings = dict(rec.get("settings") or {})
        print(f"\n{'='*60}")
        print(f"[{route}] {alpha_id}  {rec.get('expression','')[:70]}")
        print(f"  sharpe={rec.get('is_sharpe')}  fitness={rec.get('is_fitness')}  "
              f"regime={record_regime(rec)}")

        results = _workshop_alpha_iterative(
            rec, route, agent, session, out_sim_log,
            alpha_settings, fake_args,
            n_rounds=args.rounds,
            known_exprs=known_exprs,
            all_records=all_records,
            portfolio_alpha_data=portfolio_alpha_data,
        )
        run_records.extend(results)
        all_records.extend(results)
        known_exprs.update(record_workshop_key(r) for r in results if r.get("expression"))

    # Summary
    per_alpha_path = WORKSHOP_DIR / "by_id_per_alpha_results.txt"
    _write_per_alpha_results(run_records, per_alpha_path)
    n_pass = sum(1 for r in run_records if is_passing(r))
    print(f"\nDone. {len(run_records)} variants simulated, {n_pass} passing.")
    print(f"Sim log: {out_sim_log}")
    if out_sim_log.exists():
        n_new, n_updated = merge_session_to_master(out_sim_log)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    print(f"Per-alpha summary: {per_alpha_path}")


if __name__ == "__main__":
    main()
