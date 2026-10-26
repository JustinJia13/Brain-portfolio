"""Post-grammar semantic LLM workshop runner.

Workflow:
  seed -> diagnostic agent -> generator round 1 -> simulate -> robustness card
       -> diagnostic update -> generator round 2 -> simulate -> robustness card
       -> diagnostic update -> generator round 3 -> simulate -> final summary
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wq_alpha.agents.semantic_workshop import (
    SemanticDiagnosticAgent,
    SemanticGeneratorAgent,
    GeneratedVariant,
)
from wq_alpha.brain.submit import DEFAULT_SETTINGS, _regime_key
from wq_alpha.logs import MASTER, merge_session_to_master
from wq_alpha.semantic_templates.core import (
    DEFAULT_PLAN_JSONL as DEFAULT_TEMPLATE_PLAN_JSONL,
    DEFAULT_PROBE_LOG as DEFAULT_TEMPLATE_PROBE_LOG,
    DEFAULT_ROUTES_JSONL as DEFAULT_TEMPLATE_ROUTES_JSONL,
    DEFAULT_SELECTIONS_JSONL as DEFAULT_TEMPLATE_SELECTIONS_JSONL,
    DEFAULT_TEMPLATES_JSONL as DEFAULT_TEMPLATE_TEMPLATES_JSONL,
)
from wq_alpha.semantic_templates.handoff import (
    create_routed_templates_for_finished_seed,
    handoff_from_args,
)
from wq_alpha.workshop.batch import (
    DEFAULT_EXCLUDED_EXPRS,
    _load_workshopped_sources,
    _save_workshopped_sources,
    select_candidates,
)
from wq_alpha.workshop.core import (
    _looks_like_fastexpr,
    _make_candidate,
    failing_checks,
    find_passing_examples,
    is_passing,
    record_workshop_key,
    route_record,
)
from wq_alpha.workshop.history import WorkshopHistory
from wq_alpha.workshop.probe_cards import build_probe_card, compact_probe_card, load_probe_prior
from wq_alpha.workshop.records import load_records
from wq_alpha.workshop.robustness_card import (
    build_robustness_candidates,
    build_robustness_card,
    final_recommendation,
    format_card_text,
    load_cluster_members,
    record_score_addition,
)
from wq_alpha.workshop.semantic_context import (
    DEFAULT_MAX_CONTEXT_CHARS,
    SemanticContextBuilder,
)
from wq_alpha.workshop.scoring import score_seed_records, with_portfolio_prompt_feedback
from wq_alpha.workshop.simulation import submit_candidate_batch


SEMANTIC_DIR = ROOT / "artifacts" / "semantic_workshop"
DEFAULT_SIM_LOG = SEMANTIC_DIR / "semantic_workshop_sim.jsonl"
DEFAULT_ROBUST_SIM_LOG = SEMANTIC_DIR / "semantic_robustness_sim.jsonl"
DEFAULT_DIAG_JSONL = SEMANTIC_DIR / "diagnostics.jsonl"
DEFAULT_CONTEXT_REPORT_JSONL = SEMANTIC_DIR / "context_reports.jsonl"
DEFAULT_CARD_JSONL = SEMANTIC_DIR / "robustness_cards.jsonl"
DEFAULT_SUMMARY_TXT = SEMANTIC_DIR / "workshop_summary.txt"
DEFAULT_SUMMARY_JSONL = SEMANTIC_DIR / "workshop_summary.jsonl"
DEFAULT_CLUSTER_PATH = ROOT / "alpha_grammar" / "artifacts" / "all_clusters.json"
DEFAULT_PROBE_PRIOR = ROOT / "artifacts" / "probe" / "grammar_prior_from_probe.json"
DEFAULT_SEED_RESIM_LOG = SEMANTIC_DIR / "semantic_seed_resim_sim.jsonl"
DEFAULT_TEMPLATE_EVENTS_JSONL = DEFAULT_TEMPLATE_TEMPLATES_JSONL.with_name("template_events.jsonl")

DEFAULT_INPUT_GLOBS = [
    str(MASTER),
    str(ROOT / "artifacts" / "sim" / "sessions" / "grammar_*.jsonl"),
    str(ROOT / "alpha_grammar" / "artifacts" / "all_sim.jsonl"),
    str(ROOT / "artifacts" / "workshop" / "workshop_sim.jsonl"),
    str(DEFAULT_SEED_RESIM_LOG),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-log", nargs="*", default=[], help="Input sim logs. Defaults to grammar/master history.")
    parser.add_argument("--source", choices=("grammar", "all"), default="grammar")
    parser.add_argument("--seed-id", nargs="*", default=[], help="Specific alpha_id seed(s).")
    parser.add_argument("--seed-expr", nargs="*", default=[], help="Specific expression seed(s).")
    parser.add_argument("--max-seeds", type=int, default=5)
    parser.add_argument("--min-abs-fitness", type=float, default=0.8, help=argparse.SUPPRESS)
    parser.add_argument("--max-abs-sharpe", type=float, default=10.0, help=argparse.SUPPRESS)
    parser.add_argument("--max-abs-fitness", type=float, default=20.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--lane",
        nargs="*",
        choices=("repair", "enhance"),
        default=["repair", "enhance"],
        help="Autoworkshop source lanes allowed into semantic workshop (default: repair enhance).",
    )
    parser.add_argument(
        "--exclude-expr",
        nargs="*",
        default=[],
        help="Expressions to exclude from semantic workshop source selection.",
    )
    parser.set_defaults(score_seeds=True)
    parser.add_argument(
        "--score-seeds",
        dest="score_seeds",
        action="store_true",
        help=(
            "Fetch official Brain score deltas for source seeds before semantic workshop "
            "(default). Repair/enhance seeds continue only with positive fetched score delta."
        ),
    )
    parser.add_argument(
        "--no-score-seeds",
        dest="score_seeds",
        action="store_false",
        help="Disable official seed score gating; keeps only the structural autoworkshop route gate.",
    )
    parser.add_argument(
        "--bypass-correlation-check",
        action="store_true",
        help="Bypass the autoworkshop self-correlation gate during source routing.",
    )
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--settings-json", default="")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--competition-id", default="IQC2026S1")
    parser.add_argument("--skip-portfolio-pnl", action="store_true")
    parser.add_argument("--pnl-sleep", type=float, default=1.0)
    parser.add_argument(
        "--yearly-stats-attempts",
        type=int,
        default=6,
        help="Retry yearly-stats fetch this many times after each completed simulation.",
    )
    parser.add_argument(
        "--yearly-stats-sleep",
        type=float,
        default=20.0,
        help="Seconds to wait between yearly-stats retry attempts.",
    )
    parser.add_argument("--clusters", default=str(DEFAULT_CLUSTER_PATH))
    parser.add_argument("--probe-prior", default=str(DEFAULT_PROBE_PRIOR))
    parser.add_argument(
        "--robust-top-k",
        type=int,
        default=1,
        help="Round-1 max source candidates for light robustness. Round 2 is skipped; Round 3 uses top 1.",
    )
    parser.add_argument("--max-expression-variants", type=int, default=2)
    parser.add_argument("--max-window-variants", type=int, default=0)
    parser.add_argument("--max-field-variants", type=int, default=0)
    parser.add_argument("--max-settings-variants", type=int, default=2)
    parser.add_argument("--final-max-expression-variants", type=int, default=3)
    parser.add_argument("--final-max-window-variants", type=int, default=2)
    parser.add_argument("--final-max-field-variants", type=int, default=1)
    parser.add_argument("--final-max-settings-variants", type=int, default=6)
    parser.set_defaults(semantic_template_auto=True)
    parser.add_argument(
        "--semantic-template-auto",
        dest="semantic_template_auto",
        action="store_true",
        help="After each completed semantic seed, create/route semantic templates into artifacts/semantic_templates (default).",
    )
    parser.add_argument(
        "--no-semantic-template-auto",
        dest="semantic_template_auto",
        action="store_false",
        help="Disable automatic semantic template creation/routing after each completed seed.",
    )
    parser.add_argument("--template-out", default=str(DEFAULT_TEMPLATE_TEMPLATES_JSONL))
    parser.add_argument("--template-selections-out", default=str(DEFAULT_TEMPLATE_SELECTIONS_JSONL))
    parser.add_argument("--template-routes-out", default=str(DEFAULT_TEMPLATE_ROUTES_JSONL))
    parser.add_argument("--template-plan-out", default=str(DEFAULT_TEMPLATE_PLAN_JSONL))
    parser.add_argument("--template-events-out", default=str(DEFAULT_TEMPLATE_EVENTS_JSONL))
    parser.add_argument("--template-probe-log", default=str(DEFAULT_TEMPLATE_PROBE_LOG))
    parser.add_argument("--template-max-active", type=int, default=3)
    parser.add_argument("--template-catalog-limit", type=int, default=160, help="Field selector catalog limit per template. 0 means all clusters.")
    parser.add_argument("--template-selector-chunk-size", type=int, default=40)
    parser.add_argument("--template-route-chunk-size", type=int, default=10)
    parser.add_argument("--template-route-candidate-limit", type=int, default=80)
    parser.add_argument("--template-max-selections-per-chunk", type=int, default=5)
    parser.add_argument("--template-max-windows", type=int, default=2)
    parser.add_argument("--template-max-groups", type=int, default=2)
    parser.add_argument("--template-max-candidates-per-route", type=int, default=3)
    parser.add_argument("--out-sim-log", default=str(DEFAULT_SIM_LOG))
    parser.add_argument("--robust-sim-log", default=str(DEFAULT_ROBUST_SIM_LOG))
    parser.add_argument("--seed-resim-log", default=str(DEFAULT_SEED_RESIM_LOG))
    parser.add_argument("--diagnostics-jsonl", default=str(DEFAULT_DIAG_JSONL))
    parser.add_argument("--context-reports-jsonl", default=str(DEFAULT_CONTEXT_REPORT_JSONL))
    parser.add_argument("--cards-jsonl", default=str(DEFAULT_CARD_JSONL))
    parser.add_argument("--summary-txt", default=str(DEFAULT_SUMMARY_TXT))
    parser.add_argument("--summary-jsonl", default=str(DEFAULT_SUMMARY_JSONL))
    parser.set_defaults(resimulate_missing_score_seeds=True)
    parser.add_argument(
        "--resimulate-missing-score-seeds",
        dest="resimulate_missing_score_seeds",
        action="store_true",
        help="Resimulate repair/enhance seeds on the active Brain account when official score is unavailable.",
    )
    parser.add_argument(
        "--no-resimulate-missing-score-seeds",
        dest="resimulate_missing_score_seeds",
        action="store_false",
        help="Do not resimulate score-unavailable seeds before source gating.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Select seeds and exit without LLM or Brain calls.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help="Optional positive char limit for semantic context packets. Default 0 disables the guard.",
    )
    return parser.parse_args()


def main() -> None:
    _run_semantic_workshop(parse_args())


def _run_semantic_workshop(args: argparse.Namespace) -> None:
    input_paths = _input_paths(args)
    all_records = load_records(input_paths, tag_source=True)
    all_records.extend(load_records([args.out_sim_log, args.robust_sim_log], tag_source=True))
    seeds = _select_seed_records(all_records, args)
    if not seeds:
        raise SystemExit("No seed records selected.")

    settings = _load_settings(args.settings_json)
    out_sim_log = Path(args.out_sim_log)
    robust_sim_log = Path(args.robust_sim_log)
    seed_resim_log = Path(args.seed_resim_log)
    out_sim_log.parent.mkdir(parents=True, exist_ok=True)
    robust_sim_log.parent.mkdir(parents=True, exist_ok=True)
    seed_resim_log.parent.mkdir(parents=True, exist_ok=True)
    probe_index = load_probe_prior(args.probe_prior)
    if probe_index:
        print(f"Loaded probe prior cards: {len(probe_index)} cluster/field entries from {args.probe_prior}")
    else:
        print(f"[warn] No probe prior cards loaded from {args.probe_prior}")

    session = None
    if args.score_seeds:
        from wq_alpha.brain.api import sign_in

        print("\nScoring semantic workshop source seeds with official Brain score deltas...")
        session = sign_in(args.credentials)
        seeds = score_seed_records(
            seeds,
            args,
            session,
            settings=settings,
            seed_resim_log=seed_resim_log,
        )
        if not seeds:
            raise SystemExit("No semantic workshop seed remained after autoworkshop score gating.")

    route_counts = _route_counts(seeds, args)
    print(f"Selected {len(seeds)} semantic workshop seed(s):")
    print(
        "  routes: "
        + " | ".join(f"{name}={route_counts.get(name, 0)}" for name in ("repair", "enhance"))
    )
    for seed in seeds:
        score_delta, fallback_delta = record_score_addition(seed)
        print(
            f"  id={seed.get('alpha_id') or '?'} "
            f"fit={_fmt(seed.get('is_fitness'))} sh={_fmt(seed.get('is_sharpe'))} "
            f"route={seed.get('_route') or '?'} "
            f"score_delta={_fmt(score_delta, 0)} fallback_delta={_fmt(fallback_delta)} "
            f"probe={build_probe_card(seed, probe_index).get('computed_trust') if probe_index else 'missing'} "
            f"{seed.get('expression', '')[:90]}"
        )
    if args.dry_run:
        return

    if session is None:
        from wq_alpha.brain.api import sign_in

        session = sign_in(args.credentials)
    portfolio_alpha_data = None
    if not args.skip_portfolio_pnl:
        from alpha_grammar.run_pipeline import load_portfolio_alpha_data

        portfolio_alpha_data = load_portfolio_alpha_data(session, competition_id=args.competition_id)

    diagnostic_agent = SemanticDiagnosticAgent(api_key=args.api_key, model=args.model)
    generator_agent = SemanticGeneratorAgent(api_key=args.api_key, model=args.model)
    template_handoff = handoff_from_args(args)
    cluster_members = load_cluster_members(args.clusters)
    known_expr_keys = {
        record_workshop_key(record)
        for record in all_records
        if record.get("expression")
    }
    robust_known = _completed_pairs_from_records(all_records + load_records([args.robust_sim_log]))
    history = WorkshopHistory([record_workshop_key(seed) for seed in seeds if seed.get("expression")])
    completed_source_keys: set[str] = set()

    run_records: list[dict[str, Any]] = []
    card_by_root: dict[str, dict[str, Any]] = {}
    seed_by_root: dict[str, dict[str, Any]] = {}

    for seed_index, raw_seed in enumerate(seeds, 1):
        print(f"\n{'=' * 72}\nSEMANTIC SEED {seed_index}/{len(seeds)}\n{'=' * 72}")
        seed = with_portfolio_prompt_feedback(
            raw_seed,
            session,
            portfolio_alpha_data,
            competition_id=args.competition_id,
            pnl_sleep=args.pnl_sleep,
            performance_max_attempts=5,
            performance_sleep_seconds=args.pnl_sleep,
        )
        root_expression = _root_expression(seed)
        seed_by_root[root_expression] = seed
        seed_key = record_workshop_key(seed)
        history.register_seed(seed_key)
        root_key = history.root_for(seed_key)
        probe_card = build_probe_card(seed, probe_index) if probe_index else {
            "computed_trust": "missing",
            "trust_reason": "probe prior file unavailable",
        }
        context_builder = SemanticContextBuilder(
            all_records,
            known_expr_keys=known_expr_keys,
            max_context_chars=args.max_context_chars,
        )
        diagnostic_context = context_builder.diagnostic_packet(
            seed=seed,
            probe_card=probe_card,
            history_records=history.context_for_root(root_key),
        )
        _append_jsonl(
            args.context_reports_jsonl,
            _context_report_row(seed, 0, "diagnostic", diagnostic_context.report),
        )
        diagnosis = diagnostic_agent.diagnose(
            seed_record=seed,
            probe_card=probe_card,
            context_packet=diagnostic_context.packet,
        ).payload
        _append_jsonl(args.diagnostics_jsonl, _diagnostic_row(seed, 0, diagnosis, probe_card))
        print(f"Diagnostic: {diagnosis.get('semantic_template')} / risk={diagnosis.get('overfit_risk')} / rec={diagnosis.get('recommendation')}")
        if diagnosis.get("recommendation") == "abandon":
            print("  Diagnostic agent recommended abandon; skipping generation.")
            create_routed_templates_for_finished_seed(
                template_handoff,
                seed=seed,
                root_expression=root_expression,
                seed_records=[],
                all_records=all_records,
                latest_card=None,
                reason="diagnostic_abandon",
            )
            _save_workshopped_sources({seed_key})
            completed_source_keys.add(seed_key)
            print("  Marked semantic source complete.")
            continue

        previous_results: list[dict[str, Any]] = []
        round_records_all: list[dict[str, Any]] = []
        latest_card: dict[str, Any] | None = None
        source_complete = False

        for round_index in (1, 2, 3):
            context_builder = SemanticContextBuilder(
                all_records,
                known_expr_keys=known_expr_keys,
                max_context_chars=args.max_context_chars,
            )
            generator_context = context_builder.generator_packet(
                seed=seed,
                diagnosis=diagnosis,
                round_index=round_index,
                target_count="exactly 6" if round_index in {1, 2} else "3 to 6",
                previous_results=previous_results,
                robustness_card=latest_card,
                history_records=history.context_for_root(root_key),
            )
            _append_jsonl(
                args.context_reports_jsonl,
                _context_report_row(seed, round_index, "generator", generator_context.report),
            )
            result = generator_agent.generate(
                seed_record=seed,
                diagnosis=diagnosis,
                round_index=round_index,
                previous_results=previous_results,
                robustness_card=latest_card,
                context_packet=generator_context.packet,
            )
            variants = _filter_variants(result.variants, round_index=round_index)
            candidates = _variants_to_candidates(
                variants,
                seed,
                diagnosis,
                round_index=round_index,
                known_expr_keys=known_expr_keys,
                probe_card=probe_card,
            )
            if not candidates:
                print(f"  round {round_index}: no new valid expressions; stopping seed.")
                source_complete = True
                break
            for candidate in candidates:
                history.register_lineage(record_workshop_key(candidate), seed_key)

            print(f"  round {round_index}: submitting {len(candidates)} semantic candidates")
            round_records = submit_candidate_batch(
                candidates,
                session,
                out_sim_log,
                settings,
                args.credentials,
                args.batch_size,
                portfolio_alpha_data=portfolio_alpha_data,
                pnl_sleep=args.pnl_sleep,
                yearly_stats_attempts=args.yearly_stats_attempts,
                yearly_stats_sleep=args.yearly_stats_sleep,
            )
            all_records.extend(round_records)
            run_records.extend(round_records)
            round_records_all.extend(round_records)
            previous_results = round_records
            known_expr_keys.update(record_workshop_key(record) for record in round_records if record.get("expression"))
            history.update(round_records, iteration=round_index)

            if round_index == 1 and not any(_submit_positive_score_candidate(record) for record in round_records):
                print(
                    "  round 1: no standalone-passing positive-score candidate; "
                    "abandoning seed before robustness and later rounds."
                )
                source_complete = True
                break

            robust_sources = _robustness_sources_for_round(
                round_records=round_records,
                round_records_all=round_records_all,
                round_index=round_index,
                round1_top_k=args.robust_top_k,
            )
            robust_limits = _robustness_limits_for_round(args, round_index)
            robust_candidates = build_robustness_candidates(
                robust_sources,
                root_expression=root_expression,
                cluster_members=cluster_members,
                known_keys=robust_known,
                max_expression_variants=robust_limits["max_expression_variants"],
                max_window_variants=robust_limits["max_window_variants"],
                max_field_variants=robust_limits["max_field_variants"],
                max_settings_variants=robust_limits["max_settings_variants"],
            )
            robust_records: list[dict[str, Any]] = []
            if robust_candidates:
                for candidate in robust_candidates:
                    history.register_lineage(record_workshop_key(candidate), seed_key)
                print(f"  round {round_index}: submitting {len(robust_candidates)} robustness-card candidates")
                robust_records = submit_candidate_batch(
                    robust_candidates,
                    session,
                    robust_sim_log,
                    settings,
                    args.credentials,
                    args.batch_size,
                    portfolio_alpha_data=portfolio_alpha_data,
                    pnl_sleep=args.pnl_sleep,
                    yearly_stats_attempts=args.yearly_stats_attempts,
                    yearly_stats_sleep=args.yearly_stats_sleep,
                )
                all_records.extend(robust_records)
                known_expr_keys.update(record_workshop_key(record) for record in robust_records if record.get("expression"))
                robust_known.update(_completed_pairs_from_records(robust_records))
                history.update(robust_records, iteration=round_index)
            elif not robust_sources:
                print(f"  round {round_index}: robustness-card skipped by policy")

            if robust_sources:
                all_robust_records = load_records([str(robust_sim_log)]) + robust_records
                latest_card = build_robustness_card(
                    root_expression=root_expression,
                    source_records=robust_sources,
                    robustness_records=all_robust_records,
                )
                card_by_root[root_expression] = latest_card
                _append_jsonl(args.cards_jsonl, {
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "round": round_index,
                    "seed_alpha_id": seed.get("alpha_id"),
                    "card": latest_card,
                })
                print(format_card_text(latest_card))

            if round_index in {1, 2}:
                update_context_builder = SemanticContextBuilder(
                    all_records,
                    known_expr_keys=known_expr_keys,
                    max_context_chars=args.max_context_chars,
                )
                update_context = update_context_builder.diagnostic_packet(
                    seed=seed,
                    prior_diagnosis=diagnosis,
                    round_results=round_records,
                    robustness_card=latest_card,
                    history_records=history.context_for_root(root_key),
                    probe_card=probe_card,
                )
                _append_jsonl(
                    args.context_reports_jsonl,
                    _context_report_row(seed, round_index, "diagnostic_update", update_context.report),
                )
                diagnosis = diagnostic_agent.diagnose(
                    seed_record=seed,
                    prior_diagnosis=diagnosis,
                    round_results=round_records,
                    robustness_card=latest_card,
                    probe_card=probe_card,
                    context_packet=update_context.packet,
                ).payload
                _append_jsonl(args.diagnostics_jsonl, _diagnostic_row(seed, round_index, diagnosis, probe_card))
                print(
                    f"Updated diagnostic: {diagnosis.get('semantic_template')} / "
                    f"risk={diagnosis.get('overfit_risk')} / rec={diagnosis.get('recommendation')}"
                )
                if diagnosis.get("recommendation") == "abandon":
                    print("  Diagnostic agent recommended abandon after feedback; stopping seed.")
                    source_complete = True
                    break
        else:
            source_complete = True

        if source_complete:
            create_routed_templates_for_finished_seed(
                template_handoff,
                seed=seed,
                root_expression=root_expression,
                seed_records=round_records_all,
                all_records=all_records,
                latest_card=latest_card,
                reason="completed",
            )
            _save_workshopped_sources({seed_key})
            completed_source_keys.add(seed_key)
            print("  Marked semantic source complete.")

    if completed_source_keys:
        print(f"Marked semantic-workshopped sources: {len(completed_source_keys)}")

    write_workshop_summary(
        load_records([args.out_sim_log]) + run_records,
        Path(args.summary_txt),
        Path(args.summary_jsonl),
        card_by_root=card_by_root,
        seed_by_root=seed_by_root,
    )
    try:
        n_new, n_updated = merge_session_to_master(args.out_sim_log)
        print(f"Merged semantic sim log to master: {n_new} new, {n_updated} updated")
    except Exception as exc:
        print(f"[warn] Could not merge semantic sim log to master: {exc}")


def write_workshop_summary(
    records: list[dict[str, Any]],
    out_txt: Path,
    out_jsonl: Path,
    *,
    card_by_root: dict[str, dict[str, Any]] | None = None,
    seed_by_root: dict[str, dict[str, Any]] | None = None,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    from collections import defaultdict

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get("search_lane") or "") != "semantic_enhance":
            continue
        root = _root_expression(record)
        if root:
            grouped[root].append(record)

    rows: list[dict[str, Any]] = []
    lines = ["Semantic workshop summary", "ranked by Brain score delta, fallback pool delta", ""]
    for root, group_records in sorted(grouped.items(), key=lambda kv: kv[0]):
        ranked = sorted(group_records, key=_summary_rank_key, reverse=True)
        card = (card_by_root or {}).get(root) or {}
        robust_status = card.get("overall_status")
        seed_record = (seed_by_root or {}).get(root) or _seed_record_from_group(root, group_records)
        seed_eval = _seed_eval(seed_record)
        lines.append(f"Seed: {root[:160]}")
        if seed_eval:
            lines.append(
                f"  seed id={seed_eval.get('alpha_id') or '?'} "
                f"fit={_fmt(seed_eval.get('fitness'))} sh={_fmt(seed_eval.get('sharpe'))} "
                f"turn={_fmt(seed_eval.get('turnover'))} score={_fmt(seed_eval.get('score_delta'), 0)} "
                f"year_worst_fit={_year_value(seed_eval.get('yearly_stability'), 'worst_fitness_year', 'worst_fitness')} "
                f"fit_gap={_fmt((seed_eval.get('yearly_stability') or {}).get('fitness_gap_to_overall'))}"
            )
        for rank, record in enumerate(ranked[:top_n], 1):
            params = record.get("params") if isinstance(record.get("params"), dict) else {}
            brain_delta, pool_delta = record_score_addition(record)
            source_robustness = _source_robustness_for_record(card, record)
            row_robust_status = source_robustness.get("status") or ("not_checked" if card else robust_status)
            recommendation_status = row_robust_status if row_robust_status != "not_checked" else "missing"
            recommendation = final_recommendation(record, recommendation_status)
            yearly_stability = _yearly_stability(record)
            final_eval = _final_evaluation(record, seed_record, source_robustness, yearly_stability)
            row = {
                "rank": rank,
                "root_expression": root,
                "seed": seed_eval,
                "alpha_id": record.get("alpha_id"),
                "expression": record.get("expression"),
                "round": params.get("semantic_round"),
                "is_sharpe": record.get("is_sharpe"),
                "is_fitness": record.get("is_fitness"),
                "is_turnover": record.get("is_turnover"),
                "admission_brain_score_delta": brain_delta,
                "admission_delta_pool_sharpe": pool_delta,
                "max_self_correlation": record.get("admission_global_max_corr", record.get("max_self_correlation")),
                "robustness_status": row_robust_status,
                "robustness_overall_status": robust_status,
                "robustness_expression_retention_rate": source_robustness.get("expression_retention_rate"),
                "robustness_settings_pass_rate": source_robustness.get("settings_pass_rate"),
                "robustness_field_pass_rate": source_robustness.get("field_pass_rate"),
                "robustness_top1000_retention": source_robustness.get("top1000_retention"),
                "robustness_n_expression_done": source_robustness.get("n_expression_done"),
                "robustness_n_settings_done": source_robustness.get("n_settings_done"),
                "robustness_n_field_done": source_robustness.get("n_field_done"),
                "robustness_notes": source_robustness.get("notes") or [],
                "robustness_perturbation_summary": source_robustness.get("perturbation_summary") or {},
                "robustness_worst_perturbation": source_robustness.get("worst_perturbation"),
                "robustness_best_perturbation": source_robustness.get("best_perturbation"),
                "yearly_stability": yearly_stability,
                "final_evaluation": final_eval,
                "semantic_template": params.get("semantic_template"),
                "overfit_risk": params.get("diagnostic_overfit_risk"),
                "final_recommendation": recommendation,
            }
            rows.append(row)
            lines.append(
                f"  {rank}. {recommendation:<14} id={row['alpha_id'] or '?'} "
                f"round={row['round']} score={_fmt(brain_delta, 0)} pool={_fmt(pool_delta)} "
                f"fit={_fmt(row['is_fitness'])} sh={_fmt(row['is_sharpe'])} "
                f"turn={_fmt(row['is_turnover'])} corr={_fmt(row['max_self_correlation'])} "
                f"robust={row_robust_status or 'N/A'} "
                f"expr_ret={_fmt(row['robustness_expression_retention_rate'])} "
                f"settings_pass={_fmt(row['robustness_settings_pass_rate'])} "
                f"top1000_ret={_fmt(row['robustness_top1000_retention'])} "
                f"year_worst_fit={_year_value(yearly_stability, 'worst_fitness_year', 'worst_fitness')} "
                f"fit_gap={_fmt(yearly_stability.get('fitness_gap_to_overall'))} "
                f"eval={final_eval.get('label')} "
                f"template={row['semantic_template'] or 'N/A'} "
                f"risk={row['overfit_risk'] or 'N/A'}"
            )
            lines.append(f"     {row['expression']}")
            perturb_line = _perturbation_line(source_robustness)
            if perturb_line:
                lines.append(f"     perturbation: {perturb_line}")
            if final_eval.get("notes"):
                lines.append(f"     final_eval: {'; '.join(final_eval['notes'][:4])}")
            if row["robustness_notes"]:
                lines.append(f"     robustness_notes: {'; '.join(row['robustness_notes'][:4])}")
        lines.append("")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n")
    _write_jsonl(rows, out_jsonl)
    print(f"Workshop summary: {out_txt}")
    print(f"Workshop summary JSONL: {out_jsonl}")
    return rows


def _source_robustness_for_record(card: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    expr = str(record.get("expression") or "")
    alpha_id = str(record.get("alpha_id") or "")
    candidate_id = str(record.get("candidate_id") or "")
    for source in card.get("sources") or []:
        if expr and expr == str(source.get("source_expression") or ""):
            return source
        if alpha_id and alpha_id == str(source.get("source_alpha_id") or ""):
            return source
        if candidate_id and candidate_id == str(source.get("source_candidate_id") or ""):
            return source
    return {}


def _seed_record_from_group(root: str, group_records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in group_records:
        params = record.get("params") if isinstance(record.get("params"), dict) else {}
        if params.get("seed_expression") or params.get("seed_alpha_id"):
            return {
                "alpha_id": params.get("seed_alpha_id"),
                "expression": params.get("seed_expression") or root,
                "is_sharpe": params.get("seed_sharpe"),
                "is_fitness": params.get("seed_fitness"),
                "is_turnover": params.get("seed_turnover"),
                "admission_brain_score_delta": params.get("seed_admission_brain_score_delta"),
                "admission_delta_pool_sharpe": params.get("seed_admission_delta_pool_sharpe"),
                "yearly_stats": params.get("seed_yearly_stats") or {},
            }
    return {"expression": root}


def _seed_eval(seed: dict[str, Any] | None) -> dict[str, Any]:
    if not seed:
        return {}
    brain, fallback = record_score_addition(seed)
    return {
        "alpha_id": seed.get("alpha_id"),
        "expression": seed.get("expression"),
        "sharpe": seed.get("is_sharpe"),
        "fitness": seed.get("is_fitness"),
        "turnover": seed.get("is_turnover"),
        "score_delta": brain,
        "fallback_delta_pool_sharpe": fallback,
        "yearly_stability": _yearly_stability(seed),
    }


def _yearly_stability(record: dict[str, Any]) -> dict[str, Any]:
    yearly = record.get("yearly_stats") if isinstance(record.get("yearly_stats"), dict) else {}
    rows: list[tuple[int, dict[str, Any]]] = []
    for year, stats in yearly.items():
        if not isinstance(stats, dict):
            continue
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            continue
        rows.append((year_int, stats))
    rows.sort(key=lambda item: item[0])
    overall_fitness = _float(record.get("is_fitness"))
    overall_sharpe = _float(record.get("is_sharpe"))
    if not rows:
        return {
            "n_years": 0,
            "overall_fitness": overall_fitness,
            "overall_sharpe": overall_sharpe,
            "notes": ["yearly stats unavailable"],
        }

    fitness_rows = [(year, _float(stats.get("fitness"))) for year, stats in rows]
    sharpe_rows = [(year, _float(stats.get("sharpe"))) for year, stats in rows]
    fitness_rows = [(year, value) for year, value in fitness_rows if value is not None]
    sharpe_rows = [(year, value) for year, value in sharpe_rows if value is not None]
    worst_fit_year, worst_fit = min(fitness_rows, key=lambda item: item[1]) if fitness_rows else (None, None)
    worst_sharpe_year, worst_sharpe = min(sharpe_rows, key=lambda item: item[1]) if sharpe_rows else (None, None)
    return {
        "n_years": len(rows),
        "overall_fitness": overall_fitness,
        "overall_sharpe": overall_sharpe,
        "worst_fitness_year": worst_fit_year,
        "worst_fitness": worst_fit,
        "fitness_gap_to_overall": (
            overall_fitness - worst_fit
            if overall_fitness is not None and worst_fit is not None
            else None
        ),
        "worst_sharpe_year": worst_sharpe_year,
        "worst_sharpe": worst_sharpe,
        "sharpe_gap_to_overall": (
            overall_sharpe - worst_sharpe
            if overall_sharpe is not None and worst_sharpe is not None
            else None
        ),
        "negative_fitness_years": [year for year, value in fitness_rows if value < 0],
        "fitness_below_0_5_years": [year for year, value in fitness_rows if value < 0.5],
        "fitness_below_1_0_years": [year for year, value in fitness_rows if value < 1.0],
    }


def _final_evaluation(
    record: dict[str, Any],
    seed: dict[str, Any] | None,
    robustness: dict[str, Any],
    yearly: dict[str, Any],
) -> dict[str, Any]:
    notes: list[str] = []
    label = "watch"
    brain, fallback = record_score_addition(record)
    score_ok = (brain is not None and brain > 0.0) or (brain is None and fallback is not None and fallback > 0.0)
    robust_status = str(robustness.get("status") or "not_checked")
    if not robustness:
        notes.append("final robustness not checked for this alpha")
        label = "not_checked"
    elif robust_status == "fail":
        notes.append("final robustness card failed")
        label = "fragile"
    elif robust_status == "weak":
        notes.append("final robustness card is weak")
        label = "watch"
    elif robust_status == "robust":
        label = "robust_watch" if score_ok else "robust_no_score"

    fit_gap = yearly.get("fitness_gap_to_overall")
    worst_fit = yearly.get("worst_fitness")
    if fit_gap is not None and fit_gap > 1.0:
        notes.append(f"worst-year fitness is {_fmt(fit_gap)} below overall fitness")
        if label.startswith("robust"):
            label = "watch"
    if worst_fit is not None and worst_fit < 0:
        notes.append("at least one year has negative fitness")
        label = "fragile" if label != "not_checked" else label
    if yearly.get("n_years") == 0:
        notes.append("yearly stability unavailable")

    settings_pass = robustness.get("settings_pass_rate")
    if settings_pass is not None and settings_pass < 0.5:
        notes.append("settings perturbation pass rate below 50%")
        if label.startswith("robust"):
            label = "watch"
    expr_ret = robustness.get("expression_retention_rate")
    if expr_ret is not None and expr_ret < 0.5:
        notes.append("expression perturbation retention below 50%")
        label = "fragile" if label != "not_checked" else label
    top1000 = robustness.get("top1000_retention")
    if top1000 is not None and top1000 < 0.5:
        notes.append("TOP1000 Sharpe retention below 50%")
        label = "fragile" if label != "not_checked" else label

    seed_fit = _float(seed.get("is_fitness")) if seed else None
    fit = _float(record.get("is_fitness"))
    if seed_fit is not None and fit is not None:
        diff = fit - seed_fit
        if diff < -0.25:
            notes.append(f"candidate fitness is {_fmt(abs(diff))} below seed fitness")

    if score_ok and label == "watch":
        notes.insert(0, "positive score addition but requires robustness review")
    elif score_ok and label.startswith("robust"):
        notes.insert(0, "positive score addition with acceptable deterministic robustness")
    elif not score_ok:
        notes.insert(0, "no positive score addition")

    return {
        "label": label,
        "score_positive": score_ok,
        "notes": notes,
    }


def _year_value(stability: dict[str, Any] | None, year_key: str, value_key: str) -> str:
    if not stability:
        return "N/A"
    year = stability.get(year_key)
    value = stability.get(value_key)
    if year is None or value is None:
        return "N/A"
    return f"{year}:{_fmt(value)}"


def _perturbation_line(robustness: dict[str, Any]) -> str:
    summary = robustness.get("perturbation_summary")
    if not isinstance(summary, dict):
        return ""
    parts = []
    for kind in ("expression", "window", "settings", "field_sibling"):
        row = summary.get(kind)
        if not isinstance(row, dict) or not row.get("n_done"):
            continue
        parts.append(
            f"{kind}=n{row.get('n_done')}/pass{row.get('n_pass')}/avg_ret{_fmt(row.get('avg_sharpe_retention'))}"
        )
    worst = robustness.get("worst_perturbation")
    if isinstance(worst, dict):
        parts.append(
            f"worst={worst.get('robust_kind')}:{worst.get('robust_check')} ret={_fmt(worst.get('sharpe_retention'))}"
        )
    return " | ".join(parts)


def _input_paths(args: argparse.Namespace) -> list[str]:
    patterns = args.sim_log or DEFAULT_INPUT_GLOBS
    out: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        for match in matches if matches else [pattern]:
            path = str(Path(match))
            key = str(Path(path).resolve()) if Path(path).exists() else path
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def _select_seed_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    lane_filter = list(getattr(args, "lane", None) or ["repair", "enhance"])
    bypass_self_corr = bool(getattr(args, "bypass_correlation_check", False))
    ignore_admission_scores = bool(getattr(args, "score_seeds", True))

    if args.seed_id or args.seed_expr:
        selected: list[dict[str, Any]] = []
        for alpha_id in args.seed_id:
            matches = [record for record in records if record.get("alpha_id") == alpha_id and record.get("status") == "done"]
            if matches:
                selected.append(max(matches, key=lambda record: record.get("is_sharpe") or -999.0))
        for expr in args.seed_expr:
            matches = [
                record for record in records
                if str(record.get("expression") or "").strip() == expr.strip()
                and record.get("status") == "done"
            ]
            if matches:
                selected.append(max(matches, key=lambda record: record.get("is_sharpe") or -999.0))
        return select_candidates(
            _dedupe_records(selected),
            workshopped_exprs=set(),
            excluded_exprs=set(),
            lane_filter=lane_filter,
            max_candidates=args.max_seeds,
            verbose=False,
            bypass_self_corr=bypass_self_corr,
            ignore_admission_scores=ignore_admission_scores,
        )

    source_records = [
        record for record in records
        if args.source != "grammar" or _is_grammar(record)
    ]
    excluded_exprs = DEFAULT_EXCLUDED_EXPRS | set(getattr(args, "exclude_expr", []) or [])
    return select_candidates(
        source_records,
        workshopped_exprs=_load_workshopped_sources(),
        excluded_exprs=excluded_exprs,
        lane_filter=lane_filter,
        max_candidates=args.max_seeds,
        verbose=True,
        bypass_self_corr=bypass_self_corr,
        ignore_admission_scores=ignore_admission_scores,
    )


def _route_counts(seeds: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, int]:
    from collections import Counter
    from wq_alpha.workshop.core import route_record

    return Counter(
        seed.get("_route")
        or route_record(seed, bypass_self_corr=bool(getattr(args, "bypass_correlation_check", False)))
        or "skip"
        for seed in seeds
    )


def _is_grammar(record: dict[str, Any]) -> bool:
    lane = str(record.get("search_lane") or "")
    family = str(record.get("family_id") or "")
    intention = str(record.get("intention_id") or "")
    source = str(record.get("_source_path") or "")
    return (
        lane == "grammar"
        or family.startswith("grammar|")
        or intention.startswith("grammar")
        or "grammar_" in Path(source).name
        or "alpha_grammar" in Path(source).parts
    )


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record.get("expression") or record.get("alpha_id") or record.get("candidate_id")
        if not key:
            continue
        existing = best.get(str(key))
        if existing is None or _seed_rank_key(record) > _seed_rank_key(existing):
            best[str(key)] = record
    return list(best.values())


def _seed_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    brain, fallback = record_score_addition(record)
    score = brain if brain is not None else fallback
    passing_bonus = 2.0 if _standalone_passes(record) else 0.0
    return (
        score if score is not None else -1e9,
        passing_bonus + abs(_float(record.get("is_fitness")) or 0.0),
        abs(_float(record.get("is_sharpe")) or 0.0),
        -(_float(record.get("is_turnover")) or 9.0),
    )


def _filter_variants(variants: list[GeneratedVariant], *, round_index: int) -> list[GeneratedVariant]:
    out: list[GeneratedVariant] = []
    seen: set[str] = set()
    for variant in variants:
        expr = variant.expression.strip()
        if not expr or expr in seen or not _looks_like_fastexpr(expr):
            continue
        seen.add(expr)
        out.append(variant)
    if round_index in {1, 2}:
        if len(out) != 6:
            print(f"  [warn] round {round_index} expected exactly 6 parsed expressions, got {len(out)}")
        return out[:6]
    if len(out) < 3:
        print(f"  [warn] round 3 expected 3-6 parsed expressions, got {len(out)}")
    return out[:6]


def _variants_to_candidates(
    variants: list[GeneratedVariant],
    seed: dict[str, Any],
    diagnosis: dict[str, Any],
    *,
    round_index: int,
    known_expr_keys: set[str],
    probe_card: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    root_expression = _root_expression(seed)
    for variant in variants:
        expr = variant.expression.strip()
        expr_key = f"{record_regime_safe(seed)}\t{expr}"
        if expr_key in known_expr_keys:
            continue
        known_expr_keys.add(expr_key)
        params = {
            "original_expression": seed.get("expression"),
            "root_expression": root_expression,
            "seed_alpha_id": seed.get("alpha_id"),
            "seed_expression": seed.get("expression"),
            "seed_sharpe": seed.get("is_sharpe"),
            "seed_fitness": seed.get("is_fitness"),
            "seed_turnover": seed.get("is_turnover"),
            "seed_yearly_stats": seed.get("yearly_stats") or {},
            "seed_admission_brain_score_delta": record_score_addition(seed)[0],
            "seed_admission_delta_pool_sharpe": record_score_addition(seed)[1],
            "semantic_round": round_index,
            "semantic_template": diagnosis.get("semantic_template"),
            "signal_identity": diagnosis.get("signal_identity"),
            "diagnostic_overfit_risk": diagnosis.get("overfit_risk"),
            "diagnostic_recommendation": diagnosis.get("recommendation"),
            "diagnostic_probe_relevance": diagnosis.get("probe_relevance"),
            "diagnostic_probe_use": diagnosis.get("probe_use"),
            "seed_vs_probe_alignment": diagnosis.get("seed_vs_probe_alignment"),
            "field_roles": diagnosis.get("field_roles"),
            "valid_mutation_axes": diagnosis.get("valid_mutation_axes"),
            "invalid_mutations": diagnosis.get("invalid_mutations"),
            "structural_purpose": variant.structural_purpose,
            "mutation_axes": variant.mutation_axes,
            "preserved_identity": variant.preserved_identity,
            "expected_fix": variant.expected_fix,
            "variant_overfit_risk": variant.overfit_risk,
        }
        if probe_card:
            params["probe_card"] = compact_probe_card(probe_card)
        candidates.append(_make_candidate(expr, seed, "semantic_enhance", params))
    return candidates


def _robustness_sources_for_round(
    *,
    round_records: list[dict[str, Any]],
    round_records_all: list[dict[str, Any]],
    round_index: int,
    round1_top_k: int,
) -> list[dict[str, Any]]:
    """Select a small diagnostic robustness set for the current semantic round."""
    if round_index == 1:
        return _top_done_records(round_records, max(1, round1_top_k))
    if round_index == 2:
        # Lightweight check: top-1 from round 2 only, so the round-2 diagnostic
        # update (which informs round-3 generation) has signal-stability feedback.
        return _top_done_records(round_records, 1)
    if round_index == 3:
        return _top_done_records(round_records_all, 1)
    return []


def _robustness_limits_for_round(args: argparse.Namespace, round_index: int) -> dict[str, int]:
    if round_index == 3:
        return {
            "max_expression_variants": args.final_max_expression_variants,
            "max_window_variants": args.final_max_window_variants,
            "max_field_variants": args.final_max_field_variants,
            "max_settings_variants": args.final_max_settings_variants,
        }
    if round_index == 2:
        # Lightweight: expression-check and one window neighbour only.
        # No field-sibling or settings sweeps — those run in rounds 1 and 3.
        return {
            "max_expression_variants": 2,
            "max_window_variants": 1,
            "max_field_variants": 0,
            "max_settings_variants": 0,
        }
    return {
        "max_expression_variants": args.max_expression_variants,
        "max_window_variants": args.max_window_variants,
        "max_field_variants": args.max_field_variants,
        "max_settings_variants": args.max_settings_variants,
    }


def _top_done_records(records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    done = [record for record in records if record.get("status") == "done" and record.get("expression")]
    done.sort(key=_summary_rank_key, reverse=True)
    return done[:max(0, top_k)]


def _summary_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    brain, fallback = record_score_addition(record)
    score = brain if brain is not None else fallback
    corr = _float(record.get("admission_global_max_corr", record.get("max_self_correlation")))
    turnover = _float(record.get("is_turnover"))
    return (
        score if score is not None else -1e9,
        _float(record.get("is_fitness")) or -1e9,
        _float(record.get("is_sharpe")) or -1e9,
        -(corr if corr is not None else 9.0),
        -(turnover if turnover is not None else 9.0),
    )


def _standalone_passes(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "done"
        and (_float(record.get("is_sharpe")) or -999.0) >= 1.25
        and (_float(record.get("is_fitness")) or -999.0) >= 1.0
        and is_passing(record)
    )


def _submit_positive_score_candidate(record: dict[str, Any]) -> bool:
    if not _standalone_passes(record):
        return False
    brain, fallback = record_score_addition(record)
    if not ((brain is not None and brain > 0.0) or (brain is None and fallback is not None and fallback > 0.0)):
        return False
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    decision = str(record.get("admission_decision") or admission.get("decision") or "").upper()
    if decision and decision != "PROMOTE":
        return False
    return True


def _diagnostic_row(
    seed: dict[str, Any],
    round_index: int,
    diagnosis: dict[str, Any],
    probe_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "round": round_index,
        "seed_alpha_id": seed.get("alpha_id"),
        "root_expression": _root_expression(seed),
        "diagnosis": diagnosis,
        "probe_card": compact_probe_card(probe_card),
    }


def _context_report_row(seed: dict[str, Any], round_index: int, phase: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "round": round_index,
        "phase": phase,
        "seed_alpha_id": seed.get("alpha_id"),
        "root_expression": _root_expression(seed),
        "report": report,
    }


def _root_expression(record: dict[str, Any]) -> str:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    return str(params.get("root_expression") or params.get("original_expression") or record.get("expression") or "")


def record_regime_safe(record: dict[str, Any]) -> str:
    settings = record.get("settings")
    if isinstance(settings, dict):
        return _regime_key({**DEFAULT_SETTINGS, **settings})
    override = record.get("settings_override")
    if isinstance(override, dict):
        return _regime_key({**DEFAULT_SETTINGS, **override})
    return str(record.get("regime") or _regime_key(DEFAULT_SETTINGS))


def _completed_pairs_from_records(records: list[dict[str, Any]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for record in records:
        expr = record.get("expression")
        if expr:
            out.add((str(expr), record_regime_safe(record)))
    return out


def _load_settings(path: str) -> dict[str, Any]:
    if not path:
        return dict(DEFAULT_SETTINGS)
    with open(path) as handle:
        settings = json.load(handle)
    return {**DEFAULT_SETTINGS, **settings}


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
