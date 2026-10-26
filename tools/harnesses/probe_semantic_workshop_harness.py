"""A/B test probe-card context for the semantic workshop.

Default pilot:
  same frozen grammar seed batch
  -> baseline semantic workshop round 1
  -> probe-card diagnostic semantic workshop round 1
  -> paired per-seed score-addition summary

This script keeps outputs isolated under artifacts/probe_semantic_ab/ and does
not write to the production semantic_workshop artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_enhancement.agents.semantic_workshop import (  # noqa: E402
    GeneratedVariant,
    SemanticDiagnosticAgent,
    SemanticGeneratorAgent,
)
from alpha_enhancement.agents.testing import StubDiagnosticAgent, StubGeneratorAgent  # noqa: E402
from core.brain.submit import DEFAULT_SETTINGS, _regime_key  # noqa: E402
from alpha_enhancement.workshop import semantic as semantic_workshop  # noqa: E402
from alpha_enhancement.workshop.core import (  # noqa: E402
    _looks_like_fastexpr,
    _make_candidate,
    find_passing_examples,
    is_passing,
    record_workshop_key,
    route_record,
)
from alpha_enhancement.workshop.records import load_records  # noqa: E402
from alpha_enhancement.workshop.robustness_card import record_score_addition  # noqa: E402
from alpha_enhancement.workshop.scoring import score_seed_records, with_portfolio_prompt_feedback  # noqa: E402
from alpha_enhancement.workshop.simulation import submit_candidate_batch  # noqa: E402


DEFAULT_OUT_ROOT = ROOT / "artifacts" / "probe_semantic_ab"
DEFAULT_PROBE_PRIOR = ROOT / "artifacts" / "probe" / "grammar_prior_from_probe.json"
DEFAULT_AUTOWORKSHOP_SUBMITTABLE = ROOT / "artifacts" / "workshop" / "workshop_submittable.jsonl"
ARMS = ("baseline", "probe_diagnostic")
_FIELD_PROBE_KEYS: set[str] = set()


@dataclass(frozen=True)
class ArmResult:
    arm: str
    records: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-log", nargs="*", default=[], help="Input sim logs. Defaults to semantic workshop inputs.")
    parser.add_argument("--source", choices=("grammar", "all"), default="grammar")
    parser.add_argument("--seed-id", nargs="*", default=[], help="Specific alpha_id seed(s).")
    parser.add_argument("--seed-expr", nargs="*", default=[], help="Specific expression seed(s).")
    parser.add_argument("--seed-file", default="", help="Reuse a frozen seed JSONL file instead of selecting a new batch.")
    parser.add_argument(
        "--seed-source",
        choices=("selector", "autoworkshop_submittable"),
        default="selector",
        help=(
            "Seed batch source. 'selector' uses the semantic/autoworkshop router; "
            "'autoworkshop_submittable' uses positive-score source seeds from workshop_submittable.jsonl."
        ),
    )
    parser.add_argument(
        "--autoworkshop-submittable",
        default=str(DEFAULT_AUTOWORKSHOP_SUBMITTABLE),
        help="Autoworkshop submittable JSONL used for seed-source autoworkshop_submittable.",
    )
    parser.add_argument(
        "--rescore-autoworkshop-seeds",
        action="store_true",
        help="Re-fetch official score deltas for autoworkshop_submittable seeds before freezing them.",
    )
    parser.add_argument("--max-seeds", type=int, default=5)
    parser.add_argument("--min-abs-fitness", type=float, default=0.8, help=argparse.SUPPRESS)
    parser.add_argument("--max-abs-sharpe", type=float, default=10.0, help=argparse.SUPPRESS)
    parser.add_argument("--max-abs-fitness", type=float, default=20.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--lane",
        nargs="*",
        choices=("repair", "enhance"),
        default=["repair", "enhance"],
        help="Autoworkshop source lanes allowed into the A/B seed batch (default: repair enhance).",
    )
    parser.add_argument("--exclude-expr", nargs="*", default=[], help="Expressions to exclude from seed selection.")
    parser.add_argument("--rounds", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--settings-json", default="")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--competition-id", default="IQC2026S1")
    parser.add_argument("--skip-portfolio-pnl", action="store_true")
    parser.add_argument("--pnl-sleep", type=float, default=1.0)
    parser.add_argument("--yearly-stats-attempts", type=int, default=6)
    parser.add_argument("--yearly-stats-sleep", type=float, default=20.0)
    parser.add_argument("--seed-resim-log", default="")
    parser.add_argument("--probe-prior", default=str(DEFAULT_PROBE_PRIOR))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true", help="Freeze seed batch and write manifest; no LLM or Brain calls.")
    parser.add_argument(
        "--llm-dry-run",
        action="store_true",
        help="With --dry-run, call LLM agents and synthesize local records, but do not submit to Brain.",
    )
    parser.add_argument("--stub-agents", action="store_true", help="Use deterministic local agents for script tests.")
    parser.add_argument("--no-submit", action="store_true", help="Do not call Brain; write deterministic synthetic sim records.")
    parser.set_defaults(score_seeds=True)
    parser.add_argument(
        "--score-seeds",
        dest="score_seeds",
        action="store_true",
        help=(
            "Fetch official Brain score deltas before freezing a new seed batch (default for live runs). "
            "Repair/enhance seeds continue only with positive fetched score delta."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_resim_log = Path(args.seed_resim_log) if args.seed_resim_log else run_dir / "semantic_seed_resim_sim.jsonl"
    seed_resim_log.parent.mkdir(parents=True, exist_ok=True)

    input_paths = semantic_workshop._input_paths(args)
    all_records = load_records(input_paths, tag_source=True)
    autoworkshop_benchmarks: dict[str, dict[str, Any]] = {}
    if args.seed_file:
        seeds = _load_seed_file(args.seed_file)
    elif args.seed_source == "autoworkshop_submittable":
        seeds, autoworkshop_benchmarks = load_autoworkshop_seed_batch(
            all_records,
            Path(args.autoworkshop_submittable),
            max_seeds=args.max_seeds,
        )
    else:
        seeds = semantic_workshop._select_seed_records(all_records, args)
    if not seeds:
        raise SystemExit("No seed records selected.")

    no_submit = args.no_submit or args.stub_agents or args.llm_dry_run
    session = None
    should_score_seeds = bool(
        args.score_seeds
        and not args.seed_file
        and not args.dry_run
        and not no_submit
        and (args.seed_source != "autoworkshop_submittable" or args.rescore_autoworkshop_seeds)
    )
    if should_score_seeds:
        from core.brain.api import sign_in

        print("\nScoring A/B source seeds with official Brain score deltas...")
        session = sign_in(args.credentials)
        settings = semantic_workshop._load_settings(args.settings_json)
        seeds = score_seed_records(
            seeds,
            args,
            session,
            settings=settings,
            seed_resim_log=seed_resim_log,
        )
        if not seeds:
            raise SystemExit("No A/B seed remained after autoworkshop score gating.")

    seeds_path = run_dir / "seeds.jsonl"
    _write_jsonl(seeds, seeds_path)

    probe_index = load_probe_prior(Path(args.probe_prior))
    seed_probe_cards = {
        _seed_key(seed): build_probe_card(seed, probe_index)
        for seed in seeds
    }
    manifest = _manifest(args, run_id, run_dir, input_paths, seeds_path, len(seeds), len(probe_index))
    _write_json(manifest, run_dir / "manifest.json")

    print(f"Frozen {len(seeds)} seed(s): {seeds_path}")
    route_counts = semantic_workshop._route_counts(seeds, args)
    print(
        "  routes: "
        + " | ".join(f"{name}={route_counts.get(name, 0)}" for name in ("repair", "enhance"))
    )
    for seed in seeds:
        card = seed_probe_cards[_seed_key(seed)]
        score_delta, fallback_delta = record_score_addition(seed)
        print(
            f"  id={seed.get('alpha_id') or '?'} fit={_fmt(seed.get('is_fitness'))} "
            f"sh={_fmt(seed.get('is_sharpe'))} route={seed.get('_route') or '?'} "
            f"score_delta={_fmt(score_delta, 0)} fallback_delta={_fmt(fallback_delta)} "
            f"probe={card.get('computed_trust')} "
            f"{str(seed.get('expression') or '')[:80]}"
        )
    if args.dry_run and not args.llm_dry_run:
        print(f"Dry run complete. Manifest: {run_dir / 'manifest.json'}")
        return

    settings = semantic_workshop._load_settings(args.settings_json)
    if not args.stub_agents:
        preflight_llm(args.api_key)

    diagnostic_agent: Any
    generator_agent: Any
    if args.stub_agents:
        diagnostic_agent = StubDiagnosticAgent()
        generator_agent = StubGeneratorAgent()
    else:
        diagnostic_agent = SemanticDiagnosticAgent(api_key=args.api_key, model=args.model)
        generator_agent = SemanticGeneratorAgent(api_key=args.api_key, model=args.model)

    portfolio_alpha_data = None
    if not no_submit:
        from core.brain.api import sign_in

        if session is None:
            session = sign_in(args.credentials)
        if not args.skip_portfolio_pnl:
            from core.selfcorr import load_portfolio_alpha_data

            portfolio_alpha_data = load_portfolio_alpha_data(
                session,
                competition_id=args.competition_id,
            )

    pre_run_known = {
        record_workshop_key(record)
        for record in all_records
        if record.get("expression")
    }
    cross_arm_cache: dict[str, dict[str, Any]] = {}
    arm_results: dict[str, ArmResult] = {}

    for arm in ARMS:
        arm_results[arm] = run_arm(
            arm=arm,
            seeds=seeds,
            all_records=all_records,
            pre_run_known=pre_run_known,
            cross_arm_cache=cross_arm_cache,
            probe_cards=seed_probe_cards,
            diagnostic_agent=diagnostic_agent,
            generator_agent=generator_agent,
            session=session,
            settings=settings,
            args=args,
            run_dir=run_dir,
            no_submit=no_submit,
            portfolio_alpha_data=portfolio_alpha_data,
        )

    rows = write_summary(
        seeds=seeds,
        arm_results=arm_results,
        probe_cards=seed_probe_cards,
        run_dir=run_dir,
        autoworkshop_benchmarks=autoworkshop_benchmarks,
    )
    print(f"Summary: {run_dir / 'summary.txt'}")
    print(f"Summary JSONL: {run_dir / 'summary.jsonl'}")
    print(f"Compared {len(rows)} seed(s).")


def run_arm(
    *,
    arm: str,
    seeds: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    pre_run_known: set[str],
    cross_arm_cache: dict[str, dict[str, Any]],
    probe_cards: dict[str, dict[str, Any]],
    diagnostic_agent: Any,
    generator_agent: Any,
    session: Any,
    settings: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    no_submit: bool,
    portfolio_alpha_data: dict[str, Any] | None,
) -> ArmResult:
    arm_dir = run_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    sim_log = arm_dir / "sim.jsonl"
    diagnostics_path = arm_dir / "diagnostics.jsonl"
    generations_path = arm_dir / "generations.jsonl"
    arm_known = set(pre_run_known)
    arm_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    print(f"\n{'=' * 72}\nARM {arm}\n{'=' * 72}")
    for seed_index, raw_seed in enumerate(seeds, 1):
        print(f"\n[{arm}] seed {seed_index}/{len(seeds)}")
        seed = (
            raw_seed
            if no_submit
            else with_portfolio_prompt_feedback(
                raw_seed,
                session,
                portfolio_alpha_data,
                competition_id=args.competition_id,
                pnl_sleep=args.pnl_sleep,
                performance_max_attempts=5,
                performance_sleep_seconds=args.pnl_sleep,
            )
        )
        probe_card = probe_cards.get(_seed_key(seed)) if arm == "probe_diagnostic" else None
        context_records = find_passing_examples(all_records, seed, n=8)
        diagnosis = diagnostic_agent.diagnose(
            seed_record=seed,
            context_records=context_records,
            probe_card=probe_card,
        ).payload
        diag_row = _diagnostic_row(arm, seed, 0, diagnosis, probe_card)
        diagnostics.append(diag_row)
        _append_jsonl(diagnostics_path, diag_row)
        print(
            f"  diagnostic={diagnosis.get('semantic_template')} "
            f"probe_use={diagnosis.get('probe_use')} rec={diagnosis.get('recommendation')}"
        )
        if diagnosis.get("recommendation") == "abandon":
            continue

        previous_results: list[dict[str, Any]] = []
        for round_index in range(1, args.rounds + 1):
            result = generator_agent.generate(
                seed_record=seed,
                diagnosis=diagnosis,
                round_index=round_index,
                previous_results=previous_results,
                robustness_card=None,
                known_expressions=[key.partition("\t")[2] for key in sorted(arm_known)[-80:]],
            )
            variants = semantic_workshop._filter_variants(result.variants, round_index=round_index)
            _append_jsonl(
                generations_path,
                _generation_row(arm, seed, round_index, result, variants, probe_card),
            )
            candidates, reused = variants_to_candidates(
                variants,
                seed,
                diagnosis,
                arm=arm,
                round_index=round_index,
                arm_known=arm_known,
                cross_arm_cache=cross_arm_cache,
                probe_card=probe_card,
            )
            for record in reused:
                _append_jsonl(sim_log, record)
            new_records: list[dict[str, Any]]
            if no_submit:
                new_records = synthesize_records(candidates, settings)
                for record in new_records:
                    _append_jsonl(sim_log, record)
            elif candidates:
                new_records = submit_candidate_batch(
                    candidates,
                    session,
                    sim_log,
                    settings,
                    args.credentials,
                    args.batch_size,
                    portfolio_alpha_data=portfolio_alpha_data,
                    pnl_sleep=args.pnl_sleep,
                )
            else:
                new_records = []

            round_records = reused + new_records
            for record in round_records:
                expr_key = record_workshop_key(record)
                arm_known.add(expr_key)
                cross_arm_cache.setdefault(expr_key, record)
            arm_records.extend(round_records)
            previous_results = round_records
            print(f"  round {round_index}: submitted={len(candidates)} reused={len(reused)}")
            if not round_records:
                break

    return ArmResult(arm=arm, records=arm_records, diagnostics=diagnostics)


def variants_to_candidates(
    variants: list[GeneratedVariant],
    seed: dict[str, Any],
    diagnosis: dict[str, Any],
    *,
    arm: str,
    round_index: int,
    arm_known: set[str],
    cross_arm_cache: dict[str, dict[str, Any]],
    probe_card: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    seed_regime = semantic_workshop.record_regime_safe(seed)
    seen_this_round: set[str] = set()
    for variant in variants:
        expr = variant.expression.strip()
        if not expr or expr in seen_this_round or not _looks_like_fastexpr(expr):
            continue
        seen_this_round.add(expr)
        expr_key = f"{seed_regime}\t{expr}"
        if expr_key in arm_known:
            continue
        cached = cross_arm_cache.get(expr_key)
        if cached is not None:
            reused_record = _reuse_record(cached, seed, diagnosis, arm, round_index, probe_card)
            reused.append(reused_record)
            arm_known.add(expr_key)
            continue
        params = _candidate_params(seed, diagnosis, variant, arm, round_index, probe_card)
        candidate = _make_candidate(expr, seed, "semantic_enhance", params)
        candidate["experiment_stage"] = "probe_semantic_ab"
        candidate["template_family"] = "semantic_ab"
        candidates.append(candidate)
        arm_known.add(expr_key)
    return candidates, reused


def _candidate_params(
    seed: dict[str, Any],
    diagnosis: dict[str, Any],
    variant: GeneratedVariant,
    arm: str,
    round_index: int,
    probe_card: dict[str, Any] | None,
) -> dict[str, Any]:
    root_expression = semantic_workshop._root_expression(seed)
    params = {
        "experiment_name": "probe_semantic_ab",
        "experiment_arm": arm,
        "original_expression": seed.get("expression"),
        "root_expression": root_expression,
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
    return params


def _reuse_record(
    cached: dict[str, Any],
    seed: dict[str, Any],
    diagnosis: dict[str, Any],
    arm: str,
    round_index: int,
    probe_card: dict[str, Any] | None,
) -> dict[str, Any]:
    reused = copy.deepcopy(cached)
    source_params = cached.get("params") if isinstance(cached.get("params"), dict) else {}
    variant = GeneratedVariant(
        expression=str(cached.get("expression") or ""),
        structural_purpose=str(source_params.get("structural_purpose") or "cross-arm duplicate reuse"),
        mutation_axes=list(source_params.get("mutation_axes") or []),
        preserved_identity=str(source_params.get("preserved_identity") or ""),
        expected_fix=str(source_params.get("expected_fix") or ""),
        overfit_risk=str(source_params.get("variant_overfit_risk") or "medium"),
    )
    params = _candidate_params(seed, diagnosis, variant, arm, round_index, probe_card)
    params["cross_arm_duplicate_of"] = source_params.get("experiment_arm") or cached.get("cross_arm_duplicate_of") or "unknown"
    params["reused_from_candidate_id"] = cached.get("candidate_id")
    params["reused_from_alpha_id"] = cached.get("alpha_id")
    reused["logged_at"] = datetime.now(timezone.utc).isoformat()
    reused["candidate_id"] = _stable_id(f"{arm}\n{round_index}\n{cached.get('expression')}")
    reused["params"] = params
    reused["experiment_stage"] = "probe_semantic_ab"
    reused["template_family"] = "semantic_ab"
    reused["search_lane"] = "semantic_enhance"
    reused["cross_arm_duplicate_of"] = params["cross_arm_duplicate_of"]
    return reused


def synthesize_records(candidates: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        expr = str(candidate.get("expression") or "")
        digest = hashlib.sha1(expr.encode()).hexdigest()
        n = int(digest[:8], 16)
        candidate_settings = {**settings, **candidate.get("settings_override", {})}
        sharpe = round(0.5 + (n % 220) / 100.0, 3)
        fitness = round(0.4 + ((n // 220) % 180) / 100.0, 3)
        turnover = round(0.05 + ((n // 997) % 45) / 100.0, 3)
        brain_delta = float((n % 41) - 20)
        pool_delta = round(((n // 41) % 41 - 20) / 1000.0, 4)
        records.append({
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "candidate_id": candidate.get("candidate_id"),
            "template_id": candidate.get("template_id"),
            "experiment_stage": candidate.get("experiment_stage"),
            "template_family": candidate.get("template_family"),
            "dataset_id": candidate.get("dataset_id"),
            "category": candidate.get("category"),
            "search_lane": candidate.get("search_lane", "semantic_enhance"),
            "expression": expr,
            "fields": candidate.get("fields"),
            "cluster_ids": candidate.get("cluster_ids"),
            "group": candidate.get("group"),
            "params": candidate.get("params"),
            "status": "done",
            "alpha_id": "stub_" + digest[:8],
            "settings": candidate_settings,
            "regime": _regime_key(candidate_settings),
            "is_sharpe": sharpe,
            "is_fitness": fitness,
            "is_turnover": turnover,
            "is_returns": round(sharpe / 100.0, 4),
            "is_drawdown": round(-abs(fitness) / 100.0, 4),
            "grade": "PASS",
            "checks": [],
            "admission": {
                "decision": "PROMOTE" if brain_delta > 0 else "REJECT",
                "reason": "STUB",
                "brain_score_delta": brain_delta,
                "delta_pool_sharpe": pool_delta,
                "global_max_corr": None,
            },
            "admission_decision": "PROMOTE" if brain_delta > 0 else "REJECT",
            "admission_reason": "STUB",
            "admission_brain_score_delta": brain_delta,
            "admission_delta_pool_sharpe": pool_delta,
            "admission_global_max_corr": None,
        })
    return records


def load_autoworkshop_seed_batch(
    all_records: list[dict[str, Any]],
    submittable_path: Path,
    *,
    max_seeds: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return positive-score source seeds from the latest autoworkshop output.

    The benchmark map is keyed by semantic root expression and points to the
    best autoworkshop-promoted row for that same root.
    """
    rows = load_records([str(submittable_path)], tag_source=False)
    if not rows:
        raise SystemExit(f"No autoworkshop submittable rows found: {submittable_path}")

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("admission_decision") != "PROMOTE":
            continue
        params = row.get("params") if isinstance(row.get("params"), dict) else {}
        seed_delta = _float(params.get("seed_admission_brain_score_delta"))
        if seed_delta is None or seed_delta <= 0.0:
            continue
        root = str(params.get("root_expression") or row.get("seed_expression") or params.get("seed_expression") or "")
        if not root:
            continue
        existing = grouped.get(root)
        if existing is None or _rank_key(row) > _rank_key(existing):
            grouped[root] = row

    if not grouped:
        raise SystemExit(f"No positive-score autoworkshop source seeds found in {submittable_path}")

    roots = sorted(grouped, key=lambda root: _rank_key(grouped[root]), reverse=True)
    if max_seeds > 0:
        roots = roots[:max_seeds]

    by_alpha_id = {
        str(record.get("alpha_id")): record
        for record in all_records
        if record.get("alpha_id")
    }
    by_expr = {
        str(record.get("expression") or "").strip(): record
        for record in all_records
        if record.get("expression") and record.get("status") == "done"
    }

    seeds: list[dict[str, Any]] = []
    benchmarks: dict[str, dict[str, Any]] = {}
    for root in roots:
        benchmark = grouped[root]
        params = benchmark.get("params") if isinstance(benchmark.get("params"), dict) else {}
        seed_id = params.get("seed_alpha_id")
        seed_expr = str(params.get("seed_expression") or params.get("original_expression") or root)
        source = by_alpha_id.get(str(seed_id)) if seed_id else None
        if source is None:
            source = by_expr.get(seed_expr.strip())
        if source is None:
            source = _synthesized_autoworkshop_seed(benchmark, root, seed_expr)
        seed = copy.deepcopy(source)
        seed_params = dict(seed.get("params") if isinstance(seed.get("params"), dict) else {})
        seed_params.update({
            "root_expression": root,
            "original_expression": seed_expr,
            "autoworkshop_benchmark_alpha_id": benchmark.get("alpha_id"),
            "autoworkshop_benchmark_expression": benchmark.get("expression"),
            "autoworkshop_benchmark_score_delta": benchmark.get("admission_brain_score_delta"),
        })
        seed["params"] = seed_params
        seed["admission_decision"] = params.get("seed_admission_decision") or "PROMOTE"
        seed["admission_reason"] = params.get("seed_admission_reason") or "POSITIVE_BRAIN_SCORE_DELTA"
        seed["admission_signal"] = params.get("seed_admission_signal") or "brain_score_delta"
        seed["admission_brain_score_before"] = params.get("seed_admission_brain_score_before")
        seed["admission_brain_score_after"] = params.get("seed_admission_brain_score_after")
        seed["admission_brain_score_delta"] = params.get("seed_admission_brain_score_delta")
        seed["admission"] = {
            **(seed.get("admission") if isinstance(seed.get("admission"), dict) else {}),
            "decision": seed["admission_decision"],
            "reason": seed["admission_reason"],
            "signal": seed["admission_signal"],
            "brain_score_before": seed["admission_brain_score_before"],
            "brain_score_after": seed["admission_brain_score_after"],
            "brain_score_delta": seed["admission_brain_score_delta"],
        }
        route = route_record(seed)
        seed["_route"] = route if route in {"repair", "enhance"} else (
            benchmark.get("search_lane") if benchmark.get("search_lane") in {"repair", "enhance"} else "enhance"
        )
        seeds.append(seed)
        benchmarks[root] = benchmark

    print(
        f"Loaded {len(seeds)} positive-score autoworkshop seed(s) "
        f"from {submittable_path} ({len(grouped)} available)."
    )
    return seeds, benchmarks


def _synthesized_autoworkshop_seed(benchmark: dict[str, Any], root: str, seed_expr: str) -> dict[str, Any]:
    params = benchmark.get("params") if isinstance(benchmark.get("params"), dict) else {}
    return {
        "alpha_id": params.get("seed_alpha_id"),
        "candidate_id": "autoworkshop_seed_" + _stable_id(seed_expr or root),
        "expression": seed_expr or root,
        "status": "done",
        "search_lane": "autoworkshop_seed",
        "dataset_id": benchmark.get("dataset_id"),
        "category": benchmark.get("category"),
        "fields": benchmark.get("fields") or [],
        "cluster_ids": benchmark.get("cluster_ids") or [],
        "settings": benchmark.get("settings") or benchmark.get("settings_override"),
        "is_sharpe": params.get("seed_sharpe"),
        "is_fitness": params.get("seed_fitness"),
        "is_turnover": benchmark.get("is_turnover"),
        "checks": [],
        "params": {"root_expression": root, "original_expression": seed_expr or root},
    }


def load_probe_prior(path: Path) -> dict[str, dict[str, Any]]:
    global _FIELD_PROBE_KEYS
    _FIELD_PROBE_KEYS = set()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("cluster_priors") or []:
        cluster_id = str(row.get("cluster_id") or "")
        if cluster_id:
            out[cluster_id] = row
        field = str(row.get("field") or "")
        if field:
            _FIELD_PROBE_KEYS.add(field)
            existing = out.get(f"field::{field}")
            if existing is None or (_float(row.get("prior_score")) or -999.0) > (_float(existing.get("prior_score")) or -999.0):
                out[f"field::{field}"] = row
    return out


def preflight_llm(api_key: str | None = None) -> None:
    """Fail before Brain login/PnL work when LLM dependencies or key are missing."""
    from core import llm

    try:
        import openai  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OpenAI package is missing in the active Python environment.\n"
            "Use the ibkr env Python, for example:\n"
            "  /home/justinjj_/miniconda3/envs/ibkr/bin/python -m tools.test_probe_semantic_workshop ...\n"
            "or install the package into the active env:\n"
            "  python -m pip install openai"
        ) from exc
    if not llm.load_api_key(api_key):
        raise SystemExit(
            "No OpenAI API key found. Set OPENAI_API_KEY, pass --api-key, "
            "or place the key in openaiapi.txt."
        )


def build_probe_card(seed: dict[str, Any], probe_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    clusters = [str(cluster) for cluster in (seed.get("cluster_ids") or []) if cluster]
    cluster_id = clusters[0] if clusters else ""
    row = probe_index.get(cluster_id)
    match_basis = "cluster"
    seed_fields = _seed_fields(seed)
    if row is None:
        for field in seed_fields:
            row = probe_index.get(f"field::{field}")
            if row is not None:
                cluster_id = str(row.get("cluster_id") or cluster_id)
                match_basis = f"field:{field}"
                break
    if row is None and not seed_fields:
        expr = str(seed.get("expression") or "")
        for key, candidate in probe_index.items():
            if not key.startswith("field::"):
                continue
            field = key.split("::", 1)[1]
            if field and re.search(rf"\b{re.escape(field)}\b", expr):
                row = candidate
                cluster_id = str(candidate.get("cluster_id") or cluster_id)
                match_basis = f"expression_field:{field}"
                break
    if row is None:
        return {
            "cluster_id": cluster_id,
            "computed_trust": "missing",
            "seed_fields": seed_fields,
            "trust_reason": "no probe prior row matched seed primary cluster or fields",
        }
    card = {
        "cluster_id": cluster_id,
        "match_basis": match_basis,
        "seed_fields": seed_fields,
        "field": row.get("field"),
        "probe_family": row.get("probe_family"),
        "probe_template": row.get("probe_template"),
        "preferred_direction": row.get("preferred_direction"),
        "signal_bucket": row.get("signal_bucket"),
        "probe_quality": _float(row.get("probe_quality")),
        "prior_score": _float(row.get("prior_score")),
        "is_fitness": _float(row.get("is_fitness")),
        "is_sharpe": _float(row.get("is_sharpe")),
        "is_turnover": _float(row.get("is_turnover")),
        "max_concentration": _float(row.get("max_concentration")),
        "weak_plus_rate": _float(row.get("weak_plus_rate")),
        "best_probe_expression": row.get("expression"),
    }
    trust, reason = compute_probe_trust(seed, card)
    card["computed_trust"] = trust
    card["trust_reason"] = reason
    return card


def compute_probe_trust(seed: dict[str, Any], card: dict[str, Any]) -> tuple[str, str]:
    quality = _float(card.get("probe_quality")) or 0.0
    bucket = str(card.get("signal_bucket") or "")
    max_conc = _float(card.get("max_concentration"))
    seed_strength = max(abs(_float(seed.get("is_fitness")) or 0.0), abs(_float(seed.get("is_sharpe")) or 0.0))
    if max_conc is not None and max_conc >= 0.40:
        return "contradictory", "best probe had extreme concentration"
    if bucket == "background_noise" or (quality < 0.50 and seed_strength >= 1.0):
        return "contradictory", "probe evidence is weak relative to seed strength"

    family = str(card.get("probe_family") or "")
    seed_ops = expression_ops(str(seed.get("expression") or ""))
    family_ops = FAMILY_OPERATOR_HINTS.get(family, set())
    same_field = bool(card.get("field") and card.get("field") in set(_seed_fields(seed)))
    if seed_ops & family_ops:
        if same_field:
            return "high", f"seed uses probe-family operator(s): {sorted(seed_ops & family_ops)}"
        return "medium", f"seed operator family aligns but uses a sibling field: {sorted(seed_ops & family_ops)}"
    if seed_ops & RELATED_OPERATOR_HINTS.get(family, set()):
        return "medium", "seed uses related transform family"
    return "low", "same primary cluster but seed expression does not match probe transform family"


def _seed_fields(seed: dict[str, Any]) -> list[str]:
    fields = [str(field) for field in (seed.get("fields") or []) if field]
    if fields:
        return fields
    expr = str(seed.get("expression") or "")
    # Fallback for older/autoworkshop seed records that lack field metadata.
    fields = []
    for key in sorted(k for k in globals().get("_FIELD_PROBE_KEYS", set())):
        if re.search(rf"\b{re.escape(key)}\b", expr):
            fields.append(key)
    return fields


FAMILY_OPERATOR_HINTS: dict[str, set[str]] = {
    "level": {"rank", "zscore", "winsorize", "scale"},
    "relative_value": {"group_rank", "group_zscore", "group_neutralize", "group_scale"},
    "time_series_rank": {"ts_rank"},
    "time_series_zscore": {"ts_zscore"},
    "smooth_level": {"ts_mean", "ts_decay_linear", "ts_backfill", "group_backfill"},
    "delta_reversal": {"ts_delta", "last_diff_value", "ts_av_diff"},
    "self_residual": {"ts_regression", "ts_corr", "ts_covariance"},
}

RELATED_OPERATOR_HINTS: dict[str, set[str]] = {
    "level": {"group_rank", "group_zscore"},
    "relative_value": {"rank", "zscore", "ts_rank", "ts_zscore"},
    "time_series_rank": {"rank", "group_rank", "ts_zscore"},
    "time_series_zscore": {"zscore", "group_zscore", "ts_rank"},
    "smooth_level": {"ts_rank", "ts_zscore", "kth_element"},
    "delta_reversal": {"reverse", "ts_rank", "group_rank"},
    "self_residual": {"-", "subtract", "ts_av_diff"},
}


def expression_ops(expr: str) -> set[str]:
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
    infix = {op for op in ("+", "-", "*", "/") if op in expr}
    return tokens | infix


def compact_probe_card(card: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "cluster_id",
        "match_basis",
        "seed_fields",
        "field",
        "probe_family",
        "probe_template",
        "preferred_direction",
        "signal_bucket",
        "probe_quality",
        "prior_score",
        "is_fitness",
        "is_sharpe",
        "is_turnover",
        "max_concentration",
        "weak_plus_rate",
        "best_probe_expression",
        "computed_trust",
        "trust_reason",
    )
    return {key: card.get(key) for key in keys if key in card}


def write_summary(
    *,
    seeds: list[dict[str, Any]],
    arm_results: dict[str, ArmResult],
    probe_cards: dict[str, dict[str, Any]],
    run_dir: Path,
    autoworkshop_benchmarks: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = [
        "Probe + semantic workshop A/B summary",
        "primary: admission_brain_score_delta; fallback: admission_delta_pool_sharpe",
        "",
    ]
    records_by_arm_root: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for arm, result in arm_results.items():
        for record in result.records:
            params = record.get("params") if isinstance(record.get("params"), dict) else {}
            root = str(params.get("root_expression") or params.get("original_expression") or "")
            if root:
                records_by_arm_root.setdefault((arm, root), []).append(record)

    for seed in seeds:
        root = semantic_workshop._root_expression(seed)
        baseline = _top_record(records_by_arm_root.get(("baseline", root), []))
        probe = _top_record(records_by_arm_root.get(("probe_diagnostic", root), []))
        autoworkshop = (autoworkshop_benchmarks or {}).get(root)
        semantic_best = _top_record([record for record in (baseline, probe) if record is not None])
        base_brain, base_pool = record_score_addition(baseline or {})
        probe_brain, probe_pool = record_score_addition(probe or {})
        semantic_brain, semantic_pool = record_score_addition(semantic_best or {})
        auto_brain, auto_pool = record_score_addition(autoworkshop or {})
        winner = _winner(baseline, probe)
        card = probe_cards.get(_seed_key(seed), {})
        row = {
            "seed_alpha_id": seed.get("alpha_id"),
            "seed_expression": seed.get("expression"),
            "seed_sharpe": seed.get("is_sharpe"),
            "seed_fitness": seed.get("is_fitness"),
            "winner": winner,
            "score_delta_lift": _diff(probe_brain, base_brain),
            "fallback_delta_lift": _diff(probe_pool, base_pool),
            "semantic_vs_autoworkshop_score_lift": _diff(semantic_brain, auto_brain),
            "semantic_vs_autoworkshop_fallback_lift": _diff(semantic_pool, auto_pool),
            "probe_relevance": card.get("computed_trust"),
            "probe_use": _diagnostic_probe_use(arm_results.get("probe_diagnostic"), root),
            "baseline": _summary_record(baseline),
            "probe_diagnostic": _summary_record(probe),
            "autoworkshop": _summary_record(autoworkshop),
        }
        rows.append(row)
        lines.append(f"Seed: {str(seed.get('expression') or '')[:140]}")
        lines.append(
            f"  winner={winner} score_lift={_fmt(row['score_delta_lift'], 0)} "
            f"fallback_lift={_fmt(row['fallback_delta_lift'])} "
            f"probe_relevance={row['probe_relevance']} probe_use={row['probe_use']}"
        )
        lines.append(f"  baseline: {_record_text(baseline)}")
        lines.append(f"  probe:    {_record_text(probe)}")
        if autoworkshop is not None:
            lines.append(
                f"  autoworkshop: {_record_text(autoworkshop)} "
                f"semantic_minus_auto={_fmt(row['semantic_vs_autoworkshop_score_lift'], 0)}"
            )
        lines.append("")

    _write_jsonl(rows, run_dir / "summary.jsonl")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    return rows


def _top_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    passers = [record for record in records if is_passing(record)]
    return sorted(passers or records, key=_rank_key, reverse=True)[0]


def _rank_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
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


def _winner(baseline: dict[str, Any] | None, probe: dict[str, Any] | None) -> str:
    baseline_ok = baseline is not None and is_passing(baseline)
    probe_ok = probe is not None and is_passing(probe)
    if not baseline_ok and not probe_ok:
        return "none"
    if not baseline_ok:
        return "probe_diagnostic"
    if not probe_ok:
        return "baseline"
    b_key = _rank_key(baseline)
    p_key = _rank_key(probe)
    if p_key > b_key:
        return "probe_diagnostic"
    if b_key > p_key:
        return "baseline"
    return "tie"


def _summary_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    brain, pool = record_score_addition(record)
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    return {
        "alpha_id": record.get("alpha_id"),
        "candidate_id": record.get("candidate_id"),
        "expression": record.get("expression"),
        "round": params.get("semantic_round"),
        "is_sharpe": record.get("is_sharpe"),
        "is_fitness": record.get("is_fitness"),
        "is_turnover": record.get("is_turnover"),
        "admission_brain_score_delta": brain,
        "admission_delta_pool_sharpe": pool,
        "max_self_correlation": record.get("admission_global_max_corr", record.get("max_self_correlation")),
        "checks_passing": is_passing(record),
        "failing_checks": _failing_checks(record),
        "cross_arm_duplicate_of": params.get("cross_arm_duplicate_of") or record.get("cross_arm_duplicate_of"),
        "semantic_template": params.get("semantic_template"),
        "diagnostic_probe_relevance": params.get("diagnostic_probe_relevance"),
        "diagnostic_probe_use": params.get("diagnostic_probe_use"),
    }


def _record_text(record: dict[str, Any] | None) -> str:
    if record is None:
        return "N/A"
    brain, pool = record_score_addition(record)
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    dup = params.get("cross_arm_duplicate_of") or record.get("cross_arm_duplicate_of")
    dup_text = f" duplicate_of={dup}" if dup else ""
    return (
        f"id={record.get('alpha_id') or '?'} score={_fmt(brain, 0)} pool={_fmt(pool)} "
        f"fit={_fmt(record.get('is_fitness'))} sh={_fmt(record.get('is_sharpe'))} "
        f"turn={_fmt(record.get('is_turnover'))}{dup_text} "
        f"{str(record.get('expression') or '')[:110]}"
    )


def _diagnostic_probe_use(result: ArmResult | None, root: str) -> str | None:
    if result is None:
        return None
    for row in result.diagnostics:
        if row.get("root_expression") == root:
            diagnosis = row.get("diagnosis") if isinstance(row.get("diagnosis"), dict) else {}
            return diagnosis.get("probe_use")
    return None


def _diagnostic_row(
    arm: str,
    seed: dict[str, Any],
    round_index: int,
    diagnosis: dict[str, Any],
    probe_card: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "probe_semantic_ab",
        "arm": arm,
        "round": round_index,
        "seed_alpha_id": seed.get("alpha_id"),
        "root_expression": semantic_workshop._root_expression(seed),
        "diagnosis": diagnosis,
        "probe_card": compact_probe_card(probe_card or {}) if probe_card else None,
    }


def _generation_row(
    arm: str,
    seed: dict[str, Any],
    round_index: int,
    result: GenerationResult,
    filtered_variants: list[GeneratedVariant],
    probe_card: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "probe_semantic_ab",
        "arm": arm,
        "round": round_index,
        "seed_alpha_id": seed.get("alpha_id"),
        "root_expression": semantic_workshop._root_expression(seed),
        "raw_variant_count": len(result.variants),
        "filtered_variant_count": len(filtered_variants),
        "raw_variants": [variant.expression for variant in result.variants],
        "filtered_expressions": [variant.expression for variant in filtered_variants],
        "raw_response": result.raw_response,
        "probe_card": compact_probe_card(probe_card or {}) if probe_card else None,
    }


def _load_seed_file(path: str) -> list[dict[str, Any]]:
    return load_records([path], tag_source=False)


def _manifest(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    input_paths: list[str],
    seeds_path: Path,
    seed_count: int,
    probe_prior_count: int,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "arms": list(ARMS),
        "rounds": args.rounds,
        "seed_count": seed_count,
        "seed_source": args.seed_source,
        "seeds_path": str(seeds_path),
        "input_paths": input_paths,
        "probe_prior": args.probe_prior,
        "probe_prior_count": probe_prior_count,
        "autoworkshop_submittable": args.autoworkshop_submittable,
        "model": args.model,
        "batch_size": args.batch_size,
        "dry_run": bool(args.dry_run),
        "stub_agents": bool(args.stub_agents),
        "no_submit": bool(args.no_submit or args.stub_agents),
        "argv": sys.argv,
    }


def _seed_key(seed: dict[str, Any]) -> str:
    return str(seed.get("expression") or seed.get("alpha_id") or seed.get("candidate_id") or "")


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:16]


def _failing_checks(record: dict[str, Any]) -> list[str]:
    return [
        str(check.get("name"))
        for check in record.get("checks") or []
        if check.get("result") != "PASS" and check.get("name") != "SELF_CORRELATION"
    ]


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_json(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
