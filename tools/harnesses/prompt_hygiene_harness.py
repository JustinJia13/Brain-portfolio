"""Render and test semantic-workshop prompt hygiene on frozen seeds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_discovery.grammar_method.novelty import fingerprint_jaccard, normalized_fingerprint  # noqa: E402
from alpha_enhancement.agents.semantic_workshop import (  # noqa: E402
    SemanticDiagnosticAgent,
    SemanticGeneratorAgent,
    _diagnostic_message,
    _generator_message,
)
from core.brain.submit import DEFAULT_SETTINGS  # noqa: E402
from alpha_enhancement.workshop.core import find_passing_examples, record_workshop_key  # noqa: E402
from alpha_enhancement.workshop.probe_cards import build_probe_card, load_probe_prior  # noqa: E402
from alpha_enhancement.workshop.records import load_records  # noqa: E402
from alpha_enhancement.workshop.semantic import (  # noqa: E402
    DEFAULT_PROBE_PRIOR,
    _filter_variants,
    _input_paths,
    _select_seed_records,
    _variants_to_candidates,
)
from alpha_enhancement.workshop.semantic_context import (  # noqa: E402
    DEFAULT_MAX_CONTEXT_CHARS,
    SemanticContextBuilder,
)
from alpha_enhancement.workshop.simulation import submit_candidate_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("render-only", "llm-dry-run", "submit"), default="render-only")
    parser.add_argument("--sim-log", nargs="*", default=[])
    parser.add_argument("--source", choices=("grammar", "all"), default="grammar")
    parser.add_argument("--seed-id", nargs="*", default=[])
    parser.add_argument("--seed-expr", nargs="*", default=[])
    parser.add_argument("--max-seeds", type=int, default=5)
    parser.add_argument("--lane", nargs="*", choices=("repair", "enhance"), default=["repair", "enhance"])
    parser.add_argument("--exclude-expr", nargs="*", default=[])
    parser.add_argument("--bypass-correlation-check", action="store_true")
    parser.add_argument("--no-score-seeds", dest="score_seeds", action="store_false")
    parser.set_defaults(score_seeds=False)
    parser.add_argument("--probe-prior", default=str(DEFAULT_PROBE_PRIOR))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-submit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts" / "prompt_hygiene" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_packets_path = out_dir / "prompt_packets.jsonl"
    context_reports_path = out_dir / "context_reports.jsonl"
    llm_outputs_path = out_dir / "llm_outputs.jsonl"
    parse_report_path = out_dir / "parse_report.jsonl"
    sim_log_path = out_dir / "sim.jsonl"
    summary_path = out_dir / "summary.txt"

    input_paths = _input_paths(args)
    all_records = load_records(input_paths, tag_source=True)
    seeds = _select_seed_records(all_records, args)
    probe_index = load_probe_prior(args.probe_prior)
    known_expr_keys = {record_workshop_key(record) for record in all_records if record.get("expression")}

    diagnostic_agent = None
    generator_agent = None
    if args.mode in {"llm-dry-run", "submit"}:
        diagnostic_agent = SemanticDiagnosticAgent(api_key=args.api_key, model=args.model)
        generator_agent = SemanticGeneratorAgent(api_key=args.api_key, model=args.model)

    submitted_records: list[dict[str, Any]] = []
    lines = [
        "Prompt hygiene harness",
        f"mode={args.mode}",
        f"seeds={len(seeds)}",
        f"out_dir={out_dir}",
        "",
    ]

    for seed_index, seed in enumerate(seeds, 1):
        probe_card = build_probe_card(seed, probe_index) if probe_index else {
            "computed_trust": "missing",
            "trust_reason": "probe prior file unavailable",
        }
        builder = SemanticContextBuilder(
            all_records,
            known_expr_keys=known_expr_keys,
            max_context_chars=args.max_context_chars,
        )
        diagnostic_context = builder.diagnostic_packet(seed=seed, probe_card=probe_card)
        legacy_context = find_passing_examples(all_records, seed, n=8)
        legacy_prompt = _diagnostic_message(
            seed,
            prior_diagnosis=None,
            round_results=[],
            robustness_card=None,
            context_records=legacy_context,
            probe_card=probe_card,
        )
        _append_jsonl(prompt_packets_path, {
            "seed_index": seed_index,
            "seed_alpha_id": seed.get("alpha_id"),
            "phase": "diagnostic",
            "new_packet": diagnostic_context.packet,
            "legacy_prompt": legacy_prompt if args.include_legacy else None,
        })
        _append_jsonl(context_reports_path, {
            "seed_index": seed_index,
            "seed_alpha_id": seed.get("alpha_id"),
            "phase": "diagnostic",
            "report": diagnostic_context.report,
        })

        if args.mode == "render-only":
            continue

        diagnosis_result = diagnostic_agent.diagnose(seed_record=seed, probe_card=probe_card, context_packet=diagnostic_context.packet)
        diagnosis = diagnosis_result.payload
        generator_context = builder.generator_packet(
            seed=seed,
            diagnosis=diagnosis,
            round_index=1,
            target_count="exactly 6",
            previous_results=[],
        )
        legacy_generator_prompt = _generator_message(
            seed,
            diagnosis,
            round_index=1,
            target_count="exactly 6",
            previous_results=[],
            robustness_card=None,
            known_expressions=[],
        )
        _append_jsonl(prompt_packets_path, {
            "seed_index": seed_index,
            "seed_alpha_id": seed.get("alpha_id"),
            "phase": "generator",
            "new_packet": generator_context.packet,
            "legacy_prompt": legacy_generator_prompt if args.include_legacy else None,
        })
        _append_jsonl(context_reports_path, {
            "seed_index": seed_index,
            "seed_alpha_id": seed.get("alpha_id"),
            "phase": "generator",
            "report": generator_context.report,
        })
        generation_result = generator_agent.generate(
            seed_record=seed,
            diagnosis=diagnosis,
            round_index=1,
            context_packet=generator_context.packet,
        )
        variants = _filter_variants(generation_result.variants, round_index=1)
        metrics = _variant_metrics(seed, variants)
        _append_jsonl(llm_outputs_path, {
            "seed_index": seed_index,
            "seed_alpha_id": seed.get("alpha_id"),
            "diagnosis": diagnosis,
            "diagnostic_raw": diagnosis_result.raw_response,
            "generator_raw": generation_result.raw_response,
            "expressions": [variant.expression for variant in variants],
        })
        _append_jsonl(parse_report_path, {
            "seed_index": seed_index,
            "seed_alpha_id": seed.get("alpha_id"),
            "diagnostic_parse_success": bool(diagnosis.get("semantic_template")),
            "generator_parsed_count": len(variants),
            "exact_required_count_success": len(variants) == 6,
            **metrics,
        })

        lines.append(
            f"seed {seed_index}: id={seed.get('alpha_id') or '?'} "
            f"parsed={len(variants)} exact6={len(variants) == 6} "
            f"dup={metrics['duplicate_expression_count']} "
            f"max_jaccard={metrics['max_pairwise_jaccard']:.3f}"
        )

        if args.mode == "submit":
            candidates = _variants_to_candidates(
                variants,
                seed,
                diagnosis,
                round_index=1,
                known_expr_keys=known_expr_keys,
                probe_card=probe_card,
            )
            if args.max_submit > 0:
                remaining = max(0, args.max_submit - len(submitted_records))
                candidates = candidates[:remaining]
            if candidates:
                from core.brain.api import sign_in

                session = sign_in(args.credentials)
                submitted = submit_candidate_batch(
                    candidates,
                    session,
                    sim_log_path,
                    dict(DEFAULT_SETTINGS),
                    args.credentials,
                    args.batch_size,
                    portfolio_alpha_data=None,
                )
                submitted_records.extend(submitted)
            if args.max_submit > 0 and len(submitted_records) >= args.max_submit:
                break

    if submitted_records:
        lines.append("")
        lines.append("Submitted results:")
        for record in submitted_records:
            brain = record.get("admission_brain_score_delta")
            pool = record.get("admission_delta_pool_sharpe")
            lines.append(
                f"  id={record.get('alpha_id') or '?'} fit={_fmt(record.get('is_fitness'))} "
                f"sh={_fmt(record.get('is_sharpe'))} score={_fmt(brain, 0)} pool={_fmt(pool)} "
                f"{str(record.get('expression') or '')[:100]}"
            )

    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Prompt packets: {prompt_packets_path}")
    print(f"Context reports: {context_reports_path}")
    if args.mode in {"llm-dry-run", "submit"}:
        print(f"LLM outputs: {llm_outputs_path}")
        print(f"Parse report: {parse_report_path}")
    if args.mode == "submit":
        print(f"Sim log: {sim_log_path}")
    print(f"Summary: {summary_path}")


def _variant_metrics(seed: dict[str, Any], variants: list[Any]) -> dict[str, Any]:
    expressions = [variant.expression for variant in variants if variant.expression]
    duplicate_count = len(expressions) - len(set(expressions))
    fields = seed.get("fields") or []
    fps = [normalized_fingerprint(expr, fields) for expr in expressions]
    jaccards: list[float] = []
    for i, fp1 in enumerate(fps):
        for fp2 in fps[i + 1:]:
            jaccards.append(fingerprint_jaccard(fp1, fp2))
    return {
        "duplicate_expression_count": duplicate_count,
        "max_pairwise_jaccard": max(jaccards) if jaccards else 0.0,
        "avg_pairwise_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


if __name__ == "__main__":
    main()
