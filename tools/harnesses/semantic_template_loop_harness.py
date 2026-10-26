"""Pilot semantic-template loop on a previous probe/semantic A/B batch.

Default mode builds artifacts only.  Add --submit to simulate the rendered
template plan on Brain.
"""

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

from wq_alpha.agents.semantic_templates import (  # noqa: E402
    TemplateArchitectAgent,
    TemplateFieldSelectorAgent,
    TemplateRouterAgent,
)
from wq_alpha.brain.submit import run as submit_plan  # noqa: E402
from wq_alpha.semantic_templates.core import (  # noqa: E402
    DEFAULT_CLUSTER_PATH,
    DEFAULT_PROBE_LOG,
    build_architect_evidence_groups,
    build_template_plan,
    chunk_rows,
    load_cluster_catalog,
    load_probe_summaries,
    normalize_route,
    normalize_template,
    read_jsonl,
    merge_route_candidate_metadata,
    route_candidates_from_field_selections,
    select_template_fields_with_agent,
    summarize_template_feedback,
    validate_route,
    validate_template,
    write_jsonl,
)


DEFAULT_AB_SUMMARY = ROOT / "artifacts" / "probe_semantic_ab" / "20260512T200632Z" / "summary.jsonl"
DEFAULT_AB_CONTEXT = (
    ROOT / "artifacts" / "probe_semantic_ab" / "20260512T200632Z" / "baseline" / "sim.jsonl",
    ROOT / "artifacts" / "probe_semantic_ab" / "20260512T200632Z" / "probe_diagnostic" / "sim.jsonl",
    ROOT / "artifacts" / "workshop" / "workshop_submittable.jsonl",
    ROOT / "artifacts" / "workshop" / "workshop_sim.jsonl",
    ROOT / "artifacts" / "sim" / "master.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-summary", default=str(DEFAULT_AB_SUMMARY))
    parser.add_argument("--context-log", nargs="*", default=[str(path) for path in DEFAULT_AB_CONTEXT])
    parser.add_argument("--probe-log", default=str(DEFAULT_PROBE_LOG))
    parser.add_argument("--clusters", default=str(DEFAULT_CLUSTER_PATH))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--max-roots", type=int, default=5)
    parser.add_argument("--max-templates", type=int, default=5)
    parser.add_argument("--catalog-limit", type=int, default=80, help="0 means all clusters, still chunked.")
    parser.add_argument("--selector-chunk-size", type=int, default=40)
    parser.add_argument("--route-chunk-size", type=int, default=10)
    parser.add_argument("--max-selections-per-chunk", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=2)
    parser.add_argument("--max-candidates-per-route", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--submit", action="store_true", help="Submit the rendered plan to Brain.")
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-submit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts" / "semantic_template_tests" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    templates_path = out_dir / "templates.jsonl"
    selections_path = out_dir / "field_selections.jsonl"
    routes_path = out_dir / "routes.jsonl"
    plan_path = out_dir / "template_plan.jsonl"
    sim_log = out_dir / "template_plan_sim.jsonl"
    feedback_path = out_dir / "template_feedback.jsonl"
    manifest_path = out_dir / "manifest.json"

    clusters = load_cluster_catalog(args.clusters)
    probe_records = read_jsonl([args.probe_log])
    probe_summaries = load_probe_summaries(probe_records, clusters)
    context_records = read_jsonl(args.context_log)

    templates = _create_templates(args, probe_records, context_records)
    write_jsonl(templates, templates_path)
    print(f"Templates: {templates_path} ({len(templates)})")

    selections = _select_fields(args, templates, clusters, probe_summaries)
    write_jsonl(selections, selections_path)
    print(f"Field selections: {selections_path} ({len(selections)})")

    routes = _route_selections(args, templates, selections, clusters, probe_summaries)
    write_jsonl(routes, routes_path)
    print(f"Routes: {routes_path} ({len(routes)})")

    known_records = context_records + probe_records
    plan = build_template_plan(
        templates,
        routes,
        known_records,
        max_windows=args.max_windows,
        max_groups=args.max_groups,
        max_candidates_per_route=args.max_candidates_per_route,
    )
    write_jsonl(plan, plan_path)
    print(f"Plan: {plan_path} ({len(plan)})")

    manifest = {
        "run_id": run_id,
        "ab_summary": args.ab_summary,
        "context_log": args.context_log,
        "probe_log": args.probe_log,
        "clusters": args.clusters,
        "templates": str(templates_path),
        "field_selections": str(selections_path),
        "routes": str(routes_path),
        "plan": str(plan_path),
        "sim_log": str(sim_log),
        "feedback": str(feedback_path),
        "submit": bool(args.submit),
        "model": args.model,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if not args.submit:
        print("No Brain simulation submitted. Re-run with --submit to simulate this plan.")
        print(f"Submit command: python -m wq_alpha.brain.submit --plan {plan_path} --sim-log {sim_log} --batch-size {args.batch_size} --credentials {args.credentials}")
        return

    records = submit_plan(
        str(plan_path),
        str(sim_log),
        batch_size=args.batch_size,
        credentials=args.credentials,
        max_candidates=args.max_submit,
    )
    feedback = summarize_template_feedback(records)
    write_jsonl(feedback, feedback_path)
    print(f"Feedback: {feedback_path} ({len(feedback)})")


def _create_templates(args: argparse.Namespace, probe_records: list[dict[str, Any]], context_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed_records = read_jsonl([args.ab_summary])
    groups = build_architect_evidence_groups(
        seed_records,
        context_records,
        probe_records,
        max_groups=args.max_roots,
    )
    print(f"Architect groups: {len(groups)}")
    agent = TemplateArchitectAgent(api_key=args.api_key, model=args.model)
    templates: list[dict[str, Any]] = []
    for idx, group in enumerate(groups, 1):
        print(f"Architect {idx}/{len(groups)}: {str(group.get('source_root_expression') or '')[:90]}")
        result = agent.create_templates(group)
        raw_templates = result.templates or [_reject_template_from_group(group)]
        for raw in raw_templates:
            template = normalize_template(raw)
            template["source_root_expression"] = template.get("source_root_expression") or group.get("source_root_expression")
            errors = validate_template(template)
            if errors:
                print(f"  reject malformed template: {errors}")
                continue
            templates.append(template)
            print(f"  {template['status']} {template['template_id']}")
    return templates


def _reject_template_from_group(group: dict[str, Any]) -> dict[str, Any]:
    promoted = group.get("promoted") or []
    best = promoted[0] if promoted else {}
    return {
        "template_id": "",
        "status": "reject",
        "semantic_identity": "No parseable reusable template returned for this source root.",
        "expression_pattern": "",
        "slots": {
            "main_field": {
                "role": "unresolved",
                "required": False,
                "compatible_field_types": [],
                "compatible_probe_families": [],
                "expected_direction": "either",
                "allow_sign_flip": False,
            }
        },
        "default_windows": [],
        "default_groups": [],
        "default_settings": {},
        "invalid_mutations": ["Template architect returned no parseable JSON object for this root."],
        "source_alpha_ids": [str(row.get("alpha_id")) for row in promoted if row.get("alpha_id")],
        "source_root_expression": group.get("source_root_expression") or best.get("root_expression") or "",
        "robustness_status": "missing",
        "admission_brain_score_delta": best.get("admission_brain_score_delta"),
        "overfit_risk": "high",
    }


def _select_fields(
    args: argparse.Namespace,
    templates: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    active = [row for row in templates if row.get("status") == "active"][: args.max_templates]
    selector = TemplateFieldSelectorAgent(api_key=args.api_key, model=args.model)
    selections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for template in active:
        for selection in select_template_fields_with_agent(
            template,
            selector,
            clusters,
            probe_summaries,
            catalog_limit=args.catalog_limit,
            chunk_size=args.selector_chunk_size,
            slot_shortlist_limit=20,
            max_selections_per_chunk=args.max_selections_per_chunk,
        ):
            key = selection["selector_candidate_id"]
            if key in seen:
                continue
            seen.add(key)
            selections.append(selection)
    return selections


def _route_selections(
    args: argparse.Namespace,
    templates: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    active = [row for row in templates if row.get("status") == "active"][: args.max_templates]
    router = TemplateRouterAgent(api_key=args.api_key, model=args.model)
    routes: list[dict[str, Any]] = []
    for template in active:
        candidates = route_candidates_from_field_selections(
            template,
            selections,
            clusters,
            probe_summaries,
            limit=80,
        )
        print(f"Route {template['template_id']}: candidates={len(candidates)}")
        for chunk in chunk_rows(candidates, max(1, args.route_chunk_size)):
            by_selector = {str(row.get("selector_candidate_id")): row for row in chunk if row.get("selector_candidate_id")}
            by_cluster = {str(row.get("cluster_id")): row for row in chunk}
            result = router.route(template, chunk)
            for raw in result.routes:
                candidate = by_selector.get(str(raw.get("selector_candidate_id") or "")) or by_cluster.get(str(raw.get("cluster_id") or ""))
                route = normalize_route(merge_route_candidate_metadata(raw, candidate), template)
                errors = validate_route(route)
                if errors:
                    print(f"  reject malformed route: {errors}")
                    continue
                routes.append(route)
    return routes


if __name__ == "__main__":
    main()
