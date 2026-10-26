"""Semantic template architect/router CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wq_alpha.agents.semantic_templates import TemplateArchitectAgent, TemplateFieldSelectorAgent, TemplateRouterAgent
from wq_alpha.semantic_templates.core import (
    DEFAULT_CLUSTER_PATH,
    DEFAULT_CONTEXT_PATHS,
    DEFAULT_FEEDBACK_JSONL,
    DEFAULT_KNOWN_LOGS,
    DEFAULT_PLAN_JSONL,
    DEFAULT_PROBE_LOG,
    DEFAULT_ROUTES_JSONL,
    DEFAULT_SELECTIONS_JSONL,
    DEFAULT_SEED_PATHS,
    DEFAULT_TEMPLATES_JSONL,
    build_architect_evidence_groups,
    build_selector_catalog,
    build_selector_slot_shortlists,
    build_template_plan,
    load_cluster_catalog,
    load_probe_summaries,
    normalize_route,
    normalize_template,
    read_jsonl,
    route_candidates_for_template,
    route_template_fields_with_agent,
    select_template_fields_with_agent,
    summarize_template_feedback,
    validate_route,
    validate_template,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    architect = sub.add_parser("architect", help="Create semantic templates from promoted alpha roots.")
    architect.add_argument("--seed-log", nargs="*", default=[str(path) for path in DEFAULT_SEED_PATHS])
    architect.add_argument("--context-log", nargs="*", default=[str(path) for path in DEFAULT_CONTEXT_PATHS])
    architect.add_argument("--probe-log", default=str(DEFAULT_PROBE_LOG))
    architect.add_argument("--out", default=str(DEFAULT_TEMPLATES_JSONL))
    architect.add_argument("--max-roots", type=int, default=5)
    architect.add_argument("--api-key", default=None)
    architect.add_argument("--model", default="gpt-5.5")
    architect.add_argument("--dry-run", action="store_true")

    route = sub.add_parser("route", help="Route semantic templates to compatible field clusters.")
    route.add_argument("--templates", default=str(DEFAULT_TEMPLATES_JSONL))
    route.add_argument("--selections", default="", help="Optional grounded field selections JSONL to route instead of prefilter candidates.")
    route.add_argument("--clusters", default=str(DEFAULT_CLUSTER_PATH))
    route.add_argument("--probe-log", default=str(DEFAULT_PROBE_LOG))
    route.add_argument("--out", default=str(DEFAULT_ROUTES_JSONL))
    route.add_argument("--max-templates", type=int, default=5)
    route.add_argument("--candidate-limit", type=int, default=80)
    route.add_argument("--chunk-size", type=int, default=20)
    route.add_argument("--api-key", default=None)
    route.add_argument("--model", default="gpt-5.5")
    route.add_argument("--dry-run", action="store_true")
    route.add_argument(
        "--code-only",
        action="store_true",
        help="Write deterministic prefilter routes without calling the LLM. Useful for smoke tests.",
    )

    select = sub.add_parser("select-fields", help="Use an LLM to select grounded field bindings for templates.")
    select.add_argument("--templates", default=str(DEFAULT_TEMPLATES_JSONL))
    select.add_argument("--clusters", default=str(DEFAULT_CLUSTER_PATH))
    select.add_argument("--probe-log", default=str(DEFAULT_PROBE_LOG))
    select.add_argument("--out", default=str(DEFAULT_SELECTIONS_JSONL))
    select.add_argument("--max-templates", type=int, default=5)
    select.add_argument("--catalog-limit", type=int, default=0, help="0 means all clusters, still chunked.")
    select.add_argument("--chunk-size", type=int, default=40)
    select.add_argument("--include-members", type=int, default=4)
    select.add_argument("--slot-shortlist-limit", type=int, default=20)
    select.add_argument("--max-selections-per-chunk", type=int, default=5)
    select.add_argument("--api-key", default=None)
    select.add_argument("--model", default="gpt-5.5")
    select.add_argument("--dry-run", action="store_true")

    build = sub.add_parser("build-plan", help="Render routed semantic templates into a Brain submit plan.")
    build.add_argument("--templates", default=str(DEFAULT_TEMPLATES_JSONL))
    build.add_argument("--routes", default=str(DEFAULT_ROUTES_JSONL))
    build.add_argument("--known-log", nargs="*", default=[str(path) for path in DEFAULT_KNOWN_LOGS])
    build.add_argument("--out", default=str(DEFAULT_PLAN_JSONL))
    build.add_argument("--max-windows", type=int, default=2)
    build.add_argument("--max-groups", type=int, default=2)
    build.add_argument("--max-candidates-per-route", type=int, default=3)
    build.add_argument("--dry-run", action="store_true")

    feedback = sub.add_parser("feedback", help="Summarize semantic-template simulation feedback.")
    feedback.add_argument("--sim-log", nargs="*", default=[str(DEFAULT_PLAN_JSONL.with_name("template_plan_sim.jsonl"))])
    feedback.add_argument("--out", default=str(DEFAULT_FEEDBACK_JSONL))
    feedback.add_argument("--dry-run", action="store_true")

    validate = sub.add_parser("validate", help="Validate semantic template artifacts.")
    validate.add_argument("--templates", default=str(DEFAULT_TEMPLATES_JSONL))
    validate.add_argument("--routes", default=str(DEFAULT_ROUTES_JSONL))
    validate.add_argument("--plan", default=str(DEFAULT_PLAN_JSONL))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "architect":
        _cmd_architect(args)
    elif args.cmd == "select-fields":
        _cmd_select_fields(args)
    elif args.cmd == "route":
        _cmd_route(args)
    elif args.cmd == "build-plan":
        _cmd_build_plan(args)
    elif args.cmd == "feedback":
        _cmd_feedback(args)
    elif args.cmd == "validate":
        _cmd_validate(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


def _cmd_architect(args: argparse.Namespace) -> None:
    seed_records = read_jsonl(args.seed_log)
    context_records = read_jsonl(args.context_log)
    probe_records = read_jsonl([args.probe_log])
    groups = build_architect_evidence_groups(
        seed_records,
        context_records,
        probe_records,
        max_groups=args.max_roots,
    )
    print(f"Template architect groups: {len(groups)}")
    if args.dry_run:
        for idx, group in enumerate(groups[: args.max_roots], 1):
            promoted = group.get("promoted") or []
            best = promoted[0] if promoted else {}
            print(
                f"{idx}. root={str(group.get('source_root_expression') or '')[:100]} "
                f"best_id={best.get('alpha_id')} score={_score_label(best)} "
                f"fields={best.get('fields')}"
            )
        print("Dry run: no LLM call and no templates written.")
        return

    agent = TemplateArchitectAgent(api_key=args.api_key, model=args.model)
    templates: list[dict[str, Any]] = []
    for idx, group in enumerate(groups[: args.max_roots], 1):
        print(f"Architect {idx}/{len(groups[: args.max_roots])}: {str(group.get('source_root_expression') or '')[:90]}")
        result = agent.create_templates(group)
        raw_templates = result.templates
        if not raw_templates:
            raw_templates = [_empty_architect_reject(group)]
        for raw_template in raw_templates:
            template = normalize_template(raw_template)
            template["source_root_expression"] = template.get("source_root_expression") or group.get("source_root_expression")
            errors = validate_template(template)
            if errors:
                print(f"  reject malformed template: {errors}")
                continue
            templates.append(template)
            print(f"  {template['status']} {template['template_id']}: {template['semantic_identity'][:90]}")
    write_jsonl(templates, args.out)
    print(f"Wrote templates: {args.out} ({len(templates)})")


def _cmd_route(args: argparse.Namespace) -> None:
    templates = [normalize_template(row) for row in read_jsonl([args.templates])]
    templates = [row for row in templates if row.get("status") == "active"][: args.max_templates]
    clusters = load_cluster_catalog(args.clusters)
    probe_summaries = load_probe_summaries(read_jsonl([args.probe_log]), clusters)
    selections = read_jsonl([args.selections]) if args.selections else []
    print(f"Templates: {len(templates)}  clusters: {len(clusters)}  probe summaries: {len(probe_summaries)}")
    if args.dry_run:
        for template in templates:
            candidates = route_candidates_for_template(
                template,
                selections,
                clusters,
                probe_summaries,
                limit=args.candidate_limit,
            )
            n_chunks = (len(candidates) + max(1, args.chunk_size) - 1) // max(1, args.chunk_size)
            print(
                f"{template['template_id']}: candidates={len(candidates)} "
                f"chunk_size={args.chunk_size} chunks={n_chunks}"
            )
            for candidate in candidates[:3]:
                print(
                    f"  {candidate['code_prefilter_score']:.2f} {candidate['cluster_id']} "
                    f"field={candidate['champion_field'].get('field_id')} "
                    f"probe={candidate.get('probe_summary', {}).get('best_probe_family')}"
                )
        print("Dry run: no LLM call and no routes written.")
        return

    routes: list[dict[str, Any]] = []
    router = None if args.code_only else TemplateRouterAgent(api_key=args.api_key, model=args.model)
    for template in templates:
        routes.extend(
            route_template_fields_with_agent(
                template,
                selections,
                router,
                clusters,
                probe_summaries,
                candidate_limit=args.candidate_limit,
                chunk_size=args.chunk_size,
                code_only=args.code_only,
            )
        )
        print(f"Routed {template['template_id']}: {sum(1 for r in routes if r.get('template_id') == template['template_id'])}")
    write_jsonl(routes, args.out)
    print(f"Wrote routes: {args.out} ({len(routes)})")


def _cmd_select_fields(args: argparse.Namespace) -> None:
    templates = [normalize_template(row) for row in read_jsonl([args.templates])]
    templates = [row for row in templates if row.get("status") == "active"][: args.max_templates]
    clusters = load_cluster_catalog(args.clusters)
    probe_summaries = load_probe_summaries(read_jsonl([args.probe_log]), clusters)
    print(f"Templates: {len(templates)}  clusters: {len(clusters)}  probe summaries: {len(probe_summaries)}")
    if args.dry_run:
        for template in templates:
            catalog = build_selector_catalog(
                template,
                clusters,
                probe_summaries,
                limit=args.catalog_limit,
                include_members=args.include_members,
            )
            shortlists = build_selector_slot_shortlists(
                template,
                clusters,
                probe_summaries,
                limit=args.slot_shortlist_limit,
            )
            n_chunks = (len(catalog) + max(1, args.chunk_size) - 1) // max(1, args.chunk_size)
            print(f"{template['template_id']}: catalog={len(catalog)} chunks={n_chunks} shortlists={list(shortlists)}")
            for row in catalog[:3]:
                print(
                    f"  {row['cluster_id']} field={row['champion_field'].get('field_id')} "
                    f"slot_scores={row.get('slot_fit_scores')}"
                )
        print("Dry run: no LLM call and no selections written.")
        return

    selector = TemplateFieldSelectorAgent(api_key=args.api_key, model=args.model)
    selections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for template in templates:
        for selection in select_template_fields_with_agent(
            template,
            selector,
            clusters,
            probe_summaries,
            catalog_limit=args.catalog_limit,
            chunk_size=args.chunk_size,
            include_members=args.include_members,
            slot_shortlist_limit=args.slot_shortlist_limit,
            max_selections_per_chunk=args.max_selections_per_chunk,
        ):
            key = selection["selector_candidate_id"]
            if key in seen:
                continue
            seen.add(key)
            selections.append(selection)
    write_jsonl(selections, args.out)
    print(f"Wrote field selections: {args.out} ({len(selections)})")


def _cmd_build_plan(args: argparse.Namespace) -> None:
    templates = [normalize_template(row) for row in read_jsonl([args.templates])]
    routes = [normalize_route(row) for row in read_jsonl([args.routes])]
    known_records = read_jsonl(args.known_log)
    plan = build_template_plan(
        templates,
        routes,
        known_records,
        max_windows=args.max_windows,
        max_groups=args.max_groups,
        max_candidates_per_route=args.max_candidates_per_route,
    )
    print(f"Rendered semantic-template candidates: {len(plan)}")
    if args.dry_run:
        for row in plan[:10]:
            print(f"  {row['template_id']} {row['cluster_ids'][0]} {row['expression'][:120]}")
        print("Dry run: no plan written.")
        return
    write_jsonl(plan, args.out)
    print(f"Wrote plan: {args.out}")


def _cmd_feedback(args: argparse.Namespace) -> None:
    records = read_jsonl(args.sim_log)
    rows = summarize_template_feedback(records)
    print(f"Feedback rows: {len(rows)}")
    for row in rows[:10]:
        print(
            f"  {row['route_feedback_status']:<6} {row['template_id']} {row['cluster_id']} "
            f"score={_fmt(row.get('admission_brain_score_delta'))} "
            f"pool={_fmt(row.get('admission_delta_pool_sharpe'))} "
            f"fit={_fmt(row.get('best_fitness'))}"
        )
    if args.dry_run:
        print("Dry run: no feedback written.")
        return
    write_jsonl(rows, args.out)
    print(f"Wrote feedback: {args.out}")


def _cmd_validate(args: argparse.Namespace) -> None:
    templates = [normalize_template(row) for row in read_jsonl([args.templates])]
    routes = [normalize_route(row) for row in read_jsonl([args.routes])]
    plan = read_jsonl([args.plan])
    template_errors = [(row.get("template_id"), validate_template(row)) for row in templates]
    route_errors = [(row.get("template_id"), row.get("cluster_id"), validate_route(row)) for row in routes]
    template_errors = [item for item in template_errors if item[1]]
    route_errors = [item for item in route_errors if item[2]]
    plan_errors = _validate_plan(plan)
    print(f"Templates: {len(templates)}  errors: {len(template_errors)}")
    print(f"Routes:    {len(routes)}  errors: {len(route_errors)}")
    print(f"Plan:      {len(plan)}  errors: {len(plan_errors)}")
    for item in template_errors[:10]:
        print(f"  template error {item}")
    for item in route_errors[:10]:
        print(f"  route error {item}")
    for item in plan_errors[:10]:
        print(f"  plan error {item}")
    if template_errors or route_errors or plan_errors:
        raise SystemExit(1)


def _empty_architect_reject(group: dict[str, Any]) -> dict[str, Any]:
    promoted = group.get("promoted") or []
    best = promoted[0] if promoted else {}
    return {
        "template_id": "",
        "status": "reject",
        "semantic_identity": (
            "LLM returned no reusable semantic template for this promoted root; "
            "treat as unstable template identity until manually reviewed."
        ),
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
        "invalid_mutations": [
            "No stable typed field roles were returned by the architect agent.",
            "Do not route until the root is manually converted into a semantic identity.",
        ],
        "source_alpha_ids": [str(row.get("alpha_id")) for row in promoted if row.get("alpha_id")],
        "source_root_expression": group.get("source_root_expression") or best.get("root_expression") or "",
        "robustness_status": "missing",
        "admission_brain_score_delta": best.get("admission_brain_score_delta"),
        "overfit_risk": "high",
    }


def _validate_plan(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    from wq_alpha.brain.submit import DEFAULT_SETTINGS, _regime_key

    for idx, row in enumerate(rows):
        expr = str(row.get("expression") or "")
        if not expr or "{" in expr or "}" in expr:
            errors.append(f"row {idx}: unrendered or missing expression")
        if row.get("search_lane") != "semantic_template":
            errors.append(f"row {idx}: search_lane is not semantic_template")
        settings = dict(DEFAULT_SETTINGS)
        if isinstance(row.get("settings_override"), dict):
            settings.update(row["settings_override"])
        key = (expr, _regime_key(settings))
        if key in seen:
            errors.append(f"row {idx}: duplicate expression/regime")
        seen.add(key)
        params = row.get("params") if isinstance(row.get("params"), dict) else {}
        if not params.get("field_bindings"):
            errors.append(f"row {idx}: missing field_bindings")
    return errors


def _score_label(record: dict[str, Any]) -> str:
    score = record.get("admission_brain_score_delta")
    if score is None:
        score = record.get("admission_delta_pool_sharpe")
    return _fmt(score)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
