"""Semantic-workshop handoff into the semantic-template pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from wq_alpha.agents.semantic_templates import (
    TemplateArchitectAgent,
    TemplateFieldSelectorAgent,
    TemplateRouterAgent,
)
from wq_alpha.semantic_templates.core import (
    DEFAULT_KNOWN_LOGS,
    build_template_plan,
    load_cluster_catalog,
    load_probe_summaries,
    merge_jsonl_rows,
    normalize_template,
    read_jsonl,
    route_template_fields_with_agent,
    select_template_fields_with_agent,
    validate_template,
)
from wq_alpha.workshop.core import is_passing, record_workshop_key
from wq_alpha.workshop.robustness_card import record_score_addition


@dataclass(frozen=True)
class SemanticTemplateHandoff:
    architect_agent: TemplateArchitectAgent
    selector_agent: TemplateFieldSelectorAgent
    router_agent: TemplateRouterAgent
    clusters: list[dict[str, Any]]
    probe_summaries: dict[str, dict[str, Any]]
    template_out: str | Path
    selections_out: str | Path
    routes_out: str | Path
    plan_out: str | Path
    events_out: str | Path
    max_active: int = 3
    catalog_limit: int = 160
    selector_chunk_size: int = 40
    route_chunk_size: int = 10
    route_candidate_limit: int = 80
    max_selections_per_chunk: int = 5
    max_windows: int = 2
    max_groups: int = 2
    max_candidates_per_route: int = 3
    known_logs: Sequence[str | Path] = DEFAULT_KNOWN_LOGS


def handoff_from_args(args: Any) -> SemanticTemplateHandoff | None:
    if not getattr(args, "semantic_template_auto", False):
        return None
    clusters = load_cluster_catalog(args.clusters)
    probe_records = read_jsonl([args.template_probe_log])
    probe_summaries = load_probe_summaries(probe_records, clusters)
    print(
        "Semantic template auto-routing enabled: "
        f"clusters={len(clusters)} probe_summaries={len(probe_summaries)}"
    )
    return SemanticTemplateHandoff(
        architect_agent=TemplateArchitectAgent(api_key=args.api_key, model=args.model),
        selector_agent=TemplateFieldSelectorAgent(api_key=args.api_key, model=args.model),
        router_agent=TemplateRouterAgent(api_key=args.api_key, model=args.model),
        clusters=clusters,
        probe_summaries=probe_summaries,
        template_out=args.template_out,
        selections_out=args.template_selections_out,
        routes_out=args.template_routes_out,
        plan_out=args.template_plan_out,
        events_out=args.template_events_out,
        max_active=args.template_max_active,
        catalog_limit=args.template_catalog_limit,
        selector_chunk_size=args.template_selector_chunk_size,
        route_chunk_size=args.template_route_chunk_size,
        route_candidate_limit=args.template_route_candidate_limit,
        max_selections_per_chunk=args.template_max_selections_per_chunk,
        max_windows=args.template_max_windows,
        max_groups=args.template_max_groups,
        max_candidates_per_route=args.template_max_candidates_per_route,
    )


def create_routed_templates_for_finished_seed(
    handoff: SemanticTemplateHandoff | None,
    *,
    seed: dict[str, Any],
    root_expression: str,
    seed_records: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    latest_card: dict[str, Any] | None,
    reason: str,
) -> None:
    if handoff is None:
        return

    event_base = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "seed_alpha_id": seed.get("alpha_id"),
        "seed_expression": seed.get("expression"),
        "root_expression": root_expression,
        "reason": reason,
    }
    source_records = _template_source_records(root_expression, seed_records, all_records)
    promoted = [record for record in source_records if _template_source_eligible(record)]
    promoted_keys = {record_workshop_key(record) for record in promoted}
    failed_or_context = sorted(
        [
            record
            for record in source_records
            if (
                record_workshop_key(record) not in promoted_keys
                and record.get("status") == "done"
                and record.get("expression")
            )
        ],
        key=lambda record: abs(_float(record.get("is_fitness")) or 0.0),
        reverse=True,
    )
    if not promoted:
        print("  semantic templates: no eligible promoted source; wrote no-template audit row.")
        _append_jsonl(handoff.events_out, {
            **event_base,
            "status": "no_template",
            "message": "no standalone-passing positive-score semantic candidate for template architect",
            "semantic_candidate_count": len(seed_records),
            "same_root_candidate_count": len(source_records),
        })
        return

    evidence = {
        "source_root_expression": root_expression,
        "seed": _template_compact_alpha(seed),
        "promoted": [_template_compact_alpha(record) for record in promoted[:5]],
        "failed_or_context": [_template_compact_alpha(record) for record in failed_or_context[:8]],
        "robustness_card": _template_compact_card(latest_card),
        "probe_evidence": _template_probe_evidence(promoted, handoff.probe_summaries),
    }

    try:
        result = handoff.architect_agent.create_templates(evidence)
    except Exception as exc:
        print(f"  [warn] semantic template architect failed: {exc}")
        _append_jsonl(handoff.events_out, {**event_base, "status": "error", "stage": "architect", "error": str(exc)})
        return

    templates: list[dict[str, Any]] = []
    for raw_template in result.templates or []:
        template = normalize_template(raw_template)
        template["source_root_expression"] = template.get("source_root_expression") or root_expression
        if not template.get("source_alpha_ids"):
            template["source_alpha_ids"] = [str(record.get("alpha_id")) for record in promoted if record.get("alpha_id")]
        if template.get("admission_brain_score_delta") is None:
            template["admission_brain_score_delta"] = record_score_addition(promoted[0])[0]
        if template.get("robustness_status") == "missing" and latest_card:
            template["robustness_status"] = str(latest_card.get("overall_status") or "missing")
        errors = validate_template(template)
        if errors:
            _append_jsonl(handoff.events_out, {
                **event_base,
                "status": "reject",
                "stage": "architect",
                "template_id": template.get("template_id"),
                "errors": errors,
            })
            continue
        templates.append(template)

    if not templates:
        print("  semantic templates: architect produced no valid templates.")
        _append_jsonl(handoff.events_out, {
            **event_base,
            "status": "no_template",
            "message": "architect returned no valid template objects",
            "promoted_alpha_ids": [record.get("alpha_id") for record in promoted[:5]],
        })
        return

    merged_templates, template_new, template_updated = merge_jsonl_rows(
        handoff.template_out,
        templates,
        lambda row: str(row.get("template_id") or ""),
    )
    active_templates = [row for row in templates if row.get("status") == "active"][: max(0, handoff.max_active)]

    selections: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    for template in active_templates:
        template_selections = select_template_fields_with_agent(
            template,
            handoff.selector_agent,
            handoff.clusters,
            handoff.probe_summaries,
            catalog_limit=handoff.catalog_limit,
            chunk_size=handoff.selector_chunk_size,
            max_selections_per_chunk=handoff.max_selections_per_chunk,
            continue_on_error=True,
        )
        selections.extend(template_selections)
        routes.extend(
            route_template_fields_with_agent(
                template,
                template_selections,
                handoff.router_agent,
                handoff.clusters,
                handoff.probe_summaries,
                candidate_limit=handoff.route_candidate_limit,
                chunk_size=handoff.route_chunk_size,
                continue_on_error=True,
            )
        )

    _, selection_new, _ = merge_jsonl_rows(
        handoff.selections_out,
        selections,
        lambda row: str(row.get("selector_candidate_id") or ""),
    )
    merged_routes, route_new, _ = merge_jsonl_rows(
        handoff.routes_out,
        routes,
        lambda row: "|".join([
            str(row.get("template_id") or ""),
            str(row.get("cluster_id") or ""),
            json.dumps(row.get("field_bindings") or {}, sort_keys=True),
        ]),
    )

    existing_plan = read_jsonl([handoff.plan_out])
    known_records = read_jsonl(handoff.known_logs) + all_records + existing_plan
    new_plan = build_template_plan(
        merged_templates,
        merged_routes,
        known_records,
        max_windows=handoff.max_windows,
        max_groups=handoff.max_groups,
        max_candidates_per_route=handoff.max_candidates_per_route,
    )
    _, plan_new, _ = merge_jsonl_rows(
        handoff.plan_out,
        new_plan,
        record_workshop_key,
    )

    print(
        "  semantic templates: "
        f"templates +{template_new}/~{template_updated}, "
        f"selections +{selection_new}, routes +{route_new}, plan +{plan_new}"
    )
    _append_jsonl(handoff.events_out, {
        **event_base,
        "status": "routed" if route_new or routes else "template_only",
        "promoted_alpha_ids": [record.get("alpha_id") for record in promoted[:5]],
        "templates_new": template_new,
        "templates_updated": template_updated,
        "templates_total_for_seed": len(templates),
        "active_templates_for_seed": len(active_templates),
        "selections_new": selection_new,
        "routes_new": route_new,
        "plan_new": plan_new,
        "template_ids": [row.get("template_id") for row in templates],
    })


def _template_source_eligible(record: dict[str, Any]) -> bool:
    if not _standalone_passes(record):
        return False
    brain, fallback = record_score_addition(record)
    if not ((brain is not None and brain > 0.0) or (brain is None and fallback is not None and fallback > 0.0)):
        return False
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    decision = str(record.get("admission_decision") or admission.get("decision") or "").upper()
    return not decision or decision == "PROMOTE"


def _template_source_records(
    root_expression: str,
    seed_records: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in [*seed_records, *all_records]:
        if not isinstance(record, dict):
            continue
        if str(record.get("search_lane") or "") != "semantic_enhance":
            continue
        if _root_expression(record) != root_expression:
            continue
        key = record_workshop_key(record)
        existing = by_key.get(key)
        if existing is None or _summary_rank_key(record) > _summary_rank_key(existing):
            by_key[key] = record
    return sorted(by_key.values(), key=_summary_rank_key, reverse=True)


def _template_compact_alpha(record: dict[str, Any]) -> dict[str, Any]:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    brain, fallback = record_score_addition(record)
    return {
        "alpha_id": record.get("alpha_id"),
        "expression": record.get("expression"),
        "root_expression": _root_expression(record),
        "fields": record.get("fields") or [],
        "cluster_ids": record.get("cluster_ids") or [],
        "is_sharpe": record.get("is_sharpe"),
        "is_fitness": record.get("is_fitness"),
        "is_turnover": record.get("is_turnover"),
        "grade": record.get("grade"),
        "admission_decision": record.get("admission_decision"),
        "admission_reason": record.get("admission_reason"),
        "admission_brain_score_delta": brain,
        "admission_delta_pool_sharpe": fallback,
        "max_self_correlation": record.get("admission_global_max_corr", record.get("max_self_correlation")),
        "semantic_template": params.get("semantic_template"),
        "overfit_risk": params.get("diagnostic_overfit_risk") or record.get("overfit_risk"),
    }


def _template_compact_card(card: dict[str, Any] | None) -> dict[str, Any] | None:
    if not card:
        return None
    return {
        "overall_status": card.get("overall_status"),
        "root_expression": card.get("root_expression"),
        "sources": [
            {
                "source_alpha_id": source.get("source_alpha_id"),
                "status": source.get("status"),
                "expression_retention_rate": source.get("expression_retention_rate"),
                "settings_pass_rate": source.get("settings_pass_rate"),
                "field_pass_rate": source.get("field_pass_rate"),
                "top1000_retention": source.get("top1000_retention"),
                "notes": source.get("notes") or [],
            }
            for source in (card.get("sources") or [])[:5]
        ],
    }


def _template_probe_evidence(
    records: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        fields = [str(field) for field in (record.get("fields") or []) if field]
        clusters = [str(cluster) for cluster in (record.get("cluster_ids") or []) if cluster]
        for idx, field in enumerate(fields):
            cluster_id = clusters[idx] if idx < len(clusters) else (clusters[0] if clusters else "")
            summary = probe_summaries.get(cluster_id) if cluster_id else None
            if summary:
                evidence.setdefault(field, []).append(summary)
    return evidence


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


def _root_expression(record: dict[str, Any]) -> str:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    return str(params.get("root_expression") or params.get("original_expression") or record.get("expression") or "")


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "SemanticTemplateHandoff",
    "create_routed_templates_for_finished_seed",
    "handoff_from_args",
]
