"""Deterministic helpers for semantic templates and template routing."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from core.brain.submit import DEFAULT_SETTINGS, _regime_key
from alpha_enhancement.workshop.core import is_passing, record_workshop_key, workshop_self_corr_eligible
from alpha_enhancement.workshop.robustness_card import record_score_addition


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "artifacts" / "semantic_templates"
DEFAULT_TEMPLATES_JSONL = DEFAULT_DIR / "templates.jsonl"
DEFAULT_SELECTIONS_JSONL = DEFAULT_DIR / "field_selections.jsonl"
DEFAULT_ROUTES_JSONL = DEFAULT_DIR / "routes.jsonl"
DEFAULT_PLAN_JSONL = DEFAULT_DIR / "template_plan.jsonl"
DEFAULT_FEEDBACK_JSONL = DEFAULT_DIR / "template_feedback.jsonl"

DEFAULT_SEED_PATHS = (
    ROOT / "artifacts" / "workshop" / "workshop_submittable.jsonl",
    ROOT / "artifacts" / "workshop" / "workshop_admission.jsonl",
    ROOT / "artifacts" / "semantic_workshop" / "workshop_summary.jsonl",
)
DEFAULT_CONTEXT_PATHS = (
    ROOT / "artifacts" / "sim" / "master.jsonl",
    ROOT / "artifacts" / "workshop" / "workshop_sim.jsonl",
    ROOT / "artifacts" / "workshop" / "by_id_sim.jsonl",
    ROOT / "alpha_discovery/grammar_method" / "artifacts" / "all_sim.jsonl",
    ROOT / "artifacts" / "probe" / "probe_sim.jsonl",
)
DEFAULT_CLUSTER_PATH = ROOT / "alpha_discovery/grammar_method" / "artifacts" / "all_clusters.json"
DEFAULT_PROBE_LOG = ROOT / "artifacts" / "probe" / "probe_sim.jsonl"
DEFAULT_KNOWN_LOGS = (
    ROOT / "artifacts" / "sim" / "master.jsonl",
    ROOT / "artifacts" / "workshop" / "workshop_sim.jsonl",
    ROOT / "artifacts" / "workshop" / "by_id_sim.jsonl",
    ROOT / "artifacts" / "probe" / "probe_sim.jsonl",
    ROOT / "artifacts" / "semantic_workshop" / "semantic_workshop_sim.jsonl",
    ROOT / "artifacts" / "semantic_templates" / "template_plan_sim.jsonl",
)


def read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stable_id(prefix: str, *parts: Any) -> str:
    source = "\n".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(source.encode('utf-8')).hexdigest()[:12]}"


def normalize_template(row: dict[str, Any]) -> dict[str, Any]:
    template = dict(row)
    pattern = str(template.get("expression_pattern") or template.get("pattern") or "").strip()
    identity = str(template.get("semantic_identity") or "").strip()
    source_root = str(template.get("source_root_expression") or "").strip()
    template_id = str(template.get("template_id") or "").strip()
    if not template_id:
        template_id = stable_id("sem_tpl", identity, pattern, source_root)
    status = _choice(template.get("status"), {"active", "watch", "reject"}, "watch")
    risk = _choice(template.get("overfit_risk"), {"low", "medium", "high"}, "medium")
    robustness = _choice(template.get("robustness_status"), {"robust", "weak", "fail", "missing"}, "missing")
    slots = template.get("slots") if isinstance(template.get("slots"), dict) else {}
    if "main_field" not in slots:
        slots["main_field"] = {
            "role": "primary signal field",
            "required": True,
            "compatible_field_types": [],
            "compatible_probe_families": [],
            "expected_direction": "either",
            "allow_sign_flip": False,
        }
    normalized_slots = {str(name): _normalize_slot(value, name) for name, value in slots.items()}
    default_settings = dict(DEFAULT_SETTINGS)
    if isinstance(template.get("default_settings"), dict):
        default_settings.update(template["default_settings"])
    return {
        "template_id": template_id,
        "status": status,
        "semantic_identity": identity or "unknown semantic identity",
        "expression_pattern": pattern,
        "slots": normalized_slots,
        "default_windows": _int_list(template.get("default_windows")) or [60],
        "default_groups": _string_list(template.get("default_groups")) or ["industry"],
        "default_settings": _compact_settings(default_settings),
        "invalid_mutations": _string_list(template.get("invalid_mutations")),
        "source_alpha_ids": _string_list(template.get("source_alpha_ids")),
        "source_root_expression": source_root,
        "robustness_status": robustness,
        "admission_brain_score_delta": _float_or_none(template.get("admission_brain_score_delta")),
        "overfit_risk": risk,
    }


def normalize_route(row: dict[str, Any], template: dict[str, Any] | None = None) -> dict[str, Any]:
    route = dict(row)
    template_id = str(route.get("template_id") or (template or {}).get("template_id") or "").strip()
    cluster_id = str(route.get("cluster_id") or "").strip()
    bindings = route.get("field_bindings") if isinstance(route.get("field_bindings"), dict) else {}
    normalized_bindings = {str(k): (None if v in ("", None) else str(v)) for k, v in bindings.items()}
    if "main_field" not in normalized_bindings and route.get("main_field"):
        normalized_bindings["main_field"] = str(route["main_field"])
    normalized = {
        "template_id": template_id,
        "cluster_id": cluster_id,
        "selector_candidate_id": str(route.get("selector_candidate_id") or route.get("candidate_id") or ""),
        "expansion_stage": _choice(route.get("expansion_stage"), {"medoid", "sibling"}, "medoid"),
        "route_status": _choice(route.get("route_status"), {"active", "watch", "reject"}, "watch"),
        "route_confidence": _choice(route.get("route_confidence"), {"high", "medium", "low"}, "medium"),
        "field_bindings": normalized_bindings,
        "slot_cluster_ids": route.get("slot_cluster_ids") if isinstance(route.get("slot_cluster_ids"), dict) else {},
        "window_candidates": _int_list(route.get("window_candidates")),
        "group_candidates": _string_list(route.get("group_candidates")),
        "selector_rationale": str(route.get("selector_rationale") or route.get("economic_rationale") or ""),
        "selector_evidence": _string_list(route.get("selector_evidence")),
        "evidence": _string_list(route.get("evidence")),
        "rejection_reason": str(route.get("rejection_reason") or ""),
        "max_initial_candidates": max(1, int(_float_or_none(route.get("max_initial_candidates")) or 1)),
    }
    return normalized


def validate_template(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not row.get("template_id"):
        errors.append("missing template_id")
    if row.get("status") == "reject":
        if not row.get("semantic_identity"):
            errors.append("reject template missing reason/semantic_identity")
        return errors
    if not row.get("expression_pattern"):
        errors.append("missing expression_pattern")
    if "{main_field}" not in str(row.get("expression_pattern", "")) and "{field}" not in str(row.get("expression_pattern", "")):
        errors.append("expression_pattern must contain {main_field} or {field}")
    slots = row.get("slots")
    if not isinstance(slots, dict) or "main_field" not in slots:
        errors.append("missing main_field slot")
    return errors


def validate_route(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not row.get("template_id"):
        errors.append("missing template_id")
    if not row.get("cluster_id"):
        errors.append("missing cluster_id")
    bindings = row.get("field_bindings")
    if not isinstance(bindings, dict) or not bindings.get("main_field"):
        errors.append("missing field_bindings.main_field")
    return errors


def build_architect_evidence_groups(
    seed_records: list[dict[str, Any]],
    context_records: list[dict[str, Any]],
    probe_records: list[dict[str, Any]],
    *,
    max_groups: int = 10,
    max_promoted_per_root: int = 5,
    max_failed_per_root: int = 8,
) -> list[dict[str, Any]]:
    expanded_seed_records = _expand_template_seed_records(seed_records, context_records)
    promoted = [_template_seed_record(row) for row in expanded_seed_records if _is_template_seed(row)]
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in promoted:
        by_root[_root_expression(record)].append(record)
    probe_by_field = _probe_by_field(probe_records)
    context_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in context_records:
        root = _root_expression(record)
        if root in by_root and not _is_template_seed(record):
            context_by_root[root].append(record)
    groups: list[dict[str, Any]] = []
    for root, records in by_root.items():
        records.sort(key=_score_key, reverse=True)
        failed = sorted(context_by_root.get(root, []), key=lambda row: abs(float(row.get("is_fitness") or 0.0)), reverse=True)
        fields = sorted({str(field) for rec in records for field in (rec.get("fields") or [])})
        groups.append({
            "source_root_expression": root,
            "promoted": [_compact_alpha(row) for row in records[:max_promoted_per_root]],
            "failed_or_context": [_compact_alpha(row) for row in failed[:max_failed_per_root]],
            "probe_evidence": {field: probe_by_field.get(field, [])[:4] for field in fields},
        })
    groups.sort(key=lambda group: _score_key(group["promoted"][0]) if group["promoted"] else (-999.0,), reverse=True)
    return groups[:max_groups]


def load_cluster_catalog(path: str | Path = DEFAULT_CLUSTER_PATH) -> list[dict[str, Any]]:
    cluster_path = Path(path)
    if not cluster_path.exists():
        return []
    try:
        payload = json.loads(cluster_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    clusters: list[dict[str, Any]] = []
    for row in payload if isinstance(payload, list) else []:
        members = [dict(member) for member in row.get("members") or [] if member.get("field_id")]
        if not members:
            continue
        champion_id = row.get("champion_field_id")
        champion = next((member for member in members if member.get("field_id") == champion_id), members[0])
        clusters.append({
            "cluster_id": str(row.get("cluster_id") or champion.get("cluster_id") or champion.get("field_id")),
            "category": row.get("category") or champion.get("category") or "",
            "dataset_id": row.get("dataset_id") or champion.get("dataset_id") or "",
            "label": row.get("label") or champion.get("description") or "",
            "champion_field_id": champion.get("field_id"),
            "sleeper_field_id": row.get("sleeper_field_id"),
            "champion": champion,
            "members": members,
        })
    return clusters


def load_probe_summaries(
    probe_records: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    field_to_cluster = {
        str(member.get("field_id")): cluster["cluster_id"]
        for cluster in clusters
        for member in cluster.get("members", [])
        if member.get("field_id")
    }
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in probe_records:
        if record.get("status") != "done":
            continue
        fields = [str(field) for field in (record.get("fields") or []) if field]
        if not fields:
            continue
        cluster_ids = [str(cluster) for cluster in (record.get("cluster_ids") or []) if cluster]
        cluster_id = cluster_ids[0] if cluster_ids else field_to_cluster.get(fields[0])
        if not cluster_id:
            continue
        params = record.get("params") if isinstance(record.get("params"), dict) else {}
        by_cluster[cluster_id].append({
            "field": fields[0],
            "probe_template": params.get("probe_template") or record.get("operator_skeleton"),
            "probe_family": params.get("probe_family") or record.get("template_family"),
            "probe_direction": params.get("probe_direction"),
            "probe_horizon": params.get("probe_horizon"),
            "concentration_risk": params.get("concentration_risk"),
            "is_sharpe": record.get("is_sharpe"),
            "is_fitness": record.get("is_fitness"),
            "is_turnover": record.get("is_turnover"),
            "grade": record.get("grade"),
            "score": abs(float(record.get("is_fitness") or 0.0)) + 0.35 * abs(float(record.get("is_sharpe") or 0.0)),
        })
    summaries: dict[str, dict[str, Any]] = {}
    for cluster_id, rows in by_cluster.items():
        rows.sort(key=lambda row: row["score"], reverse=True)
        best = rows[0]
        summaries[cluster_id] = {
            "best_probe_family": best.get("probe_family"),
            "best_probe_direction": best.get("probe_direction"),
            "best_probe_template": best.get("probe_template"),
            "best_sharpe": best.get("is_sharpe"),
            "best_fitness": best.get("is_fitness"),
            "best_turnover": best.get("is_turnover"),
            "best_field": best.get("field"),
            "concentration_risk": best.get("concentration_risk"),
            "families": sorted({str(row.get("probe_family")) for row in rows if row.get("probe_family")}),
            "directions": sorted({str(row.get("probe_direction")) for row in rows if row.get("probe_direction")}),
            "top_probes": rows[:5],
        }
    return summaries


def prefilter_route_candidates(
    template: dict[str, Any],
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
    *,
    limit: int = 80,
) -> list[dict[str, Any]]:
    slots = template.get("slots", {}) if isinstance(template.get("slots"), dict) else {}
    main_slot = slots.get("main_field", {})
    pattern = str(template.get("expression_pattern") or "")
    secondary_slot = slots.get("secondary_field", {})
    needs_secondary = (
        bool(secondary_slot.get("required"))
        or "{secondary_field}" in pattern
        or "{field2}" in pattern
    )
    secondary_candidates = (
        _slot_candidate_fields(template, "secondary_field", secondary_slot, clusters, probe_summaries, limit=12)
        if needs_secondary
        else []
    )
    candidates: list[tuple[float, dict[str, Any]]] = []
    for cluster in clusters:
        champion = cluster.get("champion") or {}
        probe = probe_summaries.get(cluster["cluster_id"], {})
        score, status, evidence, rejection = _route_prefilter(template, main_slot, cluster, champion, probe)
        row = {
            "cluster_id": cluster["cluster_id"],
            "dataset_id": cluster.get("dataset_id"),
            "category": cluster.get("category"),
            "label": cluster.get("label"),
            "champion_field": _compact_field(champion),
            "member_count": len(cluster.get("members") or []),
            "probe_summary": probe,
            "code_prefilter_status": status,
            "code_prefilter_score": round(score, 6),
            "code_evidence": evidence,
            "code_rejection_reason": rejection,
            "secondary_candidates": secondary_candidates,
        }
        if status != "reject":
            candidates.append((score, row))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in candidates[:limit]]


def build_selector_catalog(
    template: dict[str, Any],
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
    *,
    limit: int = 0,
    include_members: int = 4,
) -> list[dict[str, Any]]:
    """Build a compact real-field catalog for the LLM field selector.

    The selector sees actual cluster/field ids only.  Rows are sorted by a soft
    deterministic slot-fit score so the first chunks are most relevant, but a
    zero limit still allows passing the full catalog in bounded chunks.
    """
    slots = template.get("slots") if isinstance(template.get("slots"), dict) else {}
    rows: list[tuple[float, dict[str, Any]]] = []
    for cluster in clusters:
        champion = cluster.get("champion") or {}
        probe = probe_summaries.get(cluster["cluster_id"], {})
        text = _cluster_text(template, cluster, champion)
        slot_scores = {
            slot_name: round(_constraint_score(slot.get("compatible_field_types"), text), 6)
            for slot_name, slot in slots.items()
            if isinstance(slot, dict)
        }
        main_score = float(slot_scores.get("main_field") or 0.0)
        other_score = max(slot_scores.values() or [0.0])
        slot_score = main_score if main_score > 0.0 else 0.25 * other_score
        score = 10.0 * slot_score
        if slot_score <= 0.0:
            score -= 1.0
        if probe.get("best_fitness") is not None:
            score += 0.08 * abs(float(probe.get("best_fitness") or 0.0))
        if probe.get("best_sharpe") is not None:
            score += 0.03 * abs(float(probe.get("best_sharpe") or 0.0))
        score += 0.05 * float(champion.get("coverage") or 0.0)
        rows.append((score, {
            "cluster_id": cluster["cluster_id"],
            "dataset_id": cluster.get("dataset_id"),
            "category": cluster.get("category"),
            "label": cluster.get("label"),
            "champion_field": _compact_field(champion),
            "member_fields": _compact_cluster_members(cluster, limit=include_members),
            "member_count": len(cluster.get("members") or []),
            "slot_fit_scores": slot_scores,
            "probe_summary": _compact_probe_summary(probe),
        }))
    rows.sort(key=lambda item: item[0], reverse=True)
    catalog = [row for _, row in rows]
    return catalog[:limit] if limit and limit > 0 else catalog


def build_selector_slot_shortlists(
    template: dict[str, Any],
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
    *,
    limit: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    slots = template.get("slots") if isinstance(template.get("slots"), dict) else {}
    pattern = str(template.get("expression_pattern") or "")
    shortlists: dict[str, list[dict[str, Any]]] = {}
    for slot_name, slot in slots.items():
        if slot_name == "main_field" or not isinstance(slot, dict):
            continue
        if not slot.get("required") and f"{{{slot_name}}}" not in pattern:
            continue
        shortlists[slot_name] = _slot_candidate_fields(
            template,
            slot_name,
            slot,
            clusters,
            probe_summaries,
            limit=limit,
        )
    return shortlists


def normalize_field_selection(
    row: dict[str, Any],
    template: dict[str, Any],
    clusters: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    field_to_cluster, field_index, cluster_by_id = _catalog_indexes(clusters)
    bindings = row.get("field_bindings") if isinstance(row.get("field_bindings"), dict) else {}
    normalized_bindings = {str(k): (None if v in ("", None, "null") else str(v)) for k, v in bindings.items()}
    main_field = normalized_bindings.get("main_field") or str(row.get("main_field") or "")
    if not main_field:
        errors.append("missing field_bindings.main_field")
        return None, errors
    if main_field not in field_index:
        errors.append(f"unknown main_field {main_field}")
        return None, errors
    normalized_bindings["main_field"] = main_field
    slot_cluster_ids = row.get("slot_cluster_ids") if isinstance(row.get("slot_cluster_ids"), dict) else {}
    normalized_clusters: dict[str, str | None] = {
        str(k): (None if v in ("", None, "null") else str(v))
        for k, v in slot_cluster_ids.items()
    }
    normalized_clusters["main_field"] = normalized_clusters.get("main_field") or field_to_cluster.get(main_field)
    main_cluster = str(row.get("main_cluster_id") or normalized_clusters.get("main_field") or field_to_cluster.get(main_field) or "")
    if main_cluster not in cluster_by_id:
        errors.append(f"unknown main_cluster_id {main_cluster}")
        return None, errors
    for slot_name, field_id in list(normalized_bindings.items()):
        if slot_name == "main_field" or not field_id:
            continue
        if field_id not in field_index:
            errors.append(f"unknown {slot_name} {field_id}")
            normalized_bindings.pop(slot_name, None)
            continue
        normalized_clusters[slot_name] = normalized_clusters.get(slot_name) or field_to_cluster.get(field_id)
    selection = {
        "template_id": str(row.get("template_id") or template.get("template_id") or ""),
        "selector_candidate_id": stable_id(
            "sem_sel",
            template.get("template_id"),
            main_cluster,
            json.dumps(normalized_bindings, sort_keys=True),
            row.get("window_candidates"),
            row.get("group_candidates"),
        ),
        "selection_status": _choice(row.get("selection_status"), {"active", "watch", "reject"}, "watch"),
        "route_confidence": _choice(row.get("route_confidence"), {"high", "medium", "low"}, "medium"),
        "main_cluster_id": main_cluster,
        "slot_cluster_ids": normalized_clusters,
        "field_bindings": normalized_bindings,
        "window_candidates": _int_list(row.get("window_candidates")),
        "group_candidates": _string_list(row.get("group_candidates")),
        "economic_rationale": str(row.get("economic_rationale") or ""),
        "evidence": _string_list(row.get("evidence")),
        "rejection_reason": str(row.get("rejection_reason") or ""),
        "max_initial_candidates": max(1, int(_float_or_none(row.get("max_initial_candidates")) or 1)),
    }
    return selection, errors


def route_candidates_from_field_selections(
    template: dict[str, Any],
    selections: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
    *,
    limit: int = 80,
) -> list[dict[str, Any]]:
    field_to_cluster, field_index, cluster_by_id = _catalog_indexes(clusters)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for selection in selections:
        if selection.get("template_id") != template.get("template_id"):
            continue
        if selection.get("selection_status") == "reject":
            continue
        bindings = selection.get("field_bindings") if isinstance(selection.get("field_bindings"), dict) else {}
        main_field_id = bindings.get("main_field")
        if not main_field_id or main_field_id not in field_index:
            continue
        main_cluster_id = selection.get("main_cluster_id") or field_to_cluster.get(main_field_id)
        cluster = cluster_by_id.get(str(main_cluster_id or ""))
        if not cluster:
            continue
        dedupe_key = (str(main_cluster_id), json.dumps(bindings, sort_keys=True))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        secondary_candidates: list[dict[str, Any]] = []
        for slot_name, field_id in bindings.items():
            if slot_name == "main_field" or not field_id or field_id not in field_index:
                continue
            slot_cluster_id = selection.get("slot_cluster_ids", {}).get(slot_name) or field_to_cluster.get(field_id)
            secondary_candidates.append({
                "slot_name": slot_name,
                "cluster_id": slot_cluster_id,
                "field": _compact_field(field_index[field_id]),
                "probe_summary": _compact_probe_summary(probe_summaries.get(str(slot_cluster_id or ""), {})),
                "score": 0.0,
            })
        out.append({
            "selector_candidate_id": selection.get("selector_candidate_id"),
            "cluster_id": cluster["cluster_id"],
            "dataset_id": cluster.get("dataset_id"),
            "category": cluster.get("category"),
            "label": cluster.get("label"),
            "champion_field": _compact_field(field_index[main_field_id]),
            "member_count": len(cluster.get("members") or []),
            "probe_summary": _compact_probe_summary(probe_summaries.get(cluster["cluster_id"], {})),
            "code_prefilter_status": selection.get("selection_status") or "watch",
            "code_prefilter_score": 0.0,
            "code_evidence": selection.get("evidence") or [],
            "code_rejection_reason": selection.get("rejection_reason") or "",
            "secondary_candidates": secondary_candidates,
            "selector_window_candidates": selection.get("window_candidates") or [],
            "selector_group_candidates": selection.get("group_candidates") or [],
            "selector_rationale": selection.get("economic_rationale") or "",
            "selector_evidence": selection.get("evidence") or [],
            "selector_field_bindings": bindings,
            "selector_slot_cluster_ids": selection.get("slot_cluster_ids") or {},
        })
        if len(out) >= limit:
            break
    return out


def chunk_rows(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + size] for idx in range(0, len(rows), max(1, size))]


def route_from_prefilter_candidate(template: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    secondary_candidates = candidate.get("secondary_candidates") or []
    secondary_field = None
    if secondary_candidates:
        secondary_field = (secondary_candidates[0].get("field") or {}).get("field_id")
    bindings = dict(candidate.get("selector_field_bindings") or {})
    if not bindings:
        bindings = {
            "main_field": candidate["champion_field"].get("field_id"),
            "secondary_field": secondary_field,
        }
    return {
        "template_id": template["template_id"],
        "cluster_id": candidate["cluster_id"],
        "selector_candidate_id": candidate.get("selector_candidate_id") or "",
        "expansion_stage": "medoid",
        "route_status": "watch",
        "route_confidence": "medium",
        "field_bindings": bindings,
        "slot_cluster_ids": candidate.get("selector_slot_cluster_ids") or {},
        "window_candidates": candidate.get("selector_window_candidates") or [],
        "group_candidates": candidate.get("selector_group_candidates") or [],
        "selector_rationale": candidate.get("selector_rationale") or "",
        "selector_evidence": candidate.get("selector_evidence") or [],
        "evidence": candidate.get("code_evidence") or [],
        "rejection_reason": candidate.get("code_rejection_reason") or "",
        "max_initial_candidates": 1,
    }


def route_candidates_for_template(
    template: dict[str, Any],
    selections: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
    *,
    limit: int = 80,
) -> list[dict[str, Any]]:
    if selections:
        candidates = route_candidates_from_field_selections(
            template,
            selections,
            clusters,
            probe_summaries,
            limit=limit,
        )
        if candidates:
            return candidates
    return prefilter_route_candidates(template, clusters, probe_summaries, limit=limit)


def merge_route_candidate_metadata(raw_route: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return raw_route
    route = dict(raw_route)
    route.setdefault("selector_candidate_id", candidate.get("selector_candidate_id") or "")
    route.setdefault("window_candidates", candidate.get("selector_window_candidates") or candidate.get("window_candidates") or [])
    route.setdefault("group_candidates", candidate.get("selector_group_candidates") or candidate.get("group_candidates") or [])
    route.setdefault("selector_rationale", candidate.get("selector_rationale") or "")
    route.setdefault("selector_evidence", candidate.get("selector_evidence") or candidate.get("code_evidence") or [])
    route.setdefault("slot_cluster_ids", candidate.get("selector_slot_cluster_ids") or {})
    if candidate.get("selector_field_bindings") and not route.get("field_bindings"):
        route["field_bindings"] = candidate["selector_field_bindings"]
    return route


def merge_jsonl_rows(
    path: str | Path,
    new_rows: list[dict[str, Any]],
    key_fn,
) -> tuple[list[dict[str, Any]], int, int]:
    out_path = Path(path)
    by_key: dict[str, dict[str, Any]] = {}
    for row in read_jsonl([out_path]):
        key = key_fn(row)
        if key:
            by_key[key] = row
    n_new = 0
    n_updated = 0
    for row in new_rows:
        key = key_fn(row)
        if not key:
            continue
        if key in by_key:
            n_updated += 1
        else:
            n_new += 1
        by_key[key] = row
    merged = list(by_key.values())
    write_jsonl(merged, out_path)
    return merged, n_new, n_updated


def select_template_fields_with_agent(
    template: dict[str, Any],
    selector_agent: Any,
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
    *,
    catalog_limit: int = 0,
    chunk_size: int = 40,
    include_members: int = 4,
    slot_shortlist_limit: int = 20,
    max_selections_per_chunk: int = 5,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    catalog = build_selector_catalog(
        template,
        clusters,
        probe_summaries,
        limit=catalog_limit,
        include_members=include_members,
    )
    shortlists = build_selector_slot_shortlists(
        template,
        clusters,
        probe_summaries,
        limit=slot_shortlist_limit,
    )
    selections: list[dict[str, Any]] = []
    seen: set[str] = set()
    chunks = chunk_rows(catalog, chunk_size)
    print(f"  template field selector {template.get('template_id')}: catalog={len(catalog)} chunks={len(chunks)}")
    for idx, chunk in enumerate(chunks, 1):
        try:
            result = selector_agent.select_fields(
                template,
                chunk,
                slot_shortlists=shortlists,
                chunk_index=idx,
                total_chunks=len(chunks),
                max_selections=max_selections_per_chunk,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"  [warn] field selector failed for {template.get('template_id')} chunk {idx}: {exc}")
            continue
        kept = 0
        for raw in result.selections:
            selection, errors = normalize_field_selection(raw, template, clusters)
            if errors or not selection:
                if errors and not continue_on_error:
                    print(f"  reject malformed selection: {errors}")
                continue
            key = selection["selector_candidate_id"]
            if key in seen:
                continue
            seen.add(key)
            selections.append(selection)
            kept += 1
        if not continue_on_error:
            print(f"  chunk {idx}/{len(chunks)} selections={kept}")
    return selections


def route_template_fields_with_agent(
    template: dict[str, Any],
    selections: list[dict[str, Any]],
    router_agent: Any,
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
    *,
    candidate_limit: int = 80,
    chunk_size: int = 20,
    code_only: bool = False,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    candidates = route_candidates_for_template(
        template,
        selections,
        clusters,
        probe_summaries,
        limit=candidate_limit,
    )
    print(f"  template router {template.get('template_id')}: candidates={len(candidates)}")
    routes: list[dict[str, Any]] = []
    if code_only:
        return [normalize_route(route_from_prefilter_candidate(template, candidate), template) for candidate in candidates]
    for chunk in chunk_rows(candidates, chunk_size):
        by_selector = {
            str(row.get("selector_candidate_id")): row
            for row in chunk
            if row.get("selector_candidate_id")
        }
        by_cluster = {str(row.get("cluster_id")): row for row in chunk}
        try:
            result = router_agent.route(template, chunk)
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"  [warn] template router failed for {template.get('template_id')}: {exc}")
            continue
        for raw_route in result.routes:
            candidate = (
                by_selector.get(str(raw_route.get("selector_candidate_id") or ""))
                or by_cluster.get(str(raw_route.get("cluster_id") or ""))
            )
            route = normalize_route(merge_route_candidate_metadata(raw_route, candidate), template)
            errors = validate_route(route)
            if errors:
                if not continue_on_error:
                    print(f"  reject malformed route: {errors}")
                continue
            routes.append(route)
    return routes


def build_template_plan(
    templates: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    known_records: list[dict[str, Any]],
    *,
    max_windows: int = 2,
    max_groups: int = 2,
    max_candidates_per_route: int = 3,
) -> list[dict[str, Any]]:
    templates_by_id = {row["template_id"]: row for row in templates if row.get("status") == "active"}
    known_keys = {record_workshop_key(row) for row in known_records if row.get("expression")}
    plan: list[dict[str, Any]] = []
    for route in routes:
        if route.get("route_status") not in {"active", "watch"}:
            continue
        template = templates_by_id.get(route.get("template_id"))
        if not template:
            continue
        expressions = _render_template_expressions(
            template,
            route,
            max_windows=max_windows,
            max_groups=max_groups,
            max_candidates=max_candidates_per_route,
        )
        for expr, params in expressions:
            settings = dict(DEFAULT_SETTINGS)
            settings.update(template.get("default_settings") or {})
            regime = _regime_key(settings)
            key = f"{regime}\t{expr}"
            if key in known_keys:
                continue
            known_keys.add(key)
            plan.append({
                "candidate_id": stable_id("sem_route", route.get("template_id"), route.get("cluster_id"), regime, expr),
                "template_id": route.get("template_id"),
                "dataset_id": "",
                "category": "",
                "search_lane": "semantic_template",
                "expression": expr,
                "fields": [field for field in (route.get("field_bindings") or {}).values() if field],
                "cluster_ids": [route.get("cluster_id")],
                "group": params.get("group", ""),
                "params": {
                    "semantic_identity": template.get("semantic_identity"),
                    "source_root_expression": template.get("source_root_expression"),
                    "field_bindings": route.get("field_bindings"),
                    "slot_cluster_ids": route.get("slot_cluster_ids"),
                    "cluster_id": route.get("cluster_id"),
                    "expansion_stage": route.get("expansion_stage"),
                    "selector_candidate_id": route.get("selector_candidate_id"),
                    "selector_rationale": route.get("selector_rationale"),
                    "template_generation_version": "semantic_template_v1",
                    **params,
                },
                "settings_override": _settings_delta(settings),
                "skip_workshop": False,
            })
    return plan


def summarize_template_feedback(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("search_lane") != "semantic_template":
            continue
        params = record.get("params") if isinstance(record.get("params"), dict) else {}
        bindings = json.dumps(params.get("field_bindings") or {}, sort_keys=True)
        grouped[(str(record.get("template_id") or ""), str(params.get("cluster_id") or ""), bindings)].append(record)
    rows: list[dict[str, Any]] = []
    for (template_id, cluster_id, bindings_json), group in grouped.items():
        group.sort(key=_score_key, reverse=True)
        best = group[0]
        brain, fallback = record_score_addition(best)
        if _positive_score(best) and _passes_with_corr(best):
            status = "active"
            reason = "positive score addition and checks/correlation pass"
            next_stage = "sibling"
        elif is_passing(best) and best.get("status") == "done":
            status = "watch"
            reason = "standalone pass without positive score addition"
            next_stage = "medoid"
        else:
            status = "reject"
            reason = "negative score addition or failed checks"
            next_stage = "medoid"
        rows.append({
            "template_id": template_id,
            "cluster_id": cluster_id,
            "field_bindings": json.loads(bindings_json),
            "n_records": len(group),
            "route_feedback_status": status,
            "eligible_next_expansion_stage": next_stage,
            "reason": reason,
            "best_alpha_id": best.get("alpha_id"),
            "best_expression": best.get("expression"),
            "best_sharpe": best.get("is_sharpe"),
            "best_fitness": best.get("is_fitness"),
            "best_turnover": best.get("is_turnover"),
            "admission_brain_score_delta": brain,
            "admission_delta_pool_sharpe": fallback,
            "max_self_correlation": best.get("admission_global_max_corr", best.get("max_self_correlation")),
        })
    rows.sort(key=lambda row: (
        {"active": 2, "watch": 1, "reject": 0}.get(row["route_feedback_status"], 0),
        row.get("admission_brain_score_delta") if row.get("admission_brain_score_delta") is not None else row.get("admission_delta_pool_sharpe") or -999.0,
        row.get("best_fitness") or -999.0,
    ), reverse=True)
    return rows


def _normalize_slot(value: Any, name: str) -> dict[str, Any]:
    slot = value if isinstance(value, dict) else {}
    return {
        "role": str(slot.get("role") or name.replace("_", " ")),
        "required": bool(slot.get("required", name == "main_field")),
        "compatible_field_types": _string_list(slot.get("compatible_field_types")),
        "compatible_probe_families": _string_list(slot.get("compatible_probe_families")),
        "expected_direction": _choice(slot.get("expected_direction"), {"positive", "negative", "either"}, "either"),
        "allow_sign_flip": bool(slot.get("allow_sign_flip", False)),
    }


def _route_prefilter(
    template: dict[str, Any],
    main_slot: dict[str, Any],
    cluster: dict[str, Any],
    champion: dict[str, Any],
    probe: dict[str, Any],
) -> tuple[float, str, list[str], str]:
    evidence: list[str] = []
    score = float(champion.get("coverage") or 0.0)
    text = _cluster_text(template, cluster, champion)
    type_score = _constraint_score(main_slot.get("compatible_field_types"), text)
    if type_score:
        score += type_score
        evidence.append("field text matches slot constraints")
    elif main_slot.get("compatible_field_types"):
        return score, "reject", evidence, "field text does not match slot constraints"
    family_constraints = {item.lower() for item in main_slot.get("compatible_probe_families") or []}
    best_family = str(probe.get("best_probe_family") or "").lower()
    family_score = 0.0
    if family_constraints:
        family_score = _probe_family_match_score(family_constraints, best_family)
        if family_score:
            score += family_score
            evidence.append("probe family matches")
        elif best_family:
            score -= 0.15
            evidence.append(f"probe family differs: {best_family}")
        else:
            score -= 0.5
            evidence.append("no probe family available")
    if probe.get("best_fitness") is not None:
        score += abs(float(probe.get("best_fitness") or 0.0))
    if probe.get("best_sharpe") is not None:
        score += 0.35 * abs(float(probe.get("best_sharpe") or 0.0))
    direction = str(probe.get("best_probe_direction") or "").lower()
    expected = str(main_slot.get("expected_direction") or "either").lower()
    direction_is_applicable = not family_constraints or bool(family_score)
    if expected != "either" and direction and direction_is_applicable and not _direction_compatible(expected, direction):
        if not main_slot.get("allow_sign_flip"):
            return score, "reject", evidence, f"probe direction {direction} contradicts expected {expected}"
        evidence.append("direction mismatch allowed by sign flip")
    elif expected != "either" and direction and not direction_is_applicable:
        evidence.append("probe direction ignored because family differs")
    if _is_sparse_or_event(champion, probe) and not _template_allows_sparse(template):
        return score, "reject", evidence, "sparse/event cluster not compatible with template"
    if champion.get("is_size_proxy") and not _template_allows_size(template):
        return score, "reject", evidence, "size proxy not compatible with template"
    evidence.append("coverage acceptable")
    return score, "watch", evidence, ""


def _slot_candidate_fields(
    template: dict[str, Any],
    slot_name: str,
    slot: dict[str, Any],
    clusters: list[dict[str, Any]],
    probe_summaries: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not slot:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    family_constraints = {item.lower() for item in slot.get("compatible_probe_families") or []}
    expected = str(slot.get("expected_direction") or "either").lower()
    for cluster in clusters:
        champion = cluster.get("champion") or {}
        probe = probe_summaries.get(cluster["cluster_id"], {})
        text = _cluster_text(template, cluster, champion)
        type_score = _constraint_score(slot.get("compatible_field_types"), text)
        if slot.get("compatible_field_types") and type_score <= 0.0:
            continue
        score = float(champion.get("coverage") or 0.0) + type_score
        best_family = str(probe.get("best_probe_family") or "").lower()
        if family_constraints:
            family_score = _probe_family_match_score(family_constraints, best_family)
            if family_score:
                score += 0.85 * family_score
            elif best_family:
                score -= 0.25
            else:
                score -= 0.1
        direction = str(probe.get("best_probe_direction") or "").lower()
        if expected != "either" and direction and not _direction_compatible(expected, direction):
            if slot_name == "main_field" and not slot.get("allow_sign_flip"):
                continue
            score -= 0.25
        if champion.get("is_size_proxy") and not _slot_allows_size(slot):
            continue
        score += _slot_semantic_bonus(slot, champion)
        if probe.get("best_fitness") is not None:
            score += 0.5 * abs(float(probe.get("best_fitness") or 0.0))
        scored.append((score, {
            "slot_name": slot_name,
            "cluster_id": cluster["cluster_id"],
            "field": _compact_field(champion),
            "probe_summary": probe,
            "score": round(score, 6),
        }))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:limit]]


def _slot_allows_size(slot: dict[str, Any]) -> bool:
    text = json.dumps(slot).lower()
    return any(word in text for word in ("size", "asset", "capital", "scale", "denominator", "neutralizer", "control"))


def _slot_semantic_bonus(slot: dict[str, Any], field: dict[str, Any]) -> float:
    slot_text = json.dumps(slot).lower()
    field_id = str(field.get("field_id") or "").lower()
    description = str(field.get("description") or "").lower()
    bonus = 0.0
    if "asset" in slot_text or "denominator" in slot_text:
        if field_id == "assets" or description == "assets - total":
            bonus += 2.0
        elif "assets - total" in description:
            bonus += 1.25
        if "current assets" in description or "other" in description:
            bonus -= 0.35
    if "liabil" in slot_text or "debt" in slot_text or "obligation" in slot_text:
        if "liabil" in field_id or "liabil" in description or "debt" in field_id or "debt" in description:
            bonus += 0.5
    return bonus


def _render_template_expressions(
    template: dict[str, Any],
    route: dict[str, Any],
    *,
    max_windows: int,
    max_groups: int,
    max_candidates: int,
) -> list[tuple[str, dict[str, Any]]]:
    pattern = str(template.get("expression_pattern") or "")
    bindings = route.get("field_bindings") or {}
    if "{secondary_field}" in pattern and not bindings.get("secondary_field"):
        return []
    if "{gate_field}" in pattern and not bindings.get("gate_field"):
        return []
    windows = (_int_list(route.get("window_candidates")) or template.get("default_windows") or [60])[:max_windows]
    groups = (_string_list(route.get("group_candidates")) or template.get("default_groups") or ["industry"])[:max_groups]
    rendered: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for window in windows:
        for group in groups:
            expr = pattern
            expr = expr.replace("{main_field}", str(bindings.get("main_field") or ""))
            expr = expr.replace("{field}", str(bindings.get("main_field") or ""))
            expr = expr.replace("{secondary_field}", str(bindings.get("secondary_field") or ""))
            expr = expr.replace("{field2}", str(bindings.get("secondary_field") or ""))
            expr = expr.replace("{gate_field}", str(bindings.get("gate_field") or ""))
            expr = expr.replace("{neutralizer_field}", str(bindings.get("neutralizer_field") or group))
            expr = expr.replace("{window2}", str(window))
            expr = expr.replace("{window}", str(window))
            expr = expr.replace("{group}", str(group))
            if not expr or "{" in expr or "}" in expr or expr in seen:
                continue
            seen.add(expr)
            rendered.append((expr, {"window": window, "group": group}))
            if len(rendered) >= max_candidates:
                return rendered
    return rendered


def _template_seed_record(row: dict[str, Any]) -> dict[str, Any]:
    record = dict(row)
    if "params" not in record or not isinstance(record.get("params"), dict):
        record["params"] = {}
    if record.get("root_expression") and not record["params"].get("root_expression"):
        record["params"]["root_expression"] = record.get("root_expression")
    if record.get("seed_expression") and not record["params"].get("seed_expression"):
        record["params"]["seed_expression"] = record.get("seed_expression")
    return record


def _expand_template_seed_records(
    seed_records: list[dict[str, Any]],
    context_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return flat template-seed records from native and A/B summary inputs."""
    out: list[dict[str, Any]] = []
    context_by_alpha = {
        str(row.get("alpha_id")): row
        for row in context_records
        if row.get("alpha_id")
    }
    context_by_expr = {
        str(row.get("expression")): row
        for row in context_records
        if row.get("expression")
    }
    for row in seed_records:
        if _is_probe_semantic_ab_summary(row):
            out.extend(_expand_probe_semantic_ab_row(row, context_by_alpha, context_by_expr))
        else:
            out.append(row)
    return out


def _is_probe_semantic_ab_summary(row: dict[str, Any]) -> bool:
    return any(isinstance(row.get(key), dict) for key in ("baseline", "probe_diagnostic", "autoworkshop"))


def _expand_probe_semantic_ab_row(
    row: dict[str, Any],
    context_by_alpha: dict[str, dict[str, Any]],
    context_by_expr: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten one probe-semantic A/B summary row into candidate template seeds."""
    out: list[dict[str, Any]] = []
    source_root = str(row.get("seed_expression") or row.get("source_root_expression") or "")
    for arm in ("probe_diagnostic", "baseline", "autoworkshop"):
        nested = row.get(arm)
        if not isinstance(nested, dict) or not nested.get("expression"):
            continue
        context = (
            context_by_alpha.get(str(nested.get("alpha_id") or ""))
            or context_by_expr.get(str(nested.get("expression") or ""))
            or {}
        )
        record = dict(context)
        record.update({key: value for key, value in nested.items() if value is not None})
        params = dict(record.get("params") if isinstance(record.get("params"), dict) else {})
        context_params = context.get("params") if isinstance(context.get("params"), dict) else {}
        root = str(
            context_params.get("root_expression")
            or params.get("root_expression")
            or source_root
            or nested.get("expression")
        )
        params.update(
            {
                "root_expression": root,
                "seed_expression": source_root,
                "probe_semantic_ab_arm": arm,
                "probe_semantic_ab_winner": row.get("winner"),
                "semantic_vs_autoworkshop_score_lift": row.get("semantic_vs_autoworkshop_score_lift"),
                "semantic_vs_autoworkshop_fallback_lift": row.get("semantic_vs_autoworkshop_fallback_lift"),
            }
        )
        record["params"] = params
        record["source_root_expression"] = root
        record["search_lane"] = record.get("search_lane") or f"probe_semantic_ab_{arm}"
        record["status"] = record.get("status") or "done"
        brain, fallback = record_score_addition(record)
        score_positive = (brain is not None and brain > 0.0) or (brain is None and fallback is not None and fallback > 0.0)
        checks_passing = bool(nested.get("checks_passing"))
        if score_positive and checks_passing:
            record["admission_decision"] = "PROMOTE"
            record["final_recommendation"] = "submit"
        elif score_positive:
            record["final_recommendation"] = "keep-watchlist"
        else:
            record["final_recommendation"] = "reject"
        out.append(record)
    return out


def _is_template_seed(record: dict[str, Any]) -> bool:
    decision = str(record.get("admission_decision") or "").upper()
    recommendation = str(record.get("final_recommendation") or "").lower()
    if recommendation:
        return recommendation in {"submit", "keep-watchlist"}
    if decision:
        return decision == "PROMOTE"
    if "rank" in record and bool(record.get("alpha_id")):
        return _positive_score(record) or _passes_with_corr(record)
    return False


def _root_expression(record: dict[str, Any]) -> str:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    return str(
        params.get("root_expression")
        or record.get("source_root_expression")
        or params.get("seed_expression")
        or record.get("seed_expression")
        or record.get("expression")
        or ""
    )


def _compact_alpha(record: dict[str, Any]) -> dict[str, Any]:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    brain, fallback = record_score_addition(record)
    return {
        "alpha_id": record.get("alpha_id"),
        "expression": record.get("expression"),
        "root_expression": _root_expression(record),
        "dataset_id": record.get("dataset_id"),
        "category": record.get("category"),
        "fields": record.get("fields") or [],
        "cluster_ids": record.get("cluster_ids") or [],
        "is_sharpe": record.get("is_sharpe"),
        "is_fitness": record.get("is_fitness"),
        "is_turnover": record.get("is_turnover"),
        "grade": record.get("grade"),
        "admission_brain_score_delta": brain,
        "admission_delta_pool_sharpe": fallback,
        "semantic_template": params.get("semantic_template"),
        "overfit_risk": params.get("diagnostic_overfit_risk") or record.get("overfit_risk"),
    }


def _probe_by_field(probe_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in probe_records:
        for field in record.get("fields") or []:
            out[str(field)].append(_compact_alpha(record))
    for rows in out.values():
        rows.sort(key=_score_key, reverse=True)
    return out


def _score_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    brain, fallback = record_score_addition(record)
    score = brain if brain is not None else fallback
    return (
        score if score is not None else -999.0,
        float(record.get("is_fitness") or -999.0),
        float(record.get("is_sharpe") or -999.0),
        -(float(record.get("is_turnover") or 9.0)),
    )


def _positive_score(record: dict[str, Any]) -> bool:
    brain, fallback = record_score_addition(record)
    return (brain is not None and brain > 0.0) or (brain is None and fallback is not None and fallback > 0.0)


def _passes_with_corr(record: dict[str, Any]) -> bool:
    return record.get("status") == "done" and is_passing(record) and workshop_self_corr_eligible(record)


def _cluster_text(template: dict[str, Any], cluster: dict[str, Any], champion: dict[str, Any]) -> str:
    parts = [
        cluster.get("cluster_id"),
        cluster.get("dataset_id"),
        cluster.get("category"),
        cluster.get("label"),
        champion.get("field_id"),
        champion.get("field_type"),
        champion.get("description"),
        champion.get("economic_type"),
    ]
    return " ".join(str(part).lower() for part in parts if part)


def _constraint_score(constraints: Any, text: str) -> float:
    raw_text = str(text or "").lower()
    spaced_text = re.sub(r"[^a-z0-9]+", " ", raw_text.replace("_", " ")).strip()
    compact_text = re.sub(r"[^a-z0-9]+", "", raw_text)
    text_tokens = set(re.findall(r"[a-z0-9]+", spaced_text))
    score = 0.0
    for constraint in _string_list(constraints):
        constraint_text = constraint.lower()
        words = [
            word
            for word in re.split(r"[^a-z0-9]+", constraint_text)
            if len(word) > 2 and word not in _GENERIC_CONSTRAINT_WORDS
        ]
        positive = _constraint_positive_keywords(constraint_text)
        negative = _constraint_negative_keywords(constraint_text)
        direct_hits = {word for word in words if _text_keyword_hit(word, spaced_text, compact_text, text_tokens)}
        positive_hits = {word for word in positive if _text_keyword_hit(word, spaced_text, compact_text, text_tokens)}
        negative_hits = {word for word in negative if _text_keyword_hit(word, spaced_text, compact_text, text_tokens)}
        if "return" in constraint_text and not _text_has_return_signal(spaced_text, compact_text, text_tokens):
            continue
        if not direct_hits and not positive_hits:
            continue
        word_score = (len(direct_hits) / len(words)) if words else 0.0
        keyword_score = min(1.5, 0.55 * len(positive_hits))
        candidate_score = word_score + keyword_score - min(1.25, 0.45 * len(negative_hits))
        score = max(score, candidate_score)
    return score


_GENERIC_CONSTRAINT_WORDS = {
    "fundamental",
    "balance",
    "sheet",
    "field",
    "fields",
    "level",
    "levels",
    "signal",
    "signals",
    "series",
    "cross",
    "sectional",
    "group",
    "grouped",
    "scale",
    "scaled",
    "base",
    "measure",
    "metric",
    "type",
    "data",
    "return",
}


def _constraint_positive_keywords(constraint: str) -> set[str]:
    keywords: set[str] = set()
    if any(word in constraint for word in ("liability", "liabilities", "debt", "obligation")):
        keywords.update({
            "liability",
            "liabilities",
            "debt",
            "borrowings",
            "borrowing",
            "obligation",
            "obligations",
            "payable",
            "payables",
            "lease liability",
            "deferred tax liability",
            "current liabilities",
            "long term debt",
            "subordinated debt",
        })
    if any(word in constraint for word in ("asset", "capital_base", "size_scale", "denominator")):
        keywords.update({
            "asset",
            "assets",
            "total assets",
            "capital",
            "capitalization",
            "book equity",
            "common equity",
            "stockholders equity",
            "shareholders equity",
            "enterprise value",
            "invested capital",
        })
    if any(word in constraint for word in ("relationship", "peer", "product", "supplier", "customer", "network")):
        keywords.update({
            "rel ret",
            "relret",
            "relationship return",
            "peer return",
            "network return",
            "one day return",
            "averaged one day return",
            "customer return",
            "supplier return",
            "partner return",
            "competitor return",
        })
    if any(word in constraint for word in ("participation", "count", "breadth", "coverage")):
        keywords.update({
            "relationship count",
            "participant",
            "participation",
            "coverage",
            "breadth",
            "count",
        })
    return keywords


def _constraint_negative_keywords(constraint: str) -> set[str]:
    keywords: set[str] = set()
    if any(word in constraint for word in ("liability", "liabilities", "debt", "obligation")):
        keywords.update({
            "asset",
            "assets",
            "ppe",
            "property plant equipment",
            "cash",
            "sales",
            "revenue",
            "volatility",
            "risk free",
            "option",
            "return on assets",
            "return on equity",
        })
    if any(word in constraint for word in ("asset", "capital_base", "size_scale", "denominator")):
        keywords.update({
            "liability",
            "liabilities",
            "debt",
            "payable",
            "volatility",
            "risk free",
            "option",
        })
    if any(word in constraint for word in ("relationship", "peer", "product", "supplier", "customer", "network")):
        keywords.update({
            "return on assets",
            "return on equity",
            "number of",
            "count",
            "pagerank",
            "hits hub",
            "authority score",
            "sentiment",
            "news",
            "buzz",
            "social",
            "dividend",
        })
    return keywords


def _text_keyword_hit(keyword: str, spaced_text: str, compact_text: str, text_tokens: set[str]) -> bool:
    key = keyword.lower().strip()
    if not key:
        return False
    spaced_key = re.sub(r"[^a-z0-9]+", " ", key.replace("_", " ")).strip()
    compact_key = re.sub(r"[^a-z0-9]+", "", key)
    if " " in spaced_key:
        return spaced_key in spaced_text or bool(compact_key and compact_key in compact_text)
    if spaced_key in text_tokens:
        return True
    return bool(len(compact_key) > 4 and compact_key in compact_text)


def _text_has_return_signal(spaced_text: str, compact_text: str, text_tokens: set[str]) -> bool:
    return (
        "return" in text_tokens
        or "returns" in text_tokens
        or "ret" in text_tokens
        or "return" in compact_text
        or "returns" in compact_text
        or "relret" in compact_text
    )


def _direction_compatible(expected: str, direction: str) -> bool:
    if expected == "positive":
        return "neg" not in direction and "revers" not in direction
    if expected == "negative":
        return "neg" in direction or "revers" in direction or "low" in direction
    return True


def _probe_family_match_score(family_constraints: set[str], best_family: str) -> float:
    if not family_constraints or not best_family:
        return 0.0
    actual = best_family.lower()
    if actual in family_constraints:
        return 1.5
    actual_words = {
        word
        for word in re.split(r"[^a-z0-9]+", actual)
        if len(word) > 2 and word not in _GENERIC_CONSTRAINT_WORDS
    }
    for constraint in family_constraints:
        words = {
            word
            for word in re.split(r"[^a-z0-9]+", constraint.lower())
            if len(word) > 2 and word not in _GENERIC_CONSTRAINT_WORDS
        }
        if actual_words and actual_words & words:
            return 0.75
        if actual in constraint or constraint in actual:
            return 0.5
    if actual in {"level", "smooth_level", "relative_value"} and any("level" in item for item in family_constraints):
        return 0.5
    return 0.0


def _is_sparse_or_event(champion: dict[str, Any], probe: dict[str, Any]) -> bool:
    coverage = champion.get("coverage")
    sparse = coverage is not None and float(coverage or 0.0) < 0.35
    family = str(probe.get("best_probe_family") or "").lower()
    risk = str(probe.get("concentration_risk") or "").lower()
    return sparse or "event" in family or "staleness" in family or risk == "high"


def _template_allows_sparse(template: dict[str, Any]) -> bool:
    text = json.dumps(template).lower()
    return any(word in text for word in ("sparse", "event", "staleness", "change", "update", "backfill"))


def _template_allows_size(template: dict[str, Any]) -> bool:
    text = json.dumps(template).lower()
    return any(word in text for word in ("size", "cap", "market cap", "neutralizer", "control"))


def _compact_field(field: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_id": field.get("field_id"),
        "field_type": field.get("field_type"),
        "description": field.get("description"),
        "coverage": field.get("coverage"),
        "user_count": field.get("user_count"),
        "alpha_count": field.get("alpha_count"),
        "is_size_proxy": field.get("is_size_proxy"),
        "economic_type": field.get("economic_type"),
    }


def _compact_cluster_members(cluster: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    members = [member for member in cluster.get("members") or [] if member.get("field_id")]
    champion_id = cluster.get("champion_field_id")
    sleeper_id = cluster.get("sleeper_field_id")
    ordered: list[dict[str, Any]] = []
    for field_id in (champion_id, sleeper_id):
        if not field_id:
            continue
        match = next((member for member in members if member.get("field_id") == field_id), None)
        if match and match not in ordered:
            ordered.append(match)
    remaining = [member for member in members if member not in ordered]
    remaining.sort(key=lambda member: (
        float(member.get("coverage") or 0.0),
        int(member.get("user_count") or 0),
        int(member.get("alpha_count") or 0),
    ), reverse=True)
    ordered.extend(remaining)
    return [_compact_field(member) for member in ordered[: max(1, limit)]]


def _compact_probe_summary(probe: dict[str, Any]) -> dict[str, Any]:
    if not probe:
        return {}
    return {
        "best_probe_family": probe.get("best_probe_family"),
        "best_probe_direction": probe.get("best_probe_direction"),
        "best_probe_template": probe.get("best_probe_template"),
        "best_sharpe": probe.get("best_sharpe"),
        "best_fitness": probe.get("best_fitness"),
        "best_turnover": probe.get("best_turnover"),
        "best_field": probe.get("best_field"),
        "concentration_risk": probe.get("concentration_risk"),
        "families": probe.get("families") or [],
        "directions": probe.get("directions") or [],
    }


def _catalog_indexes(
    clusters: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    field_to_cluster: dict[str, str] = {}
    field_index: dict[str, dict[str, Any]] = {}
    cluster_by_id = {str(cluster["cluster_id"]): cluster for cluster in clusters}
    for cluster in clusters:
        cluster_id = str(cluster["cluster_id"])
        for member in cluster.get("members") or []:
            field_id = str(member.get("field_id") or "")
            if not field_id:
                continue
            field_to_cluster[field_id] = cluster_id
            field_index[field_id] = member
    return field_to_cluster, field_index, cluster_by_id


def _compact_settings(settings: dict[str, Any]) -> dict[str, Any]:
    keys = ("universe", "delay", "neutralization", "decay", "truncation")
    return {key: settings.get(key) for key in keys if key in settings}


def _settings_delta(settings: dict[str, Any]) -> dict[str, Any]:
    delta = {}
    for key, value in settings.items():
        if DEFAULT_SETTINGS.get(key) != value:
            delta[key] = value
    return delta


def _choice(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or default).lower()
    return text if text in allowed else default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _int_list(value: Any) -> list[int]:
    out: list[int] = []
    for item in _string_list(value):
        try:
            out.append(int(float(item)))
        except (TypeError, ValueError):
            continue
    return out


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_CLUSTER_PATH",
    "DEFAULT_CONTEXT_PATHS",
    "DEFAULT_DIR",
    "DEFAULT_FEEDBACK_JSONL",
    "DEFAULT_KNOWN_LOGS",
    "DEFAULT_PLAN_JSONL",
    "DEFAULT_PROBE_LOG",
    "DEFAULT_ROUTES_JSONL",
    "DEFAULT_SELECTIONS_JSONL",
    "DEFAULT_SEED_PATHS",
    "DEFAULT_TEMPLATES_JSONL",
    "build_architect_evidence_groups",
    "build_selector_catalog",
    "build_selector_slot_shortlists",
    "build_template_plan",
    "chunk_rows",
    "load_cluster_catalog",
    "load_probe_summaries",
    "merge_jsonl_rows",
    "merge_route_candidate_metadata",
    "normalize_field_selection",
    "normalize_route",
    "normalize_template",
    "prefilter_route_candidates",
    "read_jsonl",
    "route_candidates_for_template",
    "route_candidates_from_field_selections",
    "route_from_prefilter_candidate",
    "route_template_fields_with_agent",
    "select_template_fields_with_agent",
    "stable_id",
    "summarize_template_feedback",
    "validate_route",
    "validate_template",
    "write_jsonl",
]
