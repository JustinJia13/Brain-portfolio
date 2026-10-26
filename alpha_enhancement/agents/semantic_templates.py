"""LLM agents for semantic template creation, field selection, and routing.

The Template Architect turns promoted alpha families into reusable typed
templates.  The Field Selector proposes concrete, grounded field bindings from
the available catalog.  The Router validates those bindings.  None of these
agents submits Brain simulations directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from core import llm as anthropic
from core.json_utils import parse_json_array


_FASTEXPR_TEMPLATE_RULES = """
FASTEXPR template rules:
- Use placeholders only for typed slots: {main_field}, {secondary_field},
  {gate_field}, {neutralizer_field}, {window}, {window2}, {group}.
- Keep the expression valid after placeholder substitution.
- Do not hard-code a secondary data field unless it is part of the original
  semantic identity and cannot be routed safely.
- Do not emit raw candidate alphas; emit reusable expression_pattern values.
"""


_ARCHITECT_SYSTEM = f"""You are a WorldQuant Brain Semantic Template Architect.

Your job is to convert promoted, robust alpha families into reusable semantic
templates with typed field roles.  You do not generate candidate alphas for
simulation.  You create templates that can later be routed to compatible fields.

Reject or mark watch when the alpha family has no stable semantic identity, when
the expression is a one-off parameter fit, or when field roles cannot be stated.
For each source alpha family you are given, emit at least one JSON object.  If
no reusable template is safe, emit one object with status="reject" and put the
concrete reason in semantic_identity and invalid_mutations.  Never return an
empty array for a non-empty source family.

{_FASTEXPR_TEMPLATE_RULES}

Return ONLY a JSON array.  Each item must have:
{{
  "template_id": "optional stable slug",
  "status": "active|watch|reject",
  "semantic_identity": "...",
  "expression_pattern": "FASTEXPR template with typed slots",
  "slots": {{
    "main_field": {{
      "role": "...",
      "required": true,
      "compatible_field_types": ["..."],
      "compatible_probe_families": ["..."],
      "expected_direction": "positive|negative|either",
      "allow_sign_flip": false
    }},
    "secondary_field": {{
      "role": "...",
      "required": false,
      "compatible_field_types": ["..."],
      "compatible_probe_families": ["..."],
      "expected_direction": "positive|negative|either",
      "allow_sign_flip": true
    }}
  }},
  "default_windows": [20, 60, 120],
  "default_groups": ["industry", "subindustry", "sector"],
  "default_settings": {{
    "universe": "TOP3000",
    "delay": 1,
    "neutralization": "SUBINDUSTRY",
    "decay": 0,
    "truncation": 0.01
  }},
  "invalid_mutations": ["..."],
  "source_alpha_ids": ["..."],
  "source_root_expression": "...",
  "robustness_status": "robust|weak|missing",
  "admission_brain_score_delta": 0.0,
  "overfit_risk": "low|medium|high"
}}
"""


_ROUTER_SYSTEM = """You are a WorldQuant Brain Semantic Template Router.

Given one semantic template and a compact candidate cluster catalog, decide which
clusters can safely fill each field slot.  You do not rewrite the template and
you do not create new alpha ideas.  You only emit route decisions.

Be conservative:
- emit exactly one route decision for every item in candidate_clusters.  Never
  omit a candidate.  If a route is unsafe, emit route_status="reject" with a
  concrete rejection_reason.
- active routes should have clear semantic and probe evidence.
- watch routes are plausible but need medoid testing.
- reject routes when field semantics, probe direction, sparsity, size exposure,
  or coverage are incompatible.
- start with medoid/champion field bindings only.
- when a template has a required secondary_field, or the expression_pattern uses
  {secondary_field} or {field2}, choose the secondary binding from the
  candidate's secondary_candidates list.  If none is safe, mark the route watch
  or reject instead of leaving the required slot null.

Return ONLY a JSON array.  Each item must have:
{
  "template_id": "...",
  "selector_candidate_id": "copy from candidate if present, else empty string",
  "cluster_id": "...",
  "expansion_stage": "medoid|sibling",
  "route_status": "active|watch|reject",
  "route_confidence": "high|medium|low",
  "field_bindings": {
    "main_field": "field_id",
    "secondary_field": "field_id_or_null"
  },
  "evidence": ["..."],
  "rejection_reason": "",
  "max_initial_candidates": 1
}
"""


_FIELD_SELECTOR_SYSTEM = """You are a WorldQuant Brain Semantic Template Field Selector.

Given one semantic template, its economic intent, typed field slots, and a
compact chunk of the real available field/cluster catalog, propose concrete
field bindings that are worth testing.  You are allowed to be semantically
creative, but you must be grounded:
- Use only field_id and cluster_id values that appear in the provided catalog
  chunk or slot_shortlists.
- Do not invent field names.
- For required secondary slots, choose the secondary field from slot_shortlists
  unless the catalog chunk contains a clearly better concrete field.
- Prefer economically coherent field families over standalone probe strength.
- Treat probe metrics as noisy prior evidence, not ground truth.
- Return only selected bindings; do not emit reject rows for every unused
  catalog entry.
- Choose compact parameter ranges.  Windows should be 1-4 values.  Groups
  should usually be industry, subindustry, or sector.

Return ONLY a JSON array.  Each item must have:
{
  "template_id": "...",
  "selection_status": "active|watch",
  "route_confidence": "high|medium|low",
  "main_cluster_id": "...",
  "slot_cluster_ids": {
    "main_field": "...",
    "secondary_field": "cluster_id_or_null"
  },
  "field_bindings": {
    "main_field": "field_id",
    "secondary_field": "field_id_or_null"
  },
  "window_candidates": [20, 60],
  "group_candidates": ["industry", "subindustry"],
  "economic_rationale": "...",
  "evidence": ["..."],
  "rejection_reason": "",
  "max_initial_candidates": 1
}
"""


@dataclass
class ArchitectResult:
    templates: list[dict[str, Any]]
    messages: list[dict[str, str]] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class RouterResult:
    routes: list[dict[str, Any]]
    messages: list[dict[str, str]] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class FieldSelectionResult:
    selections: list[dict[str, Any]]
    messages: list[dict[str, str]] = field(default_factory=list)
    raw_response: str = ""


class TemplateArchitectAgent:
    """LLM agent that creates semantic template objects from promoted roots."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-5.5", max_tokens: int = 3000) -> None:
        resolved_key = anthropic.load_api_key(api_key)
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    def create_templates(self, evidence: dict[str, Any]) -> ArchitectResult:
        message = _architect_message(evidence)
        messages = [{"role": "user", "content": message}]
        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=_ARCHITECT_SYSTEM,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": raw})
        return ArchitectResult(templates=_parse_json_array(raw), messages=messages, raw_response=raw)


class TemplateFieldSelectorAgent:
    """LLM agent that proposes grounded field bindings for a semantic template."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-5.5", max_tokens: int = 4000) -> None:
        resolved_key = anthropic.load_api_key(api_key)
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    def select_fields(
        self,
        template: dict[str, Any],
        catalog_chunk: list[dict[str, Any]],
        *,
        slot_shortlists: dict[str, list[dict[str, Any]]] | None = None,
        chunk_index: int = 1,
        total_chunks: int = 1,
        max_selections: int = 8,
    ) -> FieldSelectionResult:
        message = _field_selector_message(
            template,
            catalog_chunk,
            slot_shortlists=slot_shortlists or {},
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            max_selections=max_selections,
        )
        messages = [{"role": "user", "content": message}]
        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=_FIELD_SELECTOR_SYSTEM,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": raw})
        return FieldSelectionResult(selections=_parse_json_array(raw), messages=messages, raw_response=raw)


class TemplateRouterAgent:
    """LLM agent that maps semantic templates to compatible cluster bindings."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-5.5", max_tokens: int = 3000) -> None:
        resolved_key = anthropic.load_api_key(api_key)
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    def route(self, template: dict[str, Any], cluster_candidates: list[dict[str, Any]]) -> RouterResult:
        message = _router_message(template, cluster_candidates)
        messages = [{"role": "user", "content": message}]
        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=_ROUTER_SYSTEM,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": raw})
        return RouterResult(routes=_parse_json_array(raw), messages=messages, raw_response=raw)


def _architect_message(evidence: dict[str, Any]) -> str:
    packet = {
        "packet_type": "template_architect",
        "evidence": evidence,
        "context_report": {
            "promoted_count": len(evidence.get("promoted") or []),
            "failed_or_context_count": len(evidence.get("failed_or_context") or []),
            "probe_field_count": len(evidence.get("probe_evidence") or {}),
        },
    }
    compact = json.dumps(packet, sort_keys=True)
    return (
        "Create 1-3 semantic templates from this promoted alpha family. "
        "Use failed variants and probe evidence only as diagnostic context. "
        "If identity is unstable, return a reject template with the reason in invalid_mutations. "
        "Return at least one JSON object for this family; do not return [].\n\n"
        f"{compact}"
    )


def _router_message(template: dict[str, Any], cluster_candidates: list[dict[str, Any]]) -> str:
    payload = {
        "packet_type": "template_router",
        "template": template,
        "candidate_clusters": cluster_candidates,
        "context_report": {
            "candidate_count": len(cluster_candidates),
            "template_id": template.get("template_id"),
        },
        "policy": {
            "expansion_stage": "medoid",
            "max_initial_candidates": 1,
            "do_not_expand_siblings_yet": True,
            "required_output_count": len(cluster_candidates),
            "emit_one_decision_per_candidate_cluster": True,
        },
    }
    return json.dumps(payload, sort_keys=True)


def _field_selector_message(
    template: dict[str, Any],
    catalog_chunk: list[dict[str, Any]],
    *,
    slot_shortlists: dict[str, list[dict[str, Any]]],
    chunk_index: int,
    total_chunks: int,
    max_selections: int,
) -> str:
    payload = {
        "packet_type": "template_field_selector",
        "template": template,
        "catalog_chunk": catalog_chunk,
        "slot_shortlists": slot_shortlists,
        "context_report": {
            "catalog_chunk_count": len(catalog_chunk),
            "slot_shortlist_counts": {key: len(value) for key, value in slot_shortlists.items()},
            "template_id": template.get("template_id"),
        },
        "policy": {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "max_selections_from_this_chunk": max_selections,
            "select_only_real_fields": True,
            "no_hallucinated_field_ids": True,
            "probe_is_noisy_prior": True,
            "prefer_economic_compatibility": True,
        },
    }
    return json.dumps(payload, sort_keys=True)


def _parse_json_array(raw_text: str) -> list[dict[str, Any]]:
    payload = parse_json_array(raw_text, allow_object=True)
    return [item for item in payload if isinstance(item, dict)]


__all__ = [
    "ArchitectResult",
    "FieldSelectionResult",
    "RouterResult",
    "TemplateArchitectAgent",
    "TemplateFieldSelectorAgent",
    "TemplateRouterAgent",
]
