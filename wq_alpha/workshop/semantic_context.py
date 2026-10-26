"""Auditable context packets for semantic workshop LLM prompts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from alpha_grammar.qtable import extract_primary_op
from wq_alpha.brain.submit import DEFAULT_SETTINGS, _regime_key
from wq_alpha.workshop.core import record_workshop_key
from wq_alpha.workshop.probe_cards import compact_probe_card
from wq_alpha.workshop.robustness_card import record_score_addition


DEFAULT_MAX_CONTEXT_CHARS = 0


class SemanticContextTooLargeError(RuntimeError):
    """Raised when an auditable context packet is too large to send safely."""


@dataclass(frozen=True)
class SemanticContextResult:
    packet: dict[str, Any]
    report: dict[str, Any]


class SemanticContextBuilder:
    """Build deterministic, lineage-aware prompt context packets.

    This builder filters by relevance, not by a fixed record cap.  By default
    it records context size but does not reject large packets; pass a positive
    max_context_chars to fail loudly on oversized context.
    """

    def __init__(
        self,
        all_records: list[dict[str, Any]],
        *,
        known_expr_keys: set[str] | None = None,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        self.all_records = list(all_records)
        self.known_expr_keys = set(known_expr_keys or set())
        self.max_context_chars = max_context_chars

    def diagnostic_packet(
        self,
        *,
        seed: dict[str, Any],
        probe_card: dict[str, Any] | None = None,
        robustness_card: dict[str, Any] | None = None,
        prior_diagnosis: dict[str, Any] | None = None,
        round_results: list[dict[str, Any]] | None = None,
        history_records: list[dict[str, Any]] | None = None,
    ) -> SemanticContextResult:
        history = list(history_records or [])
        current_round = list(round_results or [])
        relevant, report = self._relevance_context(seed, extra_records=history + current_round)
        packet = {
            "packet_type": "semantic_diagnostic",
            "seed_summary": _compact_record(seed, include_expr=True),
            "seed_identity": _serializable_identity(_record_identity(seed)),
            "prior_diagnosis": prior_diagnosis or None,
            "round_results": [_compact_record(row, include_expr=True) for row in current_round],
            "lineage_history": [_compact_record(row, include_expr=True) for row in history],
            "relevant_history": relevant,
            "probe_card": compact_probe_card(probe_card),
            "robustness_card": _compact_robustness_card(robustness_card),
            "context_report": report,
        }
        return self._checked(packet, report)

    def generator_packet(
        self,
        *,
        seed: dict[str, Any],
        diagnosis: dict[str, Any],
        round_index: int,
        target_count: str,
        previous_results: list[dict[str, Any]] | None = None,
        robustness_card: dict[str, Any] | None = None,
        history_records: list[dict[str, Any]] | None = None,
    ) -> SemanticContextResult:
        history = list(history_records or [])
        previous = list(previous_results or [])
        relevant, report = self._relevance_context(seed, extra_records=history + previous)
        relevant_known = sorted({
            str(row.get("expression"))
            for rows in relevant.values()
            for row in rows
            if row.get("expression") and record_workshop_key(row) in self.known_expr_keys
        })
        packet = {
            "packet_type": "semantic_generator",
            "round_index": round_index,
            "required_count": target_count,
            "seed_summary": _compact_record(seed, include_expr=True),
            "seed_identity": _serializable_identity(_record_identity(seed)),
            "diagnostic_json": diagnosis,
            "previous_round_results": [_compact_record(row, include_expr=True) for row in previous],
            "lineage_history": [_compact_record(row, include_expr=True) for row in history],
            "relevant_history": relevant,
            "robustness_card": _compact_robustness_card(robustness_card),
            "known_expression_dedupe": {
                "total_known_expression_regimes": len(self.known_expr_keys),
                "relevant_known_expressions": relevant_known,
                "note": "Exact expression/regime dedupe is enforced deterministically outside the LLM.",
            },
            "context_report": report,
        }
        return self._checked(packet, report)

    def _relevance_context(
        self,
        seed: dict[str, Any],
        *,
        extra_records: list[dict[str, Any]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        seed_key = record_workshop_key(seed) if seed.get("expression") else ""
        tiers: dict[str, dict[str, dict[str, Any]]] = {
            "same_root": {},
            "same_cluster": {},
            "same_primary_field": {},
            "same_dataset_operator": {},
            "same_semantic_label": {},
        }
        excluded_counts: dict[str, int] = defaultdict(int)
        included_reasons: list[dict[str, Any]] = []
        seed_identity = _record_identity(seed)

        for record in self.all_records + extra_records:
            expr = record.get("expression")
            if not expr:
                excluded_counts["missing_expression"] += 1
                continue
            key = record_workshop_key(record)
            if key == seed_key:
                excluded_counts["seed_record"] += 1
                continue
            tier = _relevance_tier(seed_identity, _record_identity(record))
            if not tier:
                excluded_counts["unrelated"] += 1
                continue
            if tier != "same_root" and not _context_worthy(record):
                excluded_counts[f"{tier}_not_context_worthy"] += 1
                continue
            if key in tiers[tier]:
                excluded_counts["duplicate_relevant_key"] += 1
                continue
            compact = _compact_record(record, include_expr=True)
            tiers[tier][key] = compact
            included_reasons.append({
                "key": key,
                "tier": tier,
                "alpha_id": record.get("alpha_id"),
                "expression": str(expr)[:180],
            })

        ordered = {
            tier: sorted(rows.values(), key=_score_sort_value, reverse=True)
            for tier, rows in tiers.items()
        }
        report = {
            "seed_key": seed_key,
            "seed_identity": _serializable_identity(seed_identity),
            "included_counts": {tier: len(rows) for tier, rows in ordered.items()},
            "excluded_counts": dict(sorted(excluded_counts.items())),
            "included_reasons": included_reasons,
        }
        return ordered, report

    def _checked(self, packet: dict[str, Any], report: dict[str, Any]) -> SemanticContextResult:
        n_chars = len(json.dumps(packet, sort_keys=True))
        report["context_chars"] = n_chars
        report["max_context_chars"] = self.max_context_chars
        if self.max_context_chars > 0 and n_chars > self.max_context_chars:
            raise SemanticContextTooLargeError(
                f"Semantic context packet has {n_chars} chars, above max_context_chars={self.max_context_chars}. "
                "Refine relevance rules or raise the explicit safety limit."
            )
        return SemanticContextResult(packet=packet, report=report)


def _relevance_tier(seed: dict[str, Any], record: dict[str, Any]) -> str:
    if seed.get("root_expression") and seed.get("root_expression") == record.get("root_expression"):
        return "same_root"
    if seed.get("cluster_ids") and seed["cluster_ids"] & record.get("cluster_ids", set()):
        return "same_cluster"
    if seed.get("fields") and seed["fields"] & record.get("fields", set()):
        return "same_primary_field"
    if (
        seed.get("dataset_id")
        and seed.get("dataset_id") == record.get("dataset_id")
        and seed.get("primary_operator")
        and seed.get("primary_operator") == record.get("primary_operator")
    ):
        return "same_dataset_operator"
    if seed.get("semantic_label") and seed.get("semantic_label") == record.get("semantic_label"):
        return "same_semantic_label"
    return ""


def _record_identity(record: dict[str, Any]) -> dict[str, Any]:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    fields = {str(field) for field in record.get("fields") or [] if field}
    cluster_ids = {str(cluster) for cluster in record.get("cluster_ids") or [] if cluster}
    expression = str(record.get("expression") or "")
    primary_field = next(iter(fields), "")
    primary_operator = ""
    if primary_field:
        primary_operator = extract_primary_op(expression, primary_field) or ""
    if not primary_operator:
        primary_operator = _outer_operator(expression)
    return {
        "root_expression": str(params.get("root_expression") or params.get("original_expression") or expression),
        "fields": fields,
        "cluster_ids": cluster_ids,
        "dataset_id": str(record.get("dataset_id") or ""),
        "primary_field": primary_field,
        "primary_operator": primary_operator,
        "semantic_label": str(params.get("semantic_template") or params.get("signal_identity") or ""),
    }


def _serializable_identity(identity: dict[str, Any]) -> dict[str, Any]:
    out = dict(identity)
    for key in ("fields", "cluster_ids"):
        value = out.get(key)
        if isinstance(value, set):
            out[key] = sorted(value)
    return out


def _compact_record(record: dict[str, Any], *, include_expr: bool) -> dict[str, Any]:
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    brain, fallback = record_score_addition(record)
    corr = _first_non_none(record.get("admission_global_max_corr"), record.get("max_self_correlation"))
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    out = {
        "alpha_id": record.get("alpha_id"),
        "candidate_id": record.get("candidate_id"),
        "search_lane": record.get("search_lane"),
        "status": record.get("status"),
        "dataset_id": record.get("dataset_id"),
        "fields": record.get("fields") or [],
        "cluster_ids": record.get("cluster_ids") or [],
        "is_sharpe": record.get("is_sharpe"),
        "is_fitness": record.get("is_fitness"),
        "is_turnover": record.get("is_turnover"),
        "grade": record.get("grade"),
        "admission_decision": record.get("admission_decision") or admission.get("decision"),
        "admission_brain_score_delta": brain,
        "admission_delta_pool_sharpe": fallback,
        "max_self_correlation": corr,
        "semantic_template": params.get("semantic_template"),
        "signal_identity": params.get("signal_identity"),
        "root_expression": params.get("root_expression") or params.get("original_expression"),
        "failing_checks": [
            str(check.get("name"))
            for check in record.get("checks") or []
            if check.get("result") != "PASS" and check.get("name") != "SELF_CORRELATION"
        ],
    }
    if include_expr:
        out["expression"] = record.get("expression")
    return out


def _compact_robustness_card(card: dict[str, Any] | None) -> dict[str, Any] | None:
    if not card:
        return None
    return {
        "root_expression": card.get("root_expression"),
        "overall_status": card.get("overall_status"),
        "sources": [
            {
                "source_alpha_id": source.get("source_alpha_id"),
                "source_expression": str(source.get("source_expression") or "")[:220],
                "expression_retention_rate": source.get("expression_retention_rate"),
                "settings_pass_rate": source.get("settings_pass_rate"),
                "field_pass_rate": source.get("field_pass_rate"),
                "top1000_retention": source.get("top1000_retention"),
                "status": source.get("status"),
                "notes": source.get("notes"),
            }
            for source in card.get("sources") or []
        ],
    }


def _score_sort_value(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    brain, fallback = record_score_addition(record)
    score = brain if brain is not None else fallback
    corr = _float(_first_non_none(record.get("admission_global_max_corr"), record.get("max_self_correlation")))
    turnover = _float(record.get("is_turnover"))
    return (
        score if score is not None else -1e9,
        _float(record.get("is_fitness")) or -1e9,
        _float(record.get("is_sharpe")) or -1e9,
        -(corr if corr is not None else 9.0),
        -(turnover if turnover is not None else 9.0),
    )


def _context_worthy(record: dict[str, Any]) -> bool:
    brain, fallback = record_score_addition(record)
    if brain is not None and brain > 0:
        return True
    if brain is None and fallback is not None and fallback > 0:
        return True
    if _standalone_passes(record):
        return True
    lane = str(record.get("search_lane") or "")
    return lane in {"semantic_enhance", "semantic_template", "workshop_enhance", "workshop_repair"}


def _standalone_passes(record: dict[str, Any]) -> bool:
    sharpe = _float(record.get("is_sharpe")) or -999.0
    fitness = _float(record.get("is_fitness")) or -999.0
    if sharpe < 1.25 or fitness < 1.0:
        return False
    for check in record.get("checks") or []:
        if check.get("name") == "SELF_CORRELATION":
            continue
        if check.get("result") != "PASS":
            return False
    return record.get("status") == "done"


def _outer_operator(expression: str) -> str:
    match = re.match(r"\s*-?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(expression or ""))
    return match.group(1) if match else ""


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def record_regime_key(record: dict[str, Any]) -> str:
    settings = record.get("settings")
    if isinstance(settings, dict):
        return _regime_key({**DEFAULT_SETTINGS, **settings})
    override = record.get("settings_override")
    if isinstance(override, dict):
        return _regime_key({**DEFAULT_SETTINGS, **override})
    return str(record.get("regime") or _regime_key(DEFAULT_SETTINGS))


__all__ = [
    "DEFAULT_MAX_CONTEXT_CHARS",
    "SemanticContextBuilder",
    "SemanticContextResult",
    "SemanticContextTooLargeError",
    "record_regime_key",
]
