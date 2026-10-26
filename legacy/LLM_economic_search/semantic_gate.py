"""Score-based semantic novelty gate for generated expressions."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

from .expression_validation import AcceptedExpression
from .history_index import HistoryIndex
from .llm_agents import ExpressionSemanticCriticAgent
from .portfolio_context import compact_portfolio_context
from .schemas import TheorySpec, semantic_normalize_expression


SIMULATE_THRESHOLD = 65.0
QUARANTINE_THRESHOLD = 85.0
HIGH_CONFIDENCE_THRESHOLD = 70.0
SOFT_QUOTA_FRACTION = 0.15
NEIGHBOR_LIMIT = 12
ALLOWED_TAGS = {
    "sign_repair",
    "cross_delay_retest",
    "same_fields_new_shape",
    "new_fields_same_shape",
    "same_mechanism_new_dataset",
    "portfolio_crowded",
    "pool_hurt_neighbor",
    "high_corr_neighbor",
}


@dataclass
class SemanticGateResult:
    accepted: list[AcceptedExpression]
    quarantined: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    responses: list[dict[str, Any]]
    report: dict[str, Any]


class SemanticNoveltyGate:
    def __init__(
        self,
        *,
        history_index: HistoryIndex | None,
        target_delay: int,
        agent: ExpressionSemanticCriticAgent | None = None,
        neighbor_limit: int = NEIGHBOR_LIMIT,
        simulate_threshold: float = SIMULATE_THRESHOLD,
        quarantine_threshold: float = QUARANTINE_THRESHOLD,
    ) -> None:
        self.history_index = history_index
        self.target_delay = int(target_delay)
        self.agent = agent or ExpressionSemanticCriticAgent()
        self.neighbor_limit = max(1, int(neighbor_limit))
        self.simulate_threshold = float(simulate_threshold)
        self.quarantine_threshold = float(quarantine_threshold)

    def apply(
        self,
        *,
        theory: TheorySpec,
        candidates: list[AcceptedExpression],
        requested_count: int,
        portfolio_context: dict[str, Any],
        history_summary: dict[str, Any],
        mock: bool = False,
    ) -> SemanticGateResult:
        if not candidates:
            return SemanticGateResult([], [], [], [], _empty_report(requested_count))
        if self.history_index is None or not self.history_index.index_path.exists():
            accepted = []
            responses = []
            for i, row in enumerate(candidates):
                scorecard = _default_scorecard(f"candidate_{i}", row, [], reason="history index unavailable")
                routing = _routing("ACCEPT_NO_HISTORY", "semantic gate skipped because history index is unavailable", False)
                _apply_metadata(row, scorecard, routing, [])
                accepted.append(row)
                responses.append({"candidate_key": f"candidate_{i}", "scorecard": scorecard, "routing": routing})
            return SemanticGateResult(
                accepted,
                [],
                [],
                responses,
                _report(requested_count, len(accepted), 0, responses),
            )

        request_rows = []
        for i, row in enumerate(candidates):
            key = f"candidate_{i}"
            neighbors = retrieve_semantic_neighbors(
                self.history_index,
                row,
                target_delay=self.target_delay,
                limit=self.neighbor_limit,
            )
            request_rows.append(
                {
                    "candidate_key": key,
                    "candidate": _candidate_payload(row),
                    "nearest_neighbors": neighbors,
                }
            )

        payload = {
            "task": "score_expression_semantic_similarity",
            "target_delay": self.target_delay,
            "theory": theory.to_dict(),
            "history_summary": _compact_history(history_summary),
            "current_submitted_portfolio": compact_portfolio_context(
                portfolio_context or {},
                target_delay=self.target_delay,
            ),
            "score_scale": "0 means unrelated; 100 means effectively the same alpha.",
            "routing_note": "Return scorecards only. Deterministic code will decide simulate versus quarantine.",
            "candidates": request_rows,
        }
        if mock:
            raw = [_mock_scorecard(row["candidate_key"], candidates[i], row["nearest_neighbors"]) for i, row in enumerate(request_rows)]
        else:
            raw = self.agent.score_candidates(payload=payload)
        scorecards = _coerce_scorecards(raw, len(candidates))

        direct_accepts: list[tuple[str, AcceptedExpression, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
        quota_pool: list[tuple[str, AcceptedExpression, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
        quarantined: list[dict[str, Any]] = []
        response_rows: list[dict[str, Any]] = []

        for i, row in enumerate(candidates):
            key = f"candidate_{i}"
            neighbors = request_rows[i]["nearest_neighbors"]
            scorecard = _normalize_scorecard(scorecards.get(key), key, row, neighbors)
            route = _preliminary_route(
                scorecard,
                row,
                simulate_threshold=self.simulate_threshold,
                quarantine_threshold=self.quarantine_threshold,
            )
            item = (key, row, scorecard, route, neighbors)
            if route["bucket"] == "accept":
                direct_accepts.append(item)
            elif route["bucket"] == "quota":
                quota_pool.append(item)
            else:
                quarantined.append(_quarantine_row(row, scorecard, route, neighbors))
                response_rows.append({"candidate_key": key, "scorecard": scorecard, "routing": route})

        budget = max(2, math.ceil(max(1, requested_count) * SOFT_QUOTA_FRACTION))
        quota_pool.sort(key=lambda item: (_score(item[2], "overall_similarity_score"), -_score(item[2], "confidence")))
        accepted = []
        for key, row, scorecard, route, neighbors in direct_accepts:
            _apply_metadata(row, scorecard, route, neighbors)
            accepted.append(row)
            response_rows.append({"candidate_key": key, "scorecard": scorecard, "routing": route})
        for i, (key, row, scorecard, route, neighbors) in enumerate(quota_pool):
            if i < budget:
                route = dict(route)
                route.update(
                    {
                        "decision": "ACCEPT_SOFT_QUOTA",
                        "allowed_by_quota": True,
                        "reason": route["reason"] + f"; allowed by soft quota {i + 1}/{budget}",
                    }
                )
                _apply_metadata(row, scorecard, route, neighbors)
                accepted.append(row)
                response_rows.append({"candidate_key": key, "scorecard": scorecard, "routing": route})
            else:
                route = dict(route)
                route.update(
                    {
                        "decision": "QUARANTINE_SOFT_QUOTA_EXCEEDED",
                        "allowed_by_quota": False,
                        "reason": route["reason"] + f"; soft quota exhausted ({budget})",
                    }
                )
                quarantined.append(_quarantine_row(row, scorecard, route, neighbors))
                response_rows.append({"candidate_key": key, "scorecard": scorecard, "routing": route})

        return SemanticGateResult(
            accepted=accepted,
            quarantined=quarantined,
            requests=[payload],
            responses=response_rows,
            report=_report(requested_count, len(accepted), len(quarantined), response_rows, budget=budget),
        )


def retrieve_semantic_neighbors(
    history_index: HistoryIndex,
    candidate: AcceptedExpression,
    *,
    target_delay: int,
    limit: int = NEIGHBOR_LIMIT,
) -> list[dict[str, Any]]:
    if not history_index.index_path.exists():
        return []
    candidate_fields = set(candidate.fields)
    candidate_skeleton = set(candidate.operator_skeleton)
    candidate_datasets = {d.lower() for d in candidate.datasets}
    candidate_categories = {c.lower() for c in candidate.categories}
    candidate_semantic = semantic_normalize_expression(candidate.expression)
    candidate_windowless = _windowless_semantic(candidate.expression)

    conn = sqlite3.connect(history_index.index_path)
    try:
        rows = conn.execute(
            """
            SELECT id, source_path, expression, delay, alpha_id, status, dataset_id, category,
	                   search_lane, template_id, fields_json, skeleton_json, failed_checks_json,
	                   is_sharpe, is_fitness, is_turnover, max_self_correlation,
	                   admission_decision, admission_reason, admission_global_max_corr,
	                   admission_brain_score_delta, admission_delta_pool_sharpe,
	                   admission_signal, standalone_pass
            FROM records
            """
        ).fetchall()
    finally:
        conn.close()

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        fields = set(_json_loads(row[10], []))
        skeleton = set(_json_loads(row[11], []))
        dataset = str(row[6] or "").lower()
        category = str(row[7] or "").lower()
        expression = str(row[2] or "")
        same_delay = int(row[3]) == int(target_delay)
        components: dict[str, float] = {}

        if semantic_normalize_expression(expression) == candidate_semantic:
            components["semantic_expression_match"] = 100.0
        if _windowless_semantic(expression) == candidate_windowless:
            components["windowless_match"] = 30.0
        components["field_jaccard"] = 35.0 * _jaccard(candidate_fields, fields)
        components["skeleton_jaccard"] = 25.0 * _jaccard(candidate_skeleton, skeleton)
        if dataset and dataset in candidate_datasets:
            components["dataset_match"] = 10.0
        if category and category in candidate_categories:
            components["category_match"] = 5.0
        if same_delay:
            components["same_delay"] = 4.0
        else:
            components["cross_delay"] = 3.0
        if _safe_float(row[19]) >= 0.70:
            components["high_corr_neighbor"] = 12.0
        contribution_delta = row[20] if row[20] is not None else row[21]
        if contribution_delta is not None and _safe_float(contribution_delta) < 0:
            components["pool_hurt_neighbor"] = 12.0
        if row[17] in {"REJECT", "WATCH"}:
            components["admission_risk"] = 4.0
        score = sum(components.values())
        if score <= 0:
            continue
        scored.append(
            (
                score,
                {
                    "history_id": row[0],
                    "score": round(score, 4),
                    "score_components": components,
                    "source_path": row[1],
                    "expression": expression[:260],
                    "delay": row[3],
                    "same_delay": same_delay,
                    "alpha_id": row[4],
                    "status": row[5],
                    "dataset_id": row[6],
                    "category": row[7],
                    "search_lane": row[8],
                    "template_id": row[9],
                    "fields": sorted(fields),
                    "operator_skeleton": _json_loads(row[11], []),
                    "failed_checks": _json_loads(row[12], []),
                    "is_sharpe": row[13],
                    "is_fitness": row[14],
                    "is_turnover": row[15],
                    "max_self_correlation": row[16],
                    "admission_decision": row[17],
                    "admission_reason": row[18],
                    "admission_global_max_corr": row[19],
                    "admission_brain_score_delta": row[20],
                    "admission_delta_pool_sharpe": row[21],
                    "admission_signal": row[22],
                    "standalone_pass": bool(row[23]),
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:limit]]


def _preliminary_route(
    scorecard: dict[str, Any],
    row: AcceptedExpression,
    *,
    simulate_threshold: float,
    quarantine_threshold: float,
) -> dict[str, Any]:
    score = _score(scorecard, "overall_similarity_score")
    confidence = _score(scorecard, "confidence")
    tags = set(scorecard.get("semantic_tags") or [])
    if row.cross_delay_retest:
        tags.add("cross_delay_retest")
    if "sign_flip_of_prior_expression" in row.validation_notes:
        tags.add("sign_repair")
    scorecard["semantic_tags"] = sorted(tags)
    if "sign_repair" in tags:
        return _routing("ACCEPT_SIGN_REPAIR", "sign repair retest is allowed with low novelty metadata", False, "accept")
    if "cross_delay_retest" in tags:
        return _routing("ACCEPT_CROSS_DELAY_RETEST", "cross-delay retest is allowed because delay branches are separate", False, "accept")
    high_conf_portfolio_risk = confidence >= HIGH_CONFIDENCE_THRESHOLD and bool(
        tags & {"portfolio_crowded", "pool_hurt_neighbor"}
    )
    if score < simulate_threshold and not high_conf_portfolio_risk:
        return _routing("ACCEPT_LOW_SIMILARITY", f"overall similarity {score:.1f} < {simulate_threshold:.1f}", False, "accept")
    if score >= quarantine_threshold:
        return _routing("QUARANTINE_HIGH_SIMILARITY", f"overall similarity {score:.1f} >= {quarantine_threshold:.1f}", False, "quarantine")
    if high_conf_portfolio_risk:
        return _routing("SOFT_QUOTA_PORTFOLIO_RISK", "high-confidence portfolio crowding/pool-hurt risk", False, "quota")
    return _routing("SOFT_QUOTA_MEDIUM_SIMILARITY", f"overall similarity {score:.1f} is in review band", False, "quota")


def _apply_metadata(
    row: AcceptedExpression,
    scorecard: dict[str, Any],
    routing: dict[str, Any],
    neighbors: list[dict[str, Any]],
) -> None:
    row.source["semantic_similarity_score"] = _score(scorecard, "overall_similarity_score")
    row.source["semantic_similarity_dimensions"] = {
        "economic_mechanism_similarity": _score(scorecard, "economic_mechanism_similarity"),
        "field_role_similarity": _score(scorecard, "field_role_similarity"),
        "operator_structure_similarity": _score(scorecard, "operator_structure_similarity"),
        "trading_condition_similarity": _score(scorecard, "trading_condition_similarity"),
        "portfolio_redundancy_risk": _score(scorecard, "portfolio_redundancy_risk"),
    }
    row.source["semantic_similarity_confidence"] = _score(scorecard, "confidence")
    row.source["semantic_tags"] = list(scorecard.get("semantic_tags") or [])
    row.source["semantic_nearest_neighbors"] = _neighbor_refs(neighbors)
    row.source["semantic_gate_decision"] = routing["decision"]
    row.source["semantic_gate_reason"] = routing["reason"]
    row.source["semantic_gate_allowed_by_quota"] = bool(routing.get("allowed_by_quota"))
    row.source["nearest_matching_mechanism"] = scorecard.get("nearest_matching_mechanism", "")
    if "sign_repair" in row.source["semantic_tags"]:
        row.novelty_score = min(row.novelty_score, 0.25)
    if "cross_delay_retest" in row.source["semantic_tags"]:
        row.cross_delay_retest = True
        row.novelty_score = min(row.novelty_score, 0.5)


def _normalize_scorecard(
    raw: Any,
    key: str,
    row: AcceptedExpression,
    neighbors: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _default_scorecard(key, row, neighbors, reason="critic returned missing or malformed scorecard")
    required_scores = {
        "overall_similarity_score",
        "economic_mechanism_similarity",
        "field_role_similarity",
        "operator_structure_similarity",
        "trading_condition_similarity",
        "portfolio_redundancy_risk",
        "confidence",
    }
    if any(field not in raw for field in required_scores):
        return _default_scorecard(key, row, neighbors, reason="critic scorecard missing required numeric fields")
    out = dict(raw)
    out["candidate_key"] = str(out.get("candidate_key") or key)
    for field in (
        "overall_similarity_score",
        "economic_mechanism_similarity",
        "field_role_similarity",
        "operator_structure_similarity",
        "trading_condition_similarity",
        "portfolio_redundancy_risk",
        "confidence",
    ):
        out[field] = _clamp_score(out.get(field))
    out["reason"] = str(out.get("reason") or "")
    out["nearest_matching_mechanism"] = str(out.get("nearest_matching_mechanism") or _neighbor_mechanism(neighbors))
    tags = []
    for tag in out.get("semantic_tags") or []:
        clean = str(tag).strip().lower()
        if clean in ALLOWED_TAGS:
            tags.append(clean)
    out["semantic_tags"] = sorted(set(tags))
    return out


def _coerce_scorecards(data: Any, expected_count: int) -> dict[str, dict[str, Any]]:
    if isinstance(data, dict):
        if isinstance(data.get("scorecards"), list):
            items = data["scorecards"]
        elif isinstance(data.get("results"), list):
            items = data["results"]
        else:
            items = [dict(v, candidate_key=k) for k, v in data.items() if isinstance(v, dict)]
    elif isinstance(data, list):
        items = data
    else:
        items = []
    out: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        key = str(item.get("candidate_key") or f"candidate_{i}")
        out[key] = item
    for i in range(expected_count):
        out.setdefault(f"candidate_{i}", {})
    return out


def _mock_scorecard(key: str, row: AcceptedExpression, neighbors: list[dict[str, Any]]) -> dict[str, Any]:
    tags = []
    if row.cross_delay_retest:
        tags.append("cross_delay_retest")
    if "sign_flip_of_prior_expression" in row.validation_notes:
        tags.append("sign_repair")
    score = 35.0
    if neighbors:
        score = min(62.0, max(20.0, float(neighbors[0].get("score") or 0.0) * 0.5))
    if tags:
        score = max(score, 88.0)
    return {
        "candidate_key": key,
        "overall_similarity_score": score,
        "economic_mechanism_similarity": score,
        "field_role_similarity": score,
        "operator_structure_similarity": score,
        "trading_condition_similarity": min(score, 50.0),
        "portfolio_redundancy_risk": 20.0,
        "confidence": 60.0,
        "reason": "mock semantic scorecard",
        "nearest_matching_mechanism": _neighbor_mechanism(neighbors),
        "semantic_tags": tags,
    }


def _default_scorecard(key: str, row: AcceptedExpression, neighbors: list[dict[str, Any]], *, reason: str) -> dict[str, Any]:
    return {
        "candidate_key": key,
        "overall_similarity_score": 90.0,
        "economic_mechanism_similarity": 90.0,
        "field_role_similarity": 90.0,
        "operator_structure_similarity": 90.0,
        "trading_condition_similarity": 90.0,
        "portfolio_redundancy_risk": 90.0,
        "confidence": 0.0,
        "reason": reason,
        "nearest_matching_mechanism": _neighbor_mechanism(neighbors),
        "semantic_tags": [],
    }


def _candidate_payload(row: AcceptedExpression) -> dict[str, Any]:
    return {
        "expression": row.expression,
        "fields": row.fields,
        "datasets": row.datasets,
        "categories": row.categories,
        "field_roles": row.field_roles,
        "operator_skeleton": row.operator_skeleton,
        "validation_notes": row.validation_notes,
        "cross_delay_retest": row.cross_delay_retest,
        "mechanism_translation": row.source.get("mechanism_translation", ""),
        "novelty_argument": row.source.get("novelty_argument", ""),
        "expected_sign": row.source.get("expected_sign", ""),
    }


def _quarantine_row(
    row: AcceptedExpression,
    scorecard: dict[str, Any],
    routing: dict[str, Any],
    neighbors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "expression": row.expression,
        "fields": row.fields,
        "datasets": row.datasets,
        "operator_skeleton": row.operator_skeleton,
        "source": row.source,
        "scorecard": scorecard,
        "routing": routing,
        "nearest_neighbors": neighbors,
    }


def _routing(decision: str, reason: str, allowed_by_quota: bool, bucket: str = "accept") -> dict[str, Any]:
    return {
        "decision": decision,
        "reason": reason,
        "allowed_by_quota": allowed_by_quota,
        "bucket": bucket,
    }


def _report(
    requested_count: int,
    accepted_count: int,
    quarantined_count: int,
    responses: list[dict[str, Any]],
    *,
    budget: int | None = None,
) -> dict[str, Any]:
    decisions: dict[str, int] = {}
    for row in responses:
        decision = str((row.get("routing") or {}).get("decision") or "UNKNOWN")
        decisions[decision] = decisions.get(decision, 0) + 1
    return {
        "requested_count": requested_count,
        "accepted_count": accepted_count,
        "quarantined_count": quarantined_count,
        "soft_quota_budget": budget,
        "decisions": decisions,
    }


def _empty_report(requested_count: int) -> dict[str, Any]:
    return {
        "requested_count": requested_count,
        "accepted_count": 0,
        "quarantined_count": 0,
        "soft_quota_budget": max(2, math.ceil(max(1, requested_count) * SOFT_QUOTA_FRACTION)),
        "decisions": {},
    }


def _compact_history(summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in ("total_records", "standalone_pass_records", "dataset", "category", "operator", "delay", "pool_impact"):
        value = summary.get(key)
        if isinstance(value, list):
            value = value[:15]
        out[key] = value
    return out


def _neighbor_refs(neighbors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = []
    for row in neighbors[:5]:
        refs.append(
            {
                "history_id": row.get("history_id"),
                "score": row.get("score"),
                "delay": row.get("delay"),
                "same_delay": row.get("same_delay"),
                "dataset_id": row.get("dataset_id"),
                "operator_skeleton": row.get("operator_skeleton"),
                "admission_decision": row.get("admission_decision"),
                "admission_brain_score_delta": row.get("admission_brain_score_delta"),
                "admission_delta_pool_sharpe": row.get("admission_delta_pool_sharpe"),
                "admission_signal": row.get("admission_signal"),
                "admission_global_max_corr": row.get("admission_global_max_corr"),
            }
        )
    return refs


def _neighbor_mechanism(neighbors: list[dict[str, Any]]) -> str:
    if not neighbors:
        return ""
    top = neighbors[0]
    return (
        f"{top.get('dataset_id') or 'unknown'} / "
        f"{'>'.join(top.get('operator_skeleton') or []) or 'unknown'} / "
        f"{', '.join((top.get('fields') or [])[:4])}"
    )


def _score(scorecard: dict[str, Any], field: str) -> float:
    return _clamp_score(scorecard.get(field))


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(100.0, score))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _windowless_semantic(expression: str) -> str:
    import re

    expr = semantic_normalize_expression(expression)
    return re.sub(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?", "#", expr)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
