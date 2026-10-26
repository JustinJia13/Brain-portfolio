"""Alpha-LLM semantic novelty gate.

This module is intentionally local to alpha_discovery/legacy.llm_grammar_control_method. It uses the alpha_discovery/legacy.llm_grammar_control_method
similarity history for deterministic neighbor retrieval, then asks an LLM critic
only about candidates with meaningful neighbor overlap or portfolio-risk
neighbors.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.expr.models import CandidateSpec

from .similarity import SimilarityFilter

_LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-6"
ALLOWED_TAGS = {
    "same_fields_new_shape",
    "new_fields_same_shape",
    "same_mechanism_new_dataset",
    "portfolio_crowded",
    "pool_hurt_neighbor",
    "high_corr_neighbor",
    "window_only_variant",
    "operator_relabel",
}

_SYSTEM_PROMPT = """\
You are a semantic novelty critic for WorldQuant Brain FASTEXPR alpha candidates.

You receive proposed alpha candidates and nearest prior candidates retrieved by
deterministic history search. Score whether each candidate is economically and
structurally redundant with the history.

Use this scale:
- 0 means unrelated.
- 65 means similar enough that simulation should be rationed.
- 85 means effectively the same alpha or highly redundant with portfolio risk.
- 100 means exact duplicate.

Return JSON only:
{
  "scorecards": [
    {
      "candidate_key": "candidate_0",
      "overall_similarity_score": 0-100,
      "economic_mechanism_similarity": 0-100,
      "field_role_similarity": 0-100,
      "operator_structure_similarity": 0-100,
      "trading_condition_similarity": 0-100,
      "portfolio_redundancy_risk": 0-100,
      "confidence": 0-100,
      "reason": "short reason",
      "nearest_matching_mechanism": "short description",
      "semantic_tags": ["same_fields_new_shape"]
    }
  ]
}

Allowed semantic_tags are: same_fields_new_shape, new_fields_same_shape,
same_mechanism_new_dataset, portfolio_crowded, pool_hurt_neighbor,
high_corr_neighbor, window_only_variant, operator_relabel.
"""


@dataclass(frozen=True)
class SemanticThresholds:
    review_threshold: float = 55.0
    simulate_threshold: float = 65.0
    quarantine_threshold: float = 85.0
    high_confidence_threshold: float = 70.0
    soft_quota_fraction: float = 0.15
    neighbor_limit: int = 12


@dataclass
class SemanticNoveltyResult:
    accepted: list[CandidateSpec]
    quarantined: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    report: dict[str, Any]


class AlphaSemanticCriticAgent:
    """Anthropic-backed semantic critic for alpha_discovery/legacy.llm_grammar_control_method candidates."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def score_candidates(self, *, payload: dict[str, Any]) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError("No OpenAI API key for alpha_discovery/legacy.llm_grammar_control_method semantic critic")
            from core import llm as anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        response = self._client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)}],
        )
        raw = response.content[0].text.strip()
        return _extract_json(raw)


class AlphaSemanticNoveltyGate:
    def __init__(
        self,
        *,
        thresholds: SemanticThresholds | None = None,
        agent: AlphaSemanticCriticAgent | None = None,
        artifact_path: Path | None = None,
    ) -> None:
        self.thresholds = thresholds or SemanticThresholds()
        self.agent = agent or AlphaSemanticCriticAgent()
        self.artifact_path = artifact_path

    def filter_candidates(
        self,
        candidates: list[CandidateSpec],
        *,
        similarity_filter: SimilarityFilter,
        requested_count: int,
        batch_index: int | None = None,
    ) -> SemanticNoveltyResult:
        if not candidates:
            return SemanticNoveltyResult([], [], [], _report(0, 0, 0, 0))

        prepared: list[dict[str, Any]] = []
        review_rows: list[dict[str, Any]] = []
        review_keys: set[str] = set()
        for index, candidate in enumerate(candidates):
            key = f"candidate_{index}"
            neighbors = similarity_filter.nearest_neighbors(
                candidate,
                limit=self.thresholds.neighbor_limit,
                min_score=self.thresholds.review_threshold,
            )
            max_neighbor_score = max((float(n.get("score") or 0.0) for n in neighbors), default=0.0)
            portfolio_risk = any(_neighbor_has_portfolio_risk(n) for n in neighbors)
            row = {
                "candidate_key": key,
                "candidate": candidate,
                "candidate_payload": _candidate_payload(candidate),
                "nearest_neighbors": neighbors,
                "max_neighbor_score": max_neighbor_score,
                "portfolio_risk_neighbor": portfolio_risk,
            }
            prepared.append(row)
            if neighbors and (max_neighbor_score >= self.thresholds.review_threshold or portfolio_risk):
                review_rows.append(row)
                review_keys.add(key)

        scorecards: dict[str, dict[str, Any]] = {}
        agent_error = ""
        if review_rows:
            payload = {
                "task": "score_alpha_llm_semantic_novelty",
                "thresholds": {
                    "simulate_threshold": self.thresholds.simulate_threshold,
                    "quarantine_threshold": self.thresholds.quarantine_threshold,
                    "review_threshold": self.thresholds.review_threshold,
                },
                "candidates": [
                    {
                        "candidate_key": row["candidate_key"],
                        "candidate": row["candidate_payload"],
                        "nearest_neighbors": row["nearest_neighbors"],
                    }
                    for row in review_rows
                ],
            }
            try:
                scorecards = _coerce_scorecards(self.agent.score_candidates(payload=payload))
            except Exception as exc:
                agent_error = str(exc)
                _LOG.warning("[SemanticNovelty] critic failed: %s", exc)

        direct: list[dict[str, Any]] = []
        quota: list[dict[str, Any]] = []
        quarantined: list[dict[str, Any]] = []

        for row in prepared:
            if row["candidate_key"] not in review_keys:
                scorecard = _default_scorecard(row, reason="no semantic review threshold reached")
                route = _routing("ACCEPT_NO_REVIEW", "no meaningful semantic neighbor risk", "accept")
            elif agent_error:
                scorecard = _default_scorecard(row, reason=f"semantic critic failed: {agent_error}")
                route = _route_agent_failure(row, self.thresholds)
            else:
                scorecard = _normalize_scorecard(scorecards.get(row["candidate_key"]), row)
                route = _route_scorecard(scorecard, self.thresholds)

            item = {
                "row": row,
                "scorecard": scorecard,
                "route": route,
            }
            if route["bucket"] == "accept":
                direct.append(item)
            elif route["bucket"] == "quota":
                quota.append(item)
            else:
                quarantined.append(item)

        accepted_items: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        for item in direct:
            if len(accepted_items) < requested_count:
                accepted_items.append(item)
            else:
                overflow.append(_reroute(item, "DROP_BATCH_FULL", "batch target already filled", "drop"))

        quota_budget = max(1, math.ceil(max(1, requested_count) * self.thresholds.soft_quota_fraction))
        quota.sort(
            key=lambda item: (
                _score(item["scorecard"], "overall_similarity_score"),
                -_score(item["scorecard"], "confidence"),
            )
        )
        quota_used = 0
        for item in quota:
            if quota_used < quota_budget and len(accepted_items) < requested_count:
                quota_used += 1
                accepted_items.append(
                    _reroute(
                        item,
                        "ACCEPT_SOFT_QUOTA",
                        f"{item['route']['reason']}; allowed by soft quota {quota_used}/{quota_budget}",
                        "accept",
                    )
                )
            else:
                quarantined.append(
                    _reroute(
                        item,
                        "QUARANTINE_SOFT_QUOTA",
                        f"{item['route']['reason']}; soft quota exhausted",
                        "quarantine",
                    )
                )

        decisions = [
            _decision_row(item, batch_index=batch_index)
            for item in [*accepted_items, *quarantined, *overflow]
        ]
        self._persist(decisions)
        accepted = [item["row"]["candidate"] for item in accepted_items]
        quarantined_rows = [_decision_row(item, batch_index=batch_index) for item in quarantined]
        return SemanticNoveltyResult(
            accepted=accepted,
            quarantined=quarantined_rows,
            decisions=decisions,
            report=_report(
                len(candidates),
                len(accepted),
                len(quarantined_rows),
                len(review_rows),
                quota_budget=quota_budget,
                quota_used=quota_used,
                agent_error=agent_error,
            ),
        )

    def _persist(self, decisions: list[dict[str, Any]]) -> None:
        if not self.artifact_path or not decisions:
            return
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with self.artifact_path.open("a") as f:
            for row in decisions:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _route_scorecard(scorecard: dict[str, Any], thresholds: SemanticThresholds) -> dict[str, Any]:
    score = _score(scorecard, "overall_similarity_score")
    confidence = _score(scorecard, "confidence")
    portfolio_score = _score(scorecard, "portfolio_redundancy_risk")
    tags = set(scorecard.get("semantic_tags") or [])
    high_conf_portfolio_risk = confidence >= thresholds.high_confidence_threshold and bool(
        tags & {"portfolio_crowded", "pool_hurt_neighbor", "high_corr_neighbor"} or portfolio_score >= 80.0
    )
    if score < thresholds.simulate_threshold and not high_conf_portfolio_risk:
        return _routing("ACCEPT_LOW_SIMILARITY", f"overall similarity {score:.1f}", "accept")
    if score >= thresholds.quarantine_threshold:
        return _routing("QUARANTINE_HIGH_SIMILARITY", f"overall similarity {score:.1f}", "quarantine")
    if high_conf_portfolio_risk:
        return _routing("SOFT_QUOTA_PORTFOLIO_RISK", "high-confidence portfolio redundancy risk", "quota")
    return _routing("SOFT_QUOTA_MEDIUM_SIMILARITY", f"overall similarity {score:.1f}", "quota")


def _route_agent_failure(row: dict[str, Any], thresholds: SemanticThresholds) -> dict[str, Any]:
    max_score = float(row.get("max_neighbor_score") or 0.0)
    if max_score >= thresholds.quarantine_threshold:
        return _routing(
            "QUARANTINE_SEMANTIC_AGENT_FAILED",
            f"semantic critic failed and deterministic neighbor score is {max_score:.1f}",
            "quarantine",
        )
    return _routing(
        "ACCEPT_SEMANTIC_AGENT_FAILED",
        "semantic critic failed; deterministic neighbor risk below hard quarantine threshold",
        "accept",
    )


def _routing(decision: str, reason: str, bucket: str) -> dict[str, Any]:
    return {"decision": decision, "reason": reason, "bucket": bucket}


def _reroute(item: dict[str, Any], decision: str, reason: str, bucket: str) -> dict[str, Any]:
    out = dict(item)
    route = dict(item["route"])
    route.update({"decision": decision, "reason": reason, "bucket": bucket})
    out["route"] = route
    return out


def _decision_row(item: dict[str, Any], *, batch_index: int | None) -> dict[str, Any]:
    row = item["row"]
    route = item["route"]
    scorecard = item["scorecard"]
    return {
        "batch_index": batch_index,
        "candidate_key": row["candidate_key"],
        "expression": row["candidate"].expression,
        "fields": list(row["candidate"].fields),
        "template_id": row["candidate"].template_id,
        "dataset_id": row["candidate"].dataset_id,
        "category": row["candidate"].category,
        "semantic_gate_decision": route["decision"],
        "semantic_gate_reason": route["reason"],
        "semantic_similarity_score": _score(scorecard, "overall_similarity_score"),
        "semantic_similarity_confidence": _score(scorecard, "confidence"),
        "semantic_similarity_dimensions": {
            "economic_mechanism_similarity": _score(scorecard, "economic_mechanism_similarity"),
            "field_role_similarity": _score(scorecard, "field_role_similarity"),
            "operator_structure_similarity": _score(scorecard, "operator_structure_similarity"),
            "trading_condition_similarity": _score(scorecard, "trading_condition_similarity"),
            "portfolio_redundancy_risk": _score(scorecard, "portfolio_redundancy_risk"),
        },
        "semantic_tags": list(scorecard.get("semantic_tags") or []),
        "nearest_matching_mechanism": scorecard.get("nearest_matching_mechanism", ""),
        "nearest_neighbors": _neighbor_refs(row["nearest_neighbors"]),
    }


def _candidate_payload(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "expression": candidate.expression,
        "fields": list(candidate.fields),
        "template_id": candidate.template_id,
        "dataset_id": candidate.dataset_id,
        "category": candidate.category,
        "search_lane": candidate.search_lane,
        "params": dict(candidate.params),
    }


def _neighbor_refs(neighbors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = []
    for row in neighbors[:5]:
        refs.append(
            {
                "score": row.get("score"),
                "expression": row.get("expression"),
                "fields": row.get("fields"),
                "operator_skeleton": row.get("operator_skeleton"),
                "dataset_id": row.get("dataset_id"),
                "category": row.get("category"),
                "search_lane": row.get("search_lane"),
                "max_self_correlation": row.get("max_self_correlation"),
                "admission_decision": row.get("admission_decision"),
                "admission_global_max_corr": row.get("admission_global_max_corr"),
                "admission_brain_score_delta": row.get("admission_brain_score_delta"),
                "admission_delta_pool_sharpe": row.get("admission_delta_pool_sharpe"),
                "admission_signal": row.get("admission_signal"),
            }
        )
    return refs


def _neighbor_has_portfolio_risk(neighbor: dict[str, Any]) -> bool:
    components = neighbor.get("score_components") or {}
    return bool(components.get("high_corr_neighbor") or components.get("pool_hurt_neighbor"))


def _default_scorecard(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    score = float(row.get("max_neighbor_score") or 0.0)
    tags = []
    if row.get("portfolio_risk_neighbor"):
        tags.append("portfolio_crowded")
    return {
        "candidate_key": row["candidate_key"],
        "overall_similarity_score": score,
        "economic_mechanism_similarity": score,
        "field_role_similarity": score,
        "operator_structure_similarity": score,
        "trading_condition_similarity": score,
        "portfolio_redundancy_risk": 80.0 if row.get("portfolio_risk_neighbor") else 0.0,
        "confidence": 50.0,
        "reason": reason,
        "nearest_matching_mechanism": _neighbor_mechanism(row["nearest_neighbors"]),
        "semantic_tags": tags,
    }


def _normalize_scorecard(raw: Any, row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _default_scorecard(row, reason="critic returned missing or malformed scorecard")
    out = dict(raw)
    out["candidate_key"] = str(out.get("candidate_key") or row["candidate_key"])
    for key in (
        "overall_similarity_score",
        "economic_mechanism_similarity",
        "field_role_similarity",
        "operator_structure_similarity",
        "trading_condition_similarity",
        "portfolio_redundancy_risk",
        "confidence",
    ):
        out[key] = _clamp_score(out.get(key))
    out["reason"] = str(out.get("reason") or "")
    out["nearest_matching_mechanism"] = str(out.get("nearest_matching_mechanism") or _neighbor_mechanism(row["nearest_neighbors"]))
    tags = []
    for tag in out.get("semantic_tags") or []:
        clean = str(tag).strip().lower()
        if clean in ALLOWED_TAGS:
            tags.append(clean)
    out["semantic_tags"] = sorted(set(tags))
    return out


def _coerce_scorecards(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, dict):
        rows = raw.get("scorecards")
        if rows is None and "candidate_key" in raw:
            rows = [raw]
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []
    out = {}
    for row in rows or []:
        if isinstance(row, dict):
            key = str(row.get("candidate_key") or "")
            if key:
                out[key] = row
    return out


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```"))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidates = []
    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        for match in re.finditer(re.escape(start_ch), text):
            start = match.start()
            depth = 0
            in_string = False
            escaped = False
            for index, char in enumerate(text[start:], start):
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == start_ch:
                    depth += 1
                elif char == end_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            candidates.append(json.loads(text[start : index + 1]))
                        except json.JSONDecodeError:
                            pass
                        break
    if not candidates:
        raise ValueError("semantic critic did not return JSON")
    return max(candidates, key=_json_payload_score)


def _json_payload_score(value: Any) -> int:
    if isinstance(value, dict):
        rows = value.get("scorecards")
        if isinstance(rows, list):
            return 1000 + len(rows)
        return 10
    if isinstance(value, list):
        return 100 + len(value)
    return 0


def _neighbor_mechanism(neighbors: list[dict[str, Any]]) -> str:
    if not neighbors:
        return ""
    top = neighbors[0]
    return f"{top.get('dataset_id') or '?'} {top.get('operator_skeleton') or []}"


def _score(scorecard: dict[str, Any], key: str) -> float:
    return _clamp_score(scorecard.get(key))


def _clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, number))


def _report(
    candidate_count: int,
    accepted_count: int,
    quarantined_count: int,
    reviewed_count: int,
    *,
    quota_budget: int = 0,
    quota_used: int = 0,
    agent_error: str = "",
) -> dict[str, Any]:
    return {
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "quarantined_count": quarantined_count,
        "reviewed_count": reviewed_count,
        "quota_budget": quota_budget,
        "quota_used": quota_used,
        "agent_error": agent_error,
    }


__all__ = [
    "AlphaSemanticCriticAgent",
    "AlphaSemanticNoveltyGate",
    "SemanticNoveltyResult",
    "SemanticThresholds",
]
