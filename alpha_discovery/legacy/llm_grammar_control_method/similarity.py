"""Novelty filter for LLM-proposed alpha candidates.

Prevents submitting expressions that are too similar to already-submitted ones.
Hard-filter layers (cheapest first):
  1. Exact expression string match
  2. Operator skeleton count >= max_skeleton_count
  3. Feature Jaccard (fields + operator tokens) >= jaccard_threshold against recent hot set

The filter can also retain external history for semantic neighbor retrieval
without letting large external logs saturate the hard skeleton counter.
"""

from __future__ import annotations

import collections
import logging
import re
from copy import deepcopy
from typing import Any
from typing import NamedTuple

from core.expr.models import CandidateSpec
from core.qtable import SkeletonCounter

try:
    from core.expr.operators import OPERATOR_REGISTRY
except Exception:  # pragma: no cover - import should normally work
    OPERATOR_REGISTRY = {}

_LOG = logging.getLogger(__name__)

_FUNC_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_KNOWN_OPERATORS = frozenset(OPERATOR_REGISTRY.keys())
_RESERVED_IDENTIFIERS = frozenset(
    {
        "market",
        "sector",
        "industry",
        "subindustry",
        "country",
        "exchange",
        "none",
        "true",
        "false",
        "nan",
        "inf",
        "k",
        "buckets",
        "bucket",
        "filter",
        "dense",
        "rettype",
        "days",
    }
)


def normalize_expression(expression: str) -> str:
    """Return a stable key for exact-expression duplicate checks."""
    return re.sub(r"\s+", "", str(expression or "")).lower()


def _operator_tokens_from_expression(expression: str) -> tuple[str, ...]:
    return tuple(_FUNC_RE.findall(str(expression or "")))


def _operator_tokens_from_template(template_id: str) -> tuple[str, ...]:
    if not template_id or "|" not in template_id:
        return ()
    return tuple(part for part in template_id.split("|") if part and not part.startswith("enhance:"))


def _record_operator_tokens(record: dict[str, Any]) -> tuple[str, ...]:
    template_ops = _operator_tokens_from_template(str(record.get("template_id") or ""))
    expression_ops = _operator_tokens_from_expression(str(record.get("expression") or ""))
    return template_ops or expression_ops


def _candidate_operator_tokens(candidate: CandidateSpec) -> tuple[str, ...]:
    template_ops = _operator_tokens_from_template(candidate.template_id)
    expression_ops = _operator_tokens_from_expression(candidate.expression)
    return template_ops or expression_ops


def _infer_fields_from_expression(expression: str, operators: tuple[str, ...]) -> tuple[str, ...]:
    op_set = set(operators) | _KNOWN_OPERATORS
    fields: list[str] = []
    seen: set[str] = set()
    for token in _IDENT_RE.findall(str(expression or "")):
        clean = token.strip()
        lower = clean.lower()
        if lower in _RESERVED_IDENTIFIERS or clean in op_set or lower in op_set:
            continue
        if clean not in seen:
            fields.append(clean)
            seen.add(clean)
    return tuple(fields)


def _record_fields(record: dict[str, Any], operators: tuple[str, ...]) -> tuple[str, ...]:
    fields = record.get("fields") or []
    if isinstance(fields, str):
        fields = [fields]
    clean = tuple(str(field) for field in fields if field)
    if clean:
        return clean
    return _infer_fields_from_expression(str(record.get("expression") or ""), operators)


def _record_feature_row(record: dict[str, Any]) -> dict[str, Any] | None:
    expression = str(record.get("expression") or "")
    operators = _record_operator_tokens(record)
    fields = _record_fields(record, operators)
    fingerprint = frozenset([*fields, *operators])
    if not expression and not fingerprint:
        return None
    return {
        "expression": expression,
        "normalized_expression": normalize_expression(expression),
        "fields": fields,
        "operators": operators,
        "fingerprint": fingerprint,
        "template_id": str(record.get("template_id") or ""),
        "dataset_id": str(record.get("dataset_id") or ""),
        "category": str(record.get("category") or ""),
        "search_lane": str(record.get("search_lane") or ""),
        "alpha_id": record.get("alpha_id"),
        "status": record.get("status"),
        "max_self_correlation": record.get("max_self_correlation"),
        "admission_decision": record.get("admission_decision"),
        "admission_reason": record.get("admission_reason"),
        "admission_global_max_corr": record.get("admission_global_max_corr"),
        "admission_brain_score_delta": record.get("admission_brain_score_delta"),
        "admission_delta_pool_sharpe": record.get("admission_delta_pool_sharpe"),
        "admission_signal": record.get("admission_signal"),
    }


def _candidate_feature_row(candidate: CandidateSpec) -> dict[str, Any]:
    operators = _candidate_operator_tokens(candidate)
    fields = tuple(candidate.fields)
    return {
        "expression": candidate.expression,
        "normalized_expression": normalize_expression(candidate.expression),
        "fields": fields,
        "operators": operators,
        "fingerprint": frozenset([*fields, *operators]),
        "template_id": candidate.template_id,
        "dataset_id": candidate.dataset_id,
        "category": candidate.category,
        "search_lane": candidate.search_lane,
        "alpha_id": None,
        "status": "candidate",
        "max_self_correlation": None,
        "admission_decision": None,
        "admission_reason": None,
        "admission_global_max_corr": None,
        "admission_brain_score_delta": None,
        "admission_delta_pool_sharpe": None,
        "admission_signal": None,
    }


def _candidate_fingerprint(candidate: CandidateSpec) -> frozenset[str]:
    """Composite fingerprint: all field IDs plus all operator tokens from template_id."""
    return _candidate_feature_row(candidate)["fingerprint"]


def _record_fingerprint(record: dict) -> frozenset[str] | None:
    """Same fingerprint computed from a sim log record dict."""
    row = _record_feature_row(record)
    if not row:
        return None
    fp = row["fingerprint"]
    return fp if fp else None


def _jaccard(fp1: set[str] | frozenset[str], fp2: set[str] | frozenset[str]) -> float:
    union = set(fp1) | set(fp2)
    if not union:
        return 1.0
    return len(set(fp1) & set(fp2)) / len(union)


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class FilterVerdict(NamedTuple):
    skip: bool
    reason: str


class SimilarityFilter:
    """Three-layer similarity filter.

    Bootstrapped from existing sim records (master.jsonl) at construction time.
    After filtering a candidate, call register() to track it.
    """

    def __init__(
        self,
        max_skeleton_count: int = 20,
        jaccard_threshold: float = 0.85,
        hot_set_size: int = 5000,
    ) -> None:
        self._max_skeleton_count = max_skeleton_count
        self._jaccard_threshold = jaccard_threshold
        self._submitted_exprs: set[str] = set()
        self._submitted_norm_exprs: set[str] = set()
        self._skeleton_counter: SkeletonCounter = SkeletonCounter()
        self._hot_fingerprints: collections.deque[frozenset[str]] = collections.deque(
            maxlen=hot_set_size
        )
        self._history: collections.deque[dict[str, Any]] = collections.deque(maxlen=hot_set_size * 2)

    def clone(self) -> "SimilarityFilter":
        """Return an in-memory copy for provisional batch filtering."""
        other = type(self)(
            max_skeleton_count=self._max_skeleton_count,
            jaccard_threshold=self._jaccard_threshold,
            hot_set_size=self._hot_fingerprints.maxlen or 5000,
        )
        other._submitted_exprs = set(self._submitted_exprs)
        other._submitted_norm_exprs = set(self._submitted_norm_exprs)
        other._skeleton_counter = deepcopy(self._skeleton_counter)
        other._hot_fingerprints = collections.deque(
            self._hot_fingerprints,
            maxlen=self._hot_fingerprints.maxlen,
        )
        other._history = collections.deque(self._history, maxlen=self._history.maxlen)
        return other

    def bootstrap(
        self,
        sim_records: list[dict],
        *,
        count_skeletons: bool = True,
        source_name: str = "",
    ) -> None:
        """Populate filter state from existing sim log records.

        count_skeletons should be True for the active alpha_discovery/legacy.llm_grammar_control_method log and False
        for broad external logs. External history should block duplicates and
        provide semantic neighbors without making common skeletons impossible.

        Layer 3 hot set is populated from the most recent hot_set_size records.
        """
        for rec in sim_records:
            row = _record_feature_row(rec)
            if not row:
                continue
            expr = row["expression"]
            if expr:
                self._submitted_exprs.add(expr)
                self._submitted_norm_exprs.add(row["normalized_expression"])
            tid = row["template_id"]
            if count_skeletons and tid:
                self._skeleton_counter.increment(tid)
            self._history.append(row)

        # Hot fingerprints: only the most recent records (ring buffer order)
        maxlen = self._hot_fingerprints.maxlen or 500
        recent = sim_records[-maxlen:]
        for rec in recent:
            fp = _record_fingerprint(rec)
            if fp:
                self._hot_fingerprints.append(fp)

        _LOG.info(
            "SimilarityFilter bootstrapped%s: %d exprs, %d norm exprs, %d skeletons, %d hot fps",
            f" from {source_name}" if source_name else "",
            len(self._submitted_exprs),
            len(self._submitted_norm_exprs),
            len(self._skeleton_counter),
            len(self._hot_fingerprints),
        )

    def is_too_similar(self, candidate: CandidateSpec) -> FilterVerdict:
        """Return (skip=True, reason) if candidate should be filtered out."""
        # Layer 1: exact expression string
        if candidate.expression in self._submitted_exprs:
            return FilterVerdict(True, "exact_match")
        if normalize_expression(candidate.expression) in self._submitted_norm_exprs:
            return FilterVerdict(True, "normalized_exact_match")

        # Layer 2: skeleton count
        count = self._skeleton_counter.count(candidate.template_id)
        if count >= self._max_skeleton_count:
            return FilterVerdict(True, f"skeleton_count={count}>={self._max_skeleton_count}")

        # Layer 3: Jaccard against hot ring buffer
        if self._hot_fingerprints:
            fp = _candidate_fingerprint(candidate)
            for existing_fp in self._hot_fingerprints:
                j = _jaccard(fp, existing_fp)
                if j >= self._jaccard_threshold:
                    return FilterVerdict(True, f"jaccard={j:.3f}>={self._jaccard_threshold}")

        return FilterVerdict(False, "")

    def register(self, candidate: CandidateSpec) -> None:
        """Mark a candidate as accepted.  Call after is_too_similar() returns False."""
        self._submitted_exprs.add(candidate.expression)
        self._submitted_norm_exprs.add(normalize_expression(candidate.expression))
        self._skeleton_counter.increment(candidate.template_id)
        row = _candidate_feature_row(candidate)
        self._hot_fingerprints.append(row["fingerprint"])
        self._history.append(row)

    def nearest_neighbors(
        self,
        candidate: CandidateSpec,
        *,
        limit: int = 12,
        min_score: float = 55.0,
    ) -> list[dict[str, Any]]:
        """Return semantic-neighbor candidates from retained history.

        Scores are deterministic retrieval scores, not final LLM similarity
        judgments. They decide whether a candidate deserves semantic review and
        give the critic compact evidence.
        """
        candidate_row = _candidate_feature_row(candidate)
        candidate_fields = set(candidate_row["fields"])
        candidate_ops = set(candidate_row["operators"])
        candidate_fp = set(candidate_row["fingerprint"])
        candidate_dataset = str(candidate_row["dataset_id"]).lower()
        candidate_category = str(candidate_row["category"]).lower()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._history:
            components: dict[str, float] = {}
            if row["normalized_expression"] and row["normalized_expression"] == candidate_row["normalized_expression"]:
                components["normalized_expression_match"] = 100.0

            fields = set(row["fields"])
            operators = set(row["operators"])
            fp = set(row["fingerprint"])
            field_jaccard = _jaccard(candidate_fields, fields)
            op_jaccard = _jaccard(candidate_ops, operators)
            components["field_jaccard"] = 35.0 * field_jaccard
            components["operator_jaccard"] = 25.0 * op_jaccard

            dataset = str(row.get("dataset_id") or "").lower()
            category = str(row.get("category") or "").lower()
            if dataset and dataset == candidate_dataset:
                components["dataset_match"] = 10.0
            if category and category == candidate_category:
                components["category_match"] = 5.0

            high_corr = max(
                _safe_float(row.get("max_self_correlation")),
                _safe_float(row.get("admission_global_max_corr")),
            ) >= 0.70
            contribution_delta = row.get("admission_brain_score_delta")
            if contribution_delta is None:
                contribution_delta = row.get("admission_delta_pool_sharpe")
            pool_hurt = contribution_delta is not None and _safe_float(contribution_delta) < 0
            if high_corr:
                components["high_corr_neighbor"] = 12.0
            if pool_hurt:
                components["pool_hurt_neighbor"] = 12.0
            if str(row.get("admission_decision") or "").upper() in {"REJECT", "WATCH"}:
                components["admission_risk"] = 4.0

            score = sum(components.values())
            overlap_score = components.get("field_jaccard", 0.0) + components.get("operator_jaccard", 0.0)
            portfolio_risk_neighbor = high_corr or pool_hurt
            if score < min_score and not (portfolio_risk_neighbor and overlap_score >= 20.0):
                continue
            if not fp and not candidate_fp:
                continue

            scored.append(
                (
                    score,
                    {
                        "score": round(score, 4),
                        "score_components": components,
                        "expression": str(row.get("expression") or "")[:260],
                        "fields": list(row["fields"]),
                        "operator_skeleton": list(row["operators"]),
                        "template_id": row.get("template_id") or "",
                        "dataset_id": row.get("dataset_id") or "",
                        "category": row.get("category") or "",
                        "search_lane": row.get("search_lane") or "",
                        "alpha_id": row.get("alpha_id"),
                        "status": row.get("status"),
                        "max_self_correlation": row.get("max_self_correlation"),
                        "admission_decision": row.get("admission_decision"),
                        "admission_reason": row.get("admission_reason"),
                        "admission_global_max_corr": row.get("admission_global_max_corr"),
                        "admission_brain_score_delta": row.get("admission_brain_score_delta"),
                        "admission_delta_pool_sharpe": row.get("admission_delta_pool_sharpe"),
                        "admission_signal": row.get("admission_signal"),
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]

    @property
    def n_submitted(self) -> int:
        return len(self._submitted_exprs)


__all__ = [
    "SimilarityFilter",
    "FilterVerdict",
    "normalize_expression",
]
