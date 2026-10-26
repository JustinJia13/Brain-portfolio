"""Probe-prior grammar sampling helpers.

The sampler is intentionally a thin adapter around the existing grammar
interfaces: it supplies a field sampler and a choice sampler.  The Q-table,
fingerprint vetoes, and skeleton novelty logic remain owned by
``GrammarPlanner``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from alpha_discovery.grammar_method.grammar import _stratified_field
from core.expr.models import DataProfile, FieldCluster


@dataclass(frozen=True)
class PriorRow:
    cluster_id: str
    field: str
    probability: float
    probe_family: str
    probe_template: str
    preferred_direction: str
    prior_score: float
    probe_quality: float


class ProbePriorGrammarSampler:
    """Probe-weighted field sampler plus probe-family operator bias."""

    def __init__(
        self,
        rows: list[PriorRow],
        *,
        preferred_field_prob: float = 0.70,
        sleeper_prob: float = 0.20,
        operator_prior_prob: float = 0.50,
    ) -> None:
        self._rows = list(rows)
        self._rows_by_cluster = {row.cluster_id: row for row in rows}
        self._preferred_field_prob = preferred_field_prob
        self._sleeper_prob = sleeper_prob
        self._operator_prior_prob = operator_prior_prob
        self._active_row: PriorRow | None = None
        self._field_calls = 0
        self._cache_key: tuple[str, ...] | None = None
        self._cache_clusters: list[FieldCluster] = []
        self._cache_probs: np.ndarray | None = None

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        probability_column: str = "raw_probe_probability",
        preferred_field_prob: float = 0.70,
        sleeper_prob: float = 0.20,
        operator_prior_prob: float = 0.50,
    ) -> "ProbePriorGrammarSampler":
        payload = json.loads(Path(path).read_text())
        rows: list[PriorRow] = []
        for raw in payload.get("cluster_priors", []):
            cluster_id = str(raw.get("cluster_id") or "")
            field = str(raw.get("field") or "")
            if not cluster_id or not field:
                continue
            prob = _float(raw.get(probability_column))
            if prob <= 0.0 and probability_column != "mixed_probability":
                prob = _float(raw.get("mixed_probability"))
            if prob <= 0.0:
                continue
            rows.append(
                PriorRow(
                    cluster_id=cluster_id,
                    field=field,
                    probability=prob,
                    probe_family=str(raw.get("probe_family") or ""),
                    probe_template=str(raw.get("probe_template") or ""),
                    preferred_direction=str(raw.get("preferred_direction") or ""),
                    prior_score=_float(raw.get("prior_score")),
                    probe_quality=_float(raw.get("probe_quality")),
                )
            )
        return cls(
            rows,
            preferred_field_prob=preferred_field_prob,
            sleeper_prob=sleeper_prob,
            operator_prior_prob=operator_prior_prob,
        )

    @property
    def row_count(self) -> int:
        return len(self._rows)

    def begin_candidate(self, rng: np.random.Generator, clusters: list[FieldCluster]) -> None:
        """Choose one active probe row to define this candidate's first field and operator theme."""
        self._ensure_cache(clusters)
        self._field_calls = 0
        self._active_row = None
        if not self._cache_clusters or self._cache_probs is None:
            return
        idx = int(rng.choice(len(self._cache_clusters), p=self._cache_probs))
        cluster = self._cache_clusters[idx]
        self._active_row = self._rows_by_cluster.get(cluster.cluster_id)

    def __call__(self, rng: np.random.Generator, clusters: list[FieldCluster]) -> DataProfile:
        self._ensure_cache(clusters)
        if not self._cache_clusters or self._cache_probs is None:
            return _stratified_field(rng, clusters)
        active_cluster = self._active_cluster(clusters)
        if self._field_calls == 0 and active_cluster is not None:
            cluster = active_cluster
        else:
            idx = int(rng.choice(len(self._cache_clusters), p=self._cache_probs))
            cluster = self._cache_clusters[idx]
        self._field_calls += 1
        return self._sample_member(rng, cluster)

    def choice(self, rng: np.random.Generator, kind: str, choices: Sequence[Any]) -> Any:
        """Return a probe-biased grammar choice with probability ``operator_prior_prob``."""
        if self._active_row is not None and choices and rng.random() < self._operator_prior_prob:
            preferred = _preferred_choices(self._active_row, kind, choices)
            if preferred:
                return preferred[int(rng.integers(len(preferred)))]
        return choices[int(rng.integers(len(choices)))]

    def _ensure_cache(self, clusters: list[FieldCluster]) -> None:
        key = tuple(cluster.cluster_id for cluster in clusters)
        if key == self._cache_key:
            return
        weighted_clusters: list[FieldCluster] = []
        weights: list[float] = []
        for cluster in clusters:
            row = self._rows_by_cluster.get(cluster.cluster_id)
            if row is None:
                continue
            if not _numeric_members(cluster):
                continue
            weighted_clusters.append(cluster)
            weights.append(max(row.probability, 0.0))
        if not weighted_clusters or sum(weights) <= 0.0:
            self._cache_key = key
            self._cache_clusters = []
            self._cache_probs = None
            return
        probs = np.array(weights, dtype=float)
        probs /= probs.sum()
        self._cache_key = key
        self._cache_clusters = weighted_clusters
        self._cache_probs = probs

    def _sample_member(self, rng: np.random.Generator, cluster: FieldCluster) -> DataProfile:
        members = _numeric_members(cluster)
        if not members:
            return _stratified_field(rng, [cluster])
        row = self._rows_by_cluster.get(cluster.cluster_id)
        by_field = {member.field_id: member for member in members}
        preferred = by_field.get(row.field) if row else None
        sleeper = cluster.sleeper if cluster.sleeper and cluster.sleeper.field_id in by_field else None
        r = float(rng.random())
        if preferred is not None and r < self._preferred_field_prob:
            return preferred
        if sleeper is not None and r < self._preferred_field_prob + self._sleeper_prob:
            return sleeper
        return members[int(rng.integers(len(members)))]

    def _active_cluster(self, clusters: list[FieldCluster]) -> FieldCluster | None:
        row = self._active_row
        if row is None:
            return None
        for cluster in clusters:
            if cluster.cluster_id == row.cluster_id and _numeric_members(cluster):
                return cluster
        return None


def _preferred_choices(row: PriorRow, kind: str, choices: Sequence[Any]) -> list[Any]:
    values = set(_preferred_values(row, kind))
    if not values:
        return []
    return [choice for choice in choices if choice in values]


def _preferred_values(row: PriorRow, kind: str) -> tuple[Any, ...]:
    family = row.probe_family
    template = row.probe_template
    negative = row.preferred_direction == "negative" or template.endswith("_neg")
    window = _template_window(template)

    if kind == "alpha_production":
        if family == "level":
            return ("raw_expr", "cs_op", "group_op")
        if family in {"relative_value", "time_series_rank", "time_series_zscore", "delta_reversal"}:
            return ("group_op", "double_group_op")
        if family == "smooth_level":
            return ("group_op", "group_window_op", "raw_expr")
        if family == "self_residual":
            return ("group_op", "raw_expr")

    if kind == "expr_production":
        if family == "level":
            return ("field", "unary", "expr_cs_op", "expr_group_op")
        if family == "relative_value":
            return ("expr_group_op", "expr_cs_op", "field")
        if family in {"time_series_rank", "time_series_zscore", "smooth_level"}:
            return ("ts_rolling", "expr_group_op")
        if family == "delta_reversal":
            return ("ts_delta_op", "ts_rolling", "expr_group_op")
        if family == "self_residual":
            return ("ts_regression", "ts_paired", "ts_rolling")

    if kind == "group_op":
        if "grp_zscore" in template or family == "time_series_zscore":
            return ("group_zscore", "group_neutralize")
        if "grp_rank" in template or family in {"relative_value", "time_series_rank", "delta_reversal"}:
            return ("group_rank", "group_zscore")
        return ("group_rank", "group_zscore", "group_neutralize")

    if kind == "cs_op":
        if family == "level" or "cs_rank" in template:
            return ("rank", "zscore", "winsorize")
        return ("rank", "zscore")

    if kind == "ts_rolling_op":
        if family == "time_series_rank" or template.startswith("ts_rank"):
            return ("ts_rank",)
        if family == "time_series_zscore" or template.startswith("ts_zscore"):
            return ("ts_zscore",)
        if family == "smooth_level" or template.startswith("smooth_level"):
            return ("ts_mean", "ts_decay_linear", "ts_backfill")
        if family == "delta_reversal":
            return ("ts_av_diff", "ts_rank", "ts_mean")
        if family == "self_residual":
            return ("ts_av_diff", "ts_zscore", "ts_mean")

    if kind == "ts_delta_op":
        if family == "delta_reversal" or template.startswith("delta_rev"):
            return ("ts_delta", "last_diff_value")
        if negative:
            return ("ts_delta",)

    if kind == "ts_paired_op":
        if family == "self_residual":
            return ("ts_regression", "ts_corr", "ts_covariance")
        if family == "delta_reversal":
            return ("ts_corr", "ts_covariance")

    if kind == "unary_op":
        if negative:
            return ("reverse",)
        if family == "level":
            return ("log", "winsorize", "hump")
        if family == "smooth_level":
            return ("hump", "winsorize")

    if kind == "window":
        if window is not None:
            return (window,)
        if family == "smooth_level":
            return (20, 40)
        if family == "time_series_rank":
            return (60, 120)
        if family == "time_series_zscore":
            return (120,)
        if family == "self_residual":
            return (60, 120)

    if kind == "delta_window":
        if window is not None:
            return (window,)
        if family == "delta_reversal":
            return (5, 20)

    if kind == "binary_op" and family == "self_residual":
        return ("-", "subtract")

    return ()


def _template_window(template: str) -> int | None:
    for part in template.split("_"):
        if part.isdigit():
            return int(part)
    return None


def _numeric_members(cluster: FieldCluster) -> list[DataProfile]:
    return [
        member for member in cluster.members
        if member.field_type.upper() in {"MATRIX", "VECTOR"}
    ]


def _float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["PriorRow", "ProbePriorGrammarSampler"]
