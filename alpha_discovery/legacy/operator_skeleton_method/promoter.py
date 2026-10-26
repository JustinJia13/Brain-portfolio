"""Promote top-scoring cluster pairs from screening to the full template suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha_discovery.legacy.operator_skeleton_method.screener import ScreeningResult
from alpha_discovery.legacy.operator_skeleton_method.templates import (
    RelationshipTemplate,
    PROMOTION_TEMPLATES,
)


@dataclass(frozen=True)
class PromotionCandidate:
    expression: str
    template_id: str
    field_a: str
    field_b: str
    field_c: str | None
    label_a: str
    label_b: str
    label_c: str | None
    window: int | None
    source_screening_score: float

    def as_candidate_dict(self) -> dict[str, Any]:
        """Dict format compatible with run_batch_submission in run_pipeline.py."""
        import hashlib
        cid = hashlib.md5(self.expression.encode()).hexdigest()[:16]
        fields = [self.field_a, self.field_b]
        if self.field_c:
            fields.append(self.field_c)
        return {
            "candidate_id": cid,
            "family_id": f"alpha_discovery.legacy.operator_skeleton_method|{self.template_id}",
            "intention_id": "operator_search_promotion",
            "intention_label": f"Operator-first promotion: {self.template_id}",
            "search_lane": "alpha_discovery/legacy.operator_skeleton_method",
            "category": "alpha_discovery/legacy.operator_skeleton_method",
            "dataset_id": "mixed",
            "expression": self.expression,
            "fields": fields,
            "field_types": ["MATRIX"] * len(fields),
            "cluster_ids": [],
            "group": "market",
            "params": {
                "template_id": self.template_id,
                "window": self.window,
                "source_screening_score": self.source_screening_score,
                "field_a": self.field_a,
                "field_b": self.field_b,
                "field_c": self.field_c,
                "label_a": self.label_a,
                "label_b": self.label_b,
            },
            "horizon_bucket": "medium",
            "tree_depth": 3,
            "priority_score": self.source_screening_score,
            "template_id": self.template_id,
        }


def select_top_pairs(
    results: list[ScreeningResult],
    top_k: int = 20,
    sharpe_threshold: float = 1.25,
) -> list[ScreeningResult]:
    eligible = [r for r in results if r.passes_promotion(sharpe_threshold)]
    return sorted(eligible, key=lambda r: r.screening_score, reverse=True)[:top_k]


def build_promotion_candidates(
    top_results: list[ScreeningResult],
    templates: list[RelationshipTemplate] | None = None,
) -> list[PromotionCandidate]:
    """Apply top field pairs to all promotion templates × window sweep."""
    if templates is None:
        templates = PROMOTION_TEMPLATES

    seen_exprs: set[str] = set()
    candidates: list[PromotionCandidate] = []

    for sr in top_results:
        for tmpl in templates:
            if tmpl.arity == 3 and sr.field_c is None:
                continue
            windows_iter: list[int | None] = list(tmpl.windows) if tmpl.needs_window else [None]
            for w in windows_iter:
                if tmpl.arity == 3:
                    expr = tmpl.render(sr.field_a, sr.field_b, w=w, c=sr.field_c)
                else:
                    expr = tmpl.render(sr.field_a, sr.field_b, w=w)
                if expr in seen_exprs:
                    continue
                seen_exprs.add(expr)
                candidates.append(PromotionCandidate(
                    expression=expr,
                    template_id=tmpl.template_id,
                    field_a=sr.field_a,
                    field_b=sr.field_b,
                    field_c=sr.field_c if tmpl.arity == 3 else None,
                    label_a=sr.label_a,
                    label_b=sr.label_b,
                    label_c=sr.label_c if tmpl.arity == 3 else None,
                    window=w,
                    source_screening_score=sr.screening_score,
                ))

    return candidates


def build_all_screening_candidates(
    clusters,
    templates: list[RelationshipTemplate],
    n_pairs_per_template: int,
    rng,
    pruned_fields_map: dict[str, frozenset[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build screening candidate dicts from templates × sampled cluster pairs."""
    from alpha_discovery.legacy.operator_skeleton_method.field_sampler import sample_cluster_pairs, sample_cluster_triples

    pruned_map = pruned_fields_map or {}
    seen_exprs: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for tmpl in templates:
        pruned = pruned_map.get(tmpl.template_id, frozenset())
        windows_iter: list[int | None] = list(tmpl.windows) if tmpl.needs_window else [None]

        if tmpl.arity == 3:
            triples = sample_cluster_triples(clusters, tmpl, n_pairs_per_template, rng, pruned)
            for a, b, c in triples:
                for w in windows_iter:
                    expr = tmpl.render(a.champion_field_id, b.champion_field_id, w=w,
                                       c=c.champion_field_id)
                    if expr in seen_exprs:
                        continue
                    seen_exprs.add(expr)
                    candidates.append({
                        "expression": expr,
                        "template_id": tmpl.template_id,
                        "field_a": a.champion_field_id,
                        "field_b": b.champion_field_id,
                        "field_c": c.champion_field_id,
                        "label_a": a.label,
                        "label_b": b.label,
                        "label_c": c.label,
                        "window": w,
                    })
        else:
            pairs = sample_cluster_pairs(clusters, tmpl, n_pairs_per_template, rng, pruned)
            for a, b in pairs:
                for w in windows_iter:
                    expr = tmpl.render(a.champion_field_id, b.champion_field_id, w=w)
                    if expr in seen_exprs:
                        continue
                    seen_exprs.add(expr)
                    candidates.append({
                        "expression": expr,
                        "template_id": tmpl.template_id,
                        "field_a": a.champion_field_id,
                        "field_b": b.champion_field_id,
                        "field_c": None,
                        "label_a": a.label,
                        "label_b": b.label,
                        "label_c": None,
                        "window": w,
                    })

    return candidates
