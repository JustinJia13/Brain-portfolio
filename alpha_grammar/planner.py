"""GrammarPlanner: generates alpha candidate batches via grammar sampling.

Field selection is uniform over datasets, then clusters, then fields.
Before emitting a candidate, the planner checks:
  1. Q-table Level 1 pruning: (cluster_id, specific_op) in pruned set → skip
  2. Q-table Level 2 pruning: (field_id,   specific_op) in pruned set → skip

Batch workflow:
    qtable = QTable.load(path) or QTable()
    planner = GrammarPlanner(config, qtable=qtable)
    candidates = planner.build_plan(clusters)
    # ... submit candidates to Brain, get results ...
    for result in brain_results:
        qtable.update(result.cluster_ids[0], result.fields[0], specific_op, reward)
    # repeat
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .models import CandidateSpec, DataProfile, FieldCluster
from .feedback import SimResult
from .grammar import (
    BrainGrammar,
    GrammarConfig,
    _extract_field_ids,
    _horizon_from_window,
    _top_window,
)
from .qtable import QTable, SkeletonCounter, extract_primary_op
from .novelty import TestedFingerprintRegistry


def _has_nested_is_nan(expression: str) -> bool:
    return "is_nan(is_nan(" in expression.replace(" ", "")


def _begin_candidate_samplers(
    rng: np.random.Generator,
    eligible: list[FieldCluster],
    *samplers: object,
) -> None:
    """Reset stateful samplers once per candidate without double-calling shared objects."""
    seen: set[int] = set()
    for sampler in samplers:
        if sampler is None:
            continue
        key = id(sampler)
        if key in seen:
            continue
        seen.add(key)
        begin = getattr(sampler, "begin_candidate", None)
        if callable(begin):
            begin(rng, eligible)


class GrammarPlanner:
    """Generates batches of novel alpha candidates using grammar sampling.

    Field selection is uniform over datasets, then clusters, then fields.
    No UCB weighting.

    Diversity within a batch is enforced by exact expression string dedup.
    Cross-batch diversity is enforced by the tested registry (persisted to disk).
    """

    def __init__(
        self,
        config: GrammarConfig,
        qtable: QTable | None = None,
        skeleton_counter: SkeletonCounter | None = None,
        group_profiles: Sequence[DataProfile] | None = None,
        fingerprint_registry: TestedFingerprintRegistry | None = None,
        jaccard_threshold: float | None = None,
    ) -> None:
        self._cfg = config
        self._grammar = BrainGrammar(config)
        self._qtable = qtable if qtable is not None else QTable()
        self._skeleton_counter = skeleton_counter
        self._group_profiles = tuple(group_profiles or ())
        self._fingerprint_registry = fingerprint_registry
        self._jaccard_threshold = (
            float(jaccard_threshold)
            if jaccard_threshold is not None
            else float(config.diversity_threshold)
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build_plan(
        self,
        clusters: Sequence[FieldCluster],
        *,
        flat_profiles: Sequence[DataProfile] | None = None,
        seed: int = 42,
        exploit_clusters: Sequence[FieldCluster] | None = None,
        field_sampler: Callable[[np.random.Generator, list[FieldCluster]], DataProfile] | None = None,
        choice_sampler: Callable[[np.random.Generator, str, Sequence[Any]], Any] | None = None,
    ) -> list[CandidateSpec]:
        """Generate up to n_samples novel candidates for one batch.

        Steps:
          1. Filter clusters by champion.coverage >= min_coverage.
          2. Build field-id → DataProfile and field-id → cluster_id lookup tables.
          3. Grammar samples uniformly by dataset, then cluster, then field.
          4. Generate candidates.
          5. Return list of CandidateSpec objects.

        flat_profiles and exploit_clusters are accepted for API compatibility but ignored.
        """
        rng = np.random.default_rng(seed)

        # ── 1. Filter by coverage ──────────────────────────────────────
        eligible = [
            c for c in clusters
            if (c.champion.coverage or 0.0) >= self._cfg.min_coverage
        ]
        if not eligible:
            return []

        # ── 2. Build lookup tables ────────────────────────────────────
        profile_by_field: dict[str, DataProfile] = {}
        cluster_by_field: dict[str, str] = {}
        for cluster in eligible:
            for member in cluster.members:
                profile_by_field[member.field_id] = member
                cluster_by_field[member.field_id] = cluster.cluster_id
            profile_by_field[cluster.champion.field_id] = cluster.champion
            cluster_by_field[cluster.champion.field_id] = cluster.cluster_id

        # ── 3. Build proxy list (dataset-uniform for division RHS) ───
        all_members = [p for c in eligible for p in c.members]
        proxy_profiles = [p for p in all_members if p.is_size_proxy]

        # ── 4. Generate candidates ────────────────────────────────────
        seen_exprs: set[str] = set()
        candidates = self._generate_pool(
            n_target=self._cfg.n_samples,
            eligible=eligible,
            proxy_profiles=proxy_profiles,
            profile_by_field=profile_by_field,
            cluster_by_field=cluster_by_field,
            seen_exprs=seen_exprs,
            rng=rng,
            field_sampler=field_sampler,
            choice_sampler=choice_sampler,
        )

        return candidates

    def _generate_pool(
        self,
        n_target: int,
        eligible: list[FieldCluster],
        proxy_profiles: list[DataProfile],
        profile_by_field: dict[str, DataProfile],
        cluster_by_field: dict[str, str],
        seen_exprs: set[str],
        rng: np.random.Generator,
        field_sampler: Callable[[np.random.Generator, list[FieldCluster]], DataProfile] | None = None,
        choice_sampler: Callable[[np.random.Generator, str, Sequence[Any]], Any] | None = None,
    ) -> list[CandidateSpec]:
        """Generate up to n_target candidates via dataset-uniform field sampling.

        Grammar samples: uniform dataset → uniform cluster → uniform field.
        Clusters are also used for metadata lookup in CandidateSpec.

        When a SkeletonCounter is present the pool is oversampled by
        skeleton_oversample_factor then sorted by novelty_score (descending)
        before trimming to n_target.
        """
        budget = int(self._cfg.budget)
        if budget < 1:
            raise ValueError("GrammarConfig.budget must be >= 1")

        # Snapshot pruned sets once per pool call (cheap frozenset lookups).
        pruned_l1 = self._qtable.pruned_level1()
        pruned_l2 = self._qtable.pruned_level2()

        oversample = getattr(self._cfg, "skeleton_oversample_factor", 3)
        rank_by_novelty = self._skeleton_counter is not None or self._fingerprint_registry is not None
        pool_target = n_target * oversample if rank_by_novelty else n_target

        candidates: list[CandidateSpec] = []
        max_attempts = max(pool_target * 15, 100)

        for _ in range(max_attempts):
            if len(candidates) >= pool_target:
                break

            # ── Select group ────────────────────────────────────────
            group = str(self._cfg.groups[int(rng.integers(len(self._cfg.groups)))])

            _begin_candidate_samplers(rng, eligible, field_sampler, choice_sampler)

            # ── Sample expression tree ──────────────────────────────
            # Grammar picks dataset first, then cluster, then field.
            tree = self._grammar.sample(
                budget, rng, eligible, proxy_profiles,
                group_profiles=self._group_profiles,
                group=group,
                field_sampler=field_sampler,
                choice_sampler=choice_sampler,
            )
            if tree is None:
                continue

            # ── Canonicalize ────────────────────────────────────────
            tree = self._grammar.canonicalize(tree)
            expr_str = tree.render()
            if _has_nested_is_nan(expr_str):
                continue

            # ── Exact string dedup (within batch) ───────────────────
            if expr_str in seen_exprs:
                continue

            # ── Extract field metadata ──────────────────────────────
            field_ids_raw = _extract_field_ids(tree)
            if not field_ids_raw:
                continue
            seen_fids: set[str] = set()
            field_ids: list[str] = []
            for fid in field_ids_raw:
                if fid not in seen_fids:
                    field_ids.append(fid)
                    seen_fids.add(fid)

            profiles = [profile_by_field.get(fid) for fid in field_ids]
            if any(p is None for p in profiles):
                continue
            clean_profiles: list[DataProfile] = profiles  # type: ignore[assignment]

            # Use first field's cluster for metadata; fallback to dataset_id.
            primary_fid = field_ids[0]
            primary_profile = clean_profiles[0]
            primary_cluster_id = cluster_by_field.get(
                primary_fid, primary_profile.dataset_id
            )
            cluster_ids_used = tuple(
                cluster_by_field.get(fid, clean_profiles[i].dataset_id)
                for i, fid in enumerate(field_ids)
            )

            # ── Q-table pruning + tested registry check ──────────────
            primary_op = extract_primary_op(expr_str, primary_fid)

            if primary_op is not None:
                if (primary_cluster_id, primary_op) in pruned_l1:
                    continue
                if (primary_fid, primary_op) in pruned_l2:
                    continue

            max_jaccard = 0.0
            if self._fingerprint_registry is not None:
                max_jaccard = self._fingerprint_registry.max_similarity(
                    primary_cluster_id,
                    expr_str,
                    field_ids,
                )
                if max_jaccard > self._jaccard_threshold:
                    continue

            # ── Build candidate ─────────────────────────────────────
            field_types = tuple(p.field_type or "" for p in clean_profiles)
            proxy_flags = tuple(p.is_size_proxy for p in clean_profiles)

            top_window = _top_window(tree)
            params: Mapping[str, Any] = {"window": top_window} if top_window is not None else {}
            horizon = _horizon_from_window(top_window) if top_window is not None else "medium"

            cid = hashlib.md5(expr_str.encode()).hexdigest()[:16]
            fingerprint = self._grammar.structure_fingerprint(tree)
            semantic_sig = f"{primary_cluster_id}|{group}|{'_'.join(fingerprint[:4])}"

            template_id = "|".join(fingerprint)
            skeleton_novelty = (
                self._skeleton_counter.novelty_score(template_id)
                if self._skeleton_counter is not None
                else 1.0
            )
            if self._fingerprint_registry is not None:
                cluster_novelty = 1.0 / (
                    1.0 + self._fingerprint_registry.cluster_size(primary_cluster_id)
                )
                novelty = (
                    0.55 * skeleton_novelty
                    + 0.35 * (1.0 - max_jaccard)
                    + 0.10 * cluster_novelty
                )
            else:
                novelty = skeleton_novelty

            seen_exprs.add(expr_str)
            candidates.append(
                CandidateSpec(
                    candidate_id=cid,
                    family_id=f"grammar|{primary_cluster_id}|k{budget}|{group}",
                    category=primary_profile.category,
                    dataset_id=primary_profile.dataset_id,
                    intention_id="grammar_uniform",
                    intention_label="Grammar uniform sampling",
                    template_id=template_id,
                    economic_source="",
                    search_lane="grammar",
                    expression=expr_str,
                    fields=tuple(field_ids),
                    field_types=field_types,
                    size_proxy_flags=proxy_flags,
                    cluster_ids=cluster_ids_used,
                    group=group,
                    params=params,
                    horizon_bucket=horizon,
                    settings_signature="",
                    semantic_signature=semantic_sig,
                    rationale_type="exploratory",
                    tree_depth=tree.depth(),
                    priority_score=novelty,
                )
            )

        # ── Novelty sort + trim ─────────────────────────────────────────
        if rank_by_novelty and len(candidates) > n_target:
            candidates.sort(key=lambda c: c.priority_score, reverse=True)
            candidates = candidates[:n_target]

        return candidates

__all__ = ["GrammarPlanner"]
