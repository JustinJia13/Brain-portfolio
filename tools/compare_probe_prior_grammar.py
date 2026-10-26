"""Build matched raw-grammar vs probe-prior grammar plans.

This is a dry-run planner comparison. It does not sign in to Brain and does not
submit simulations. It writes two submit-ready JSONL plans:

  - raw_grammar_plan.jsonl: current uniform dataset -> cluster -> field sampler
  - probe_prior_plan.jsonl: probe-prior cluster/field sampler plus operator bias

Both plans use the same Q-table, skeleton counter, and tested-fingerprint vetoes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from alpha_discovery.grammar_method import GrammarConfig, GrammarPlanner
from alpha_discovery.grammar_method.grammar import _stratified_field
from core.expr.models import DataProfile, FieldCluster
from alpha_discovery.grammar_method.novelty import TestedFingerprintRegistry
from core.qtable import QTable, SkeletonCounter
from alpha_discovery.grammar_method.run_pipeline import _filter_clusters_by_field_types, _group_profiles_from_clusters, _load_clusters


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLUSTERS = ROOT / "alpha_discovery/grammar_method" / "artifacts" / "all_clusters.json"
DEFAULT_PRIOR = ROOT / "artifacts" / "probe" / "grammar_prior_from_probe.json"
DEFAULT_QTABLE = ROOT / "alpha_discovery/grammar_method" / "artifacts" / "all_qtable.json"
DEFAULT_SKELETONS = ROOT / "alpha_discovery/grammar_method" / "artifacts" / "all_skeleton_counts.json"
DEFAULT_FINGERPRINTS = ROOT / "alpha_discovery/grammar_method" / "artifacts" / "all_tested_fingerprints.json"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "probe_prior_compare"


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
    """Sample fields and grammar choices from compact probe evidence.

    One probe row is selected at the start of each candidate. The first field
    leaf is pulled from that row's cluster, while later leaves continue to use
    the probe-weighted cluster distribution. Grammar production/operator choices
    use the active row with a configurable mixture probability.
    """

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
        path: Path,
        *,
        probability_column: str = "raw_probe_probability",
        preferred_field_prob: float = 0.70,
        sleeper_prob: float = 0.20,
        operator_prior_prob: float = 0.50,
    ) -> "ProbePriorGrammarSampler":
        payload = json.loads(path.read_text())
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

    def begin_candidate(
        self,
        rng: np.random.Generator,
        clusters: list[FieldCluster],
    ) -> None:
        """Pick one active probe row to define this candidate's operator theme."""
        self._ensure_cache(clusters)
        self._field_calls = 0
        self._active_row = None
        if not self._cache_clusters or self._cache_probs is None:
            return
        idx = int(rng.choice(len(self._cache_clusters), p=self._cache_probs))
        cluster = self._cache_clusters[idx]
        self._active_row = self._rows_by_cluster.get(cluster.cluster_id)

    def __call__(
        self,
        rng: np.random.Generator,
        clusters: list[FieldCluster],
    ) -> DataProfile:
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

    def choice(
        self,
        rng: np.random.Generator,
        kind: str,
        choices: Sequence[Any],
    ) -> Any:
        """Return a probe-biased grammar choice with probability operator_prior_prob."""
        if (
            self._active_row is not None
            and choices
            and rng.random() < self._operator_prior_prob
        ):
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
            members = _numeric_members(cluster)
            if not members:
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
        sleeper = (
            cluster.sleeper
            if cluster.sleeper and cluster.sleeper.field_id in by_field
            else None
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", default=str(DEFAULT_CLUSTERS))
    parser.add_argument("--probe-prior", default=str(DEFAULT_PRIOR))
    parser.add_argument("--qtable", default=str(DEFAULT_QTABLE))
    parser.add_argument("--skeleton-counts", default=str(DEFAULT_SKELETONS))
    parser.add_argument("--fingerprints", default=str(DEFAULT_FINGERPRINTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--allow-vector", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--exclude-dataset", nargs="*", default=[])
    parser.add_argument(
        "--prior-prob-column",
        default="raw_probe_probability",
        choices=("raw_probe_probability", "mixed_probability"),
        help="Use raw for 100%% probe-prior sampling; mixed includes the notebook's uniform blend.",
    )
    parser.add_argument("--preferred-field-prob", type=float, default=0.70)
    parser.add_argument("--sleeper-field-prob", type=float, default=0.20)
    parser.add_argument(
        "--operator-prior-prob",
        type=float,
        default=0.50,
        help="Probability of using probe-family/template bias for each compatible grammar choice.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clusters = _load_and_filter_clusters(args)
    group_profiles = _group_profiles_from_clusters(clusters)
    qtable = QTable.load(args.qtable) if Path(args.qtable).exists() else QTable()
    skeleton_counter = (
        SkeletonCounter.load(args.skeleton_counts)
        if Path(args.skeleton_counts).exists()
        else SkeletonCounter()
    )
    fingerprint_registry = (
        TestedFingerprintRegistry.load(Path(args.fingerprints))
        if Path(args.fingerprints).exists()
        else TestedFingerprintRegistry()
    )
    config = GrammarConfig(
        budget=args.budget,
        n_samples=args.n_samples,
        min_coverage=args.min_coverage,
        exploration_fraction=1.0,
    )

    raw_planner = GrammarPlanner(
        config,
        qtable=qtable,
        skeleton_counter=skeleton_counter,
        group_profiles=group_profiles,
        fingerprint_registry=fingerprint_registry,
    )
    prior_planner = GrammarPlanner(
        config,
        qtable=qtable,
        skeleton_counter=skeleton_counter,
        group_profiles=group_profiles,
        fingerprint_registry=fingerprint_registry,
    )
    prior_sampler = ProbePriorGrammarSampler.from_json(
        Path(args.probe_prior),
        probability_column=args.prior_prob_column,
        preferred_field_prob=args.preferred_field_prob,
        sleeper_prob=args.sleeper_field_prob,
        operator_prior_prob=args.operator_prior_prob,
    )

    raw_plan = raw_planner.build_plan(clusters, seed=args.seed)
    prior_plan = prior_planner.build_plan(
        clusters,
        seed=args.seed,
        field_sampler=prior_sampler,
        choice_sampler=prior_sampler.choice,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_grammar_plan.jsonl"
    prior_path = out_dir / "probe_prior_plan.jsonl"
    summary_path = out_dir / "comparison_summary.json"
    summary_txt_path = out_dir / "comparison_summary.txt"

    _write_plan(raw_path, raw_plan, "raw_grammar")
    _write_plan(prior_path, prior_plan, "probe_prior_grammar")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_clusters": len(clusters),
        "qtable": {
            "level1_entries": len(qtable.level1),
            "level2_entries": len(qtable.level2),
            "pruned_level1": len(qtable.pruned_level1()),
            "pruned_level2": len(qtable.pruned_level2()),
            "min_obs_l1": qtable.min_obs_l1,
            "min_obs_l2": qtable.min_obs_l2,
            "min_prune_skeletons": qtable.min_prune_skeletons,
            "min_prune_contexts_l1": qtable.min_prune_contexts_l1,
        },
        "raw": _plan_summary(raw_plan),
        "probe_prior": _plan_summary(prior_plan),
        "overlap": {
            "expressions": len({c.expression for c in raw_plan} & {c.expression for c in prior_plan}),
            "primary_clusters": len(_primary_clusters(raw_plan) & _primary_clusters(prior_plan)),
            "primary_fields": len(_primary_fields(raw_plan) & _primary_fields(prior_plan)),
        },
        "weighting": {
            "cluster_probability_column": args.prior_prob_column,
            "cluster_probability_mode": "100% probe prior, renormalized over eligible clusters",
            "preferred_field_probability": args.preferred_field_prob,
            "sleeper_field_probability": args.sleeper_field_prob,
            "random_sibling_probability": max(
                0.0,
                1.0 - args.preferred_field_prob - args.sleeper_field_prob,
            ),
            "operator_prior_probability": args.operator_prior_prob,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary_txt_path.write_text(_format_summary(summary, raw_path, prior_path) + "\n")
    print(_format_summary(summary, raw_path, prior_path))


def _load_and_filter_clusters(args: argparse.Namespace) -> list[FieldCluster]:
    clusters = _load_clusters(Path(args.clusters))
    if args.exclude_dataset:
        excluded = set(args.exclude_dataset)
        clusters = [cluster for cluster in clusters if cluster.dataset_id not in excluded]
    allowed_types = ("MATRIX", "VECTOR") if args.allow_vector else ("MATRIX",)
    return _filter_clusters_by_field_types(clusters, allowed_types)


def _write_plan(path: Path, plan: list[Any], comparison_lane: str) -> None:
    with path.open("w") as handle:
        for candidate in plan:
            row = candidate.as_record()
            row["comparison_lane"] = comparison_lane
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _plan_summary(plan: list[Any]) -> dict[str, Any]:
    primary_clusters = [candidate.cluster_ids[0] for candidate in plan if candidate.cluster_ids]
    primary_fields = [candidate.fields[0] for candidate in plan if candidate.fields]
    return {
        "n_candidates": len(plan),
        "unique_primary_clusters": len(set(primary_clusters)),
        "unique_primary_fields": len(set(primary_fields)),
        "unique_templates": len({candidate.template_id for candidate in plan}),
        "top_primary_clusters": Counter(primary_clusters).most_common(20),
        "top_primary_fields": Counter(primary_fields).most_common(20),
        "top_templates": Counter(candidate.template_id for candidate in plan).most_common(20),
        "avg_priority_score": _mean([candidate.priority_score for candidate in plan]),
        "avg_tree_depth": _mean([candidate.tree_depth for candidate in plan]),
    }


def _format_summary(summary: dict[str, Any], raw_path: Path, prior_path: Path) -> str:
    lines = [
        "Probe-prior grammar comparison plans written:",
        f"  raw:         {raw_path}",
        f"  probe_prior: {prior_path}",
        "",
        "Q-table:",
        (
            f"  L1={summary['qtable']['level1_entries']} "
            f"pruned={summary['qtable']['pruned_level1']} "
            f"min_obs={summary['qtable']['min_obs_l1']}"
        ),
        (
            f"  L2={summary['qtable']['level2_entries']} "
            f"pruned={summary['qtable']['pruned_level2']} "
            f"min_obs={summary['qtable']['min_obs_l2']}"
        ),
        "",
        "Plan summary:",
    ]
    for label in ("raw", "probe_prior"):
        row = summary[label]
        lines.append(
            f"  {label:<12} n={row['n_candidates']} "
            f"clusters={row['unique_primary_clusters']} "
            f"fields={row['unique_primary_fields']} "
            f"templates={row['unique_templates']} "
            f"avg_depth={row['avg_tree_depth']:.2f}"
        )
    lines.extend([
        "",
        "Overlap:",
        f"  expressions={summary['overlap']['expressions']}",
        f"  primary_clusters={summary['overlap']['primary_clusters']}",
        f"  primary_fields={summary['overlap']['primary_fields']}",
        "",
        "Weighting:",
        f"  cluster probability: {summary['weighting']['cluster_probability_column']}",
        (
            "  field split: "
            f"{summary['weighting']['preferred_field_probability']:.0%} probe-winning field, "
            f"{summary['weighting']['sleeper_field_probability']:.0%} sleeper, "
            f"{summary['weighting']['random_sibling_probability']:.0%} random sibling"
        ),
        f"  operator prior probability: {summary['weighting']['operator_prior_probability']:.0%}",
    ])
    return "\n".join(lines)


def _primary_clusters(plan: list[Any]) -> set[str]:
    return {candidate.cluster_ids[0] for candidate in plan if candidate.cluster_ids}


def _primary_fields(plan: list[Any]) -> set[str]:
    return {candidate.fields[0] for candidate in plan if candidate.fields}


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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
