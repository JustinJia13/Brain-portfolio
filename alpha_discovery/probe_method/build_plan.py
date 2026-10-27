"""Build a JSONL signal-probe plan from field clusters.

Default usage from repo root:

    python -m alpha_discovery.probe_method.build_plan \
        --clusters alpha_discovery/grammar_method/artifacts/all_clusters.json \
        --out artifacts/probe/probe_plan.jsonl \
        --pack core \
        --cluster-limit 400

Submit later with the existing submitter:

    python -m core.brain.submit \
        --plan artifacts/probe/probe_plan.jsonl \
        --sim-log artifacts/probe/probe_sim.jsonl \
        --batch-size 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from core.fields import cluster_profiles, normalize_category
from core.expr.models import DataProfile, FieldCluster
from alpha_discovery.probe_method.templates import ProbeTemplate, default_templates


DEFAULT_OUT = Path("artifacts/probe/probe_plan.jsonl")
DEFAULT_KNOWN_LOGS = (
    Path("artifacts/sim/master.jsonl"),
    Path("artifacts/workshop/workshop_sim.jsonl"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a field signal-probe candidate plan.")
    p.add_argument("--clusters", default="alpha_discovery/grammar_method/artifacts/all_clusters.json")
    p.add_argument("--fields-csv", default="all_fields.csv")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--pack",
        choices=("core", "broad", "event", "all"),
        default="core",
        help="Template pack. core is safest for broad first-pass probing.",
    )
    p.add_argument("--cluster-limit", type=int, default=400)
    p.add_argument("--min-coverage", type=float, default=0.35)
    p.add_argument(
        "--no-balance-datasets",
        action="store_true",
        help="Keep globally ranked clusters instead of round-robin balancing by dataset.",
    )
    p.add_argument(
        "--sort-by-priority",
        action="store_true",
        help="Sort final candidates by metadata priority. Default preserves balanced probe order.",
    )
    p.add_argument(
        "--representatives",
        choices=("champion", "champion_sleeper", "all"),
        default="champion",
        help="champion = one field per cluster; champion_sleeper is a useful second pass.",
    )
    p.add_argument("--max-fields-per-cluster", type=int, default=3)
    p.add_argument("--include-dataset", nargs="*", default=[])
    p.add_argument("--exclude-dataset", nargs="*", default=[])
    p.add_argument("--allow-vector", action="store_true")
    p.add_argument(
        "--vector-reductions",
        nargs="*",
        default=["vec_avg"],
        choices=("vec_avg", "vec_sum"),
        help="Vector leaf reductions used when --allow-vector is set.",
    )
    p.add_argument(
        "--known-log",
        nargs="*",
        default=[str(p) for p in DEFAULT_KNOWN_LOGS],
        help="JSONL logs used to skip exact expressions already tested.",
    )
    p.add_argument("--no-skip-known", action="store_true")
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Print summary without writing.")
    return p.parse_args()


def _load_clusters(path: Path) -> list[FieldCluster]:
    with path.open() as handle:
        records = json.load(handle)
    clusters: list[FieldCluster] = []
    for rec in records:
        members = tuple(
            DataProfile(
                field_id=m["field_id"],
                category=m.get("category", ""),
                dataset_id=m.get("dataset_id", rec.get("dataset_id", "")),
                dataset_name=m.get("dataset_name", ""),
                subcategory_id=m.get("subcategory_id", ""),
                subcategory_name=m.get("subcategory_name", ""),
                field_type=m.get("field_type", ""),
                description=m.get("description", ""),
                coverage=m.get("coverage"),
                date_coverage=m.get("date_coverage"),
                user_count=int(m.get("user_count") or 0),
                alpha_count=int(m.get("alpha_count") or 0),
                economic_type=m.get("economic_type", "OTHER"),
                is_size_proxy=bool(m.get("is_size_proxy", False)),
            )
            for m in rec.get("members", [])
        )
        if not members:
            continue
        by_id = {m.field_id: m for m in members}
        champion = by_id.get(rec.get("champion_field_id")) or members[0]
        sleeper_id = rec.get("sleeper_field_id")
        sleeper = by_id.get(sleeper_id) if sleeper_id else None
        clusters.append(
            FieldCluster(
                cluster_id=rec.get("cluster_id", champion.cluster_id),
                category=rec.get("category", champion.category),
                dataset_id=rec.get("dataset_id", champion.dataset_id),
                label=rec.get("label", champion.cluster_label),
                champion=champion,
                sleeper=sleeper,
                members=members,
            )
        )
    return clusters


def _load_clusters_from_csv(path: Path, *, cluster_limit: int | None) -> list[FieldCluster]:
    profiles: list[DataProfile] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            field_id = (row.get("field_id") or "").strip()
            if not field_id:
                continue
            profiles.append(
                DataProfile(
                    field_id=field_id,
                    category=normalize_category(row.get("category") or ""),
                    dataset_id=(row.get("dataset_id") or "").strip(),
                    dataset_name="",
                    subcategory_name=row.get("subcategory") or "",
                    field_type=(row.get("type") or "").strip(),
                    description=row.get("field_name") or "",
                    coverage=_float_or_none(row.get("coverage")),
                    user_count=int(float(row.get("user_count") or 0)),
                    alpha_count=int(float(row.get("alpha_count") or 0)),
                )
            )
    return cluster_profiles(profiles, max_clusters=cluster_limit)


def _float_or_none(raw: object) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _known_expressions(paths: Iterable[str | Path]) -> set[str]:
    known: set[str] = set()
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
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                expr = rec.get("expression")
                if expr:
                    known.add(str(expr))
    return known


def _representatives(cluster: FieldCluster, mode: str, max_fields: int) -> list[DataProfile]:
    if mode == "champion":
        return [cluster.champion]
    if mode == "champion_sleeper":
        reps = list(cluster.representatives(include_sleeper=True))
        return reps[:max_fields]
    return list(cluster.members[:max_fields])


def _leaf_expr(profile: DataProfile, vector_reduction: str | None) -> str:
    if profile.is_vector:
        reduction = vector_reduction or "vec_avg"
        return f"{reduction}({profile.field_id})"
    return profile.field_id


def _priority(profile: DataProfile, cluster: FieldCluster, template: ProbeTemplate) -> float:
    coverage = profile.coverage or 0.0
    alpha_score = math.log1p(max(profile.alpha_count, 0))
    user_score = 0.35 * math.log1p(max(profile.user_count, 0))
    risk_penalty = 1.5 if template.concentration_risk == "high" and coverage < 0.6 else 0.0
    return round(coverage + alpha_score + user_score - risk_penalty, 6)


def _candidate(
    *,
    cluster: FieldCluster,
    profile: DataProfile,
    template: ProbeTemplate,
    expression: str,
    vector_reduction: str | None,
) -> dict:
    source = f"{cluster.cluster_id}|{profile.field_id}|{template.template_id}|{vector_reduction or ''}|{expression}"
    cid = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return {
        "candidate_id": cid,
        "template_id": f"probe:{template.template_id}",
        "operator_skeleton": template.template_id,
        "is_probe": True,
        "experiment_stage": "field_probe",
        "template_family": template.family,
        "dataset_id": profile.dataset_id,
        "category": profile.category or cluster.category,
        "search_lane": "probe",
        "expression": expression,
        "fields": [profile.field_id],
        "field_types": [profile.field_type],
        "cluster_ids": [cluster.cluster_id],
        "group": template.group,
        "params": {
            "probe_template": template.template_id,
            "probe_family": template.family,
            "probe_description": template.description,
            "probe_horizon": template.horizon,
            "probe_direction": template.direction,
            "concentration_risk": template.concentration_risk,
            "field_coverage": profile.coverage,
            "field_alpha_count": profile.alpha_count,
            "field_user_count": profile.user_count,
            "field_description": profile.description,
            "cluster_label": cluster.label,
            "representative_role": _representative_role(cluster, profile),
            "vector_reduction": vector_reduction,
        },
        "priority_score": _priority(profile, cluster, template),
        "skip_workshop": False,
        "skip_qtable_update": False,
    }


def _representative_role(cluster: FieldCluster, profile: DataProfile) -> str:
    if profile.field_id == cluster.champion.field_id:
        return "champion"
    if cluster.sleeper and profile.field_id == cluster.sleeper.field_id:
        return "sleeper"
    return "member"


def build_plan(args: argparse.Namespace) -> list[dict]:
    cluster_path = Path(args.clusters)
    if cluster_path.exists():
        clusters = _load_clusters(cluster_path)
    else:
        clusters = _load_clusters_from_csv(Path(args.fields_csv), cluster_limit=None)

    include = set(args.include_dataset)
    exclude = set(args.exclude_dataset)
    filtered: list[FieldCluster] = []
    allowed_types = {"MATRIX"} | ({"VECTOR"} if args.allow_vector else set())
    for cluster in clusters:
        if include and cluster.dataset_id not in include:
            continue
        if exclude and cluster.dataset_id in exclude:
            continue
        champion = cluster.champion
        if champion.field_type not in allowed_types:
            continue
        if (champion.coverage or 0.0) < args.min_coverage:
            continue
        filtered.append(cluster)
    filtered.sort(
        key=lambda c: (
            c.champion.alpha_count,
            c.champion.user_count,
            c.champion.coverage or 0.0,
            c.cluster_id,
        ),
        reverse=True,
    )
    if not args.no_balance_datasets:
        filtered = _balance_by_dataset(filtered)
    if args.cluster_limit > 0:
        filtered = filtered[: args.cluster_limit]

    templates = default_templates(args.pack)
    known = set() if args.no_skip_known else _known_expressions(args.known_log)
    candidates: list[dict] = []
    seen_exprs: set[str] = set()
    for cluster in filtered:
        for profile in _representatives(cluster, args.representatives, args.max_fields_per_cluster):
            if profile.field_type not in allowed_types:
                continue
            reductions = args.vector_reductions if profile.is_vector else [None]
            for reduction in reductions:
                leaf = _leaf_expr(profile, reduction)
                for template in templates:
                    expr = template.render(leaf)
                    if expr in known or expr in seen_exprs:
                        continue
                    seen_exprs.add(expr)
                    candidates.append(
                        _candidate(
                            cluster=cluster,
                            profile=profile,
                            template=template,
                            expression=expr,
                            vector_reduction=reduction,
                        )
                    )

    if args.sort_by_priority:
        candidates.sort(key=lambda c: c.get("priority_score", 0.0), reverse=True)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    return candidates


def _balance_by_dataset(clusters: list[FieldCluster]) -> list[FieldCluster]:
    buckets: dict[str, list[FieldCluster]] = {}
    order: list[str] = []
    for cluster in clusters:
        key = cluster.dataset_id or "unknown"
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(cluster)

    balanced: list[FieldCluster] = []
    while any(buckets.values()):
        for key in list(order):
            bucket = buckets.get(key) or []
            if bucket:
                balanced.append(bucket.pop(0))
    return balanced


def _summarize(candidates: list[dict]) -> None:
    from collections import Counter

    print(f"Candidates: {len(candidates)}")
    print("Datasets:", Counter(c["dataset_id"] for c in candidates).most_common(12))
    print("Families:", Counter(c["template_family"] for c in candidates).most_common(12))
    print("Templates:", Counter(c["operator_skeleton"] for c in candidates).most_common(12))
    for row in candidates[:8]:
        print(
            f"  {row['priority_score']:>7.3f}  {row['dataset_id']:<14} "
            f"{row['operator_skeleton']:<24} {row['expression'][:100]}"
        )


def main() -> None:
    args = parse_args()
    candidates = build_plan(args)
    _summarize(candidates)
    if args.dry_run:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")
    manifest = {
        "args": vars(args),
        "n_candidates": len(candidates),
        "templates": [asdict(t) for t in default_templates(args.pack)],
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote plan: {out}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
