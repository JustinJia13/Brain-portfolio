"""Field-cluster catalog construction and caching.

Builds the cluster JSON that four separate modules hardcode a path to.
Platform concern, not a grammar concern."""
from __future__ import annotations

import json
from pathlib import Path
from core.fields import build_data_profiles, cluster_profiles, filter_profiles
from core.expr.models import FieldCluster


def _numeric_allowed_types(allow_vector: bool) -> tuple[str, ...]:
    return ("MATRIX", "VECTOR") if allow_vector else ("MATRIX",)


def _save_clusters(clusters: list[FieldCluster], path: Path) -> None:
    records = []
    for c in clusters:
        members = [
            {
                "field_id": m.field_id,
                "category": m.category,
                "dataset_id": m.dataset_id,
                "dataset_name": m.dataset_name,
                "subcategory_id": m.subcategory_id,
                "subcategory_name": m.subcategory_name,
                "field_type": m.field_type,
                "description": m.description,
                "coverage": m.coverage,
                "date_coverage": m.date_coverage,
                "user_count": m.user_count,
                "alpha_count": m.alpha_count,
                "economic_type": m.economic_type,
                "is_size_proxy": m.is_size_proxy,
            }
            for m in c.members
        ]
        records.append(
            {
                "cluster_id": c.cluster_id,
                "category": c.category,
                "dataset_id": c.dataset_id,
                "label": c.label,
                "champion_field_id": c.champion.field_id,
                "sleeper_field_id": c.sleeper.field_id if c.sleeper else None,
                "members": members,
            }
        )
    with path.open("w") as f:
        json.dump(records, f, indent=2)


def load_clusters_from_csv(
    path: str,
    max_clusters: int = 400,
    exclude_datasets: list[str] | None = None,
    allow_vector: bool = False,
    include_fields: set[str] | None = None,
    balance_by_economic_type: bool = False,
) -> list[FieldCluster]:
    """Build FieldCluster list from all_fields.csv, bypassing Brain API.

    Enables multi-dataset mode: all datasets in the CSV are merged into one
    cluster pool so the grammar can generate cross-dataset expressions.

    Parameters
    ----------
    exclude_datasets:
        Dataset IDs to skip (e.g. ["analyst4"]).
    """
    import csv as _csv
    excluded = set(exclude_datasets or [])
    raw_fields: list[dict] = []
    skipped_datasets: set[str] = set()
    with open(path) as f:
        for row in _csv.DictReader(f):
            ds_id = row.get("dataset_id", "")
            if ds_id in excluded:
                skipped_datasets.add(ds_id)
                continue
            raw_fields.append({
                "id":          row["field_id"],
                "description": row.get("field_name", ""),
                "type":        row.get("type", "MATRIX"),
                "coverage":    float(row.get("coverage") or 0.0),
                "category":    row.get("category", ""),
                "subcategory": {"name": row.get("subcategory", "")},
                "alphaCount":  int(row.get("alpha_count") or 0),
                "userCount":   int(row.get("user_count") or 0),
                "dataset":     {"id": ds_id},
            })
    if skipped_datasets:
        print(f"  Excluded datasets: {sorted(skipped_datasets)}")
    profiles = build_data_profiles(raw_fields)
    allowed_types = _numeric_allowed_types(allow_vector)
    profiles = filter_profiles(profiles, allowed_types=allowed_types, min_coverage=0.0,
                               include_fields=include_fields)
    profiles = [p for p in profiles if p.field_type.upper() in allowed_types]
    return cluster_profiles(profiles, max_clusters=max_clusters,
                            balance_by_economic_type=balance_by_economic_type)


DEFAULT_CLUSTER_CATALOG = (
    Path(__file__).resolve().parents[1]
    / "alpha_discovery" / "grammar_method" / "artifacts" / "all_clusters.json"
)


def load_cluster_catalog(path: Path | None = None) -> dict[str, dict]:
    """{cluster_id: cluster} from the generated cluster catalog.

    Lives here rather than in a discovery lane: four separate modules read this
    same catalog, so the shared layer owns it.
    """
    data = json.loads((path or DEFAULT_CLUSTER_CATALOG).read_text())
    return {c["cluster_id"]: c for c in data}
