"""Field normalization, tagging, and clustering for the alpha_discovery/grammar_method pipeline.

Originally copied from the retired alpha_search package (fields.py + tagging.py); no dependency on it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Sequence

from core.expr.models import DataProfile, FieldCluster


# ---------------------------------------------------------------------------
# Category normalization
# ---------------------------------------------------------------------------

CATEGORY_ALIASES = {
    "analyst": "analyst",
    "fundamental": "fundamental",
    "model": "model",
    "news": "news",
    "option": "option",
    "options": "option",
    "price volume": "price_volume",
    "price_volume": "price_volume",
    "social media data": "social_media",
    "social_media": "social_media",
}


def normalize_category(category: str) -> str:
    return CATEGORY_ALIASES.get(category.strip().lower(), category.strip().lower())


# ---------------------------------------------------------------------------
# Field tagging (inlined from the retired alpha_search tagging module)
# ---------------------------------------------------------------------------

SAFE_DENOMINATOR_KEYWORDS = (
    "assets", "equity", "cap", "revenue", "ebit", "bookvalue",
)


def is_size_proxy(field_id: str) -> bool:
    lowered = field_id.lower()
    return any(keyword in lowered for keyword in SAFE_DENOMINATOR_KEYWORDS)


# ---------------------------------------------------------------------------
# Profile building and filtering
# ---------------------------------------------------------------------------

def build_data_profiles(
    raw_fields: Sequence[dict],
    category: str | None = None,
    dataset_id: str | None = None,
) -> list[DataProfile]:
    """Normalize raw Brain API records into typed DataProfile objects."""
    profiles = []
    for raw in raw_fields:
        raw_dataset = raw.get("dataset") or {}
        raw_subcategory = raw.get("subcategory") or {}
        raw_category = raw.get("category") or {}

        dataset = raw_dataset if isinstance(raw_dataset, dict) else {}
        subcategory = raw_subcategory if isinstance(raw_subcategory, dict) else {}
        field_category = raw_category if isinstance(raw_category, dict) else {}

        resolved_category = normalize_category(
            category
            or field_category.get("name", "")
            or raw.get("category_name", "")
            or str(raw.get("category", "") or "")
        )
        resolved_dataset_id = dataset_id or dataset.get("id", "") or raw.get("dataset_id", "")
        subcategory_name = subcategory.get("name", "") or raw.get("subcategory_name", "")
        profiles.append(DataProfile(
            field_id=raw.get("id", ""),
            category=resolved_category,
            dataset_id=resolved_dataset_id,
            dataset_name=dataset.get("name", "") or raw.get("dataset_name", ""),
            subcategory_id=subcategory.get("id", "") or raw.get("subcategory_id", ""),
            subcategory_name=subcategory_name,
            field_type=raw.get("type", ""),
            description=raw.get("description", ""),
            coverage=raw.get("coverage"),
            date_coverage=raw.get("dateCoverage"),
            user_count=int(raw.get("userCount") or 0),
            alpha_count=int(raw.get("alphaCount") or 0),
            economic_type=str(raw.get("economic_type") or ""),
            is_size_proxy=bool(raw.get("is_size_proxy", is_size_proxy(raw.get("id", "")))),
            raw=raw,
        ))
    return profiles


def filter_profiles(
    profiles: Iterable[DataProfile],
    *,
    allowed_types: Sequence[str] = ("MATRIX",),
    min_coverage: float = 0.0,
    include_fields: set[str] | None = None,
) -> list[DataProfile]:
    """Filter profiles by type, coverage, and optional explicit field pinlist.

    If include_fields is provided, only those field_ids are kept (regardless of
    type/coverage — the user explicitly opted them in).  The type/coverage filters
    still apply to all other fields.
    """
    result = []
    for p in profiles:
        if include_fields is not None and p.field_id in include_fields:
            result.append(p)  # explicitly pinned — bypass type/coverage gates
        elif p.field_type in allowed_types and (p.coverage or 0.0) >= min_coverage:
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def metadata_score(profile: DataProfile) -> float:
    coverage = profile.coverage or 0.0
    return float(profile.alpha_count) + 0.25 * float(profile.user_count) + 10.0 * coverage


def cluster_profiles(
    profiles: Sequence[DataProfile],
    *,
    max_clusters: int | None = None,
    balance_by_economic_type: bool = False,
) -> list[FieldCluster]:
    """Group fields by cluster_id and choose the highest-scoring champion per cluster."""
    grouped: dict[str, list[DataProfile]] = defaultdict(list)
    for profile in profiles:
        grouped[profile.cluster_id].append(profile)

    clusters = []
    for cluster_id, members in grouped.items():
        ranked = sorted(
            members,
            key=lambda p: (metadata_score(p), p.alpha_count, p.user_count, p.field_id),
            reverse=True,
        )
        clusters.append(FieldCluster(
            cluster_id=cluster_id,
            category=ranked[0].category,
            dataset_id=ranked[0].dataset_id,
            label=ranked[0].cluster_label,
            champion=ranked[0],
            sleeper=ranked[1] if len(ranked) > 1 else None,
            members=tuple(ranked),
        ))

    clusters.sort(key=lambda c: metadata_score(c.champion), reverse=True)

    if balance_by_economic_type and max_clusters is not None:
        return _balance_by_type(clusters, max_clusters)
    if max_clusters is not None:
        return clusters[:max_clusters]
    return clusters


def _balance_by_type(clusters: Sequence[FieldCluster], max_clusters: int) -> list[FieldCluster]:
    buckets: dict[str, deque[FieldCluster]] = defaultdict(deque)
    for c in clusters:
        buckets[c.champion.economic_type].append(c)
    bucket_order = sorted(
        buckets,
        key=lambda t: metadata_score(buckets[t][0].champion),
        reverse=True,
    )
    selected: list[FieldCluster] = []
    while len(selected) < max_clusters and bucket_order:
        next_order: list[str] = []
        for t in bucket_order:
            if len(selected) >= max_clusters:
                break
            if buckets[t]:
                selected.append(buckets[t].popleft())
                if buckets[t]:
                    next_order.append(t)
        bucket_order = next_order
    return selected


__all__ = [
    "build_data_profiles", "filter_profiles", "cluster_profiles",
    "normalize_category", "is_size_proxy",
]
