"""Catalogs and context assembly for the theory lane.

Loads the cluster catalog (alpha_discovery/grammar_method/artifacts/all_clusters.json), field
metadata (datafield_metadata.json), and wiki pages, and renders the slices
that go into the proposer prompt. The LLM never sees the full field universe —
only the candidate clusters for the current batch; validation against the full
whitelist happens in code (enumerate_/economist).
"""

from __future__ import annotations
from core.clusters import load_cluster_catalog

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLUSTERS_PATH = ROOT / "alpha_discovery/grammar_method" / "artifacts" / "all_clusters.json"
FIELD_METADATA_PATH = ROOT / "datafield_metadata.json"
WIKI_DIR = ROOT / "wiki"


def load_clusters(path: Path | None = None) -> dict[str, dict]:
    """{cluster_id: cluster} from the grammar cluster catalog.

    Thin alias; the implementation lives in core.clusters so the shared layer
    does not have to import from a discovery lane.
    """
    return load_cluster_catalog(path or CLUSTERS_PATH)


def load_field_metadata(path: Path | None = None) -> dict[str, dict]:
    """{field_id: metadata} including userCount/alphaCount crowding stats."""
    data = json.loads((path or FIELD_METADATA_PATH).read_text())
    return {f["id"]: f for f in data if isinstance(f, dict) and f.get("id")}


def field_whitelist(clusters: dict[str, dict]) -> set[str]:
    fields: set[str] = set()
    for cluster in clusters.values():
        for member in cluster.get("members") or []:
            fid = member.get("field_id")
            if fid:
                fields.add(str(fid))
    return fields


def median_user_count(metadata: dict[str, dict]) -> float:
    counts = sorted(
        int(m.get("userCount") or 0) for m in metadata.values()
    )
    if not counts:
        return 0.0
    mid = len(counts) // 2
    return float(counts[mid])


def cluster_crowding(cluster: dict, metadata: dict[str, dict]) -> dict:
    """Max crowding stats over the cluster's members (worst case binds)."""
    users = alphas = 0
    for member in cluster.get("members") or []:
        meta = metadata.get(str(member.get("field_id") or ""), {})
        users = max(users, int(meta.get("userCount") or 0))
        alphas = max(alphas, int(meta.get("alphaCount") or 0))
    return {"userCount": users, "alphaCount": alphas}


def render_cluster_catalog(
    cluster_ids: list[str],
    clusters: dict[str, dict],
    metadata: dict[str, dict],
    *,
    max_members: int = 8,
) -> str:
    """Compact per-cluster catalog block for the proposer prompt."""
    lines: list[str] = []
    for cid in cluster_ids:
        cluster = clusters.get(cid)
        if cluster is None:
            continue
        crowd = cluster_crowding(cluster, metadata)
        lines.append(
            f"- {cid} | dataset={cluster.get('dataset_id')} | {cluster.get('label')} "
            f"| userCount={crowd['userCount']} alphaCount={crowd['alphaCount']}"
        )
        for member in (cluster.get("members") or [])[:max_members]:
            desc = str(member.get("description") or "").strip()
            lines.append(f"    {member.get('field_id')}: {desc[:110]}")
    return "\n".join(lines) if lines else "(no clusters)"


def render_cluster_index(
    clusters: dict[str, dict],
    metadata: dict[str, dict],
) -> str:
    """One-line-per-cluster index of the FULL catalog (no member detail).

    Lets the proposer reference any cluster_id for relational / cross-dataset
    cards, not just the detailed batch. Validation (validate_card) and rendering
    (enumerate_family / card_fields) already accept every cluster_id, so this
    only makes the rest of the catalog visible — at one short line each.
    Sorted least-crowded first to surface combinable neglected datasets.
    """
    rows = [
        (cluster_crowding(c, metadata)["userCount"], cid, c)
        for cid, c in clusters.items()
    ]
    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [
        f"- {cid} | dataset={c.get('dataset_id')} | {c.get('label')} | userCount={uc}"
        for uc, cid, c in rows
    ]
    return "\n".join(lines) if lines else "(no clusters)"


def render_operator_menu() -> str:
    """Operator families from the grammar registry, grouped by category."""
    from core.expr.operators import OPERATOR_REGISTRY, BANNED_GENERATION_OPS

    by_category: dict[str, list[str]] = {}
    for name, spec in OPERATOR_REGISTRY.items():
        if name in BANNED_GENERATION_OPS:
            continue
        by_category.setdefault(spec.category, []).append(name)
    lines = [f"- {cat}: {', '.join(sorted(ops))}" for cat, ops in sorted(by_category.items())]
    return "\n".join(lines)


def wiki_excerpts(cluster_ids: list[str], *, max_chars_per_page: int = 2500) -> str:
    """Relevant wiki cluster pages, truncated; empty marker when none exist."""
    from core.wiki.generate import _page_name  # canonical name incl. length cap

    parts: list[str] = []
    for cid in cluster_ids:
        page = WIKI_DIR / "clusters" / f"{_page_name(cid)}.md"
        if page.exists():
            parts.append(f"### {cid}\n{page.read_text()[:max_chars_per_page]}")
    return "\n\n".join(parts) if parts else "(no wiki evidence yet for these clusters)"


__all__ = [
    "CLUSTERS_PATH",
    "FIELD_METADATA_PATH",
    "WIKI_DIR",
    "cluster_crowding",
    "field_whitelist",
    "load_clusters",
    "load_field_metadata",
    "median_user_count",
    "render_cluster_catalog",
    "render_cluster_index",
    "render_operator_menu",
    "wiki_excerpts",
]
