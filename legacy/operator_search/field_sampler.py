"""Field/cluster loading and pair sampling for operator-first alpha discovery.

Uses the existing alpha_grammar cluster cache (all_clusters.json) as the
primary abstraction.  Each cluster has a human-readable label (e.g.
"Assets - Total", "Daily market capitalization") that carries the economic
meaning of the slot.

Sampling is at cluster level:
    - Pick cluster A → use its champion field_id as the expression slot
    - Pick cluster B → use its champion field_id
    - The pair label (A.label, B.label) is displayed in dry-run output so you
      can inspect the economic relationship before ever submitting to Brain

No coverage filter.  No co-occurrence weighting.  Pure uniform random over
the cluster pool.  Q-table pruned fields are excluded at load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from operator_search.templates import RelationshipTemplate


_DEFAULT_CLUSTER_CACHE = (
    Path(__file__).parent.parent / "alpha_grammar/artifacts/all_clusters.json"
)

# Datasets whose champion fields are sensible as expression terminals.
# GROUP fields and pure-technical fields (currency codes etc.) are excluded.
_NUMERIC_DATASETS = frozenset({
    "analyst4", "fundamental2", "fundamental6",
    "model16", "model51",
    "news12", "news18",
    "option8", "option9",
    "pv1", "pv13",
    "sentiment1", "socialmedia12", "socialmedia8",
})


@dataclass(frozen=True)
class ClusterRecord:
    cluster_id: str
    label: str                  # Human-readable economic description
    dataset_id: str
    champion_field_id: str      # Primary representative field
    n_members: int

    @property
    def short_label(self) -> str:
        """Truncated label for display."""
        return self.label[:55] if len(self.label) > 55 else self.label


def load_clusters(
    path: str | Path = _DEFAULT_CLUSTER_CACHE,
    exclude_datasets: frozenset[str] | None = None,
) -> list[ClusterRecord]:
    """Load cluster records from the alpha_grammar cluster cache JSON.

    Filters to MATRIX champion fields from numeric datasets only.
    """
    excluded = exclude_datasets or frozenset()
    with open(path) as f:
        raw = json.load(f)

    records: list[ClusterRecord] = []
    for c in raw:
        ds = c.get("dataset_id", "")
        if ds in excluded or ds not in _NUMERIC_DATASETS:
            continue
        champion_id = c.get("champion_field_id", "")
        if not champion_id:
            continue
        # Skip champions that are VECTOR or GROUP type (check members)
        members = c.get("members", [])
        champion_member = next((m for m in members if m["field_id"] == champion_id), None)
        if champion_member and champion_member.get("field_type", "MATRIX").upper() != "MATRIX":
            continue
        records.append(ClusterRecord(
            cluster_id=c["cluster_id"],
            label=c["label"],
            dataset_id=ds,
            champion_field_id=champion_id,
            n_members=len(members),
        ))
    return records


def summarize_clusters(clusters: list[ClusterRecord]) -> None:
    """Print cluster count by dataset (diagnostic helper)."""
    from collections import Counter
    ds_counts: Counter[str] = Counter(c.dataset_id for c in clusters)
    print(f"Cluster pool: {len(clusters)} clusters")
    print()
    for ds, n in ds_counts.most_common():
        print(f"  {ds:20s}  {n:4d} clusters")


# ---------------------------------------------------------------------------
# Q-table integration
# ---------------------------------------------------------------------------

def _op_for_template(template: RelationshipTemplate) -> str:
    tid = template.template_id
    if "corr" in tid:
        return "ts_corr"
    if "covariance" in tid:
        return "ts_covariance"
    if "regression" in tid:
        return "ts_regression"
    if "ratio" in tid or "spread" in tid:
        return "divide"
    return ""


def pruned_fields_for_op(op: str, qtable_path: str | Path) -> frozenset[str]:
    if not Path(qtable_path).exists():
        return frozenset()
    try:
        from alpha_grammar.qtable import QTable
        qt = QTable.load(str(qtable_path))
        return frozenset(fid for (fid, op_name) in qt.pruned_level2() if op_name == op)
    except Exception:
        return frozenset()


def get_pruned_fields(
    template: RelationshipTemplate,
    qtable_path: str | Path,
) -> frozenset[str]:
    op = _op_for_template(template)
    if not op:
        return frozenset()
    return pruned_fields_for_op(op, qtable_path)


# ---------------------------------------------------------------------------
# Pair / triple sampling
# ---------------------------------------------------------------------------

def _build_cluster_pool(
    clusters: list[ClusterRecord],
    allowed_datasets: frozenset[str] | None,
    pruned_fields: frozenset[str],
) -> list[ClusterRecord]:
    return [
        c for c in clusters
        if c.champion_field_id not in pruned_fields
        and (allowed_datasets is None or c.dataset_id in allowed_datasets)
    ]


def sample_cluster_pairs(
    clusters: list[ClusterRecord],
    template: RelationshipTemplate,
    n_pairs: int,
    rng: np.random.Generator,
    pruned_fields: frozenset[str] | None = None,
) -> list[tuple[ClusterRecord, ClusterRecord]]:
    """Sample n_pairs distinct (cluster_a, cluster_b) pairs.

    Deduplication is unordered: (A, B) and (B, A) are the same pair.
    The champion field_id of each cluster fills the template slot.
    """
    pruned = pruned_fields or frozenset()
    pool_a = _build_cluster_pool(clusters, getattr(template, "slot_a_datasets", None), pruned)
    pool_b = _build_cluster_pool(clusters, getattr(template, "slot_b_datasets", None), pruned)

    if not pool_a or not pool_b:
        return []

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[ClusterRecord, ClusterRecord]] = []
    max_attempts = n_pairs * 50
    attempts = 0

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        a = pool_a[rng.integers(0, len(pool_a))]
        b = pool_b[rng.integers(0, len(pool_b))]
        if a.champion_field_id == b.champion_field_id:
            continue
        key = (
            min(a.champion_field_id, b.champion_field_id),
            max(a.champion_field_id, b.champion_field_id),
        )
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))

    return pairs


def sample_cluster_triples(
    clusters: list[ClusterRecord],
    template: RelationshipTemplate,
    n_triples: int,
    rng: np.random.Generator,
    pruned_fields: frozenset[str] | None = None,
) -> list[tuple[ClusterRecord, ClusterRecord, ClusterRecord]]:
    """Sample n_triples distinct (a, b, c) cluster triples."""
    pruned = pruned_fields or frozenset()
    pool_ab = _build_cluster_pool(clusters, getattr(template, "slot_a_datasets", None), pruned)
    pool_c = _build_cluster_pool(clusters, getattr(template, "slot_c_datasets", None), pruned)

    if len(pool_ab) < 2 or not pool_c:
        return []

    seen: set[tuple[str, str, str]] = set()
    triples: list[tuple[ClusterRecord, ClusterRecord, ClusterRecord]] = []
    max_attempts = n_triples * 100
    attempts = 0

    while len(triples) < n_triples and attempts < max_attempts:
        attempts += 1
        a = pool_ab[rng.integers(0, len(pool_ab))]
        b = pool_ab[rng.integers(0, len(pool_ab))]
        c = pool_c[rng.integers(0, len(pool_c))]
        ids = {a.champion_field_id, b.champion_field_id, c.champion_field_id}
        if len(ids) < 3:
            continue
        key = (a.champion_field_id, b.champion_field_id, c.champion_field_id)
        if key in seen:
            continue
        seen.add(key)
        triples.append((a, b, c))

    return triples
