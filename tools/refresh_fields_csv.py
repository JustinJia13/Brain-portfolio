"""Refresh all_fields.csv from Brain API and regenerate the cluster cache.

Fetches ALL datafields (all types: MATRIX, VECTOR, GROUP, SYMBOL, UNIVERSE)
for USA/TOP3000/delay1/EQUITY, writes all_fields.csv, then rebuilds
all_clusters.json with the new field list.

Usage:
    python tools/refresh_fields_csv.py
    python tools/refresh_fields_csv.py --credentials brain2.txt
    python tools/refresh_fields_csv.py --max-clusters 400 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SEARCH_SCOPE_DELAY1 = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
}

SEARCH_SCOPE_DELAY0 = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 0,
}

FIELDS_CSV = ROOT / "all_fields.csv"
CLUSTERS_JSON = ROOT / "alpha_discovery/grammar_method" / "artifacts" / "all_clusters.json"


def fetch_all_fields(session) -> list[dict]:
    from core.brain.api import _paged_get

    def _fetch(scope: dict, label: str) -> list[dict]:
        print(f"Fetching datafields ({label}) ...")
        results = _paged_get(
            session,
            "https://api.worldquantbrain.com/data-fields",
            params=scope,
            limit=50,
            max_retries=20,
            default_sleep_seconds=60.0,
        )
        print(f"  {label}: {len(results)} fields")
        return results

    delay1 = _fetch(SEARCH_SCOPE_DELAY1, "delay=1, TOP3000")
    delay0 = _fetch(SEARCH_SCOPE_DELAY0, "delay=0, TOP3000")

    # Merge: union by field_id, delay1 takes precedence for duplicates
    merged: dict[str, dict] = {}
    for f in delay0:
        merged[f["id"]] = f
    for f in delay1:
        merged[f["id"]] = f  # delay1 wins on conflict

    fields = list(merged.values())
    print(f"  Merged: {len(fields)} unique fields ({len(delay1)} delay1 + {len(delay0)} delay0, {len(delay0)+len(delay1)-len(fields)} dupes dropped)")
    return fields


def write_csv(fields: list[dict], path: Path) -> None:
    rows = []
    for f in fields:
        ds = f.get("dataset") or {}
        cat = f.get("category") or {}
        sub = f.get("subcategory") or {}
        rows.append({
            "dataset_id":  ds.get("id", "") if isinstance(ds, dict) else str(ds),
            "field_id":    f.get("id", ""),
            "field_name":  f.get("description", ""),
            "type":        f.get("type", "MATRIX"),
            "coverage":    f.get("coverage", ""),
            "category":    cat.get("name", "") if isinstance(cat, dict) else str(cat),
            "subcategory": sub.get("name", "") if isinstance(sub, dict) else str(sub),
            "alpha_count": f.get("alphaCount", 0),
            "user_count":  f.get("userCount", 0),
        })
    # Sort by dataset_id then field_id for stable diffs
    rows.sort(key=lambda r: (r["dataset_id"], r["field_id"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset_id", "field_id", "field_name", "type",
            "coverage", "category", "subcategory", "alpha_count", "user_count",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows to {path}")


def rebuild_clusters(max_clusters: int) -> None:
    from core.clusters import load_clusters_from_csv, _save_clusters
    from collections import Counter

    print(f"\nRebuilding clusters (max_clusters={max_clusters}) ...")
    clusters = load_clusters_from_csv(str(FIELDS_CSV), max_clusters=max_clusters, allow_vector=True)

    ds_count = Counter(m.dataset_id for c in clusters for m in c.members)
    print(f"  {len(clusters)} clusters across {len(ds_count)} datasets:")
    for ds, n in sorted(ds_count.items(), key=lambda x: -x[1]):
        print(f"    {ds:<20} {n:4d} fields")

    CLUSTERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    _save_clusters(clusters, CLUSTERS_JSON)
    print(f"  Saved {CLUSTERS_JSON}")


def compare_csv(old_path: Path, new_fields: list[dict]) -> None:
    """Print a diff summary between old CSV and new fetch."""
    old_ids: set[str] = set()
    old_ds: dict[str, int] = {}
    try:
        with old_path.open() as f:
            for row in csv.DictReader(f):
                key = f"{row['dataset_id']}.{row['field_id']}"
                old_ids.add(key)
                old_ds[row["dataset_id"]] = old_ds.get(row["dataset_id"], 0) + 1
    except FileNotFoundError:
        print("  (no existing CSV to compare)")
        return

    new_ids: set[str] = set()
    new_ds: dict[str, int] = {}
    for f in new_fields:
        ds = (f.get("dataset") or {}).get("id", "")
        fid = f.get("id", "")
        new_ids.add(f"{ds}.{fid}")
        new_ds[ds] = new_ds.get(ds, 0) + 1

    added = new_ids - old_ids
    removed = old_ids - new_ids

    print(f"\n  Diff vs existing all_fields.csv:")
    print(f"    Added:   {len(added)} fields")
    print(f"    Removed: {len(removed)} fields")

    if added:
        print("  New fields:")
        for key in sorted(added)[:20]:
            print(f"    + {key}")
        if len(added) > 20:
            print(f"    ... and {len(added)-20} more")

    if removed:
        print("  Removed fields:")
        for key in sorted(removed)[:20]:
            print(f"    - {key}")
        if len(removed) > 20:
            print(f"    ... and {len(removed)-20} more")

    print(f"\n  Dataset sizes (old → new):")
    all_ds = sorted(set(list(old_ds.keys()) + list(new_ds.keys())))
    for ds in all_ds:
        o = old_ds.get(ds, 0)
        n = new_ds.get(ds, 0)
        delta = n - o
        arrow = f"+{delta}" if delta >= 0 else str(delta)
        print(f"    {ds:<20} {o:4d} → {n:4d}  ({arrow})")


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh all_fields.csv from Brain API")
    p.add_argument("--credentials", default=str(ROOT / "brain.txt"))
    p.add_argument("--max-clusters", type=int, default=400)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and compare but do not write CSV or clusters",
    )
    p.add_argument(
        "--skip-clusters", action="store_true",
        help="Update CSV only, skip cluster rebuild",
    )
    args = p.parse_args()

    from core.brain.api import sign_in
    session = sign_in(args.credentials)

    fields = fetch_all_fields(session)
    compare_csv(FIELDS_CSV, fields)

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    write_csv(fields, FIELDS_CSV)

    if not args.skip_clusters:
        rebuild_clusters(args.max_clusters)

    print("\nDone.")


if __name__ == "__main__":
    main()
