"""Generate field-metadata-guided probe templates from all_fields.csv.

Produces general probe templates for high-signal fields (sorted by alpha_count)
that aren't already covered by existing templates in discovered_templates.json.

Templates generated per field:
  1. group_rank({field}, {group})
  2. group_rank(ts_zscore({field}, {window}), {group})
  3. group_rank(ts_rank({field}, {window}), {group})
  4. group_rank(ts_delta({field}, {window}), {group})
  5. group_rank(ts_mean({field}, {window}) - {field}, {group})   ← mean-reversion
  6. rank({field})

Writes to shared/discovered_templates.json (merges, deduplicates by signature).

Usage:
    python tools/generate_field_probes.py
    python tools/generate_field_probes.py --top-n 100 --min-coverage 0.5
    python tools/generate_field_probes.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIELDS_CSV = ROOT / "all_fields.csv"
OUT_PATH = ROOT / "shared" / "discovered_templates.json"

DEFAULT_TOP_N = 80
DEFAULT_MIN_COVERAGE = 0.5
PROBE_PRIORITY = 1.5
PROBE_WINDOWS = [20, 60]
PROBE_GROUPS = ["sector", "subindustry", "market"]

# Skip datasets that are too noisy or sparse for generic probes
SKIP_DATASETS = {"pv1", "pv2"}  # price/volume probes already well-covered

# Probe patterns: (pattern_key, pattern_template, needs_window)
PROBE_PATTERNS = [
    ("rankonly",  "rank({field})",                                    False),
    ("grp",       "group_rank({field}, {group})",                     False),
    ("grp_zs",    "group_rank(ts_zscore({field}, {window}), {group})", True),
    ("grp_rank",  "group_rank(ts_rank({field}, {window}), {group})",   True),
    ("grp_delta", "group_rank(ts_delta({field}, {window}), {group})",  True),
    ("grp_rev",   "group_rank(ts_mean({field}, {window}) - {field}, {group})", True),
]

DATASET_CATEGORY = {
    "model": "model", "fundamental": "fundamental",
    "analyst": "analyst", "news": "news",
    "socialmedia": "social", "pv": "price_volume",
    "options": "options", "trade": "trade",
}


def _category(dataset_id: str) -> str:
    for prefix, cat in DATASET_CATEGORY.items():
        if dataset_id.startswith(prefix):
            return cat
    return "other"


def _sig(pattern: str, dataset_id: str = "", field_id: str = "") -> str:
    import re
    normalised = re.sub(r"\s+", "", pattern.lower())
    # Include dataset and field so same structural pattern on different fields/datasets
    # gets distinct signatures
    parts = [p for p in (dataset_id, field_id, normalised) if p]
    return ":".join(parts)


def load_fields(path: Path, min_coverage: float, skip_datasets: set[str]) -> list[dict]:
    """Load and filter fields from all_fields.csv."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            dataset_id = row["dataset_id"]
            if dataset_id in skip_datasets:
                continue
            coverage = float(row.get("coverage") or 0)
            if coverage < min_coverage:
                continue
            if row.get("type", "MATRIX") not in ("MATRIX", ""):
                continue  # skip non-matrix field types
            rows.append({
                "field_id": row["field_id"],
                "dataset_id": dataset_id,
                "coverage": coverage,
                "alpha_count": int(row.get("alpha_count") or 0),
                "user_count": int(row.get("user_count") or 0),
                "category": _category(dataset_id),
            })
    # Sort by alpha_count descending
    rows.sort(key=lambda r: r["alpha_count"], reverse=True)
    return rows


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _covered_fields(existing: list[dict]) -> set[str]:
    """Return set of field_ids already covered by at least one existing template."""
    covered: set[str] = set()
    for t in existing:
        for fid in (t.get("fields") or []):
            covered.add(fid)
    return covered


def generate_probes(
    fields: list[dict],
    existing_sigs: set[str],
    covered_fields: set[str],
    top_n: int,
) -> list[dict]:
    """Generate probe templates for top-N fields not already covered."""
    probes = []
    seen_fields = 0

    for field_meta in fields:
        if seen_fields >= top_n:
            break

        fid = field_meta["field_id"]
        dataset_id = field_meta["dataset_id"]
        category = field_meta["category"]
        alpha_count = field_meta["alpha_count"]

        if fid in covered_fields:
            continue

        seen_fields += 1

        for probe_key, pattern_tpl, needs_window in PROBE_PATTERNS:
            sig = _sig(pattern_tpl, dataset_id, fid)

            if sig in existing_sigs:
                continue

            template_id = f"probe_{dataset_id}_{fid[:20]}_{probe_key}"
            windows = PROBE_WINDOWS if needs_window else []

            probes.append({
                "template_id": template_id,
                "dataset_id": dataset_id,
                "category": category,
                "description": (
                    f"Field probe: {fid} ({dataset_id}, alpha_count={alpha_count})"
                ),
                "pattern": pattern_tpl,
                "fields": [fid],
                "windows": windows,
                "groups": PROBE_GROUPS if "{group}" in pattern_tpl else [],
                "priority": PROBE_PRIORITY,
                "status": "active",
                "source": "field_probe",
                "signature": sig,
                "seed_alpha_id": "",
                "seed_sharpe": 0.0,
                "seed_expression": "",
            })

    return probes


def merge_into(existing: list[dict], new_probes: list[dict]) -> tuple[list[dict], int]:
    """Add new probes to existing list, skipping duplicate signatures."""
    existing_sigs = {t["signature"] for t in existing}
    added = 0
    result = list(existing)
    for p in new_probes:
        if p["signature"] not in existing_sigs:
            result.append(p)
            existing_sigs.add(p["signature"])
            added += 1
    # Sort by priority desc
    result.sort(key=lambda x: -x.get("priority", 0))
    return result, added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Number of top-alpha_count fields to probe (default: {DEFAULT_TOP_N})")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                        help=f"Minimum field coverage to include (default: {DEFAULT_MIN_COVERAGE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written but don't write files")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    out_path = Path(args.out)

    print(f"Loading fields from {FIELDS_CSV}...")
    fields = load_fields(FIELDS_CSV, args.min_coverage, SKIP_DATASETS)
    print(f"  Fields after coverage/type filter: {len(fields)}")

    print(f"Loading existing templates from {out_path}...")
    existing = load_existing(out_path)
    print(f"  Existing entries: {len(existing)}")
    existing_sigs = {t["signature"] for t in existing}
    covered = _covered_fields(existing)
    print(f"  Fields already covered: {len(covered)}")

    print(f"\nGenerating probes for top-{args.top_n} uncovered fields...")
    probes = generate_probes(fields, existing_sigs, covered, args.top_n)
    print(f"  Generated {len(probes)} probe templates")

    merged, added = merge_into(existing, probes)
    print(f"  Added: {added}  |  Total: {len(merged)}")

    if args.dry_run:
        print("\n[DRY RUN] Sample probes:")
        dataset_seen: set[str] = set()
        for p in probes:
            ds = p["dataset_id"]
            if ds not in dataset_seen:
                dataset_seen.add(ds)
                print(f"  [{ds}] {p['template_id']}")
                print(f"         pattern: {p['pattern']}")
                print(f"         field:   {p['fields']}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nWritten: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
