"""Field catalog loaded from all_fields.csv.

Provides coverage and alpha_count lookup so the sweep can:
- Skip fields below a coverage threshold (sparse → CONCENTRATED_WEIGHT failures)
- Surface unexplored fields (low alpha_count + good coverage)
"""

from __future__ import annotations

import csv
from pathlib import Path

_DEFAULT_CSV = Path(__file__).resolve().parents[2] / "all_fields.csv"


def load(csv_path: str | Path = _DEFAULT_CSV) -> dict[str, dict]:
    """Return a dict keyed by field_id with coverage, alpha_count, dataset_id, field_name."""
    catalog: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            catalog[row["field_id"]] = {
                "dataset_id": row["dataset_id"],
                "field_name": row["field_name"],
                "type": row["type"],
                "coverage": float(row["coverage"]),
                "category": row.get("category", ""),
                "alpha_count": int(row["alpha_count"]),
                "user_count": int(row["user_count"]),
            }
    return catalog


def filter_by_coverage(
    field_ids: list[str],
    catalog: dict[str, dict],
    min_coverage: float = 0.5,
) -> list[str]:
    """Return only fields whose coverage meets the threshold.

    Fields not present in the catalog pass through (coverage unknown — don't block them).
    """
    return [
        f for f in field_ids
        if catalog.get(f, {}).get("coverage", 1.0) >= min_coverage
    ]


def unexplored(
    dataset_id: str,
    catalog: dict[str, dict],
    min_coverage: float = 0.5,
    max_alpha_count: int = 100,
) -> list[dict]:
    """Return fields with good coverage but few existing alphas (sorted by alpha_count asc)."""
    fields = [
        v | {"field_id": k}
        for k, v in catalog.items()
        if v["dataset_id"] == dataset_id
        and v["coverage"] >= min_coverage
        and v["alpha_count"] <= max_alpha_count
        and v["type"] == "MATRIX"  # only numeric fields are usable in expressions
    ]
    return sorted(fields, key=lambda x: x["alpha_count"])
