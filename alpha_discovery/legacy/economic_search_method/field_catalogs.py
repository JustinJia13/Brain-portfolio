"""Delay-aware field catalog helpers for economic search."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from core.brain.api import sign_in


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
DEFAULT_DELAY_FIELD_DIR = PACKAGE_DIR / "artifacts" / "field_catalogs"
DEFAULT_METADATA_PATH = ROOT / "datafield_metadata.json"
DELAY_DEFINITION = (
    "delay 0 uses yesterday's available data to trade today; "
    "delay 1 uses data from two days ago to trade today."
)
CSV_COLUMNS = [
    "dataset_id",
    "field_id",
    "field_name",
    "type",
    "coverage",
    "category",
    "subcategory",
    "alpha_count",
    "user_count",
    "delay",
    "universe",
]


def parse_target_delay(value: Any) -> int:
    try:
        delay = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("delay must be 0 or 1") from exc
    if delay not in {0, 1}:
        raise ValueError("delay must be 0 or 1")
    return delay


def credential_tag(credentials_path: str | Path) -> str:
    """Return a short tag derived from the credentials filename.

    brain.txt  → ""       (default credential, no suffix)
    brain2.txt → "brain2"
    General:   stem == "brain" → "", otherwise the stem.
    """
    stem = Path(credentials_path).stem  # e.g. "brain" or "brain2"
    return "" if stem == "brain" else stem


def delay_field_csv_path(
    delay: int,
    *,
    output_dir: Path = DEFAULT_DELAY_FIELD_DIR,
    universe: str = "TOP3000",
    credentials: str | Path | None = None,
) -> Path:
    delay = parse_target_delay(delay)
    tag = credential_tag(credentials) if credentials is not None else ""
    suffix = f"_{tag}" if tag else ""
    return Path(output_dir) / f"fields_delay{delay}_{universe}{suffix}.csv"


def default_fields_csv(credentials: str | Path, root: Path = ROOT) -> Path:
    """Return the all_fields CSV path for the given credentials.

    brain.txt  → <root>/all_fields.csv
    brain2.txt → <root>/all_fields_brain2.csv
    """
    tag = credential_tag(credentials)
    suffix = f"_{tag}" if tag else ""
    return root / f"all_fields{suffix}.csv"


def csv_has_delay_metadata(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return "delay" in (reader.fieldnames or [])


def resolve_delay_field_csv(
    *,
    target_delay: int,
    fields_csv: Path,
    delay_fields_dir: Path = DEFAULT_DELAY_FIELD_DIR,
    universe: str = "TOP3000",
    credentials: str | Path | None = None,
) -> Path:
    """Resolve a delay-safe field CSV or fail closed.

    The merged legacy ``all_fields.csv`` is not safe for delay-aware generation
    unless it contains explicit delay metadata. Delay-specific generated
    catalogs are preferred.

    When ``credentials`` is provided the credential-specific delay catalog is
    checked first (e.g. brain2.txt → fields_delay1_TOP3000_brain2.csv), then
    falls back to the credential-agnostic path.
    """
    target_delay = parse_target_delay(target_delay)
    if credentials is not None:
        cred_candidate = delay_field_csv_path(
            target_delay, output_dir=delay_fields_dir, universe=universe, credentials=credentials
        )
        if cred_candidate.exists():
            return cred_candidate
    candidate = delay_field_csv_path(target_delay, output_dir=delay_fields_dir, universe=universe)
    if candidate.exists():
        return candidate
    fields_csv = Path(fields_csv)
    if fields_csv.exists() and csv_has_delay_metadata(fields_csv):
        return fields_csv
    raise FileNotFoundError(
        "Missing delay-aware field catalog for delay "
        f"{target_delay}. Expected {candidate}, or pass --fields-csv with a delay column. "
        "Build one with: python -m alpha_discovery/legacy.economic_search_method.cli fields --delay "
        f"{target_delay} --credentials brain.txt"
    )


def write_field_catalog_csv(records: Iterable[dict[str, Any]], path: Path) -> int:
    rows = [_field_row(record) for record in records]
    rows.sort(key=lambda r: (r["dataset_id"], r["field_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def records_from_metadata(
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    delay: int,
    universe: str = "TOP3000",
) -> list[dict[str, Any]]:
    delay = parse_target_delay(delay)
    if not metadata_path.exists():
        return []
    data = json.loads(metadata_path.read_text())
    if not isinstance(data, list):
        return []
    return [
        row
        for row in data
        if int(row.get("delay", -1)) == delay and str(row.get("universe") or "") == universe
    ]


def fetch_delay_records(
    *,
    credentials: str,
    delay: int,
    universe: str = "TOP3000",
    region: str = "USA",
    instrument_type: str = "EQUITY",
    limit: int = 50,
) -> list[dict[str, Any]]:
    from core.brain.api import _paged_get

    delay = parse_target_delay(delay)
    session = sign_in(credentials)
    return _paged_get(
        session,
        "https://api.worldquantbrain.com/data-fields",
        params={
            "instrumentType": instrument_type,
            "region": region,
            "universe": universe,
            "delay": delay,
        },
        limit=limit,
        max_retries=20,
        default_sleep_seconds=60.0,
    )


def write_delay_catalog(
    *,
    delay: int,
    records: list[dict[str, Any]],
    output_dir: Path = DEFAULT_DELAY_FIELD_DIR,
    universe: str = "TOP3000",
    credentials: str | Path | None = None,
) -> Path:
    path = delay_field_csv_path(
        delay, output_dir=output_dir, universe=universe, credentials=credentials
    )
    write_field_catalog_csv(records, path)
    return path


def field_catalog_summary(path: Path, *, target_delay: int, limit: int = 20) -> dict[str, Any]:
    target_delay = parse_target_delay(target_delay)
    dataset = Counter()
    category = Counter()
    field_examples: list[dict[str, Any]] = []
    n_rows = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        has_delay = "delay" in (reader.fieldnames or [])
        for row in reader:
            if has_delay and str(row.get("delay")) != str(target_delay):
                continue
            n_rows += 1
            dataset[row.get("dataset_id") or "unknown"] += 1
            category[(row.get("category") or "unknown").lower().replace(" ", "_")] += 1
            if len(field_examples) < limit:
                field_examples.append(
                    {
                        "field_id": row.get("field_id"),
                        "dataset_id": row.get("dataset_id"),
                        "category": row.get("category"),
                        "field_name": row.get("field_name"),
                    }
                )
    return {
        "target_delay": target_delay,
        "path": str(path),
        "n_fields": n_rows,
        "dataset": [{"key": k, "count": v} for k, v in dataset.most_common(limit)],
        "category": [{"key": k, "count": v} for k, v in category.most_common(limit)],
        "field_examples": field_examples,
    }


def _field_row(record: dict[str, Any]) -> dict[str, Any]:
    dataset = record.get("dataset") or {}
    category = record.get("category") or {}
    subcategory = record.get("subcategory") or {}
    return {
        "dataset_id": dataset.get("id", "") if isinstance(dataset, dict) else str(dataset or ""),
        "field_id": record.get("id", ""),
        "field_name": record.get("description", ""),
        "type": record.get("type", "MATRIX"),
        "coverage": record.get("coverage", ""),
        "category": category.get("name", "") if isinstance(category, dict) else str(category or ""),
        "subcategory": subcategory.get("name", "") if isinstance(subcategory, dict) else str(subcategory or ""),
        "alpha_count": record.get("alphaCount", 0),
        "user_count": record.get("userCount", 0),
        "delay": record.get("delay", ""),
        "universe": record.get("universe", ""),
    }
