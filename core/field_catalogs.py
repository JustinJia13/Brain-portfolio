"""Shared field catalog path helpers for active pipelines."""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Group/keyword literals that look like identifiers but are NOT data fields.
_NON_FIELD_IDENTIFIERS = frozenset({"market", "sector", "industry", "subindustry", "none"})
_IDENT_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


@lru_cache(maxsize=1)
def known_field_ids(root: Path = ROOT) -> frozenset[str]:
    """All known Brain data-field ids, from datafield_metadata.json + all_fields*.csv."""
    ids: set[str] = set()
    meta = root / "datafield_metadata.json"
    if meta.exists():
        for f in json.loads(meta.read_text()):
            if isinstance(f, dict) and f.get("id"):
                ids.add(str(f["id"]))
    for csv_path in root.glob("all_fields*.csv"):
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            col = "field_id" if "field_id" in cols else (cols[1] if len(cols) > 1 else None)
            if not col:
                continue
            for row in reader:
                v = row.get(col)
                if v:
                    ids.add(str(v))
    return frozenset(ids)


def derive_fields(expression: str, *, root: Path = ROOT) -> list[str]:
    """Extract data-field ids from a raw FASTEXPR string, in order, deduped.

    Used to recover the `fields` list for records that carry an expression but
    no parsed fields (e.g. api_backfill records), so they can feed the Q-table.
    Group literals (market/sector/...) are excluded.
    """
    known = known_field_ids(root)
    seen: list[str] = []
    for tok in _IDENT_RE.findall(expression or ""):
        if tok in known and tok not in _NON_FIELD_IDENTIFIERS and tok not in seen:
            seen.append(tok)
    return seen


def credential_tag(credentials_path: str | Path) -> str:
    """Return a short suffix derived from a Brain credentials filename."""
    stem = Path(credentials_path).stem
    return "" if stem == "brain" else stem


def default_fields_csv(credentials: str | Path, root: Path = ROOT) -> Path:
    """Return the default all-fields CSV path for a credentials file."""
    tag = credential_tag(credentials)
    suffix = f"_{tag}" if tag else ""
    return root / f"all_fields{suffix}.csv"


__all__ = ["credential_tag", "default_fields_csv", "derive_fields", "known_field_ids"]
