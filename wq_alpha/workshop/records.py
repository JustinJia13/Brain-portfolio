"""Shared workshop record loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_records(paths: list[str], *, tag_source: bool = False) -> list[dict[str, Any]]:
    """Load JSONL simulation/workshop records from existing paths.

    Invalid JSON lines are ignored to preserve the historical behavior of the
    batch workshop loader.
    """

    records: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if tag_source and isinstance(record, dict):
                        record = dict(record)
                        record["_source_path"] = str(p)
                    records.append(record)
                except json.JSONDecodeError:
                    pass
    return records


__all__ = ["load_records"]
