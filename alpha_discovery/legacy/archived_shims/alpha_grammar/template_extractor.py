"""Extract parameterized templates from grammar simulation results.

Called after each grammar batch by run_pipeline.py to convert passing
(and near-pass) alpha expressions into templates for alpha_search exploitation.

Templates are written to shared/discovered_templates.json.
The core extraction logic is shared with tools/mine_sim_templates.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Re-use the extraction utilities from tools/mine_sim_templates
from tools.mine_sim_templates import (
    is_passing,
    is_near_pass,
    mine_templates,
    load_existing,
    merge_templates,
)

DISCOVERED_PATH = ROOT / "shared" / "discovered_templates.json"


def extract_from_batch(sim_records: list[dict], out_path: Path = DISCOVERED_PATH) -> int:
    """Extract templates from a single batch of sim records and merge into out_path.

    Called by run_pipeline.py after each batch completes.

    Parameters
    ----------
    sim_records:
        List of sim record dicts from the completed batch (not the full log —
        just the new records so extraction stays fast).
    out_path:
        Path to shared/discovered_templates.json.

    Returns
    -------
    n_added : int
        Number of new templates added this batch.
    """
    if not sim_records:
        return 0

    new_templates = mine_templates(sim_records)
    if not new_templates:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(out_path)
    merged, added, updated = merge_templates(existing, new_templates)

    if added or updated:
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)

    return added


