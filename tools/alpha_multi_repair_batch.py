"""Scan a sim.jsonl for near-miss alphas and emit a repair batch for submission.

Usage:
    python -m alpha_multi.repair_batch \\
        --simulation-log alpha_grammar/artifacts/fundamental6_sim.jsonl \\
        --output alpha_multi/artifacts/repair_batch.jsonl \\
        --min-sharpe 1.25 --min-fitness 1.0 \\
        --top-n 10

Then submit the repair batch:
    python -m alpha_multi.run_plan \\
        --plan-path alpha_multi/artifacts/repair_batch.jsonl \\
        --simulation-log alpha_multi/artifacts/repair_sim.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wq_alpha.agents.repair import AlphaRepairAgent, load_sim_log, select_near_misses, RepairResult


_ARTIFACTS_DIR = Path(__file__).parent.parent / "alpha_multi" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate repair candidates for near-miss alphas.")
    p.add_argument("--simulation-log", required=True,
                   help="sim.jsonl to scan for near-misses")
    p.add_argument("--output", default="",
                   help="Output JSONL for repair candidates (defaults to artifacts/repair_batch.jsonl)")
    p.add_argument("--min-sharpe", type=float, default=1.25,
                   help="Minimum IS Sharpe to consider a near-miss (default 1.25)")
    p.add_argument("--min-fitness", type=float, default=1.0,
                   help="Minimum IS Fitness to consider a near-miss (default 1.0)")
    p.add_argument("--top-n", type=int, default=10,
                   help="Maximum number of near-misses to repair (default 10)")
    p.add_argument("--api-key", default="",
                   help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    p.add_argument("--model", default="gpt-5.5",
                   help="LLM model label for repair suggestions")
    return p.parse_args()


def make_repair_candidate(
    original_record: dict[str, Any],
    repair_expr: str,
    repair_index: int,
) -> dict[str, Any]:
    """Build a CandidateSpec-compatible dict for one repair expression."""
    orig_candidate_id = original_record.get("candidate_id", "")
    # Stable ID: MD5 of the repair expression
    cid = hashlib.md5(repair_expr.encode()).hexdigest()[:16]
    return {
        "candidate_id": cid,
        "family_id": f"repair|{orig_candidate_id}|v{repair_index}",
        "intention_id": "repair",
        "intention_label": "LLM Repair",
        "template_id": f"repair_{orig_candidate_id}",
        "search_lane": "repair",
        "rationale_type": "repair",
        "economic_source": original_record.get("economic_source", ""),
        "category": original_record.get("category", ""),
        "dataset_id": original_record.get("dataset_id", ""),
        "expression": repair_expr,
        "fields": original_record.get("fields", []),
        "field_types": original_record.get("field_types", []),
        "size_proxy_flags": original_record.get("size_proxy_flags", []),
        "cluster_ids": original_record.get("cluster_ids", []),
        "group": original_record.get("group", ""),
        "params": original_record.get("params", {}),
        "horizon_bucket": original_record.get("horizon_bucket", "medium"),
        "settings_signature": original_record.get("settings_signature", ""),
        "semantic_signature": f"repair|{orig_candidate_id}",
        "tree_depth": (original_record.get("tree_depth") or 0) + 1,
        "priority_score": float(original_record.get("is_sharpe") or 0.0),
        # Provenance
        "original_candidate_id": orig_candidate_id,
        "original_expression": original_record.get("expression", ""),
        "failure_modes": [],  # will be filled in main()
    }


def main() -> None:
    args = parse_args()
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else _ARTIFACTS_DIR / "repair_batch.jsonl"

    # ── 1. Load sim log ────────────────────────────────────────────────────
    print(f"Loading sim log: {args.simulation_log}")
    records = load_sim_log(args.simulation_log)
    print(f"Total records:   {len(records)}")

    # ── 2. Select near-misses ──────────────────────────────────────────────
    near_misses = select_near_misses(
        records,
        min_sharpe=args.min_sharpe,
        min_fitness=args.min_fitness,
        max_results=args.top_n,
    )
    print(f"Near-misses:     {len(near_misses)}")

    if not near_misses:
        print("No near-misses found. Nothing to repair.")
        return

    # ── 3. Run repair agent ────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "No OpenAI API key found. Set OPENAI_API_KEY env var or use --api-key."
        )

    agent = AlphaRepairAgent(api_key=api_key, model=args.model)

    repair_candidates: list[dict[str, Any]] = []
    for i, rec in enumerate(near_misses, 1):
        expr = rec.get("expression", "N/A")
        print(f"\n[{i}/{len(near_misses)}] Repairing: {expr[:80]}")
        print(f"  Sharpe={float(rec.get('is_sharpe') or 0.0):.3f}  Fitness={float(rec.get('is_fitness') or 0.0):.3f}")

        result: RepairResult = agent.repair(rec)
        print(f"  Failures: {result.failure_modes}")
        print(f"  Repairs suggested: {len(result.repair_expressions)}")
        for j, repair_expr in enumerate(result.repair_expressions, 1):
            print(f"    {j}. {repair_expr}")
            candidate = make_repair_candidate(rec, repair_expr, j)
            candidate["failure_modes"] = result.failure_modes
            repair_candidates.append(candidate)

    # ── 4. Write repair batch ──────────────────────────────────────────────
    # Deduplicate by expression string
    seen_exprs: set[str] = set()
    unique_candidates = []
    for c in repair_candidates:
        if c["expression"] not in seen_exprs:
            seen_exprs.add(c["expression"])
            unique_candidates.append(c)

    with output_path.open("w") as f:
        for c in unique_candidates:
            f.write(json.dumps(c) + "\n")

    print(f"\nRepair batch written → {output_path}")
    print(f"Total repair candidates: {len(unique_candidates)}")
    print(f"\nNext step:")
    print(f"  python -m alpha_multi.run_plan \\")
    print(f"      --plan-path {output_path} \\")
    print(f"      --simulation-log {output_path.parent / 'repair_sim.jsonl'}")


if __name__ == "__main__":
    main()
