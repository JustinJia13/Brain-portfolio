"""Generate negated versions of strong-inverse-signal grammar alphas.

When a grammar alpha has a strongly negative Sharpe (e.g. -1.3), the signal
is real but pointing in the wrong direction. Negating the expression flips
the signal and may yield a passing alpha without any structural changes.

How it works:
  1. Scan grammar sim log(s) for alphas with Sharpe <= threshold (default -1.0)
  2. For each, generate the negated expression using the same sign-flip logic
     as repair.py (pushes negation as deep as possible into the expression tree)
  3. Write a candidate JSONL plan ready for submission via core.brain.submit

Usage:
    python tools/try_negations.py
    python tools/try_negations.py --threshold -1.0 --out artifacts/negations_batch.jsonl
    python tools/try_negations.py --sim-log alpha_discovery/grammar_method/artifacts/all_sim.jsonl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_SIM_GLOB = str(ROOT / "*" / "artifacts" / "*sim*.jsonl")
DEFAULT_OUT = ROOT / "artifacts" / "negations_batch.jsonl"
DEFAULT_THRESHOLD = -1.0


# ---------------------------------------------------------------------------
# Re-use sign-flip from repair.py
# ---------------------------------------------------------------------------

def _candidate_id(expr: str) -> str:
    return hashlib.sha1(expr.encode()).hexdigest()[:16]


def _load_sims(paths: list[str]) -> list[dict]:
    records = []
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
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def find_strong_negatives(
    records: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """Return unique done records with Sharpe <= threshold, sorted most-negative last."""
    seen: set[str] = set()
    out = []
    for r in records:
        if r.get("status") != "done":
            continue
        sharpe = r.get("is_sharpe")
        if sharpe is None or sharpe > threshold:
            continue
        expr = r.get("expression", "")
        if not expr or expr in seen:
            continue
        seen.add(expr)
        out.append(r)
    out.sort(key=lambda r: r.get("is_sharpe") or 0)  # most negative first
    return out


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from alpha_enhancement.workshop.repair import _sign_flip

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-log", nargs="*",
                        help="Sim log paths (supports globs). Default: all pipeline logs.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Sharpe threshold for negation (default {DEFAULT_THRESHOLD}).")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output JSONL path (default: {DEFAULT_OUT}).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Resolve sim log paths
    if args.sim_log:
        paths: list[str] = []
        for pattern in args.sim_log:
            expanded = glob.glob(pattern)
            paths.extend(expanded if expanded else [pattern])
    else:
        paths = glob.glob(DEFAULT_SIM_GLOB)

    print(f"Loading from {len(paths)} sim log(s)...")
    records = _load_sims(paths)
    print(f"  Total records: {len(records)}")

    negatives = find_strong_negatives(records, args.threshold)
    print(f"  Strong-negative (Sharpe <= {args.threshold}): {len(negatives)}")

    # Load already-known expressions to avoid re-submitting
    known: set[str] = {r.get("expression", "") for r in records}

    candidates = []
    for r in negatives:
        original = r.get("expression", "")
        sharpe = r.get("is_sharpe", 0)
        negated = _sign_flip(original)

        if negated == original or negated in known:
            continue

        known.add(negated)
        candidates.append({
            "candidate_id": _candidate_id(negated),
            "template_id": f"negation:{r.get('template_id', r.get('candidate_id', 'grammar'))[:30]}",
            "dataset_id": r.get("dataset_id", ""),
            "category": r.get("category", ""),
            "search_lane": "negation",
            "expression": negated,
            "fields": r.get("fields") or [],
            "cluster_ids": r.get("cluster_ids") or [],
            "group": r.get("group", ""),
            "params": {
                "original_expression": original,
                "original_sharpe": sharpe,
            },
            "priority_score": abs(sharpe),  # higher |Sharpe| = stronger signal = try first
        })

    # Sort strongest signals first
    candidates.sort(key=lambda c: c["priority_score"], reverse=True)

    print(f"\nGenerated {len(candidates)} unique negation candidates")

    if args.dry_run:
        print("\n[DRY RUN] Sample:")
        for c in candidates[:8]:
            print(f"  original_sharpe={c['params']['original_sharpe']:.3f} | {c['expression'][:80]}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"Written: {out_path}")
    print(f"\nTo submit: python -m core.brain.submit --plan {out_path} --sim-log artifacts/negations_sim.jsonl")


if __name__ == "__main__":
    main()
