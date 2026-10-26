"""Template performance tracker.

Reads one or more simulation logs and computes per-template statistics:
- Total simulations
- Pass rate (grade in GOOD/EXCELLENT/SPECTACULAR)
- Average and max Sharpe
- Most common failing checks

Usage:
    python -m alpha_search.feedback --sim-log artifacts/model16_sim.jsonl
    python -m alpha_search.feedback --sim-log artifacts/*.jsonl
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


_GOOD_GRADES = {"GOOD", "EXCELLENT", "SPECTACULAR", "PASS"}


def read_sim_log(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("status") == "done":
                    records.append(r)
            except json.JSONDecodeError:
                pass
    return records


_OOS_YEAR = 2023
_OOS_ROBUST_THRESHOLD = 1.0  # 2023 Sharpe must be >= this to count as robust pass


def compute_stats(records: list[dict]) -> dict[str, dict]:
    """Return per-template performance statistics including 2023 robustness."""
    stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "passing": 0,
        "passing_robust": 0,   # passing AND 2023 Sharpe >= threshold
        "sharpe_sum": 0.0,
        "max_sharpe": 0.0,
        "sharpe_2023_sum": 0.0,
        "sharpe_2023_n": 0,
        "failing_checks": Counter(),
        "dataset_id": "",
    })

    for r in records:
        tid = r.get("template_id") or "unknown"
        s = stats[tid]
        s["total"] += 1
        s["dataset_id"] = r.get("dataset_id", s["dataset_id"])

        sharpe = r.get("is_sharpe") or 0.0
        s["sharpe_sum"] += sharpe
        s["max_sharpe"] = max(s["max_sharpe"], sharpe)

        passes = r.get("grade") in _GOOD_GRADES
        if passes:
            s["passing"] += 1

        # 2023 out-of-sample Sharpe (from yearly_stats or oos_sharpe_2023 field)
        oos_2023 = r.get("oos_sharpe_2023")
        if oos_2023 is None:
            yearly = r.get("yearly_stats") or {}
            oos_2023 = (yearly.get(_OOS_YEAR) or {}).get("sharpe")

        if oos_2023 is not None:
            s["sharpe_2023_sum"] += oos_2023
            s["sharpe_2023_n"] += 1
            if passes and oos_2023 >= _OOS_ROBUST_THRESHOLD:
                s["passing_robust"] += 1

        for c in (r.get("checks") or []):
            if isinstance(c, dict) and c.get("result") != "PASS":
                s["failing_checks"][c["name"]] += 1

    # Compute averages
    for s in stats.values():
        s["avg_sharpe"] = s["sharpe_sum"] / s["total"] if s["total"] else 0.0
        s["pass_rate"] = s["passing"] / s["total"] if s["total"] else 0.0
        s["pass_rate_robust"] = s["passing_robust"] / s["total"] if s["total"] else 0.0
        s["avg_sharpe_2023"] = (
            s["sharpe_2023_sum"] / s["sharpe_2023_n"] if s["sharpe_2023_n"] else None
        )
        del s["sharpe_sum"], s["sharpe_2023_sum"], s["sharpe_2023_n"]

    return dict(stats)


def print_summary(stats: dict[str, dict]) -> None:
    """Print a human-readable summary table sorted by max Sharpe."""
    rows = sorted(stats.items(), key=lambda x: x[1]["max_sharpe"], reverse=True)

    header = (
        f"\n{'Template':<42} {'Dataset':<14} {'N':>5} "
        f"{'Pass%':>6} {'Rob%':>6} {'AvgSh':>7} {'MaxSh':>7} {'Sh2023':>7}  Failing checks"
    )
    print(header)
    print("-" * 125)
    for tid, s in rows:
        top_checks = ", ".join(f"{k}:{v}" for k, v in s["failing_checks"].most_common(2))
        sh2023 = f"{s['avg_sharpe_2023']:.3f}" if s["avg_sharpe_2023"] is not None else "   N/A"
        print(
            f"{tid:<42} {s['dataset_id']:<14} {s['total']:>5} "
            f"{s['pass_rate']:>6.1%} {s['pass_rate_robust']:>6.1%} "
            f"{s['avg_sharpe']:>7.3f} {s['max_sharpe']:>7.3f} {sh2023:>7}  {top_checks}"
        )

    total = sum(s["total"] for s in stats.values())
    passing = sum(s["passing"] for s in stats.values())
    all_sharpes = [r for s in stats.values() for r in [s["avg_sharpe"]]]
    print(f"\nTotal: {total} simulations | {passing} passing ({passing/total:.1%} pass rate)" if total else "")


def status(sim_log_paths: list[str]) -> dict[str, dict]:
    """Aggregate stats across multiple sim logs."""
    all_records: list[dict] = []
    for path in sim_log_paths:
        all_records.extend(read_sim_log(path))
    print(f"Loaded {len(all_records)} completed simulations from {len(sim_log_paths)} log(s)")
    stats = compute_stats(all_records)
    print_summary(stats)
    return stats


def main() -> None:
    import argparse
    import glob

    p = argparse.ArgumentParser(description="Show template performance stats from simulation logs.")
    p.add_argument("--sim-log", nargs="+", required=True,
                   help="Simulation log JSONL paths (supports glob patterns).")
    args = p.parse_args()

    # Expand globs
    paths: list[str] = []
    for pattern in args.sim_log:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])

    status(paths)


if __name__ == "__main__":
    main()
