"""Quick result viewer for any sim log.

Usage:
    python tools/show_results.py                                  # workshop results
    python tools/show_results.py --sim-log alpha_grammar/artifacts/all_sim.jsonl
    python tools/show_results.py --min-sharpe 1.25 --passing-only
    python tools/show_results.py --lane negate repair
    python tools/show_results.py --top 20
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_LOG = ROOT / "artifacts" / "workshop" / "workshop_sim.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-log", nargs="*", help="Sim log path(s) or globs. Default: workshop sim log.")
    p.add_argument("--min-sharpe", type=float, default=None)
    p.add_argument("--passing-only", action="store_true", help="Show only sharpe>=1.25 and fitness>=1.0.")
    p.add_argument("--lane", nargs="*", help="Filter by search_lane(s): negate repair enhance sweep manual")
    p.add_argument("--top", type=int, default=0, help="Show only top N by Sharpe (0 = all).")
    p.add_argument("--show-checks", action="store_true", help="Show failing checks for each record.")
    args = p.parse_args()

    # Resolve paths
    if args.sim_log:
        paths: list[str] = []
        for pattern in args.sim_log:
            expanded = glob.glob(pattern)
            paths.extend(expanded if expanded else [pattern])
    else:
        paths = [str(DEFAULT_LOG)]

    records: list[dict] = []
    for path in paths:
        p_obj = Path(path)
        if not p_obj.exists():
            print(f"Not found: {path}")
            continue
        with p_obj.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    done = [r for r in records if r.get("status") == "done"]

    # Filters
    if args.passing_only:
        done = [r for r in done if (r.get("is_sharpe") or 0) >= 1.25 and (r.get("is_fitness") or 0) >= 1.0]
    if args.min_sharpe is not None:
        done = [r for r in done if (r.get("is_sharpe") or 0) >= args.min_sharpe]
    if args.lane:
        done = [r for r in done if r.get("search_lane") in args.lane]

    done.sort(key=lambda r: r.get("is_sharpe") or 0, reverse=True)

    if args.top:
        done = done[:args.top]

    # Counts
    n_total = len([r for r in records if r.get("status") == "done"])
    n_pass = sum(1 for r in records if r.get("status") == "done"
                 and (r.get("is_sharpe") or 0) >= 1.25
                 and (r.get("is_fitness") or 0) >= 1.0)
    n_fail = sum(1 for r in records if r.get("status") == "submit_failed")

    print(f"Logs: {', '.join(paths)}")
    print(f"Total done: {n_total}  |  Passing (sh>=1.25, fit>=1.0): {n_pass}  |  Submit failed: {n_fail}")
    if args.lane:
        print(f"Filter: lane={args.lane}")
    print()

    # Table
    print(f"{'sharpe':>7} {'fitness':>7} {'grade':<8} {'lane':<10} {'original_sharpe':>15}  expression")
    print("─" * 110)
    for r in done:
        sh   = r.get("is_sharpe") or 0
        ft   = r.get("is_fitness") or 0
        grade = (r.get("grade") or "")[:8]
        lane  = (r.get("search_lane") or "")[:10]
        expr  = (r.get("expression") or "")[:72]
        orig_sh = (r.get("params") or {}).get("original_sharpe")
        orig_str = f"{orig_sh:.3f}" if orig_sh is not None else ""
        print(f"{sh:>7.3f} {ft:>7.3f} {grade:<8} {lane:<10} {orig_str:>15}  {expr}")

        if args.show_checks:
            checks = r.get("checks") or []
            failing = [c["name"] for c in checks if c.get("result") != "PASS" and c.get("name") != "SELF_CORRELATION"]
            if failing:
                print(f"         {'':>7} {'':>8} {'':>10}   failing: {', '.join(failing)}")

    print()
    print(f"Showing {len(done)} record(s).")


if __name__ == "__main__":
    main()
