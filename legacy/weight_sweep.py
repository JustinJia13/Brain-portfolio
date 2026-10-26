"""Coordinate-descent weight sweep for the analyst4 composite alpha.

Fixes all signal weights at best-known values, then sweeps each signal
one at a time across a weight grid. This directly measures each signal's
contribution and avoids combinatorial explosion.

Usage
-----
# Build plan (inspect before submitting):
python weight_sweep.py --mode build --plan alpha_search/artifacts/weight_sweep.jsonl

# Show top results by fitness:
python weight_sweep.py --mode top --sim-log alpha_search/artifacts/weight_sweep_sim.jsonl

# Run (build + submit), shard 1 of 2:
python weight_sweep.py --shard 1/2 --credentials brain.txt  --sim-log alpha_search/artifacts/weight_sweep_sim.jsonl --history-log artifacts/all_sim.jsonl alpha_grammar/artifacts/analyst4_sim.jsonl

# Run shard 2 of 2 in another terminal:
python weight_sweep.py --shard 2/2 --credentials brain2.txt --sim-log alpha_search/artifacts/weight_sweep_sim.jsonl --history-log artifacts/all_sim.jsonl alpha_grammar/artifacts/analyst4_sim.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _merge_sim_log(path: str) -> None:
    try:
        from wq_alpha.logs import MASTER, merge_session_to_master

        n_new, n_updated = merge_session_to_master(path)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    except Exception as exc:
        print(f"[warn] Could not merge {path} to master: {exc}")


# ---------------------------------------------------------------------------
# Best-known baseline weights (fix these while sweeping others)
# ---------------------------------------------------------------------------

BASELINE = {
    # Best-known expression: fitness=1.99, Sharpe=1.62
    "anl4_totassets_flag":  2.0,
    "anl4_ffo_flag":        1.0,
    "anl4_bvps_flag":       1.19,
    "anl4_ptpr_flag":       0.3,
    "anl4_netdebt_flag":    0.5,
    "anl4_capex_flag":      0.4,
    "anl4_rd_exp_flag":     0.3,
    "anl4_netprofit_flag":  0.6,
    "anl4_tot_gw_ft":       0.2,   # NaN-safe sparse signal
}

# Rank layer — fixed, not swept
RANK_SIGNALS = [
    "anl4_totassets_flag",
    "anl4_bvps_flag",
    "anl4_ffo_flag",
    "anl4_netprofit_flag",
]

# Per-signal weight grids — always includes 0.0 (drop signal) and baseline value
SWEEP_GRIDS = {
    "anl4_totassets_flag":  [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "anl4_ffo_flag":        [0.0, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0],
    "anl4_bvps_flag":       [0.0, 0.5, 0.8, 1.0, 1.19, 1.5, 2.0],
    "anl4_ptpr_flag":       [0.0, 0.1, 0.3, 0.5, 0.8, 1.0],
    "anl4_netdebt_flag":    [0.0, 0.2, 0.4, 0.5, 0.7, 1.0],
    "anl4_capex_flag":      [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "anl4_rd_exp_flag":     [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "anl4_netprofit_flag":  [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
    "anl4_tot_gw_ft":       [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
    # cfi_flag not in current baseline — test if adding it helps
    "anl4_cfi_flag":        [0.0, 0.2, 0.3, 0.5, 0.7],
}

# NaN-safe wrapper for sparse fields
NAN_SAFE = {
    "anl4_tot_gw_ft": "if_else(is_nan(anl4_tot_gw_ft), 0, anl4_tot_gw_ft)",
}


# ---------------------------------------------------------------------------
# Expression builder
# ---------------------------------------------------------------------------

def build_expression(weights: dict[str, float]) -> str:
    raw_terms = []
    for sig, w in weights.items():
        if w == 0.0:
            continue
        expr = NAN_SAFE.get(sig, sig)
        raw_terms.append(expr if w == 1.0 else f"{w}*{expr}")

    rank_terms = [f"rank({s})" for s in RANK_SIGNALS]

    raw_sum = " + ".join(raw_terms)
    rank_sum = " + ".join(rank_terms)
    return f"normalize(\n  ({raw_sum}) +\n  ({rank_sum})\n)"


def candidate_id(expr: str) -> str:
    return hashlib.md5(expr.encode()).hexdigest()[:16]


def _infer_sweep_metadata(record: dict) -> tuple[str, float | str | None]:
    """Recover sweep signal/weight for older log rows that only preserved params."""
    sweep_signal = record.get("sweep_signal")
    sweep_weight = record.get("sweep_weight")
    if sweep_signal is not None or sweep_weight is not None:
        return (sweep_signal or record.get("template_id") or "?"), sweep_weight

    template_id = record.get("template_id") or "?"
    if template_id == "baseline":
        return "baseline", None

    params = record.get("params") or {}
    diffs = [(sig, weight) for sig, weight in params.items() if BASELINE.get(sig) != weight]
    if len(diffs) == 1:
        return diffs[0]
    return template_id, None


# ---------------------------------------------------------------------------
# Plan generation — coordinate descent
# ---------------------------------------------------------------------------

def build_plan(output_path: str, max_candidates: int = 0) -> list[dict]:
    candidates = []
    seen = set()

    # Always include the baseline itself first
    baseline_expr = build_expression(BASELINE)
    cid = candidate_id(baseline_expr)
    seen.add(cid)
    candidates.append({
        "candidate_id": cid,
        "template_id": "baseline",
        "dataset_id": "analyst4",
        "expression": baseline_expr,
        "sweep_signal": "baseline",
        "params": dict(BASELINE),
    })

    # For each signal, sweep its weight while holding all others fixed
    for sweep_sig, grid in SWEEP_GRIDS.items():
        for w in grid:
            weights = dict(BASELINE)
            weights[sweep_sig] = w
            expr = build_expression(weights)
            cid = candidate_id(expr)
            if cid in seen:
                continue
            seen.add(cid)
            candidates.append({
                "candidate_id": cid,
                "template_id": f"sweep_{sweep_sig.replace('anl4_', '').replace('_flag', '').replace('_ft', '')}",
                "dataset_id": "analyst4",
                "expression": expr,
                "sweep_signal": sweep_sig,
                "sweep_weight": w,
                "params": weights,
            })

            if max_candidates and len(candidates) >= max_candidates:
                break
        if max_candidates and len(candidates) >= max_candidates:
            break

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    total = len(candidates)
    print(f"Generated {total} candidates ({len(SWEEP_GRIDS)} signals × ~{total // len(SWEEP_GRIDS)} weights) → {output_path}")
    return candidates


# ---------------------------------------------------------------------------
# Top results display
# ---------------------------------------------------------------------------

def top_results(sim_logs: list[str], n: int = 30) -> None:
    records = []
    for path in sim_logs:
        p = Path(path)
        if not p.exists():
            print(f"  [not found] {path}")
            continue
        with p.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "done" and r.get("is_fitness") is not None:
                        records.append(r)
                except json.JSONDecodeError:
                    pass

    if not records:
        print("No completed simulations found.")
        return

    # Collapse duplicate expressions from retries / concurrent shards.
    deduped: dict[str, dict] = {}
    for record in records:
        expr = record.get("expression")
        if not expr:
            continue
        best = deduped.get(expr)
        if best is None or (record.get("is_fitness") or 0.0) > (best.get("is_fitness") or 0.0):
            deduped[expr] = record
    records = list(deduped.values())

    records.sort(key=lambda r: r.get("is_fitness") or 0.0, reverse=True)
    passing = sum(1 for r in records if r.get("grade") in {"GOOD", "EXCELLENT", "SPECTACULAR"})
    print(f"\nLoaded {len(records)} simulations | {passing} passing\n")

    # Group by sweep_signal for per-signal best
    from collections import defaultdict
    by_signal: dict[str, list] = defaultdict(list)
    for r in records:
        sig, _ = _infer_sweep_metadata(r)
        by_signal[sig].append(r)

    print("Per-signal best fitness:")
    print(f"  {'Signal':<28} {'BestFit':>8} {'BestSharpe':>10} {'BestWeight':>11} {'Sub-U':>6}")
    print("  " + "-" * 68)
    sig_bests = []
    for sig, recs in sorted(by_signal.items()):
        best = max(recs, key=lambda r: r.get("is_fitness") or 0.0)
        _, best_w = _infer_sweep_metadata(best)
        sub_u = "OK" if best.get("is_sub_universe_sharpe_ok") else "FAIL"
        sig_bests.append((sig, best))
        print(f"  {sig:<28} {best.get('is_fitness') or 0:>8.3f} {best.get('is_sharpe') or 0:>10.3f} {str(best_w if best_w is not None else 'baseline'):>11} {sub_u:>6}")

    print(f"\nTop {n} overall by fitness:")
    print(f"{'#':<4} {'Fitness':>7} {'Sharpe':>7} {'Grade':<12} {'Sub-U':>6}  {'Signal':<28} {'Weight':>7}  Expression")
    print("-" * 140)
    for i, r in enumerate(records[:n], 1):
        fitness  = r.get("is_fitness") or 0.0
        sharpe   = r.get("is_sharpe") or 0.0
        grade    = r.get("grade") or "—"
        sub_u    = "OK" if r.get("is_sub_universe_sharpe_ok") else "FAIL"
        sig, weight = _infer_sweep_metadata(r)
        expr     = (r.get("expression") or "").replace("\n", " ")[:60]
        print(f"{i:<4} {fitness:>7.3f} {sharpe:>7.3f} {grade:<12} {sub_u:>6}  {sig:<28} {str(weight if weight is not None else 'baseline'):>7}  {expr}")

    print(f"\nShowing {min(n, len(records))} of {len(records)} results")


# ---------------------------------------------------------------------------
# Dedup helper
# ---------------------------------------------------------------------------

def _load_done_expressions(sim_logs: list[str]) -> set[str]:
    done: set[str] = set()
    for path in sim_logs:
        p = Path(path)
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("expression"):
                        done.add(r["expression"])
                except json.JSONDecodeError:
                    pass
    return done


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Coordinate-descent weight sweep for analyst4 composite.")
    parser.add_argument("--mode", choices=["build", "submit", "run", "top"], default="run")
    parser.add_argument("--plan", default="alpha_search/artifacts/weight_sweep.jsonl")
    parser.add_argument("--sim-log", default="alpha_search/artifacts/weight_sweep_sim.jsonl",
                        help="Output sim log (results written here).")
    parser.add_argument("--history-log", nargs="*", default=[], metavar="PATH",
                        help="Additional sim logs to check for already-tested expressions (dedup only).")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="Limit total candidates (0 = unlimited)")
    parser.add_argument("--shard", default="1/1", metavar="N/TOTAL",
                        help="Shard the plan for parallel runs e.g. --shard 1/2 and --shard 2/2")
    parser.add_argument("--n", type=int, default=30, help="Number of top results to show (--mode top)")
    args = parser.parse_args()

    shard_idx, shard_total = (int(x) for x in args.shard.split("/"))
    assert 1 <= shard_idx <= shard_total

    if args.mode == "top":
        all_logs = [args.sim_log] + (args.history_log or [])
        top_results(all_logs, n=args.n)
        return

    if args.mode in ("build", "run"):
        build_plan(args.plan, max_candidates=args.max_candidates)

    if args.mode in ("submit", "run"):
        all_history = [args.sim_log] + (args.history_log or [])
        done = _load_done_expressions(all_history)

        plan_path = args.plan
        filtered_plan = Path(plan_path).with_suffix(".filtered.jsonl")

        kept = 0
        with open(plan_path) as fin, open(filtered_plan, "w") as fout:
            for line in fin:
                record = json.loads(line)
                if record.get("expression") in done:
                    continue
                if (kept % shard_total) != (shard_idx - 1):
                    kept += 1
                    continue
                fout.write(line)
                kept += 1

        print(f"Dedup: {len(done)} already tested | shard {shard_idx}/{shard_total} → {kept} remaining")

        from wq_alpha.brain.submit import run
        run(
            str(filtered_plan),
            args.sim_log,
            batch_size=args.batch_size,
            credentials=args.credentials,
            max_candidates=args.max_candidates,
        )
        _merge_sim_log(args.sim_log)


if __name__ == "__main__":
    main()
