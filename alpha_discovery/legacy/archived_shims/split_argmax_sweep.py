"""Constant sweep for the split ts_arg_max composite alpha (delay=0).

Alpha:
    normalize(
        w1*group_rank(ts_arg_max(split, win1), subindustry)
      + w2*group_rank(ts_arg_max(split, win2), subindustry)
      + w3*rank(ts_sum(split, win3))
    )

Runs coordinate descent: sweeps one constant at a time while holding all
others at baseline. All candidates are submitted with delay=0 via settings_override.

Usage
-----
# Build plan (inspect before submitting):
python split_argmax_sweep.py --mode build --plan artifacts/split_argmax_sweep.jsonl

# Show top results:
python split_argmax_sweep.py --mode top --sim-log artifacts/split_argmax_sim.jsonl

# After sweep completes: pick best per axis, submit combined alpha:
python split_argmax_sweep.py --mode best --sim-log artifacts/split_argmax_sim.jsonl

# Run (build + submit):
python split_argmax_sweep.py --mode run \\
    --sim-log artifacts/split_argmax_sim.jsonl \\
    --credentials brain.txt

# Parallel shard 1/2:
python split_argmax_sweep.py --mode run --shard 1/2 --credentials brain.txt \\
    --sim-log artifacts/split_argmax_sim.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _merge_sim_log(path: str) -> None:
    try:
        from core.logs import MASTER, merge_session_to_master

        n_new, n_updated = merge_session_to_master(path)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    except Exception as exc:
        print(f"[warn] Could not merge {path} to master: {exc}")


# ---------------------------------------------------------------------------
# Baseline (the original alpha constants)
# ---------------------------------------------------------------------------

BASELINE = {
    "w1":   0.6,
    "w2":   0.25,
    "w3":   0.25,
    "win1": 60,
    "win2": 126,
    "win3": 252,
}

# Baseline simulation settings (delay=0 fixed for all candidates)
BASELINE_SETTINGS = {
    "delay":         0,
    "neutralization": "SUBINDUSTRY",
    "decay":         0,
}

# Coordinate-descent grids — baseline value is always included implicitly.
# Expression params (w*, win*) vary the formula; settings params vary the sim regime.
EXPR_GRIDS: dict[str, list[float | int]] = {
    "w1":   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "w2":   [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "w3":   [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "win1": [20, 40, 60, 90, 120],
    "win2": [63, 90, 126, 189],
    "win3": [126, 189, 252],
}

SETTINGS_GRIDS: dict[str, list] = {
    "neutralization": ["INDUSTRY", "MARKET", "SECTOR"],   # baseline: SUBINDUSTRY
    "decay":          [2, 4, 6],                           # baseline: 0
}


# ---------------------------------------------------------------------------
# Expression builder
# ---------------------------------------------------------------------------

def _fmt_weight(w: float) -> str:
    """Format weight compactly: 1.0 → '1.0', 0.25 → '0.25'."""
    return f"{w:g}"


def build_expression(p: dict) -> str:
    w1, w2, w3 = p["w1"], p["w2"], p["w3"]
    win1, win2, win3 = int(p["win1"]), int(p["win2"]), int(p["win3"])
    return (
        f"normalize(\n"
        f"    {_fmt_weight(w1)}*group_rank(ts_arg_max(split, {win1}), subindustry)\n"
        f"  + {_fmt_weight(w2)}*group_rank(ts_arg_max(split, {win2}), subindustry)\n"
        f"  + {_fmt_weight(w3)}*rank(ts_sum(split, {win3}))\n"
        f")"
    )


def candidate_id(expr: str) -> str:
    return hashlib.md5(expr.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Plan generation — coordinate descent
# ---------------------------------------------------------------------------

def build_plan(output_path: str, max_candidates: int = 0) -> list[dict]:
    candidates: list[dict] = []
    # Dedup key: (expression, neutralization, decay) — same expr under different regime is a distinct run
    seen: set[tuple] = set()

    def _add(
        expr_params: dict,
        settings_override: dict,
        template_id: str,
        sweep_key: str | None,
        sweep_val,
    ) -> None:
        expr = build_expression(expr_params)
        neut  = settings_override.get("neutralization", BASELINE_SETTINGS["neutralization"])
        decay = settings_override.get("decay", BASELINE_SETTINGS["decay"])
        key = (expr, neut, decay)
        if key in seen:
            return
        seen.add(key)
        cid = hashlib.md5((expr + neut + str(decay)).encode()).hexdigest()[:16]
        candidates.append({
            "candidate_id": cid,
            "template_id": template_id,
            "dataset_id": "model16",
            "category": "model",
            "search_lane": "const_sweep",
            "expression": expr,
            "fields": ["split"],
            "group": "subindustry",
            "sweep_key": sweep_key,
            "sweep_val": sweep_val,
            "params": dict(expr_params),
            "settings_override": settings_override,
        })

    baseline_override = {
        "delay":          BASELINE_SETTINGS["delay"],
        "neutralization": BASELINE_SETTINGS["neutralization"],
        "decay":          BASELINE_SETTINGS["decay"],
    }

    # Always start with baseline
    _add(BASELINE, baseline_override, "baseline", None, None)

    # Sweep expression constants (hold settings at baseline)
    for key, grid in EXPR_GRIDS.items():
        for val in grid:
            if max_candidates and len(candidates) >= max_candidates:
                break
            _add({**BASELINE, key: val}, baseline_override, f"sweep_{key}", key, val)
        if max_candidates and len(candidates) >= max_candidates:
            break

    # Sweep simulation settings (hold expression at baseline)
    for key, grid in SETTINGS_GRIDS.items():
        for val in grid:
            if max_candidates and len(candidates) >= max_candidates:
                break
            override = {**baseline_override, key: val}
            _add(BASELINE, override, f"sweep_{key}", key, val)
        if max_candidates and len(candidates) >= max_candidates:
            break

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    n_expr = sum(len(g) for g in EXPR_GRIDS.values())
    n_sett = sum(len(g) for g in SETTINGS_GRIDS.values())
    print(
        f"Generated {len(candidates)} candidates "
        f"({n_expr} expr variants + {n_sett} settings variants + 1 baseline, delay=0) → {output_path}"
    )
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

    # Deduplicate by expression
    deduped: dict[str, dict] = {}
    for r in records:
        expr = r.get("expression")
        if not expr:
            continue
        best = deduped.get(expr)
        if best is None or (r.get("is_fitness") or 0.0) > (best.get("is_fitness") or 0.0):
            deduped[expr] = r
    records = list(deduped.values())

    records.sort(key=lambda r: r.get("is_fitness") or 0.0, reverse=True)
    passing = sum(1 for r in records if r.get("grade") in {"GOOD", "EXCELLENT", "SPECTACULAR"})
    print(f"\nLoaded {len(records)} simulations | {passing} passing\n")

    print(f"Top {n} by fitness:")
    print(f"{'#':<4} {'Fitness':>7} {'Sharpe':>7} {'Grade':<12} {'Key':<6} {'Val':>6}  Expression")
    print("-" * 110)
    for i, r in enumerate(records[:n], 1):
        fitness = r.get("is_fitness") or 0.0
        sharpe  = r.get("is_sharpe") or 0.0
        grade   = r.get("grade") or "—"
        key     = r.get("sweep_key") or "base"
        val     = r.get("sweep_val")
        val_str = f"{val:g}" if val is not None else "—"
        expr    = (r.get("expression") or "").replace("\n", " ")[:55]
        print(f"{i:<4} {fitness:>7.3f} {sharpe:>7.3f} {grade:<12} {key:<6} {val_str:>6}  {expr}")

    print(f"\nShowing {min(n, len(records))} of {len(records)} results")


# ---------------------------------------------------------------------------
# Best-per-axis selection and submission
# ---------------------------------------------------------------------------

def _load_sweep_records(sim_logs: list[str]) -> list[dict]:
    """Load all completed records that came from this sweep (have a sweep_key field)."""
    records = []
    for path in sim_logs:
        p = Path(path)
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "done":
                        records.append(r)
                except json.JSONDecodeError:
                    pass
    # Deduplicate by (expression, neutralization, decay) — keep highest fitness
    deduped: dict[tuple, dict] = {}
    for r in records:
        so = r.get("settings_override") or {}
        key = (
            r.get("expression", ""),
            so.get("neutralization", BASELINE_SETTINGS["neutralization"]),
            so.get("decay", BASELINE_SETTINGS["decay"]),
        )
        best = deduped.get(key)
        if best is None or (r.get("is_fitness") or 0.0) > (best.get("is_fitness") or 0.0):
            deduped[key] = r
    return list(deduped.values())


def pick_best(sim_logs: list[str]) -> tuple[dict, dict] | None:
    """For each sweep axis, pick the value with highest is_fitness (fallback: is_sharpe).

    Returns (best_expr_params, best_settings_override) combining the per-axis winners,
    or None if insufficient data.
    """
    records = _load_sweep_records(sim_logs)
    if not records:
        print("No completed simulations found in logs.")
        return None

    def _fitness(r: dict) -> float:
        return r.get("is_fitness") or r.get("is_sharpe") or 0.0

    # Separate baseline record for fallback
    baseline_record = next((r for r in records if r.get("sweep_key") is None), None)

    # Group by sweep_key
    from collections import defaultdict
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        k = r.get("sweep_key")
        if k is not None:
            by_key[k].append(r)

    all_axes = list(EXPR_GRIDS.keys()) + list(SETTINGS_GRIDS.keys())

    best_expr   = dict(BASELINE)
    best_sett   = dict(BASELINE_SETTINGS)
    baseline_fit = _fitness(baseline_record) if baseline_record else 0.0

    print(f"\nBaseline fitness: {baseline_fit:.3f}")
    print(f"\n{'Axis':<16} {'Best val':>10} {'Fitness':>8} {'Sharpe':>8} {'Grade':<12}  {'vs baseline':>12}")
    print("-" * 72)

    for axis in all_axes:
        recs = by_key.get(axis, [])
        if not recs:
            print(f"  {axis:<14} {'(no data)':>10}")
            continue

        winner = max(recs, key=_fitness)
        val    = winner.get("sweep_val")
        fit    = winner.get("is_fitness") or 0.0
        sharpe = winner.get("is_sharpe") or 0.0
        grade  = winner.get("grade") or "—"
        delta  = fit - baseline_fit
        delta_str = f"{delta:+.3f}"

        # Only adopt winner if it beats baseline
        if fit > baseline_fit:
            marker = " <-- adopted"
            if axis in EXPR_GRIDS:
                best_expr[axis] = val
            else:
                best_sett[axis] = val
        else:
            marker = ""

        val_str = f"{val:g}" if isinstance(val, (int, float)) else str(val)
        print(f"  {axis:<14} {val_str:>10} {fit:>8.3f} {sharpe:>8.3f} {grade:<12}  {delta_str:>12}{marker}")

    best_override = {
        "delay":          best_sett["delay"],
        "neutralization": best_sett["neutralization"],
        "decay":          best_sett["decay"],
    }

    best_expr_clean = {k: best_expr[k] for k in BASELINE}
    print(f"\nBest expr params:     {best_expr_clean}")
    print(f"Best settings override: {best_override}")
    print(f"\nBest expression:")
    print(build_expression(best_expr_clean))

    return best_expr_clean, best_override


def build_best_plan(sim_logs: list[str], output_path: str) -> list[dict] | None:
    """Pick best params from sim_logs, write a 1-candidate plan, return it."""
    result = pick_best(sim_logs)
    if result is None:
        return None
    best_expr_params, best_override = result

    expr = build_expression(best_expr_params)
    cid  = hashlib.md5(
        (expr + best_override.get("neutralization", "") + str(best_override.get("decay", 0))).encode()
    ).hexdigest()[:16]

    candidate = {
        "candidate_id": cid,
        "template_id":  "best_combined",
        "dataset_id":   "model16",
        "category":     "model",
        "search_lane":  "const_sweep_best",
        "expression":   expr,
        "fields":       ["split"],
        "group":        "subindustry",
        "sweep_key":    None,
        "sweep_val":    None,
        "params":       best_expr_params,
        "settings_override": best_override,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(json.dumps(candidate) + "\n")

    print(f"\nWrote best-combined candidate → {output_path}")
    return [candidate]


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
    parser = argparse.ArgumentParser(description="Constant sweep for split ts_arg_max composite alpha (delay=0).")
    parser.add_argument("--mode", choices=["build", "submit", "run", "top", "best"], default="run")
    parser.add_argument("--plan", default="artifacts/split_argmax_sweep.jsonl")
    parser.add_argument("--sim-log", default="artifacts/split_argmax_sim.jsonl")
    parser.add_argument("--history-log", nargs="*", default=[], metavar="PATH",
                        help="Extra sim logs to skip already-tested expressions.")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="Limit total candidates (0 = unlimited)")
    parser.add_argument("--shard", default="1/1", metavar="N/TOTAL")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    shard_idx, shard_total = (int(x) for x in args.shard.split("/"))
    assert 1 <= shard_idx <= shard_total

    if args.mode == "top":
        all_logs = [args.sim_log] + (args.history_log or [])
        top_results(all_logs, n=args.n)
        return

    if args.mode == "best":
        all_logs = [args.sim_log] + (args.history_log or [])
        best_plan = Path(args.plan).with_stem(Path(args.plan).stem + "_best")
        candidates = build_best_plan(all_logs, str(best_plan))
        if not candidates:
            return
        print(f"\nSubmitting best-combined alpha ...")
        from core.brain.submit import run
        run(str(best_plan), args.sim_log, batch_size=1, credentials=args.credentials)
        _merge_sim_log(args.sim_log)
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

        from core.brain.submit import run
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
