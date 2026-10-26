"""Targeted sweep to maximize fitness on the cash inverse signal family.

Anchor signal:  group_backfill(-inverse(cash), sector, 20)
  Sharpe 1.52 | Fitness 2.45 | CONCENTRATED_WEIGHT FAIL (13.45% > 10%)

Known fixes so far:
  winsorize wrapper   → Fitness 2.51 | passes
  normalize(winsorize) → Fitness 2.41 | passes (lower fitness, higher Sharpe)
  rank wrapper         → Fitness 0.29 | kills signal

Goal: find combinations that pass CONCENTRATED_WEIGHT AND maximize fitness.

Sweep dimensions:
  1. Pre-processing before inverse: winsorize, zscore, ts_backfill, log, scale
  2. Post-processing after inverse: winsorize, scale, normalize, zscore, none
  3. Outer wrapper around group_backfill: none, normalize, zscore, rank, group_neutralize
  4. Backfill group: sector, industry, subindustry, market
  5. Backfill window: 5, 10, 20, 40, 60, 120

Usage:
    python -m tools.cash_sweep
    python -m tools.cash_sweep --dry-run          # print candidates, don't submit
    python -m tools.cash_sweep --top 20           # submit only top 20 (by priority)
    python -m tools.cash_sweep --sim-log <path>   # custom output log
    python -m tools.cash_sweep --credentials brain.txt
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Simulation settings (match what the grammar pipeline uses)
# ---------------------------------------------------------------------------

SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.01,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}

SIM_LOG_DEFAULT = str(ROOT / "artifacts" / "cash_sweep_sim.jsonl")
CREDENTIALS_DEFAULT = str(ROOT / "brain.txt")


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

# Core signal: the proven signal family. Only vary the structural wrappers,
# not the underlying field formula (adding log/rank variants dilutes focus).
# -inverse(cash) = -(1/cash): higher rank = less cash relative to book/size.
CORE_SIGNALS = [
    "-inverse(cash)",
    "-log(cash)",             # log version — no division singularity risk
]

# Pre-processing applied to the core signal INSIDE group_backfill.
# Winsorize is proven to fix concentration; zscore(winsorize) is complementary.
PRE_OPS = [
    ("raw",        "{core}"),
    ("win",        "winsorize({core})"),
    ("win_zscore", "zscore(winsorize({core}))"),
]

# Group used for the group_backfill.
# sector is proven; industry/subindustry may improve sub-universe sharpe.
GROUPS = ["sector", "industry", "subindustry"]

# Backfill window: 20 is proven. Explore shorter (5, 10) and longer (40, 60).
BF_WINDOWS = [10, 20, 40]

# Outer wrapper: applied after group_backfill.
# Controls portfolio-level concentration and normalization.
OUTER_OPS = [
    ("none",      "{inner}"),
    ("normalize", "normalize({inner})"),
    ("zscore",    "zscore({inner})"),
    ("grank_sec", "group_rank({inner}, sector)"),
    ("grank_ind", "group_rank({inner}, industry)"),
    ("gneu_sec",  "group_neutralize({inner}, sector)"),
    ("win_norm",  "normalize(winsorize({inner}))"),
]

# Priority weights: proven combos first, known-bad last.
# Higher = submitted first.
PRIORITY_OVERRIDES: dict[tuple, float] = {
    # (pre_label, outer_label, group, window)
    ("win",        "none",      "sector", 20): 3.0,   # known: Fitness 2.51 ✓
    ("win",        "normalize", "sector", 20): 2.5,   # known: Fitness 2.41 ✓
    # Nearby variants of the proven winner — high priority
    ("win",        "none",      "sector", 40): 2.0,
    ("win",        "none",      "sector", 10): 2.0,
    ("win",        "none",      "industry", 20): 2.0,
    ("win",        "normalize", "sector", 40): 1.8,
    ("win",        "gneu_sec",  "sector", 20): 1.8,
    ("win_zscore", "none",      "sector", 20): 1.8,
    # Known failures: deprioritize but allow --include-bad to run them
    ("raw",        "none",      "sector", 20): 0.1,   # known: CONCENTRATED_WEIGHT FAIL
    ("raw",        "rank",      "sector", 20): 0.1,   # known: Fitness 0.29
}


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _candidate_id(expr: str) -> str:
    return hashlib.sha1(expr.encode()).hexdigest()[:16]


def build_candidates() -> list[dict]:
    """Expand all template combinations into candidate dicts."""
    candidates = []
    seen: set[str] = set()

    for core_sig in CORE_SIGNALS:
        for pre_label, pre_tmpl in PRE_OPS:
            pre_expr = pre_tmpl.replace("{core}", core_sig)
            for group in GROUPS:
                for window in BF_WINDOWS:
                    inner = f"group_backfill({pre_expr}, {group}, {window})"
                    for outer_label, outer_tmpl in OUTER_OPS:
                        expr = outer_tmpl.replace("{inner}", inner)
                        if expr in seen:
                            continue
                        seen.add(expr)
                        key = (pre_label, outer_label, group, window)
                        priority = PRIORITY_OVERRIDES.get(key, 1.0)
                        candidates.append({
                            "candidate_id": _candidate_id(expr),
                            "expression": expr,
                            "search_lane": "cash_sweep",
                            "template_id": f"cash_sweep:{pre_label}:{outer_label}:{group}:{window}",
                            "priority_score": priority,
                            "params": {
                                "core_signal": core_sig,
                                "pre_op": pre_label,
                                "outer_op": outer_label,
                                "group": group,
                                "window": window,
                            },
                        })

    # Sort: skip already-known bad ones (priority 0.1), prioritize promising first
    candidates.sort(key=lambda c: c["priority_score"], reverse=True)
    return candidates


def filter_remaining(candidates: list[dict], sim_log: str) -> list[dict]:
    """Skip candidates already logged in sim_log."""
    done_ids: set[str] = set()
    done_exprs: set[str] = set()
    path = Path(sim_log)
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    cid = r.get("candidate_id")
                    expr = r.get("expression")
                    if cid:
                        done_ids.add(cid)
                    if expr:
                        done_exprs.add(expr)
                except json.JSONDecodeError:
                    pass
    return [
        c for c in candidates
        if c["candidate_id"] not in done_ids and c["expression"] not in done_exprs
    ]


# ---------------------------------------------------------------------------
# Submit helpers (thin wrappers over run_pipeline internals)
# ---------------------------------------------------------------------------

def _append_jsonl(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def _merge_sim_log(path: str) -> None:
    try:
        from wq_alpha.logs import MASTER, merge_session_to_master

        n_new, n_updated = merge_session_to_master(path)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    except Exception as exc:
        print(f"[warn] Could not merge {path} to master: {exc}")


def _parse_is_checks(is_checks: list) -> dict:
    return {c["name"]: c.get("result", "FAIL") == "PASS" for c in (is_checks or [])}


def _parse_check_values(is_checks: list) -> dict:
    return {
        c["name"]: float(c["value"]) if c.get("value") is not None else None
        for c in (is_checks or [])
    }


def build_sim_record(candidate: dict, result_detail: dict, alpha_id: str | None,
                     status: str, yearly: dict) -> dict:
    is_stats = result_detail.get("is") or {}
    os_stats = result_detail.get("os") or {}
    raw_checks = is_stats.get("checks") or []
    checks = _parse_is_checks(raw_checks)
    check_vals = _parse_check_values(raw_checks)
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate.get("candidate_id"),
        "search_lane": candidate.get("search_lane", "cash_sweep"),
        "template_id": candidate.get("template_id", ""),
        "expression": candidate.get("expression"),
        "params": candidate.get("params", {}),
        "priority_score": candidate.get("priority_score", 1.0),
        "alpha_id": alpha_id,
        "status": status,
        "is_sharpe": is_stats.get("sharpe"),
        "is_fitness": is_stats.get("fitness"),
        "is_turnover": is_stats.get("turnover"),
        "is_returns": is_stats.get("returns"),
        "is_drawdown": is_stats.get("drawdown"),
        "is_concentrated_weight_ok": checks.get("CONCENTRATED_WEIGHT", True),
        "is_sub_universe_sharpe_ok": checks.get("LOW_SUB_UNIVERSE_SHARPE", True),
        "max_concentration": check_vals.get("CONCENTRATED_WEIGHT"),
        "max_self_correlation": check_vals.get("SELF_CORRELATION"),
        "grade": result_detail.get("grade"),
        "checks": raw_checks,
        "yearly_stats": yearly or {},
        "oos_sharpe_2023": (yearly or {}).get(2023, {}).get("sharpe"),
        "oos_fitness_2023": (yearly or {}).get(2023, {}).get("fitness"),
        "raw_alpha_detail": result_detail,
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

HEADER = (
    f"  {'#':<4} {'sharpe':>7} {'fitness':>7} {'23fit':>7} {'conc%':>7} "
    f"{'conc_ok':<8} {'group':<12} {'win':>4} {'pre':<12} {'outer':<10}  expression"
)
DIVIDER = "  " + "─" * 130


def print_results(records: list[dict]) -> None:
    """Print sorted results table."""
    done = [r for r in records if r.get("status") == "done" and r.get("is_sharpe") is not None]
    done.sort(key=lambda r: (
        1 if r.get("is_concentrated_weight_ok", False) else 0,
        r.get("is_fitness") or 0.0,
    ), reverse=True)

    print(f"\n{'=' * 80}")
    print(f"  CASH SWEEP RESULTS  ({len(done)} completed)")
    print(HEADER)
    print(DIVIDER)

    for i, r in enumerate(done, 1):
        sharpe = r.get("is_sharpe") or 0.0
        fitness = r.get("is_fitness") or 0.0
        f23 = r.get("oos_fitness_2023")
        f23_str = f"{f23:.2f}" if f23 is not None else " N/A"
        conc = r.get("max_concentration")
        conc_str = f"{conc*100:.1f}" if conc is not None else "  N/A"
        conc_ok = "PASS" if r.get("is_concentrated_weight_ok") else "FAIL"
        p = r.get("params") or {}
        group = p.get("group", "?")
        window = p.get("window", "?")
        pre = p.get("pre_op", "?")
        outer = p.get("outer_op", "?")
        expr = (r.get("expression") or "")[:55]
        print(
            f"  {i:<4} {sharpe:>7.3f} {fitness:>7.3f} {f23_str:>7} {conc_str:>7} "
            f"{conc_ok:<8} {group:<12} {window:>4} {pre:<12} {outer:<10}  {expr}"
        )

    print(DIVIDER)
    # Best passing
    passing = [r for r in done if r.get("is_concentrated_weight_ok")]
    if passing:
        best = max(passing, key=lambda r: r.get("is_fitness") or 0.0)
        print(f"\n  BEST PASSING  fitness={best.get('is_fitness'):.3f} "
              f"sharpe={best.get('is_sharpe'):.3f}:")
        print(f"  {best.get('expression')}")
    print()


# ---------------------------------------------------------------------------
# Main submission loop
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> None:
    from wq_alpha.brain.api import sign_in, fetch_yearly_stats_with_retry
    from alpha_grammar.run_pipeline import (
        submit_alpha, wait_for_completion, get_alpha_detail,
    )

    Path(args.sim_log).parent.mkdir(parents=True, exist_ok=True)

    all_candidates = build_candidates()
    remaining = filter_remaining(all_candidates, args.sim_log)

    # Exclude known-bad (priority 0.1) unless --include-bad flag
    if not args.include_bad:
        remaining = [c for c in remaining if c["priority_score"] > 0.1]

    if args.top > 0:
        remaining = remaining[:args.top]

    print(f"Total templates:   {len(all_candidates)}")
    print(f"Already logged:    {len(all_candidates) - len(filter_remaining(all_candidates, args.sim_log))}")
    print(f"To submit:         {len(remaining)}")
    print(f"Sim log:           {args.sim_log}")

    if args.dry_run:
        print("\n[DRY RUN] Candidates (first 30):")
        for c in remaining[:30]:
            p = c["params"]
            print(f"  pri={c['priority_score']:.1f}  {c['expression'][:80]}")
        return

    if not remaining:
        print("Nothing new to submit.")
        _load_and_print(args.sim_log)
        return

    session = sign_in(args.credentials)

    batch_size = args.batch_size
    total = len(remaining)
    logged = 0

    def _simulate_one(c: dict) -> dict:
        expr = c["expression"]
        print(f"[SWEEP]   Submit: {expr[:70]}")
        url = submit_alpha(
            session, expr, SETTINGS,
            max_retries=5, sleep_seconds=30,
            credentials_path=args.credentials,
        )
        if not url:
            print(f"[SWEEP]   FAILED to submit: {expr[:60]}")
            return build_sim_record(c, {}, None, "submit_failed", {})

        print(f"[SWEEP]   Wait:   {expr[:70]}")
        final = wait_for_completion(session, url, sleep_seconds=30,
                                    credentials_path=args.credentials)
        alpha_id = final.get("alpha")
        if alpha_id:
            detail = get_alpha_detail(session, alpha_id)
            yearly = fetch_yearly_stats_with_retry(session, alpha_id)
            status = "done"
        else:
            detail, yearly, status = {}, {}, "no_alpha_id"
        return build_sim_record(c, detail, alpha_id, status, yearly)

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_simulate_one, c): c for c in remaining}
        for future in as_completed(futures):
            record = future.result()
            _append_jsonl(args.sim_log, record)
            logged += 1

            sharpe = record.get("is_sharpe")
            fitness = record.get("is_fitness")
            conc_ok = record.get("is_concentrated_weight_ok")
            f23 = record.get("oos_fitness_2023")
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
            fitness_str = f"{fitness:.3f}" if fitness is not None else "N/A"
            f23_str = f"{f23:.2f}" if f23 is not None else "N/A"
            conc_str = "PASS" if conc_ok else "FAIL"
            print(f"[SWEEP]   Done [{logged}/{total}]: sharpe={sharpe_str} fitness={fitness_str} 23fit={f23_str} conc={conc_str}")

    print(f"\nDone. {logged} records logged to {args.sim_log}")
    _merge_sim_log(args.sim_log)
    _load_and_print(args.sim_log)


def _load_and_print(sim_log: str) -> None:
    records = []
    path = Path(sim_log)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print_results(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep operator combinations on the cash inverse signal family."
    )
    p.add_argument("--sim-log", default=SIM_LOG_DEFAULT,
                   help=f"Output sim log (default: {SIM_LOG_DEFAULT})")
    p.add_argument("--credentials", default=CREDENTIALS_DEFAULT)
    p.add_argument("--top", type=int, default=0,
                   help="Submit only top N candidates by priority (0 = all)")
    p.add_argument("--batch-size", type=int, default=3,
                   help="Candidates per Brain API batch (default 3)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print candidates without submitting")
    p.add_argument("--include-bad", action="store_true",
                   help="Include already-known-failing variants (priority 0.1)")
    p.add_argument("--results", action="store_true",
                   help="Just print results from existing sim log, no submission")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    if args.results:
        _load_and_print(args.sim_log)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
