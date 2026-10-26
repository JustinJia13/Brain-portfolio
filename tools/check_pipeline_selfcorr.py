"""Check whether the pipeline's PnL-based self-correlation is working.

The pipeline computes self-correlation itself (run_pipeline.py:548) by:
  1. Fetching PnL series for each competition portfolio alpha
  2. Fetching PnL series for each new simulation
  3. Computing max Pearson correlation (daily returns) vs the portfolio

This script verifies each step independently so you can see where it
succeeds or fails without running a full pipeline round.

Usage:
    python -m tools.check_pipeline_selfcorr
    python -m tools.check_pipeline_selfcorr --alpha-id XXXXXXXXX
    python -m tools.check_pipeline_selfcorr --n 3 --lane grammar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BRAIN_BASE = "https://api.worldquantbrain.com"
MASTER = Path("artifacts/sim/master.jsonl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--competition-id", default="IQC2026S1")
    p.add_argument("--sim-log", default=str(MASTER))
    p.add_argument("--lane", default="grammar")
    p.add_argument("--n", type=int, default=3, help="Number of unsubmitted alphas to test")
    p.add_argument("--alpha-id", default=None, help="Test a specific alpha ID")
    return p.parse_args()


def _load_recent_alphas(sim_log: str, n: int, lane: str) -> list[dict]:
    results: list[dict] = []
    p = Path(sim_log)
    if not p.exists():
        return results
    lines: list[str] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    # Walk from newest (end of file)
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") != "done":
            continue
        if not rec.get("alpha_id"):
            continue
        if rec.get("search_lane") != lane:
            continue
        results.append(rec)
        if len(results) >= n:
            break
    return results


def _pnl_to_returns(pnl: dict[str, float]) -> dict[str, float]:
    keys = sorted(pnl)
    if len(keys) < 2:
        return {}
    return {keys[i]: pnl[keys[i]] - pnl[keys[i - 1]] for i in range(1, len(keys))}


def _compute_max_corr(candidate_pnl: dict, portfolio_pnls: dict) -> float | None:
    try:
        import numpy as np
    except ImportError:
        print("  ERROR: numpy not available")
        return None
    cand_ret = _pnl_to_returns(candidate_pnl)
    if not cand_ret:
        return None
    cand_dates = set(cand_ret)
    max_corr: float | None = None
    for port_pnl in portfolio_pnls.values():
        port_ret = _pnl_to_returns(port_pnl)
        shared = sorted(cand_dates & set(port_ret))
        if len(shared) < 20:
            continue
        a = [cand_ret[d] for d in shared]
        b = [port_ret[d] for d in shared]
        import numpy as np
        av, bv = np.array(a, dtype=float), np.array(b, dtype=float)
        if av.std() < 1e-12 or bv.std() < 1e-12:
            continue
        corr = float(np.corrcoef(av, bv)[0, 1])
        if max_corr is None or corr > max_corr:
            max_corr = corr
    return max_corr


def main() -> None:
    args = parse_args()

    from wq_alpha.brain.api import sign_in, fetch_pnl_series, fetch_competition_alpha_ids
    session = sign_in(args.credentials)

    # ── Step 1: portfolio PnL ─────────────────────────────────────────────
    print(f"Step 1: Fetching competition alpha IDs ({args.competition_id}) ...")
    alpha_ids = fetch_competition_alpha_ids(session, args.competition_id)
    print(f"  Found {len(alpha_ids)} competition alphas.")
    if not alpha_ids:
        print("  FAIL: no competition alphas found — check competition ID or credentials.")
        return

    print(f"  Fetching PnL for up to {min(5, len(alpha_ids))} portfolio alphas (sample) ...")
    portfolio_pnls: dict[str, dict[str, float]] = {}
    for aid in alpha_ids:
        pnl = fetch_pnl_series(session, aid)
        if pnl is None:
            print(f"  RATE LIMITED on {aid}")
            continue
        if pnl:
            portfolio_pnls[aid] = pnl
        if len(portfolio_pnls) >= 5:
            break

    if not portfolio_pnls:
        print("  FAIL: could not fetch any portfolio PnL series.")
        return

    sample_aid, sample_pnl = next(iter(portfolio_pnls.items()))
    print(f"  OK: loaded {len(portfolio_pnls)} portfolio PnL series.")
    print(f"  Sample ({sample_aid}): {len(sample_pnl)} dates, "
          f"first={min(sample_pnl)}, last={max(sample_pnl)}")

    # ── Step 2: candidate PnL ─────────────────────────────────────────────
    if args.alpha_id:
        records = [{"alpha_id": args.alpha_id, "expression": "(specified)", "is_sharpe": "?",
                    "search_lane": args.lane}]
    else:
        records = _load_recent_alphas(args.sim_log, args.n, args.lane)

    if not records:
        print(f"\nStep 2: No {args.lane} alphas found in {args.sim_log}.")
        return

    print(f"\nStep 2: Fetching candidate PnL + computing self-corr for {len(records)} alpha(s) ...")
    for rec in records:
        alpha_id = rec.get("alpha_id") or rec.get("id", "")
        expr = str(rec.get("expression", ""))[:70]
        stored_corr = rec.get("max_self_correlation")
        print(f"\n  === {alpha_id}  sharpe={rec.get('is_sharpe')}  {expr} ===")
        print(f"  stored max_self_correlation = {stored_corr}")

        pnl = fetch_pnl_series(session, alpha_id)
        if pnl is None:
            print("  PnL fetch: RATE LIMITED")
            continue
        if not pnl:
            print("  PnL fetch: EMPTY (no data returned)")
            continue
        print(f"  PnL fetch: OK — {len(pnl)} dates, "
              f"first={min(pnl)}, last={max(pnl)}")

        # Full portfolio for actual correlation
        print(f"  Loading full portfolio PnL ({len(alpha_ids)} alphas) ...")
        full_portfolio: dict[str, dict[str, float]] = {}
        for aid in alpha_ids:
            p = fetch_pnl_series(session, aid)
            if p:
                full_portfolio[aid] = p
        print(f"  Portfolio loaded: {len(full_portfolio)}/{len(alpha_ids)} with data.")

        computed = _compute_max_corr(pnl, full_portfolio)
        if computed is None:
            print("  Computed self-corr: None (insufficient overlap with portfolio)")
        else:
            flag = "  *** HIGH ***" if computed >= 0.70 else ""
            print(f"  Computed self-corr: {computed:.4f}{flag}")
            if stored_corr is not None:
                diff = abs(computed - float(stored_corr))
                match = "OK" if diff < 0.01 else f"MISMATCH (diff={diff:.4f})"
                print(f"  vs stored {float(stored_corr):.4f}: {match}")
            else:
                print("  (stored was None — pipeline did not compute it at sim time)")

    print("\n=== Summary ===")
    print("If computed self-corr is a float above: the pipeline CAN compute correlation.")
    print("If stored=None and computed=float: pipeline was run with --skip-portfolio-pnl")
    print("  or portfolio_pnls was empty at runtime.")


if __name__ == "__main__":
    main()
