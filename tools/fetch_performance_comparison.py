"""Fetch Performance Comparison for a simulated alpha.

Replicates the Brain portal's "Performance Comparison" panel:

    Delay-1 Score   Before Submission   After Submission   Change
                         10,327              10,056           -271

Usage:
    # Show current portfolio scores + check status for an alpha
    python -m tools.fetch_performance_comparison --alpha-id <ALPHA_ID>

    # Specify competition (default: challenge)
    python -m tools.fetch_performance_comparison --alpha-id <ALPHA_ID> --competition IQC2026S1

    # Compare multiple alphas
    python -m tools.fetch_performance_comparison --alpha-ids ID1 ID2 ID3 --competition IQC2026S1

    # Compare multiple alphas at once (e.g. from sim log)
    python -m tools.fetch_performance_comparison --sim-log alpha_discovery/grammar_method/artifacts/all_sim.jsonl --top 10

    # Only show alphas that are already submitted (have "After" score)
    python -m tools.fetch_performance_comparison --sim-log ... --submitted-only

How Brain computes the scores:
    "Before Submission" = your current Delay-1 Score in the competition leaderboard.
    "After Submission"  = Brain's projected score after adding this alpha to your portfolio.
                         The Brain portal exposes it for eligible unsubmitted alphas too,
                         but the endpoint may return an empty response while it computes.
    "Change"            = After - Before.

This script fetches:
    1. Current leaderboard score for each competition (Before).
    2. Alpha's competition-level performance data (Before/After), matching the portal.
    3. IS check results and self-correlation data from /check.
    4. Yearly stats (2023 OOS fitness) from /recordsets/yearly-stats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


BASE_URL = "https://api.worldquantbrain.com"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(session, url: str, timeout: int = 15) -> dict:
    resp = session.get(url, timeout=timeout)
    if resp.status_code == 200 and resp.text.strip():
        return resp.json()
    return {}


def fetch_competition_leaderboard(session, comp_id: str) -> dict:
    """Return {'score': float, 'rank': int, 'level': str, 'alphas': int} or {}."""
    data = _get(session, f"{BASE_URL}/competitions/{comp_id}")
    return data.get("leaderboard") or {}


def fetch_alpha_detail(session, alpha_id: str) -> dict:
    return _get(session, f"{BASE_URL}/alphas/{alpha_id}")


def fetch_alpha_check(session, alpha_id: str) -> dict:
    """Return the /check response: checks list + selfCorrelated table."""
    return _get(session, f"{BASE_URL}/alphas/{alpha_id}/check")


def fetch_yearly_stats(session, alpha_id: str) -> dict:
    """Return {year_str: {sharpe, fitness, ...}} or {}."""
    data = _get(session, f"{BASE_URL}/alphas/{alpha_id}/recordsets/yearly-stats")
    if not data:
        return {}
    schema = data.get("schema", {})
    props = schema.get("properties", [])
    cols = [p["name"] for p in props]
    result = {}
    for row in data.get("records") or []:
        if not row:
            continue
        year = str(int(row[0]))
        result[year] = dict(zip(cols[1:], row[1:]))
    return result


def fetch_competition_alpha_performance(session, alpha_id: str, comp_id: str) -> dict | None:
    """Fetch the portal's official before/after score projection."""
    from core.portfolio.admission import fetch_brain_performance_comparison

    data = fetch_brain_performance_comparison(
        session,
        alpha_id,
        comp_id,
        max_attempts=5,
        sleep_seconds=2.0,
        timeout=20.0,
    )
    if not data:
        return None

    score = data.get("score") or {}
    before = score.get("before")
    after = score.get("after")
    if before is None or after is None:
        return None

    stats = data.get("stats") or {}
    before_stats = stats.get("before") or {}
    after_stats = stats.get("after") or {}
    before_sharpe = before_stats.get("sharpe")
    after_sharpe = after_stats.get("sharpe")
    return {
        "score_before": before,
        "score_after": after,
        "change": after - before,
        "pool_sharpe_before": before_sharpe,
        "pool_sharpe_after": after_sharpe,
        "pool_sharpe_change": (
            after_sharpe - before_sharpe
            if before_sharpe is not None and after_sharpe is not None
            else None
        ),
        "_source": f"{BASE_URL}/competitions/{comp_id}/alphas/{alpha_id}/before-and-after-performance",
    }

    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_score(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:,.0f}"


def _fmt_change(v: float | None) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.0f}"


def _check_symbol(result: str | None) -> str:
    return "PASS" if result == "PASS" else "FAIL"


# ---------------------------------------------------------------------------
# Core display
# ---------------------------------------------------------------------------

def show_performance_comparison(
    session,
    alpha_id: str,
    comp_ids: list[str],
    verbose: bool = False,
) -> None:
    """Print the performance comparison block for one alpha."""

    print(f"\n{'=' * 72}")
    print(f"  Alpha: {alpha_id}")
    print(f"{'=' * 72}")

    # Alpha detail
    detail = fetch_alpha_detail(session, alpha_id)
    if not detail:
        print("  ERROR: Alpha not found or API error.")
        return

    expr = (detail.get("regular") or {}).get("code", "")
    grade = detail.get("grade", "?")
    stage = detail.get("stage", "?")
    status = detail.get("status", "?")
    is_stats = detail.get("is") or {}
    sharpe = is_stats.get("sharpe")
    fitness = is_stats.get("fitness")
    turnover = is_stats.get("turnover")

    print(f"  Expression : {expr[:80]}")
    print(f"  Stage      : {stage}   Grade: {grade}   Status: {status}")
    if sharpe is not None:
        print(f"  IS Metrics : Sharpe={sharpe:.3f}  Fitness={fitness:.3f}  Turnover={turnover:.4f}")

    # IS checks
    check_data = fetch_alpha_check(session, alpha_id)
    checks = (check_data.get("is") or {}).get("checks") or []
    if checks:
        failing = [c["name"] for c in checks if c.get("result") != "PASS"
                   and c["name"] != "SELF_CORRELATION"]
        self_corr = next((c for c in checks if c["name"] == "SELF_CORRELATION"), None)
        sc_val = self_corr.get("value") if self_corr else None
        sc_str = f"{sc_val:.4f}" if sc_val is not None else "N/A"
        print(f"  IS Checks  : {'ALL PASS' if not failing else 'FAIL: ' + ', '.join(failing)}")
        print(f"  Self-Corr  : {sc_str}  (limit: {self_corr.get('limit', 'N/A') if self_corr else 'N/A'})")

        # Self-correlated alpha detail
        sc_table = (check_data.get("is") or {}).get("selfCorrelated", {})
        sc_records = sc_table.get("records") or []
        if sc_records and verbose:
            sc_cols = [p["name"] for p in (sc_table.get("schema") or {}).get("properties", [])]
            print("  Correlated with:")
            for row in sc_records:
                row_dict = dict(zip(sc_cols, row))
                print(f"    alpha={row_dict.get('id')}  corr={row_dict.get('correlation'):.4f}"
                      f"  sharpe={row_dict.get('sharpe')}  fitness={row_dict.get('fitness')}")

    # OOS yearly stats (2023 fitness)
    yearly = fetch_yearly_stats(session, alpha_id)
    if yearly:
        row_2023 = yearly.get("2023") or yearly.get(2023)
        if row_2023:
            s23 = row_2023.get("sharpe")
            f23 = row_2023.get("fitness")
            t23 = row_2023.get("turnover")
            print(f"  2023 OOS   : Sharpe={s23}  Fitness={f23}  Turnover={t23}")
    else:
        print("  2023 OOS   : not yet computed (run backfill_yearly_stats.py)")

    # Performance comparison per competition
    print()
    print(f"  {'Competition':<24}  {'Before':>10}  {'After':>10}  {'Change':>10}  "
          f"{'Rank':>6}  {'Level':<8}")
    print(f"  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*8}")

    any_perf = False
    for comp_id in comp_ids:
        lb = fetch_competition_leaderboard(session, comp_id)
        score_before = lb.get("score")
        rank = lb.get("rank")
        level = lb.get("level", "")

        # Try to get the projected "after" score
        perf = fetch_competition_alpha_performance(session, alpha_id, comp_id)
        if perf:
            any_perf = True
            score_after = perf["score_after"]
            change = perf["change"]
            score_before = perf["score_before"]
        else:
            score_after = None
            change = None

        before_str = _fmt_score(score_before)
        after_str = _fmt_score(score_after) if score_after is not None else "pending"
        change_str = _fmt_change(change) if change is not None else "pending"
        rank_str = str(rank) if rank is not None else "N/A"

        print(f"  {comp_id:<24}  {before_str:>10}  {after_str:>10}  {change_str:>10}  "
              f"{rank_str:>6}  {level:<8}")

    if not any_perf:
        print()
        print("  NOTE: Brain did not return Performance Comparison data after polling.")
        print("        The portal may still be computing it; retry shortly.")


# ---------------------------------------------------------------------------
# Sim log batch mode
# ---------------------------------------------------------------------------

def show_from_sim_log(
    session,
    sim_log: str,
    comp_ids: list[str],
    top_n: int = 10,
    submitted_only: bool = False,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
) -> None:
    """Batch mode: read sim log, find top passing alphas, show comparison."""
    path = Path(sim_log)
    if not path.exists():
        print(f"ERROR: sim log not found: {sim_log}")
        sys.exit(1)

    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "done":
                continue
            if not r.get("alpha_id"):
                continue
            sharpe = r.get("is_sharpe") or 0.0
            fitness = r.get("is_fitness") or 0.0
            if sharpe < min_sharpe or fitness < min_fitness:
                continue
            # Only include if all checks passed
            checks = r.get("checks") or []
            raw = r.get("raw_alpha_detail") or {}
            if not checks:
                checks = (raw.get("is") or {}).get("checks") or []
            failing = [c["name"] for c in checks
                       if c.get("result") != "PASS" and c["name"] != "SELF_CORRELATION"]
            if failing:
                continue
            records.append(r)

    records.sort(key=lambda r: r.get("is_sharpe") or 0.0, reverse=True)
    records = records[:top_n]

    print(f"Loaded {len(records)} passing alpha(s) from {sim_log}")
    print(f"Competitions: {comp_ids}")

    for r in records:
        show_performance_comparison(session, r["alpha_id"], comp_ids, verbose=False)
        time.sleep(0.5)  # avoid hammering Brain API


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Brain Performance Comparison (Before/After Delay-1 Score)."
    )
    p.add_argument("--alpha-id", default="", help="Specific alpha ID to check")
    p.add_argument("--alpha-ids", nargs="+", default=[], help="One or more alpha IDs to check")
    p.add_argument(
        "--competition",
        default="challenge",
        help="Competition ID(s), comma-separated (default: challenge). "
             "Use 'all' to check both challenge and IQC2026S1.",
    )
    p.add_argument(
        "--sim-log",
        default="",
        help="Path to sim JSONL log — batch mode: show top passing alphas",
    )
    p.add_argument("--top", type=int, default=10, help="Top N alphas from sim log (default 10)")
    p.add_argument("--min-sharpe", type=float, default=1.25)
    p.add_argument("--min-fitness", type=float, default=1.0)
    p.add_argument("--submitted-only", action="store_true",
                   help="Only show alphas that already have an After score")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show self-correlated alpha details")
    p.add_argument("--credentials", default="brain.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from core.brain.api import sign_in
    session = sign_in(args.credentials)

    # Resolve competition IDs
    if args.competition == "all":
        comp_ids = ["challenge", "IQC2026S1"]
    else:
        comp_ids = [c.strip() for c in args.competition.split(",") if c.strip()]

    alpha_ids = list(args.alpha_ids)
    if args.alpha_id:
        alpha_ids.insert(0, args.alpha_id)

    if alpha_ids:
        for alpha_id in alpha_ids:
            show_performance_comparison(session, alpha_id, comp_ids, verbose=args.verbose)
            time.sleep(0.2)
    elif args.sim_log:
        show_from_sim_log(
            session,
            args.sim_log,
            comp_ids,
            top_n=args.top,
            submitted_only=args.submitted_only,
            min_sharpe=args.min_sharpe,
            min_fitness=args.min_fitness,
        )
    else:
        # Default: show current portfolio scores for all competitions
        print("\nCurrent Portfolio Delay-1 Scores")
        print("=" * 50)
        for comp_id in comp_ids:
            lb = fetch_competition_leaderboard(session, comp_id)
            if lb:
                print(f"  {comp_id:<24}  score={_fmt_score(lb.get('score'))}  "
                      f"rank={lb.get('rank')}  level={lb.get('level')}  "
                      f"alphas={lb.get('alphas')}")
            else:
                print(f"  {comp_id:<24}  not enrolled or no data")
        print("\nUse --alpha-id <ID> to check a specific alpha's performance comparison.")
        print("Use --sim-log <path> for batch mode over a sim log.")


if __name__ == "__main__":
    main()
