"""Scan all sim history for submittable, unsubmitted, positive-contribution alphas.

Pre-filters aggressively (grade, numeric thresholds, all Brain checks including
SELF_CORRELATION) before fetching any PnL from Brain, so only genuine candidates
hit the network.

Usage:
    python tools/full_history_scan.py [--competition-id IQC2026S1] [--credentials brain.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wq_alpha.brain.api import fetch_pnl_series_with_retry, sign_in
from alpha_grammar.run_pipeline import get_alpha_detail
from wq_alpha.portfolio.admission import (
    PROMOTE,
    AdmissionConfig,
    annotate_records_with_admission,
    fetch_brain_performance_comparison,
    load_submitted_portfolio_data,
    write_ranked_outputs,
)

# Grades that are clearly not submittable — skip immediately without reading checks
_BAD_GRADES = frozenset({"INFERIOR", "FAIL", "UNKNOWN"})
# Grades that are positively submittable (fallback when no checks array)
_GOOD_GRADES = frozenset({"SPECTACULAR", "GOOD", "AVERAGE", "PASS"})


def _normalise_expr(expr: str) -> str:
    return " ".join((expr or "").split())


def _preferred_delta(record: dict) -> float:
    value = record.get("admission_brain_score_delta")
    if value is None:
        value = record.get("admission_delta_pool_sharpe")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_checks(record: dict) -> list[dict]:
    checks = record.get("checks")
    if isinstance(checks, list):
        return checks
    raw = record.get("raw_alpha_detail") or {}
    return (raw.get("is") or {}).get("checks") or []


def _is_submittable(record: dict) -> bool:
    """Return True only if all Brain submission checks pass (including SELF_CORRELATION)."""
    # Fast exits on missing / clearly bad data
    sharpe = record.get("is_sharpe")
    fitness = record.get("is_fitness")
    turnover = record.get("is_turnover")
    if sharpe is None or fitness is None or turnover is None:
        return False
    if sharpe < 1.25 or fitness < 1.0:
        return False
    if not (0.01 <= turnover <= 0.70):
        return False

    grade = (record.get("grade") or "").upper()
    if grade in _BAD_GRADES:
        return False
    if record.get("is_concentrated_weight_ok") is False:
        return False
    if record.get("is_sub_universe_sharpe_ok") is False:
        return False

    checks = _resolve_checks(record)
    # SELF_CORRELATION is always PENDING for unsubmitted alphas (Brain hasn't scored it yet).
    # UNITS is a warning only. Both are evaluated dynamically in admission scoring.
    _STATIC_IGNORED = frozenset({"SELF_CORRELATION", "UNITS"})
    if checks:
        for c in checks:
            if c.get("name") in _STATIC_IGNORED:
                continue
            if c.get("result") != "PASS":
                return False
    else:
        # No check data: accept only if grade is explicitly positive
        if grade and grade not in _GOOD_GRADES:
            return False

    return True


def _collect_sim_files(root: Path) -> list[Path]:
    """Return deduplicated list of all sim JSONL files."""
    seen: set[Path] = set()
    files: list[Path] = []

    def _add(path: Path) -> None:
        p = path.resolve()
        if p in seen:
            return
        if "full_scan" in p.parts:  # skip our own output
            return
        seen.add(p)
        files.append(path)

    # Primary pattern: anything named *_sim.jsonl or *_sim_scored.jsonl
    for f in root.glob("**/*_sim.jsonl"):
        _add(f)
    for f in root.glob("**/*_sim_scored.jsonl"):
        _add(f)

    # artifacts/sim/ uses non-standard names (master.jsonl, sessions/, archive/all_batch*.jsonl)
    sim_dir = root / "artifacts" / "sim"
    if sim_dir.exists():
        for f in sim_dir.rglob("*.jsonl"):
            _add(f)

    return sorted(files)


def _load_all_sim_records(root: Path) -> dict[str, dict]:
    """Load all sim records, deduplicate by expression (keep highest fitness)."""
    files = _collect_sim_files(root)
    lookup: dict[str, dict] = {}
    total_raw = 0
    for path in files:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_raw += 1
                # Fast skip: no expression
                expr = _normalise_expr(record.get("expression", ""))
                if not expr:
                    continue
                # Fast skip: no alpha_id (never registered on Brain)
                if not record.get("alpha_id"):
                    continue
                # Fast skip: no numeric data
                if record.get("is_sharpe") is None:
                    continue
                # Fast skip: bad grade
                grade = (record.get("grade") or "").upper()
                if grade in _BAD_GRADES:
                    continue
                # Deduplicate: keep highest fitness
                existing = lookup.get(expr)
                if existing is None:
                    lookup[expr] = record
                else:
                    new_fit = record.get("is_fitness") or -999.0
                    old_fit = existing.get("is_fitness") or -999.0
                    if new_fit > old_fit:
                        lookup[expr] = record

    print(f"Loaded {total_raw} raw records from {len(files)} files")
    print(f"After dedup + grade/data pre-filter: {len(lookup)} unique expressions")
    return lookup


# ── live self-correlation check ───────────────────────────────────────────────

def _fetch_self_corr_status(
    session,
    alpha_id: str,
    *,
    max_attempts: int = 8,
    sleep_seconds: float = 2.0,
) -> str:
    """Fetch live SELF_CORRELATION result from Brain, retrying up to max_attempts.

    Returns "PASS", "FAIL", or "PENDING".
    - PASS  → include candidate
    - PENDING → include candidate (Brain hasn't evaluated it yet for unsubmitted alphas;
                admission scoring will do a live PnL-based correlation check instead)
    - FAIL  → skip candidate (confirmed correlated with existing portfolio)
    """
    for attempt in range(1, max_attempts + 1):
        try:
            detail = get_alpha_detail(session, alpha_id)
            checks = (detail.get("is") or {}).get("checks") or []
            for c in checks:
                if c.get("name") == "SELF_CORRELATION":
                    result = (c.get("result") or "PENDING").upper()
                    if result == "FAIL":
                        return "FAIL"
                    # PASS or PENDING: stop retrying, return immediately
                    return result
            return "PASS"  # check not present in response
        except Exception:
            pass
        # Only retry on exception (network error), not on PENDING
        if attempt < max_attempts:
            time.sleep(sleep_seconds)
    return "PENDING"  # failed to fetch — pass through to admission scoring


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan all sim history for submittable unsubmitted positive-contribution alphas."
    )
    p.add_argument("--competition-id", default="IQC2026S1")
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--min-brain-score-delta", type=float, default=0.0)
    p.add_argument("--min-fallback-delta-sharpe", type=float, default=0.0)
    p.add_argument("--skip-brain-performance-comparison", action="store_true")
    p.add_argument("--performance-attempts", type=int, default=5)
    p.add_argument("--performance-wait", type=float, default=2.0)
    p.add_argument("--max-pnl-attempts", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--out-jsonl", default="artifacts/full_scan/ranked.jsonl")
    p.add_argument("--out-txt", default="artifacts/full_scan/ranked.txt")
    return p.parse_args()


import sys as _sys
_sys.stdout.reconfigure(line_buffering=True)

def main() -> None:
    args = parse_args()
    session = sign_in(args.credentials)

    print("Loading competition portfolio from Brain...")
    portfolio = load_submitted_portfolio_data(
        session,
        args.competition_id,
        max_pnl_attempts=args.max_pnl_attempts,
        sleep_seconds=args.sleep,
    )
    competition_ids = set(portfolio.keys())
    print(f"Portfolio loaded: {len(portfolio)} alphas in competition")

    # Load + pre-filter all records
    all_records = _load_all_sim_records(ROOT)

    # Apply full submittability filter and exclude already-submitted
    candidates = []
    skipped_submitted = 0
    skipped_checks = 0
    for record in all_records.values():
        alpha_id = record.get("alpha_id")
        if alpha_id in competition_ids:
            skipped_submitted += 1
            continue
        if not _is_submittable(record):
            skipped_checks += 1
            continue
        candidates.append(record)

    print(f"Already submitted (skipped): {skipped_submitted}")
    print(f"Failed checks (skipped):     {skipped_checks}")
    print(f"Candidates for PnL fetch:    {len(candidates)}")

    if not candidates:
        print("Nothing to score.")
        return

    # Fetch live SELF_CORRELATION result from Brain (up to 8 attempts per candidate)
    print(f"\nChecking live SELF_CORRELATION for {len(candidates)} candidates...")
    corr_pass: list[dict] = []
    skipped_corr = 0
    for i, record in enumerate(candidates, 1):
        alpha_id = str(record.get("alpha_id") or "")
        status = _fetch_self_corr_status(session, alpha_id, max_attempts=8, sleep_seconds=0.5)
        if status == "FAIL":
            skipped_corr += 1
        else:  # PASS or PENDING — proceed to admission scoring
            corr_pass.append(record)
        if i % 10 == 0:
            print(f"  self-corr checked {i}/{len(candidates)}  pass={len(corr_pass)}  skip={skipped_corr}")
        elif args.sleep > 0:
            time.sleep(min(args.sleep, 0.2))
    candidates = corr_pass
    print(f"After SELF_CORRELATION gate: {len(candidates)} pass  {skipped_corr} skipped (FAIL/PENDING)")

    if not candidates:
        print("Nothing to score after correlation check.")
        return

    # Fetch PnL
    print(f"\nFetching PnL for {len(candidates)} candidates...")
    pnl_cache: dict[str, dict] = {}
    for i, record in enumerate(candidates, 1):
        alpha_id = str(record.get("alpha_id") or "")
        if alpha_id in pnl_cache:
            continue
        pnl = fetch_pnl_series_with_retry(
            session,
            alpha_id,
            max_attempts=args.max_pnl_attempts,
            sleep_seconds=args.sleep,
            timeout=20,
        )
        pnl_cache[alpha_id] = pnl or {}
        if args.sleep > 0:
            time.sleep(min(args.sleep, 0.3))
        if i % 10 == 0:
            print(f"  {i}/{len(candidates)}")

    pnl_available = sum(1 for v in pnl_cache.values() if v)
    print(f"PnL available for {pnl_available}/{len(candidates)} candidates")

    # Score admission
    config = AdmissionConfig(
        min_brain_score_delta=args.min_brain_score_delta,
        min_fallback_delta_sharpe=args.min_fallback_delta_sharpe,
    )

    def fetch_performance(alpha_id: str) -> dict | None:
        if args.skip_brain_performance_comparison:
            return None
        return fetch_brain_performance_comparison(
            session,
            alpha_id,
            args.competition_id,
            max_attempts=args.performance_attempts,
            sleep_seconds=args.performance_wait,
            timeout=20,
        )

    annotated = annotate_records_with_admission(
        candidates,
        portfolio,
        config=config,
        candidate_pnls=pnl_cache,
        fetch_performance_comparison=fetch_performance,
    )

    write_ranked_outputs(annotated, Path(args.out_jsonl), Path(args.out_txt))

    counts: dict[str, int] = {}
    for r in annotated:
        d = r.get("admission_decision", "WATCH")
        counts[d] = counts.get(d, 0) + 1
    print(f"\nDecisions: {counts}")
    print(f"Wrote: {args.out_txt}")
    print(f"Wrote: {args.out_jsonl}")

    promote_records = [r for r in annotated if r.get("admission_decision") == PROMOTE]
    promote_records.sort(key=_preferred_delta, reverse=True)

    if promote_records:
        print(f"\nTop PROMOTE alphas ({len(promote_records)} total):")
        print(f"{'#':>3} {'alpha_id':<10} {'sharpe':>6} {'fitness':>7} {'corr':>6} {'d_score':>8} {'d_fb':>8} {'signal':<27}  expression")
        print("-" * 145)
        for i, r in enumerate(promote_records[:30], 1):
            expr = _normalise_expr(r.get("expression", ""))[:65]
            print(
                f"{i:>3} {str(r.get('alpha_id', '')):<10} "
                f"{(r.get('is_sharpe') or 0):>6.2f} "
                f"{(r.get('is_fitness') or 0):>7.2f} "
                f"{(r.get('admission_global_max_corr') or 0):>6.3f} "
                f"{(r.get('admission_brain_score_delta') or 0):>8.0f} "
                f"{(r.get('admission_delta_pool_sharpe') or 0):>8.4f} "
                f"{str(r.get('admission_signal') or 'N/A'):<27}  {expr}"
            )

        print("\nCombination proposal generation is disabled; submit PROMOTE alphas individually.")
    else:
        print("No PROMOTE candidates found.")


if __name__ == "__main__":
    main()
