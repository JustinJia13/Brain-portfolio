"""Re-simulate brain2 discovery winners on the brain (competition) account.

brain2 is a separate discovery pool ruled only by submittable + robustness +
fitness — it has no self-correlation or score-addition gate (different account
portfolio). When a brain2 theory family validates and yields submittable, robust
members, the theory runner queues them in artifacts/theory/promote_to_brain.jsonl.

This tool re-simulates those expressions on the brain account, where the
portfolio-aware gates DO apply: it loads brain's competition portfolio and runs
the same admission path as the workshop (self-correlation + Brain score-addition
delta). Output is a separate brain-pool sim log + admission rows.

Usage:
    python tools/promote_to_brain.py --dry-run             # show the queue, no sims
    python tools/promote_to_brain.py --credentials brain.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wq_alpha.brain.api import sign_in  # noqa: E402
from wq_alpha.brain.submit import DEFAULT_SETTINGS  # noqa: E402
from wq_alpha.logs import merge_session_to_master  # noqa: E402
from wq_alpha.workshop.core import DEFAULT_WORKSHOP_SETTINGS  # noqa: E402
from wq_alpha.workshop.records import load_records  # noqa: E402
from wq_alpha.workshop.simulation import submit_candidate_batch  # noqa: E402

QUEUE_PATH = Path("artifacts/theory/promote_to_brain.jsonl")
BRAIN_SIM_LOG = Path("artifacts/theory/promoted_brain_sim.jsonl")
SETTINGS = {**DEFAULT_SETTINGS, **DEFAULT_WORKSHOP_SETTINGS}


def load_queue() -> list[dict]:
    if not QUEUE_PATH.exists():
        return []
    rows: list[dict] = []
    for line in QUEUE_PATH.open():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--credentials", default="brain.txt",
                    help="competition account credentials (default brain.txt)")
    ap.add_argument("--competition-id", default="IQC2026S1")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--skip-portfolio-pnl", action="store_true",
                    help="skip loading brain portfolio (no corr/score gate; not advised)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queue = load_queue()
    if not queue:
        print(f"Promotion queue empty ({QUEUE_PATH}). Run the theory lane on brain2 first.")
        return

    # dedup by (expression, settings-regime), not expression alone: a repaired
    # alpha's fix IS its settings (decay for turnover, neutralization for
    # concentration), so two queue rows with the same expression but different
    # settings_override are DISTINCT sims and must both be scored. Skip only when
    # the same (expr, regime) is already done on brain.
    from wq_alpha.workshop.core import regime_from_settings

    def _norm(expr) -> str:
        return " ".join(str(expr or "").split())

    def _regime(settings_override) -> str:
        return regime_from_settings({**SETTINGS, **(settings_override or {})})

    already: set[tuple[str, str]] = set()
    for r in load_records([str(BRAIN_SIM_LOG)]):
        if r.get("status") != "done":
            continue
        reg = r.get("regime") or regime_from_settings({**SETTINGS, **(r.get("settings") or {})})
        already.add((_norm(r.get("expression")), reg))

    seen: set[tuple[str, str]] = set()
    candidates: list[dict] = []
    for row in queue:
        override = row.get("settings_override") or {}
        key = (_norm(row.get("expression")), _regime(override))
        if not key[0] or key in seen or key in already:
            continue
        seen.add(key)
        candidates.append({
            "candidate_id": f"promote_{row.get('theory_id')}_{len(candidates):02d}",
            "theory_id": row.get("theory_id"),
            "search_lane": "promoted_from_brain2",
            "expression": row.get("expression"),
            "cluster_ids": row.get("cluster_ids"),
            "settings_override": override,
            "params": {"source_account": row.get("source_account"),
                       "direction": row.get("direction")},
        })

    print(f"queued: {len(queue)} | new to re-sim on brain: {len(candidates)} "
          f"| already done on brain: {len(already)}")
    for c in candidates:
        print(f"  {c['expression'][:110]}")
    if args.dry_run:
        print("dry-run: stopping before submission.")
        return
    if not candidates:
        print("Nothing new to promote.")
        return

    session = sign_in(args.credentials)
    portfolio_alpha_data = None
    if not args.skip_portfolio_pnl:
        from alpha_grammar.run_pipeline import load_portfolio_alpha_data
        portfolio_alpha_data = load_portfolio_alpha_data(
            session, competition_id=args.competition_id)
        print(f"loaded brain portfolio: {len(portfolio_alpha_data or {})} alphas "
              "(self-correlation + score-addition gates active)")

    BRAIN_SIM_LOG.parent.mkdir(parents=True, exist_ok=True)
    records = submit_candidate_batch(
        candidates, session, BRAIN_SIM_LOG, dict(SETTINGS),
        args.credentials, args.batch_size,
        portfolio_alpha_data=portfolio_alpha_data,
        yearly_stats_attempts=4,
    )
    from wq_alpha.theory.dedup import lane_lock  # serialize master write vs a live lane
    with lane_lock():
        merge_session_to_master(BRAIN_SIM_LOG)
    print(f"\nre-simulated {len(records or [])} on brain -> {BRAIN_SIM_LOG}")
    print("Review admission rows there; submit the PROMOTE alphas to the brain competition pool.")


if __name__ == "__main__":
    main()
