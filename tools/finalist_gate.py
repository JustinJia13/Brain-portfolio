"""Finalist robustness gate: window-plateau + neutralization-consistency checks.

For the top N candidates that already pass base thresholds (sharpe>=1.25,
fitness>=1.0, checks pass), simulate a small designed set of perturbations
(window neighbors + neutralization variants, capped per finalist) and render
a pass/weak/fail robustness card per finalist. IS peaks that collapse under
perturbation are exactly the alphas that decay out-of-sample — they get
status "fail" and should not be submitted.

Reuses alpha_enhancement.workshop.robustness_card for variant generation and verdicts,
and the shared workshop submission boundary (which skips variants already in
the log — historical sims are free evidence).

Usage:
    python tools/finalist_gate.py --dry-run            # show planned sims, no submission
    python tools/finalist_gate.py --top 5              # gate the current top 5 finalists
    python tools/finalist_gate.py --expressions f.txt  # gate specific expressions
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.brain.api import sign_in  # noqa: E402
from core.brain.submit import DEFAULT_SETTINGS  # noqa: E402
from core.robustness import record_robustness_score  # noqa: E402
from alpha_enhancement.workshop.batch import DEFAULT_SIM_GLOBS, _rank_key, _standalone_passes  # noqa: E402
from alpha_enhancement.workshop.core import DEFAULT_WORKSHOP_SETTINGS  # noqa: E402

# submit_candidate_batch sends settings straight to the API; supply the full
# required field set, not just the workshop axes.
GATE_SETTINGS = {**DEFAULT_SETTINGS, **DEFAULT_WORKSHOP_SETTINGS}
from alpha_enhancement.workshop.records import load_records  # noqa: E402
from alpha_enhancement.workshop.robustness_card import (  # noqa: E402
    build_robustness_candidates,
    build_robustness_card,
    final_recommendation,
    format_card_text,
)
from alpha_enhancement.workshop.simulation import submit_candidate_batch  # noqa: E402

OUT_DIR = Path("artifacts/robustness")
GATE_SIM_LOG = OUT_DIR / "finalist_gate_sim.jsonl"
CARDS_PATH = OUT_DIR / "finalist_cards.jsonl"


def _gate_candidates(source: dict, max_sims: int) -> list[dict]:
    """Window + neutralization variants only, capped at max_sims per finalist."""
    expr = str(source.get("expression") or "")
    raw = build_robustness_candidates(
        [source],
        root_expression=expr,
        max_expression_variants=0,
        max_field_variants=0,
        max_window_variants=4,
    )
    kept: list[dict] = []
    for cand in raw:
        kind = cand.get("robust_kind") or (cand.get("params") or {}).get("robust_kind")
        check = str(cand.get("robust_check") or (cand.get("params") or {}).get("robust_check") or "")
        if kind == "window":
            kept.append(cand)
        elif kind == "settings" and "neutralization" in check:
            kept.append(cand)
    return kept[:max_sims]


def select_finalists(records: list[dict], top: int) -> list[dict]:
    passing = [r for r in records if _standalone_passes(r)]
    # one representative per expression, highest |sharpe|
    by_expr: dict[str, dict] = {}
    for r in passing:
        key = " ".join(str(r.get("expression") or "").split())
        old = by_expr.get(key)
        if old is None or abs(r.get("is_sharpe") or 0) > abs(old.get("is_sharpe") or 0):
            by_expr[key] = r
    ranked = sorted(by_expr.values(), key=_rank_key, reverse=True)
    return ranked[:top]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=5, help="number of finalists to gate")
    ap.add_argument("--max-sims-per-finalist", type=int, default=6)
    ap.add_argument("--expressions", help="optional file with one expression per line to gate instead")
    ap.add_argument("--credentials", default="brain.txt")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="plan only, no simulations")
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in DEFAULT_SIM_GLOBS:
        paths.extend(glob.glob(pattern))
    records = load_records(sorted(set(paths)))

    if args.expressions:
        wanted = {line.strip() for line in open(args.expressions) if line.strip()}
        sources = [r for r in records if str(r.get("expression") or "").strip() in wanted
                   and r.get("status") == "done"]
        sources = select_finalists(sources, top=len(wanted))
    else:
        sources = select_finalists(records, top=args.top)

    if not sources:
        print("No finalists found (nothing passes base thresholds).")
        return

    plans: list[tuple[dict, list[dict]]] = []
    total = 0
    for source in sources:
        cands = _gate_candidates(source, args.max_sims_per_finalist)
        plans.append((source, cands))
        total += len(cands)
        rob = record_robustness_score(source)
        print(f"finalist: sharpe={source.get('is_sharpe')} fitness={source.get('is_fitness')} "
              f"robustness={rob if rob is not None else 'n/a'}")
        print(f"  {source.get('expression', '')[:140]}")
        for cand in cands:
            params = cand.get("params") or {}
            print(f"    - [{params.get('robust_kind', cand.get('robust_kind'))}/"
                  f"{params.get('robust_check', cand.get('robust_check'))}] "
                  f"{str(cand.get('expression'))[:110]}")

    print(f"\nplanned simulations: {total} across {len(plans)} finalists")
    if args.dry_run:
        print("dry-run: stopping before submission.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = sign_in(args.credentials)
    all_gate_records: list[dict] = []
    for source, cands in plans:
        if not cands:
            continue
        new_records = submit_candidate_batch(
            cands,
            session,
            GATE_SIM_LOG,
            dict(GATE_SETTINGS),
            args.credentials,
            args.batch_size,
            yearly_stats_attempts=4,
        )
        all_gate_records.extend(new_records or [])

    gate_log = load_records([str(GATE_SIM_LOG)])
    with CARDS_PATH.open("w") as f:
        for source, _cands in plans:
            expr = str(source.get("expression") or "")
            card = build_robustness_card(
                root_expression=expr,
                source_records=[source],
                robustness_records=gate_log,
            )
            card["final_recommendation"] = final_recommendation(
                source, robustness_status=card.get("overall_status")
            )
            f.write(json.dumps(card) + "\n")
            print()
            print(format_card_text(card))
            print(f"  -> final_recommendation: {card['final_recommendation']}")
    print(f"\ncards written: {CARDS_PATH}")


if __name__ == "__main__":
    main()
