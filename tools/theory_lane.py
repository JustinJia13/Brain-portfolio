"""CLI for the theory-first hypothesis lane.

Usage:
    python tools/theory_lane.py --dry-run                      # cards + families, no sims
    python tools/theory_lane.py --dataset fnd6 --n-theories 5  # one dataset
    python tools/theory_lane.py --n-theories 8 --refine        # full run with refine round

Dry-run first: iterate the proposer prompt until the expression-rejection rate
is ~0 before burning simulation budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wq_alpha.theory.runner import run_theory_lane  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", help="restrict candidate clusters to one dataset id")
    ap.add_argument("--n-theories", type=int, default=5)
    ap.add_argument("--family-size", type=int, default=15, help="max members per family")
    ap.add_argument("--max-simulated", type=int, default=None,
                    help="simulate at most this many theories (Thompson-ranked)")
    ap.add_argument("--refine", action="store_true", help="allow one refine round per inconclusive theory")
    ap.add_argument("--rounds", type=int, default=1,
                    help="propose->simulate cycles in one run; <=0 runs perpetually "
                         "until interrupted (intended for the headless brain2 account). "
                         "For brain (face-scan), pass a bounded number.")
    ap.add_argument("--dry-run", action="store_true", help="propose + enumerate only, no simulations")
    ap.add_argument("--credentials", default="brain.txt")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    run_theory_lane(
        dataset=args.dataset,
        n_theories=args.n_theories,
        family_max=args.family_size,
        max_theories_to_simulate=args.max_simulated,
        refine=args.refine,
        dry_run=args.dry_run,
        credentials=args.credentials,
        batch_size=args.batch_size,
        model=args.model,
        seed=args.seed,
        rounds=args.rounds,
    )


if __name__ == "__main__":
    main()
