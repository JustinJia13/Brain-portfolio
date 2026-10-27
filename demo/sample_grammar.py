"""Sample candidate alphas from the bounded grammar, offline (needs only all_fields.csv).

The budget is the grammar's size limit (the paper's K): a bigger budget allows deeper
expressions. GrammarConfig is the one place the grammar's hyperparameters live, and
run_pipeline's command-line flags take their defaults from it.

    python demo/sample_grammar.py [--budget 4 6] [--n 4] [--seed 7]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_discovery.grammar_method.grammar import GrammarConfig  # noqa: E402
from alpha_discovery.grammar_method.planner import GrammarPlanner  # noqa: E402
from core.clusters import load_clusters_from_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget", type=int, nargs="+", default=[4, 6], help="budgets to sample at")
    parser.add_argument("--n", type=int, default=4, help="examples to print per budget")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    clusters = load_clusters_from_csv(str(ROOT / "all_fields.csv"))
    print(f"{len(clusters)} field clusters built from all_fields.csv")
    for budget in args.budget:
        plan = GrammarPlanner(GrammarConfig(budget=budget, n_samples=40)).build_plan(clusters, seed=args.seed)
        print(f"\nbudget={budget}: {len(plan)} candidates sampled; first {args.n}:")
        for candidate in plan[: args.n]:
            print(f"  {candidate.expression}")


if __name__ == "__main__":
    main()
