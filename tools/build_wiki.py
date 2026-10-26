"""Build/refresh the research wiki and export the grammar prior.

Usage:
    python tools/build_wiki.py                  # deterministic pages only (no LLM cost)
    python tools/build_wiki.py --llm            # LLM narrative for changed pages
    python tools/build_wiki.py --export-priors  # also write grammar_prior_from_wiki.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.wiki.generate import export_priors, generate_wiki  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", action="store_true", help="LLM narrative for changed pages")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--min-sims", type=int, default=10, help="page threshold per cluster")
    ap.add_argument("--no-rebuild-stats", action="store_true",
                    help="reuse existing cluster_stats.jsonl")
    ap.add_argument("--export-priors", action="store_true")
    args = ap.parse_args()

    counters = generate_wiki(
        use_llm=args.llm,
        model=args.model,
        min_sims=args.min_sims,
        rebuild_stats=not args.no_rebuild_stats,
    )
    print(f"wiki: {counters}")
    if args.export_priors:
        n = export_priors()
        print(f"prior export: {n} cluster_priors rows -> artifacts/probe/grammar_prior_from_wiki.json")


if __name__ == "__main__":
    main()
