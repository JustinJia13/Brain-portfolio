"""Build and summarize fixed Brain settings regime checks.

Examples
--------
Build a one-axis regime check plan from promoted workshop alphas:

    python -m tools.regime_check build \
        --input artifacts/workshop/workshop_submittable.jsonl \
        --output artifacts/regime_check/plan.jsonl

Submit the plan with the normal Brain submitter:

    python -m alpha_discovery.template_sweep_method.pipeline submit \
        --plan artifacts/regime_check/plan.jsonl \
        --sim-log artifacts/regime_check/sim.jsonl

Summarize the completed check:

    python -m tools.regime_check summary \
        --sim-log artifacts/regime_check/sim.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_discovery.template_sweep_method.regime_check import (
    DEFAULT_DELAYS,
    DEFAULT_DECAYS,
    DEFAULT_NEUTRALIZATIONS,
    DEFAULT_TRUNCATIONS,
    DEFAULT_UNIVERSES,
    build_regime_check_candidates,
    known_expression_regimes,
    load_jsonl,
    parse_float_grid,
    parse_int_grid,
    parse_text_grid,
    summarize_regime_check,
    write_jsonl,
    write_summary_text,
)


def _build(args: argparse.Namespace) -> None:
    source_records = []
    if args.input:
        source_records.extend(load_jsonl(args.input))
    if args.expression:
        source_records.append({
            "expression": args.expression,
            "settings": {
                "delay": args.delay,
                "universe": args.base_universe.upper(),
                "neutralization": args.base_neutralization.upper(),
                "truncation": args.base_truncation,
                "decay": args.base_decay,
            },
        })
    if not source_records:
        raise SystemExit("No source alphas. Pass --input JSONL or --expression.")

    known_records = []
    if args.known_sim_log:
        known_records.extend(load_jsonl(args.known_sim_log))
    if not args.include_baseline:
        known_records.extend(source_records)
    known_pairs = known_expression_regimes(known_records)

    candidates = build_regime_check_candidates(
        source_records,
        universes=parse_text_grid(args.universes, DEFAULT_UNIVERSES),
        delays=parse_int_grid(args.delays, DEFAULT_DELAYS),
        neutralizations=parse_text_grid(args.neutralizations, DEFAULT_NEUTRALIZATIONS),
        truncations=parse_float_grid(args.truncations, DEFAULT_TRUNCATIONS),
        decays=parse_int_grid(args.decays, DEFAULT_DECAYS),
        mode=args.mode,
        include_baseline=args.include_baseline,
        known_pairs=known_pairs,
        max_alphas=args.max_alphas,
        promote_only=args.promote_only,
    )

    write_jsonl(candidates, args.output)
    axes = "one axis at a time" if args.mode == "one-axis" else "full cross-product"
    print(f"Wrote {len(candidates)} regime-check candidates to {args.output}")
    print(f"Mode: {args.mode} ({axes}); no adaptive/greedy selection.")
    print("Submit with:")
    print(f"  python -m alpha_discovery.template_sweep_method.pipeline submit --plan {args.output} --sim-log {args.sim_log}")


def _summary(args: argparse.Namespace) -> None:
    records = load_jsonl(args.sim_log)
    summaries = summarize_regime_check(records, min_pass_rate=args.min_pass_rate)
    write_summary_text(summaries, args.out_txt)
    write_jsonl(summaries, args.out_jsonl)
    print(f"Wrote text summary: {args.out_txt}")
    print(f"Wrote JSONL summary: {args.out_jsonl}")
    if summaries:
        top = summaries[0]
        print(
            "Top diagnostic result: "
            f"pass_rate={top['pass_rate']:.2f} best_regime={top['best_regime']} "
            f"fitness={top.get('best_fitness')} sharpe={top.get('best_sharpe')}"
        )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build a fixed regime-check candidate plan.")
    build.add_argument("--input", nargs="*", default=[], help="Source JSONL rows with expressions.")
    build.add_argument("--expression", default="", help="Single FASTEXPR expression to check.")
    build.add_argument("--output", default="artifacts/regime_check/plan.jsonl")
    build.add_argument("--sim-log", default="artifacts/regime_check/sim.jsonl")
    build.add_argument("--known-sim-log", nargs="*", default=[], help="Existing sim logs to avoid duplicate regimes.")
    build.add_argument("--mode", choices=["one-axis", "grid"], default="one-axis")
    build.add_argument(
        "--delays",
        default=",".join(str(v) for v in DEFAULT_DELAYS),
        help="Optional delay grid. Default keeps delay fixed; pass e.g. 0,1 to test delay robustness.",
    )
    build.add_argument("--universes", default=",".join(DEFAULT_UNIVERSES))
    build.add_argument("--neutralizations", default=",".join(DEFAULT_NEUTRALIZATIONS))
    build.add_argument("--truncations", default=",".join(str(v) for v in DEFAULT_TRUNCATIONS))
    build.add_argument("--decays", default=",".join(str(v) for v in DEFAULT_DECAYS))
    build.add_argument("--include-baseline", action="store_true", help="Also simulate the source/base regime.")
    build.add_argument("--promote-only", action="store_true", help="Only use rows with admission_decision PROMOTE.")
    build.add_argument("--max-alphas", type=int, default=0, help="Limit number of source expressions (0 = no cap).")
    build.add_argument("--delay", type=int, default=1, help="Base delay for --expression mode.")
    build.add_argument("--base-universe", default="TOP3000")
    build.add_argument("--base-neutralization", default="SUBINDUSTRY")
    build.add_argument("--base-truncation", type=float, default=0.01)
    build.add_argument("--base-decay", type=int, default=0)

    summary = sub.add_parser("summary", help="Summarize completed regime-check simulations.")
    summary.add_argument("--sim-log", nargs="+", required=True)
    summary.add_argument("--out-txt", default="artifacts/regime_check/summary.txt")
    summary.add_argument("--out-jsonl", default="artifacts/regime_check/summary.jsonl")
    summary.add_argument("--min-pass-rate", type=float, default=0.5)

    return parser


def main() -> None:
    args = _make_parser().parse_args()
    if args.command == "build":
        _build(args)
    elif args.command == "summary":
        _summary(args)


if __name__ == "__main__":
    main()
