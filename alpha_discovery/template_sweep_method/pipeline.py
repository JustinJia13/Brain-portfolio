"""Unified CLI entry point for the alpha-search pipeline.

Commands
--------
build       Generate a JSONL plan from ALPHA_TEMPLATES (templates.py).
submit      Submit a JSONL plan to WorldQuant Brain.
repair      Scan sim logs for near-misses, generate repair variants.
status      Print performance statistics from simulation logs.
run         Build + submit in one step.
refine      Disabled legacy combo optimizer.

Quick-start examples
--------------------
# Build and submit all alpha template variants:
python -m alpha_discovery.template_sweep_method.pipeline run \\
    --sim-log artifacts/sweep/alpha_sim.jsonl

# Build plan only (inspect before submitting):
python -m alpha_discovery.template_sweep_method.pipeline build \\
    --output artifacts/sweep/alpha_batch.jsonl

# Submit a pre-built plan:
python -m alpha_discovery.template_sweep_method.pipeline submit \\
    --plan artifacts/sweep/alpha_batch.jsonl \\
    --sim-log artifacts/sweep/alpha_sim.jsonl

# Show stats:
python -m alpha_discovery.template_sweep_method.pipeline status \\
    --sim-log artifacts/sweep/alpha_sim.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Ensure the sweep artifacts directory exists
Path("artifacts/sweep").mkdir(parents=True, exist_ok=True)


def _build(args: argparse.Namespace) -> None:
    from alpha_discovery.template_sweep_method.sweep import build_alpha_plan
    build_alpha_plan(output_path=args.output)


def _submit(args: argparse.Namespace) -> None:
    import json
    from core.brain.submit import run

    settings = None
    if getattr(args, "settings_json", "") and args.settings_json:
        with open(args.settings_json) as f:
            settings = json.load(f)

    run(
        args.plan,
        args.sim_log,
        batch_size=args.batch_size,
        credentials=args.credentials,
        settings=settings,
        max_candidates=getattr(args, "max_candidates", 0),
        abandon_after=getattr(args, "abandon_after", 15),
        abandon_threshold=getattr(args, "abandon_threshold", 0.75),
        probe_abandon_after=getattr(args, "probe_abandon_after", 60),
        shard=getattr(args, "shard", "1/1"),
    )
    _merge_sim_log(args.sim_log)


def _repair(args: argparse.Namespace) -> None:
    from alpha_enhancement.workshop.repair import build_repair_plan
    build_repair_plan(
        sim_log_paths=args.sim_log,
        output_path=args.output,
        min_sharpe=args.min_sharpe,
        max_repairs_per_alpha=args.max_repairs,
        flip_threshold=getattr(args, "flip_threshold", -1.5),
    )




def _status(args: argparse.Namespace) -> None:
    import glob
    from alpha_discovery.template_sweep_method.feedback import status

    paths: list[str] = []
    for pattern in args.sim_log:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])
    status(paths)


def _run(args: argparse.Namespace) -> None:
    """Build + submit in one step."""
    from alpha_discovery.template_sweep_method.sweep import build_alpha_plan
    from core.brain.submit import run

    plan_path = getattr(args, "plan", None) or "artifacts/sweep/alpha_batch.jsonl"
    build_alpha_plan(output_path=plan_path)

    run(
        plan_path,
        args.sim_log,
        batch_size=args.batch_size,
        credentials=args.credentials,
        max_candidates=getattr(args, "max_candidates", 0),
        abandon_after=getattr(args, "abandon_after", 15),
        abandon_threshold=getattr(args, "abandon_threshold", 0.75),
        shard=getattr(args, "shard", "1/1"),
    )
    _merge_sim_log(args.sim_log)


def _refine(args: argparse.Namespace) -> None:
    """Disabled legacy same-delay combo optimizer."""
    seed_file = args.template_file or args.seed_file
    print("[refine] Disabled: combo optimization has been removed from the active pipeline.")
    print(f"[refine] Seed file left untouched: {seed_file}")
    print("[refine] Submit PROMOTE rows from workshop_submittable.jsonl individually after admission review.")
    print("[refine] No Brain simulations were submitted.")


def _merge_sim_log(sim_log: str) -> None:
    """Merge a sim log into master.jsonl after a pipeline run, then update Q-table."""
    p = Path(sim_log)
    if not p.exists():
        return
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from core.logs import merge_session_to_master, MASTER
        n_new, n_updated = merge_session_to_master(p)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    except Exception as e:
        print(f"[warn] Could not merge to master: {e}")
        return

    try:
        import json as _json
        from core.qtable import update_qtable_from_records
        records = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass
        n = update_qtable_from_records(records)
        if n:
            print(f"Q-table updated: {n} record(s)")
    except Exception as e:
        print(f"[warn] Could not update Q-table: {e}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m alpha_discovery.template_sweep_method.pipeline",
        description="Hypothesis-driven alpha search pipeline.",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ---- build ----
    p_build = sub.add_parser("build", help="Generate JSONL plan from ALPHA_TEMPLATES.")
    p_build.add_argument("--output", default="artifacts/sweep/alpha_batch.jsonl")

    # ---- submit ----
    p_submit = sub.add_parser("submit", help="Submit a JSONL plan to Brain.")
    p_submit.add_argument("--plan", required=True)
    p_submit.add_argument("--sim-log", required=True)
    p_submit.add_argument("--batch-size", type=int, default=3)
    p_submit.add_argument("--credentials", default="brain.txt")
    p_submit.add_argument("--settings-json", default="")
    p_submit.add_argument("--max-candidates", type=int, default=0)
    p_submit.add_argument("--abandon-after", type=int, default=15,
                          help="Abandon a hardcoded template after N results if max_sharpe < threshold. 0=disabled.")
    p_submit.add_argument("--abandon-threshold", type=float, default=0.75,
                          help="Max Sharpe threshold below which a template is abandoned (default 0.75).")
    p_submit.add_argument("--probe-abandon-after", type=int, default=60,
                          help="Abandon an auto-fields probe template after N results (default 60, 0=disabled).")
    p_submit.add_argument("--shard", default="1/1", metavar="N/TOTAL",
                          help="Shard for parallel runs, e.g. --shard 1/2 and --shard 2/2.")

    # ---- repair ----
    p_repair = sub.add_parser("repair", help="Generate repair variants for near-miss alphas.")
    p_repair.add_argument("--sim-log", nargs="+", required=True)
    p_repair.add_argument("--output", required=True)
    p_repair.add_argument("--min-sharpe", type=float, default=0.9)
    p_repair.add_argument("--flip-threshold", type=float, default=-1.5)
    p_repair.add_argument("--max-repairs", type=int, default=3)

    # ---- status ----
    p_status = sub.add_parser("status", help="Show template performance stats from simulation logs.")
    p_status.add_argument("--sim-log", nargs="+", required=True)

    # ---- run (build + submit) ----
    p_run = sub.add_parser("run", help="Build plan from ALPHA_TEMPLATES and submit in one step.")
    p_run.add_argument("--sim-log", required=True)
    p_run.add_argument("--plan", default="", help="Override auto-generated plan path.")
    p_run.add_argument("--batch-size", type=int, default=3)
    p_run.add_argument("--credentials", default="brain.txt")
    p_run.add_argument("--max-candidates", type=int, default=0)
    p_run.add_argument("--abandon-after", type=int, default=15,
                       help="Abandon a template after N results if max_sharpe < threshold. 0=disabled.")
    p_run.add_argument("--abandon-threshold", type=float, default=0.75)
    p_run.add_argument("--shard", default="1/1", metavar="N/TOTAL")

    # ---- refine ----
    p_refine = sub.add_parser(
        "refine",
        help="Disabled legacy combo optimizer; submits nothing.",
    )
    p_refine.add_argument(
        "--seed-file",
        default="artifacts/workshop/workshop_submittable.jsonl",
        metavar="JSONL",
        help="Ignored legacy seed file argument.",
    )
    p_refine.add_argument(
        "--template-file",
        default="",
        metavar="JSONL",
        help="Deprecated alias for --seed-file.",
    )
    p_refine.add_argument(
        "--sim-log",
        required=True,
        help="Ignored legacy sim-log argument retained for old scripts.",
    )

    return root


_DISPATCH = {
    "build": _build,
    "submit": _submit,
    "repair": _repair,
    "status": _status,
    "run": _run,
    "refine": _refine,
}


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    _DISPATCH[args.command](args)


if __name__ == "__main__":
    main()
