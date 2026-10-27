"""LLM enhancement loop: structural EnhanceAgent → submit → feedback → repeat.

Usage:
    # Dry run (no Brain submission — prints the plan only)
    python tools/alpha_multi_enhance_loop.py \\
        --sim-log artifacts/sim/master.jsonl \\
        --enhance-sim-log alpha_multi/artifacts/enhance_sim.jsonl \\
        --fields-csv all_fields.csv \\
        --n-iterations 1 --top-n 5 --dry-run

    # Live run
    python tools/alpha_multi_enhance_loop.py \\
        --sim-log artifacts/sim/master.jsonl \\
        --enhance-sim-log alpha_multi/artifacts/enhance_sim.jsonl \\
        --fields-csv all_fields.csv \\
        --n-iterations 3 --credentials brain.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_enhancement.agents.enhance import (
    AlphaEnhanceAgent,
    select_passing_reps,
)
from alpha_enhancement.agents.repair import load_sim_log

_ARTIFACTS_DIR = Path(__file__).parent.parent / "alpha_multi" / "artifacts"

# Tokens that are FASTEXPR operator or keyword names, not field names.
_FASTEXPR_KEYWORDS = {
    "normalize", "rank", "zscore", "scale", "winsorize",
    "group_rank", "group_zscore", "group_neutralize", "group_backfill",
    "group_mean", "group_sum",
    "ts_mean", "ts_std", "ts_rank", "ts_sum", "ts_delta", "ts_delay",
    "ts_arg_max", "ts_arg_min", "ts_zscore", "ts_regression",
    "ts_quantile", "ts_decay_linear", "ts_backfill",
    "add", "subtract", "multiply", "divide", "power", "signed_power",
    "max", "min", "abs", "log", "sign", "if_else", "is_nan",
    "bucket", "trade_when",
    "market", "sector", "industry", "subindustry",
    "true", "false",
}

_FIELD_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------

def load_known_fields(fields_csv: str) -> set[str]:
    fields: set[str] = set()
    with open(fields_csv) as f:
        for row in csv.DictReader(f):
            fid = row.get("field_id", "").strip()
            if fid:
                fields.add(fid)
    return fields


def validate_expression(expr: str, known_fields: set[str]) -> tuple[bool, list[str]]:
    """Return (is_valid, unknown_tokens).

    Unknown tokens = identifier tokens that look like field names (contain '_' or
    known dataset prefix) but are not in known_fields and not in FASTEXPR keywords.
    """
    tokens = _FIELD_TOKEN_RE.findall(expr)
    unknown: list[str] = []
    for tok in tokens:
        if tok in _FASTEXPR_KEYWORDS:
            continue
        if tok.isdigit() or (tok.replace(".", "").isdigit()):
            continue
        if len(tok) <= 3:
            continue
        if "_" not in tok:
            continue
        if tok not in known_fields:
            unknown.append(tok)
    is_valid = len(unknown) == 0
    return is_valid, list(set(unknown))


# ---------------------------------------------------------------------------
# Sim log helpers
# ---------------------------------------------------------------------------

def load_all_expressions(sim_logs: list[str]) -> set[str]:
    done: set[str] = set()
    for path in sim_logs:
        for rec in load_sim_log(path):
            expr = rec.get("expression")
            if expr:
                done.add(expr)
    return done


def load_last_batch(enhance_sim_log: str, batch_tag: str) -> list[dict]:
    """Return records from the previous enhance batch (tagged by search_lane)."""
    records = load_sim_log(enhance_sim_log)
    return [r for r in records if r.get("search_lane") == batch_tag]


# ---------------------------------------------------------------------------
# Candidate builder
# ---------------------------------------------------------------------------

def make_enhance_candidate(
    expr: str,
    rationale: str,
    iteration: int,
    source: str,
) -> dict[str, Any]:
    cid = hashlib.sha1(expr.encode()).hexdigest()[:16]
    return {
        "candidate_id": cid,
        "template_id": f"enhance_iter{iteration}_{source}",
        "dataset_id": "",
        "category": "",
        "search_lane": f"enhance_iter{iteration}",
        "expression": expr,
        "fields": [],
        "group": "",
        "params": {"rationale": rationale, "source": source},
        "priority_score": 3.0,
        "settings_override": {},
    }


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------

def load_history(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        return json.load(f)


def save_history(path: str, history: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_enhance_loop(args: argparse.Namespace) -> None:
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("No OpenAI API key. Set OPENAI_API_KEY or use --api-key.")

    known_fields = load_known_fields(args.fields_csv) if args.fields_csv else set()
    enhance_agent = AlphaEnhanceAgent(api_key=api_key or "dummy", model=args.model)

    enhance_sim_log = args.enhance_sim_log
    history_file = args.history_file or str(_ARTIFACTS_DIR / "enhance_history.json")
    history = load_history(history_file)

    for iteration in range(1, args.n_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ENHANCE ITERATION {iteration}/{args.n_iterations}")
        print(f"{'='*60}")

        # ── 1. Load passing reps ──────────────────────────────────────────
        all_records: list[dict] = []
        for path in args.sim_log:
            all_records.extend(load_sim_log(path))
        if Path(enhance_sim_log).exists():
            all_records.extend(load_sim_log(enhance_sim_log))

        passing_reps = select_passing_reps(
            all_records,
            min_sharpe=args.min_sharpe,
            min_fitness=args.min_fitness,
            top_n=args.top_n,
        )
        print(f"Passing reps (best per family): {len(passing_reps)}")
        if not passing_reps:
            print("No passing alphas found. Stopping.")
            break

        # ── 2. Last batch results (for subsequent iterations) ─────────────
        last_batch_tag = f"enhance_iter{iteration - 1}"
        last_results = load_last_batch(enhance_sim_log, last_batch_tag) if iteration > 1 else None

        # ── 3. EnhanceAgent ───────────────────────────────────────────────
        print(f"Calling EnhanceAgent ({args.model}) ...")
        enhance_result = enhance_agent.enhance(passing_reps, history, last_results)
        history = enhance_result.updated_history
        print(f"  Suggested {len(enhance_result.expressions)} expressions")
        for expr in enhance_result.expressions:
            rat = enhance_result.rationale.get(expr, "")
            print(f"  + {expr[:80]}{'...' if len(expr)>80 else ''}")
            if rat:
                print(f"    → {rat}")

        # ── 4. Collect structural candidate expressions ──────────────────
        already_run = load_all_expressions(list(args.sim_log) + [enhance_sim_log])

        candidates: list[dict] = []
        seen_this_batch: set[str] = set()

        def add_candidate(expr: str, rationale: str, source: str) -> None:
            expr = expr.strip()
            if not expr or expr in already_run or expr in seen_this_batch:
                return
            if known_fields:
                valid, unknown = validate_expression(expr, known_fields)
                if unknown:
                    print(f"  [SKIP unknown fields {unknown}]: {expr[:60]}")
                    return
            seen_this_batch.add(expr)
            candidates.append(make_enhance_candidate(expr, rationale, iteration, source))

        for expr in enhance_result.expressions:
            add_candidate(expr, enhance_result.rationale.get(expr, ""), "enhance")

        print(f"\nNew candidates after dedup: {len(candidates)}")

        if not candidates:
            print("All suggestions already run. Saving history and stopping.")
            save_history(history_file, history)
            break

        # ── 5. Write plan ─────────────────────────────────────────────────
        plan_path = _ARTIFACTS_DIR / f"enhance_batch_{iteration}.jsonl"
        with plan_path.open("w") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")
        print(f"Plan written: {plan_path} ({len(candidates)} candidates)")

        # ── 6. Dry run: stop here ─────────────────────────────────────────
        if args.dry_run:
            print("\n[DRY RUN] Stopping before submission.")
            save_history(history_file, history)
            continue

        # ── 7. Submit ─────────────────────────────────────────────────────
        from .run_plan import run as run_plan
        print(f"\nSubmitting to Brain ({args.credentials}) ...")
        run_plan(
            str(plan_path),
            enhance_sim_log,
            credentials=args.credentials,
            batch_size=args.batch_size,
        )

        # ── 8. Save history with results turn ─────────────────────────────
        iter_results = load_last_batch(enhance_sim_log, f"enhance_iter{iteration}")
        if iter_results:
            results_summary = _build_results_summary(iter_results)
            history.append({"role": "user", "content": results_summary})

        save_history(history_file, history)
        print(f"History saved: {history_file}")

    print("\nEnhancement loop complete.")
    print(f"Results: python -m alpha_discovery.template_sweep_method.pipeline status --sim-log {enhance_sim_log}")


def _build_results_summary(results: list[dict]) -> str:
    lines = [f"Results from last iteration ({len(results)} candidates submitted):"]
    for rec in sorted(results, key=lambda r: float(r.get("is_sharpe") or 0.0), reverse=True):
        expr = rec.get("expression", "")[:80]
        sharpe = rec.get("is_sharpe")
        grade = rec.get("grade", "")
        checks = rec.get("checks") or []
        failing = [c["name"] for c in checks if c.get("result") == "FAIL"]
        if sharpe is not None:
            fail_str = f" ({', '.join(failing)})" if failing else ""
            lines.append(f'- "{expr}" → Sharpe={sharpe:.2f} {grade}{fail_str}')
        else:
            lines.append(f'- "{expr}" → N/A ({rec.get("status", "")})')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM enhancement loop: suggest → submit → feedback → repeat."
    )
    p.add_argument("--sim-log", nargs="+", required=True,
                   help="Sim log(s) to read passing alphas from (e.g. all_sim.jsonl)")
    p.add_argument("--enhance-sim-log",
                   default=str(_ARTIFACTS_DIR / "enhance_sim.jsonl"),
                   help="Sim log to append enhance results to (default: artifacts/enhance_sim.jsonl)")
    p.add_argument("--history-file", default="",
                   help="JSON file for conversation history (default: artifacts/enhance_history.json)")
    p.add_argument("--fields-csv", default="",
                   help="Path to all_fields.csv for field validation (optional but recommended)")
    p.add_argument("--n-iterations", type=int, default=3,
                   help="Number of enhance → submit → feedback iterations (default 3)")
    p.add_argument("--top-n", type=int, default=8,
                   help="Max passing alpha families to include in prompt (default 8)")
    p.add_argument("--min-sharpe", type=float, default=1.25)
    p.add_argument("--min-fitness", type=float, default=1.0)
    p.add_argument("--credentials", default="brain.txt",
                   help="Brain credentials file (default brain.txt)")
    p.add_argument("--batch-size", type=int, default=3,
                   help="Brain submission batch size (default 3)")
    p.add_argument("--api-key", default="",
                   help="OpenAI API key (default: OPENAI_API_KEY env var)")
    p.add_argument("--model", default="gpt-5.5",
                   help="LLM model label for EnhanceAgent (default: gpt-5.5)")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate plan and print suggestions without submitting to Brain")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_enhance_loop(args)


if __name__ == "__main__":
    main()
