"""Repair (negation + LLM) → Enhance → Final Batch pipeline.

Chains three stages in sequence:

  Stage 1 — Repair
    Scans grammar sim logs for near-misses (sharpe >= min_sharpe, failing checks)
    and strong-inverse signals (sharpe <= flip_threshold).  Generates sign-flip
    (negation) and LLM repair candidates, submits them to Brain.

  Stage 2 — Enhance
    Runs the LLM enhancement loop over ALL passing alphas: grammar originals +
    repair results (repair_sim_log fed into enhance --sim-log).

  Stage 3 — Select
    Compares the passing pool before and after enhancement.
    For each passing original:
      - If any enhanced alpha has higher Sharpe → recommend enhanced, show original as reference.
      - Else → keep original.
    Outputs final_batch.jsonl (submittable) and final_report.txt (human comparison table).

Usage:
    python -m alpha_multi.repair_enhance_pipeline \\
        --sim-log alpha_grammar/artifacts/all_sim.jsonl \\
        --fields-csv all_fields.csv \\
        --credentials brain.txt

Skip already-done stages (resume):
    python -m alpha_multi.repair_enhance_pipeline \\
        --sim-log alpha_grammar/artifacts/all_sim.jsonl \\
        --skip-repair --skip-enhance

Dry run (no Brain submission):
    python -m alpha_multi.repair_enhance_pipeline \\
        --sim-log alpha_grammar/artifacts/all_sim.jsonl \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ARTIFACTS_DIR = Path(__file__).parent.parent / "alpha_multi" / "artifacts"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _sharpe(rec: dict) -> float:
    return float(rec.get("is_sharpe") or 0.0)


def _fitness(rec: dict) -> float:
    return float(rec.get("is_fitness") or 0.0)


def _is_passing(rec: dict, min_sharpe: float, min_fitness: float) -> bool:
    """True if the record is a done simulation that passes all submission checks."""
    if rec.get("status") != "done":
        return False
    if _sharpe(rec) < min_sharpe or _fitness(rec) < min_fitness:
        return False
    checks = rec.get("checks") or []
    if any(c.get("result") == "FAIL" for c in checks):
        return False
    if not rec.get("is_concentrated_weight_ok", True):
        return False
    if not rec.get("is_sub_universe_sharpe_ok", True):
        return False
    return True


def _collect_passing(
    records: list[dict], min_sharpe: float, min_fitness: float
) -> list[dict]:
    """Return passing records, deduplicated by expression (highest Sharpe wins)."""
    best: dict[str, dict] = {}
    for rec in records:
        if not _is_passing(rec, min_sharpe, min_fitness):
            continue
        expr = rec.get("expression", "")
        if not expr:
            continue
        if expr not in best or _sharpe(rec) > _sharpe(best[expr]):
            best[expr] = rec
    return sorted(best.values(), key=_sharpe, reverse=True)


# ---------------------------------------------------------------------------
# Stage 1: Repair (negation + LLM)
# ---------------------------------------------------------------------------

def run_repair_stage(
    sim_logs: list[str],
    repair_plan_path: str,
    repair_sim_log: str,
    credentials: str,
    api_key: str,
    min_sharpe: float,
    batch_size: int,
    dry_run: bool,
) -> None:
    print("\n" + "=" * 60)
    print("STAGE 1: REPAIR (negation + LLM)")
    print("=" * 60)

    from wq_alpha.workshop.repair import build_repair_plan

    candidates = build_repair_plan(
        sim_log_paths=sim_logs,
        output_path=repair_plan_path,
        min_sharpe=min_sharpe,
        api_key=api_key or None,
    )
    print(f"Repair plan: {len(candidates)} candidates → {repair_plan_path}")

    if not candidates:
        print("No repair candidates found. Skipping submission.")
        return

    if dry_run:
        print("[DRY RUN] Skipping Brain submission for repair stage.")
        return

    from tools.alpha_multi_run_plan import run as run_plan
    run_plan(
        repair_plan_path,
        repair_sim_log,
        credentials=credentials,
        batch_size=batch_size,
    )
    print(f"Repair results → {repair_sim_log}")


# ---------------------------------------------------------------------------
# Stage 2: Enhance
# ---------------------------------------------------------------------------

def run_enhance_stage(
    sim_logs: list[str],
    repair_sim_log: str,
    enhance_sim_log: str,
    history_file: str,
    credentials: str,
    api_key: str,
    fields_csv: str,
    n_iterations: int,
    top_n: int,
    min_sharpe: float,
    min_fitness: float,
    batch_size: int,
    dry_run: bool,
) -> None:
    print("\n" + "=" * 60)
    print("STAGE 2: ENHANCE")
    print("=" * 60)

    # Feed repair results into enhance so repaired-and-passing alphas are included.
    all_input_logs = list(sim_logs)
    if Path(repair_sim_log).exists():
        all_input_logs.append(repair_sim_log)
        print(f"Repair sim log included: {repair_sim_log}")

    from tools.alpha_multi_enhance_loop import run_enhance_loop

    ns = Namespace(
        sim_log=all_input_logs,
        enhance_sim_log=enhance_sim_log,
        history_file=history_file,
        fields_csv=fields_csv,
        n_iterations=n_iterations,
        top_n=top_n,
        min_sharpe=min_sharpe,
        min_fitness=min_fitness,
        credentials=credentials,
        batch_size=batch_size,
        api_key=api_key,
        model="gpt-5.5",
        dry_run=dry_run,
    )
    run_enhance_loop(ns)
    print(f"Enhance results → {enhance_sim_log}")


# ---------------------------------------------------------------------------
# Stage 3: Final batch selection
# ---------------------------------------------------------------------------

def build_final_batch(
    sim_logs: list[str],
    repair_sim_log: str,
    enhance_sim_log: str,
    output_path: str,
    report_path: str,
    min_sharpe: float,
    min_fitness: float,
) -> list[dict]:
    """Compare pre-enhance and post-enhance passing pools and emit the final batch.

    Selection rule per original passing alpha:
      - If any enhanced alpha has higher Sharpe → recommend best enhanced, show
        original as reference ("compare") so the user can judge.
      - Otherwise → recommend original as-is.
    Enhanced alphas that have no matching original to beat are included as
    independent new discoveries.

    Returns list of final_batch records.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: FINAL BATCH SELECTION")
    print("=" * 60)

    # Load all records from each stage
    original_records: list[dict] = []
    for log in sim_logs:
        original_records.extend(_load_jsonl(log))
    repair_records = _load_jsonl(repair_sim_log)
    enhance_records = _load_jsonl(enhance_sim_log)

    # Separate passing pools
    passing_originals = _collect_passing(original_records, min_sharpe, min_fitness)
    passing_repaired = _collect_passing(repair_records, min_sharpe, min_fitness)
    passing_enhanced = _collect_passing(enhance_records, min_sharpe, min_fitness)

    print(f"Passing grammar:  {len(passing_originals)}")
    print(f"Passing repaired: {len(passing_repaired)}")
    print(f"Passing enhanced: {len(passing_enhanced)}")

    # ── Build repaired linkage: repaired_expr → original_expr ──────────────
    # repair candidates store their origin in params.original_expression.
    repair_linkage: dict[str, str] = {}  # repaired_expr → original_expr
    for rec in repair_records:
        repaired_expr = rec.get("expression", "")
        orig_expr = (rec.get("params") or {}).get("original_expression", "")
        if repaired_expr and orig_expr:
            repair_linkage[repaired_expr] = orig_expr

    # ── Best enhanced: global pool sorted by sharpe ──────────────────────────
    best_enhanced = passing_enhanced[0] if passing_enhanced else None
    best_enhanced_sharpe = _sharpe(best_enhanced) if best_enhanced else 0.0

    # ── Per-original decision ───────────────────────────────────────────────
    # Base pool = grammar originals + repaired (both are "before enhance").
    base_pool = passing_originals + [
        r for r in passing_repaired
        if r.get("expression", "") not in {o.get("expression") for o in passing_originals}
    ]
    base_pool.sort(key=_sharpe, reverse=True)

    # Track which enhanced alphas have been "claimed" as a recommendation.
    # Since we can't link enhanced → specific original, we recommend the best
    # enhanced globally whenever it beats an original. Each enhanced alpha is
    # only listed once in the final batch even if it beats multiple originals.
    claimed_enhanced: set[str] = set()

    final_batch: list[dict] = []
    seen_exprs: set[str] = set()

    def _append(rec: dict, role: str, source: str, note: str = "") -> None:
        expr = rec.get("expression", "")
        if not expr or expr in seen_exprs:
            return
        seen_exprs.add(expr)
        final_batch.append({
            "role": role,           # "submit" | "compare"
            "source": source,       # "grammar" | "repaired" | "enhanced"
            "expression": expr,
            "is_sharpe": rec.get("is_sharpe"),
            "is_fitness": rec.get("is_fitness"),
            "is_turnover": rec.get("is_turnover"),
            "grade": rec.get("grade", ""),
            "alpha_id": rec.get("alpha_id", ""),
            "candidate_id": rec.get("candidate_id", ""),
            "dataset_id": rec.get("dataset_id", ""),
            "original_expression": repair_linkage.get(expr, ""),
            "note": note,
        })

    for orig in base_pool:
        orig_sharpe = _sharpe(orig)
        orig_source = "repaired" if orig.get("expression") in repair_linkage else "grammar"

        # Find the best unclaimed enhanced alpha that beats this original.
        better_enhanced = next(
            (e for e in passing_enhanced
             if e.get("expression") not in claimed_enhanced
             and _sharpe(e) > orig_sharpe),
            None,
        )

        if better_enhanced:
            enh_expr = better_enhanced.get("expression", "")
            enh_sharpe = _sharpe(better_enhanced)
            delta = enh_sharpe - orig_sharpe
            claimed_enhanced.add(enh_expr)
            _append(
                better_enhanced, "submit", "enhanced",
                note=f"replaces original (Sharpe +{delta:.3f})",
            )
            _append(
                orig, "compare", orig_source,
                note=f"original; superseded by enhanced (Sharpe -{delta:.3f})",
            )
        else:
            _append(
                orig, "submit", orig_source,
                note="no better enhanced found",
            )

    # Any enhanced alphas not yet claimed are new discoveries (no matching original).
    for enh in passing_enhanced:
        if enh.get("expression") not in claimed_enhanced:
            _append(enh, "submit", "enhanced", note="new enhanced discovery")

    # Sort: submit rows first (by sharpe), then compare rows.
    submit_rows = sorted(
        [r for r in final_batch if r["role"] == "submit"],
        key=lambda r: float(r.get("is_sharpe") or 0.0),
        reverse=True,
    )
    compare_rows = [r for r in final_batch if r["role"] == "compare"]

    # Pair compare rows with their submit counterparts (by note matching).
    ordered: list[dict] = []
    for sub in submit_rows:
        ordered.append(sub)
        # Include the corresponding "compare" row immediately after its submit row.
        # The compare row was added when its enhanced counterpart was chosen.
        orig_expr = sub.get("original_expression") or ""
        paired = next(
            (c for c in compare_rows if c.get("expression") == orig_expr),
            None,
        )
        if not paired:
            # Enhanced that replaced an original: compare row expression is
            # the original expression stored in the submit row's note.
            # Alternative: scan compare_rows for entries whose enhanced pair matches sub.
            # Easiest: just append all compare rows at the end.
            pass
        else:
            ordered.append(paired)

    # Append any compare rows not already paired inline.
    paired_exprs = {c.get("expression") for c in ordered if c["role"] == "compare"}
    for c in compare_rows:
        if c.get("expression") not in paired_exprs:
            ordered.append(c)

    # ── Write final_batch.jsonl ─────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in ordered:
            f.write(json.dumps(rec) + "\n")

    # ── Write human-readable report ─────────────────────────────────────────
    n_submit = sum(1 for r in ordered if r["role"] == "submit")
    n_enhanced_wins = sum(
        1 for r in ordered if r["role"] == "submit" and r["source"] == "enhanced"
    )
    n_originals_kept = sum(
        1 for r in ordered if r["role"] == "submit" and r["source"] != "enhanced"
    )

    lines: list[str] = [
        "=" * 80,
        "REPAIR → ENHANCE  FINAL BATCH REPORT",
        "=" * 80,
        f"  Grammar passing:       {len(passing_originals)}",
        f"  Repaired passing:      {len(passing_repaired)}",
        f"  Enhanced passing:      {len(passing_enhanced)}",
        f"  ─────────────────────────────────",
        f"  Total submittable:     {n_submit}",
        f"    Enhanced wins:       {n_enhanced_wins}  (enhanced beat original)",
        f"    Originals kept:      {n_originals_kept}  (no better enhanced found)",
        "",
    ]

    if not passing_enhanced:
        lines += [
            "  Enhancement stage produced NO passing alphas.",
            "  All recommendations are grammar or repair originals.",
            "",
        ]
    elif n_enhanced_wins == 0:
        lines += [
            f"  Best enhanced Sharpe ({best_enhanced_sharpe:.3f}) did not beat any original.",
            "  All recommendations are grammar or repair originals.",
            "",
        ]
    else:
        lines += [
            f"  Enhancement improved {n_enhanced_wins} slot(s).",
            f"  Best enhanced Sharpe: {best_enhanced_sharpe:.3f}",
            "",
        ]

    lines += [
        "─" * 80,
        "SUBMITTABLE BATCH  (role=submit, sorted by IS Sharpe)",
        "─" * 80,
        f"  {'#':>3}  {'Sharpe':>6}  {'Fitness':>7}  {'Source':>9}  {'Note':<30}  Expression",
        "─" * 80,
    ]

    rank = 0
    for rec in ordered:
        if rec["role"] == "submit":
            rank += 1
            sharpe_s = f"{float(rec['is_sharpe'] or 0):.3f}"
            fit_s = f"{float(rec['is_fitness'] or 0):.3f}"
            note = rec.get("note", "")[:30]
            expr = rec.get("expression", "")[:55]
            lines.append(
                f"  {rank:>3}  {sharpe_s:>6}  {fit_s:>7}  {rec['source']:>9}  {note:<30}  {expr}"
            )
        else:  # compare
            sharpe_s = f"{float(rec['is_sharpe'] or 0):.3f}"
            note = rec.get("note", "")[:30]
            expr = rec.get("expression", "")[:55]
            lines.append(
                f"  {'↑ orig':>7}  {sharpe_s:>6}            {'compare':>9}  {note:<30}  {expr}"
            )

    lines += [
        "",
        "─" * 80,
        f"JSONL:  {output_path}",
        f"REPORT: {report_path}",
        "=" * 80,
    ]

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    return ordered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repair → Enhance → Final Batch pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs
    p.add_argument("--sim-log", nargs="+", required=True,
                   help="Grammar sim log(s) to scan for near-misses and passing alphas.")
    p.add_argument("--fields-csv", default="",
                   help="all_fields.csv for field validation in enhance stage.")
    # Outputs / intermediate paths
    p.add_argument("--output-dir", default=str(_ARTIFACTS_DIR),
                   help="Directory for all pipeline outputs.")
    p.add_argument("--repair-plan", default="",
                   help="Repair plan JSONL (default: <output-dir>/repair_plan.jsonl).")
    p.add_argument("--repair-sim-log", default="",
                   help="Repair simulation log (default: <output-dir>/repair_sim.jsonl).")
    p.add_argument("--enhance-sim-log", default="",
                   help="Enhance simulation log (default: <output-dir>/pipeline_enhance_sim.jsonl).")
    p.add_argument("--enhance-history", default="",
                   help="Enhance conversation history JSON (default: <output-dir>/pipeline_enhance_history.json).")
    p.add_argument("--final-batch", default="",
                   help="Final batch JSONL (default: <output-dir>/final_batch.jsonl).")
    p.add_argument("--final-report", default="",
                   help="Final report text (default: <output-dir>/final_report.txt).")
    # Brain / API credentials
    p.add_argument("--credentials", default="brain.txt",
                   help="Brain credentials file.")
    p.add_argument("--api-key", default="",
                   help="OpenAI API key (default: OPENAI_API_KEY env var).")
    # Thresholds
    p.add_argument("--min-sharpe", type=float, default=1.25,
                   help="Minimum IS Sharpe to count as passing / near-miss.")
    p.add_argument("--min-fitness", type=float, default=1.0,
                   help="Minimum IS Fitness to count as passing.")
    # Enhance settings
    p.add_argument("--n-enhance-iterations", type=int, default=2,
                   help="Number of enhance → submit → feedback iterations.")
    p.add_argument("--enhance-top-n", type=int, default=8,
                   help="Max passing alpha families fed to the enhance agent.")
    # Submission settings
    p.add_argument("--batch-size", type=int, default=3,
                   help="Brain submission batch size.")
    # Flow control
    p.add_argument("--skip-repair", action="store_true",
                   help="Skip stage 1 (repair already done; reuse existing repair_sim_log).")
    p.add_argument("--skip-enhance", action="store_true",
                   help="Skip stage 2 (enhance already done; reuse existing enhance_sim_log).")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate plans and print suggestions without submitting to Brain.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repair_plan = args.repair_plan or str(out_dir / "repair_plan.jsonl")
    repair_sim = args.repair_sim_log or str(out_dir / "repair_sim.jsonl")
    enhance_sim = args.enhance_sim_log or str(out_dir / "pipeline_enhance_sim.jsonl")
    enhance_hist = args.enhance_history or str(out_dir / "pipeline_enhance_history.json")
    final_batch = args.final_batch or str(out_dir / "final_batch.jsonl")
    final_report = args.final_report or str(out_dir / "final_report.txt")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        _key_file = Path(__file__).parent.parent / "openaiapi.txt"
        if _key_file.exists():
            api_key = _key_file.read_text().strip()

    # ── Stage 1: Repair ──────────────────────────────────────────────────────
    if args.skip_repair:
        print("\n[SKIP] Stage 1 — using existing repair_sim_log:", repair_sim)
    else:
        run_repair_stage(
            sim_logs=args.sim_log,
            repair_plan_path=repair_plan,
            repair_sim_log=repair_sim,
            credentials=args.credentials,
            api_key=api_key,
            min_sharpe=args.min_sharpe,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

    # ── Stage 2: Enhance ─────────────────────────────────────────────────────
    if args.skip_enhance:
        print("\n[SKIP] Stage 2 — using existing enhance_sim_log:", enhance_sim)
    else:
        run_enhance_stage(
            sim_logs=args.sim_log,
            repair_sim_log=repair_sim,
            enhance_sim_log=enhance_sim,
            history_file=enhance_hist,
            credentials=args.credentials,
            api_key=api_key,
            fields_csv=args.fields_csv,
            n_iterations=args.n_enhance_iterations,
            top_n=args.enhance_top_n,
            min_sharpe=args.min_sharpe,
            min_fitness=args.min_fitness,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

    # ── Stage 3: Final batch ─────────────────────────────────────────────────
    build_final_batch(
        sim_logs=args.sim_log,
        repair_sim_log=repair_sim,
        enhance_sim_log=enhance_sim,
        output_path=final_batch,
        report_path=final_report,
        min_sharpe=args.min_sharpe,
        min_fitness=args.min_fitness,
    )


if __name__ == "__main__":
    main()
