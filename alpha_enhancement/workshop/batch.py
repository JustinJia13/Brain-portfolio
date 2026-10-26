"""Batch auto-workshop over entire simulation history.

Scans sim logs, selects the best representative alpha per field (dedup), then
routes each record through the shared workshop policy from alpha_enhancement/workshop.core.

Shared routing rules:
  negate   — fitness <= -1.2, excluding concentration >= 40%
  repair   — repairable checks fail, concentration is below 40%, and
              default batch selection has fetched a positive Brain score delta
  enhance  — all submission checks pass, correlation-eligible, and
              default batch selection has fetched a positive Brain score delta
  skip     — everything else

Records with no max_self_correlation metadata are still eligible for workshop
routing; final template promotion uses the portfolio admission gate.
Audit previews include a raw `corr` column from `max_self_correlation`.

Dedup: one representative per primary field (highest |Sharpe| kept).
Exception: if a field has both a negate AND a repair/enhance candidate, keep both.

Usage:
    # Dry-run: show tier counts and candidate list, then exit
    python tools/batch_workshop.py --dry-run

    # Preview list only (shows table, then runs)
    python tools/batch_workshop.py --list-only

    # Full auto (default: up to 200 candidates, 3 workshop iterations)
    python tools/batch_workshop.py

    # More iterations, smaller batch
    python tools/batch_workshop.py --max-candidates 50 --workshop-iterations 2

    # Exclude externally-submitted alphas
    python tools/batch_workshop.py \\
        --exclude-expr "ts_rank(operating_income/cap,250)" \\
                       "group_backfill(kth_element(kth_element(...), 60, k=2), market, 10)"

    # Negations only (no LLM cost)
    python tools/batch_workshop.py --lane negate

    # Grammar-targeted (human-checkpoint workflow):
    #   Step 1 — generate audit from grammar results only
    python tools/batch_workshop.py --source grammar --dry-run
    #   Step 2 — human edits artifacts/workshop/grammar_workshop_audit.txt
    #   Step 3 — run from edited audit
    python tools/batch_workshop.py --source grammar --from-audit artifacts/workshop/grammar_workshop_audit.txt
    #   Step 4 — review workshop_submittable.jsonl and submit promoted alphas individually
    #   Step 5 — clean up intermediate files
    python tools/batch_workshop.py --cleanup
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alpha_enhancement.workshop.core import (
    generate_candidates,
    failing_checks,
    is_passing,
    route_record,
    record_regime,
    record_workshop_key,
    regime_from_settings,
    source_settings_override,
    workshop_expr_key,
    collect_trial_history,
    find_passing_examples,
    _make_candidate,
    _looks_like_fastexpr,
)
from alpha_enhancement.workshop.repair import _outermost_call
from alpha_enhancement.workshop.records import load_records
from alpha_enhancement.workshop.scoring import (
    prescore_seeds,
    reroute_after_seed_scoring,
    with_portfolio_prompt_feedback,
)
from alpha_enhancement.workshop.simulation import submit_candidate_batch
from core.logs import merge_session_to_master
from core.robustness import record_robustness_score, reward_multiplier

DEFAULT_SIM_GLOBS = [
    str(ROOT / "*" / "artifacts" / "*sim*.jsonl"),
    str(ROOT / "artifacts" / "*sim*.jsonl"),
    str(ROOT / "artifacts" / "sim" / "master.jsonl"),
    str(ROOT / "artifacts" / "sim" / "sessions" / "*.jsonl"),
]
GRAMMAR_SIM_LOG = ROOT / "alpha_discovery/grammar_method" / "artifacts" / "all_sim.jsonl"
GRAMMAR_SIM_GLOBS = [
    str(ROOT / "artifacts" / "sim" / "master.jsonl"),
    str(ROOT / "artifacts" / "sim" / "sessions" / "grammar_*.jsonl"),
    str(GRAMMAR_SIM_LOG),
]

WORKSHOP_DIR = ROOT / "artifacts" / "workshop"
DEFAULT_WORKSHOP_SIM = WORKSHOP_DIR / "workshop_sim.jsonl"

# Expanded glob covers both current dir and old numbered run dirs
WORKSHOP_SIM_GLOBS = [
    str(WORKSHOP_DIR / "*sim*.jsonl"),
    str(ROOT / "artifacts" / "workshop_run*" / "*sim*.jsonl"),
]
WORKSHOPPED_SOURCES_FILE = WORKSHOP_DIR / "workshopped_sources.txt"
REVIEWED_SOURCES_FILE = WORKSHOP_DIR / "reviewed_sources.jsonl"

WORKSHOP_AUDIT_TXT     = WORKSHOP_DIR / "workshop_audit.txt"
WORKSHOP_AUDIT_JSONL   = WORKSHOP_DIR / "workshop_audit.jsonl"
WORKSHOP_SUMMARY_TXT   = WORKSHOP_DIR / "workshop_summary.txt"
NEW_TEMPLATES_JSONL    = WORKSHOP_DIR / "new_templates.jsonl"
SUBMITTABLE_JSONL      = WORKSHOP_DIR / "workshop_submittable.jsonl"
WORKSHOP_ADMISSION_JSONL = WORKSHOP_DIR / "workshop_admission.jsonl"

# Ephemeral files deleted by --cleanup (sim logs and persistent tracking are kept)
_CLEANUP_FILES = [
    WORKSHOP_AUDIT_TXT, WORKSHOP_AUDIT_JSONL,
    WORKSHOP_SUMMARY_TXT, NEW_TEMPLATES_JSONL, SUBMITTABLE_JSONL, WORKSHOP_ADMISSION_JSONL,
    WORKSHOP_DIR / "grammar_workshop_audit.txt",
    WORKSHOP_DIR / "grammar_workshop_audit.jsonl",
]

# Fields to exclude entirely (known to produce noisy/uninformative candidates)
EXCLUDED_FIELDS: set[str] = {
    "split",
    "anl4_adjusted_netincome",  # already submitted externally
}

# Expressions to exclude by default (already submitted externally or manually removed)
DEFAULT_EXCLUDED_EXPRS: set[str] = {
    # externally submitted
    "group_backfill(kth_element(kth_element(anl4_adjusted_netincome_ft, 40, k=4), 60, k=2), market, 10)",
    "ts_rank(operating_income/cap,250)",
    # manually removed from audit
    "group_backfill(kth_element(kth_element(anl4_ptp_flag, 40, k=3), 60, k=2), market, 10)",
    "normalize((2*anl4_totassets_flag + anl4_ffo_flag + 1.19*anl4_bvps_flag + 0.6*anl4_netprofit_flag + 0.5*anl4_netdebt_flag + 0.4*anl4_capex_flag + 0.3*anl4_rd_exp_flag + 0.2*if_else(is_nan(anl4_tot_gw_ft), 0, anl4_tot_gw_ft)) * rank(-ts_mean(returns, 60)))",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_records(paths: list[str], *, tag_source: bool = False) -> list[dict]:
    return load_records(paths, tag_source=tag_source)


def _resolve_input_paths(args) -> list[str]:
    """Resolve input sim-log paths for selection and from-audit replay."""
    if args.sim_log:
        paths: list[str] = []
        for pattern in args.sim_log:
            expanded = glob.glob(pattern)
            paths.extend(expanded if expanded else [pattern])
        return _dedupe_paths(paths)
    if args.source == "grammar":
        paths: list[str] = []
        for pattern in GRAMMAR_SIM_GLOBS:
            paths.extend(glob.glob(pattern))
        return _dedupe_paths(paths)

    paths = []
    for pattern in DEFAULT_SIM_GLOBS:
        paths.extend(glob.glob(pattern))
    return _dedupe_paths(paths)


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        key = str(Path(path).resolve()) if Path(path).exists() else path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _is_grammar_record(record: dict) -> bool:
    """Return True for records produced by the grammar generator."""
    lane = str(record.get("search_lane") or "").lower()
    if lane == "grammar":
        return True
    family = str(record.get("family_id") or "")
    if family.startswith("grammar|"):
        return True
    intention = str(record.get("intention_id") or "")
    if intention.startswith("grammar"):
        return True

    source_path = str(record.get("_source_path") or "")
    if not source_path:
        return False
    path = Path(source_path)
    if path.name.startswith("grammar_"):
        return True
    return "alpha_discovery/grammar_method" in path.parts and path.name == "all_sim.jsonl"


def _filter_records_for_source(records: list[dict], args) -> list[dict]:
    if args.source != "grammar" or args.sim_log:
        return records
    return [record for record in records if _is_grammar_record(record)]


def _load_workshopped_sources() -> set[str]:
    """Return regime-aware source keys already processed or reviewed."""
    exprs = _load_processed_source_keys()
    exprs.update(_load_reviewed_source_keys())
    return exprs


def _load_processed_source_keys() -> set[str]:
    """Return regime-aware source keys already processed by any workshop run."""
    exprs: set[str] = set()
    # Scan workshop sim logs (current dir + old numbered run dirs)
    for pat in WORKSHOP_SIM_GLOBS:
        for r in _load_records(glob.glob(pat)):
            orig = (r.get("params") or {}).get("original_expression", "")
            if orig:
                exprs.add(workshop_expr_key(orig, record_regime(r)))
            expr = r.get("expression", "")
            if expr and not _is_negate_recovery_record(r):
                exprs.add(record_workshop_key(r))
    # Persistent text file (survives log rotation / cleanup)
    if WORKSHOPPED_SOURCES_FILE.exists():
        exprs.update(
            line.strip()
            for line in WORKSHOPPED_SOURCES_FILE.read_text().splitlines()
            if line.strip()
        )
    return exprs


def _is_negate_recovery_record(record: dict) -> bool:
    """Return true for first-stage negation outputs.

    A negation simulation means we recovered the sign of a raw negative alpha;
    the original source should be marked processed, but the generated negated
    alpha should remain eligible for later semantic repair/enhancement.  If a
    negated alpha is later workshopped, that later repair/enhance run will mark
    it through normal source tracking.
    """
    if record.get("search_lane") == "negate":
        return True
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    root = str(params.get("root_expression") or "")
    original = str(params.get("original_expression") or "")
    return bool(original) and root == original and str(record.get("template_family") or "") == "negate"


def _is_seedable_promoted_semantic_robustness(record: dict) -> bool:
    """Allow positive-score robustness outputs to become semantic seeds.

    Robustness-card rows are normally marked ``skip_workshop`` so diagnostic
    perturbations do not create an automatic tuning loop.  When one of those
    rows receives a positive Brain score delta, it is no longer just diagnostic
    evidence; it is a recovered candidate that can be repaired/enhanced later.
    """
    if record.get("search_lane") != "semantic_robustness":
        return False
    if record.get("admission_decision") != "PROMOTE":
        return False
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    for value in (
        record.get("admission_brain_score_delta"),
        record.get("brain_score_delta"),
        admission.get("brain_score_delta"),
    ):
        if value is None:
            continue
        try:
            return float(value) > 0.0
        except (TypeError, ValueError):
            continue
    return False


def _save_workshopped_sources(new_exprs: set[str]) -> None:
    """Append regime-aware source keys to the persistent workshopped file."""
    WORKSHOPPED_SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_processed_source_keys()
    merged = existing | new_exprs
    WORKSHOPPED_SOURCES_FILE.write_text("\n".join(sorted(merged)) + "\n")


def _load_reviewed_source_keys() -> set[str]:
    keys: set[str] = set()
    if not REVIEWED_SOURCES_FILE.exists():
        return keys
    with REVIEWED_SOURCES_FILE.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if isinstance(key, str) and key:
                keys.add(key)
    return keys


def _append_reviewed_entries(entries: list[dict]) -> None:
    if not entries:
        return
    REVIEWED_SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    known = _load_reviewed_source_keys()
    now = datetime.now(timezone.utc).isoformat()
    with REVIEWED_SOURCES_FILE.open("a") as handle:
        for entry in entries:
            key = entry.get("key")
            if not key or key in known:
                continue
            row = dict(entry)
            row["reviewed_at"] = now
            handle.write(json.dumps(row) + "\n")
            known.add(key)


def _review_entry_from_audit_row(row: dict, *, decision: str, reason: str) -> dict | None:
    expr = row.get("expression", "")
    if not expr:
        return None
    regime = str(row.get("regime") or regime_from_settings(row.get("settings")))
    return {
        "key": workshop_expr_key(expr, regime),
        "expression": expr,
        "regime": regime,
        "alpha_id": row.get("alpha_id"),
        "route": row.get("action"),
        "decision": decision,
        "reason": reason,
        "logged_at": row.get("logged_at"),
        "source": "audit",
    }


def _review_entries_for_context(row: dict, *, decision: str, reason: str) -> list[dict]:
    entries: list[dict] = []
    for key in _audit_context_keys(row):
        regime, _, expr = key.partition("\t")
        entries.append({
            "key": key,
            "expression": expr,
            "regime": regime,
            "route": row.get("action"),
            "decision": decision,
            "reason": reason,
            "source": "audit_context",
        })
    return entries


def _extract_first_field(expr: str) -> str:
    """Extract the first identifier-looking leaf from a FASTEXPR string.

    Used as fallback when the record's `fields` list is empty.
    Tries to find a word that looks like a field name (contains underscore or digits,
    not a known operator/keyword).
    """
    _OPERATORS = {
        "rank", "scale", "zscore", "winsorize", "group_rank", "group_zscore",
        "group_neutralize", "group_scale", "group_backfill", "ts_mean", "ts_std",
        "ts_zscore", "ts_rank", "ts_delta", "ts_decay_linear", "ts_arg_max",
        "ts_sum", "ts_regression", "ts_corr", "kth_element", "if_else", "is_nan",
        "sign", "log", "inverse", "abs", "normalize", "bucket", "hump",
        "sector", "subindustry", "industry", "market",
    }
    for token in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr):
        if token.lower() not in _OPERATORS and ('_' in token or any(c.isdigit() for c in token)):
            return token
    # Fallback: first identifier
    m = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
    return m.group() if m else expr[:20]


def _display_fields(record: dict) -> str:
    """Return raw field names for display.

    Uses record["fields"] directly when populated.
    Falls back to extracting the first non-operator identifier from the expression,
    accepting any identifier (including ones without underscores like 'ppent', 'assets').
    """
    fields = record.get("fields") or []
    if fields:
        return "+".join(fields)
    expr = record.get("expression", "")
    _OPERATORS = {
        "rank", "scale", "zscore", "winsorize", "normalize", "quantile",
        "group_rank", "group_zscore", "group_neutralize", "group_scale",
        "group_backfill", "group_mean",
        "ts_mean", "ts_std", "ts_zscore", "ts_rank", "ts_delta",
        "ts_decay_linear", "ts_decay_exp", "ts_arg_max", "ts_arg_min",
        "ts_sum", "ts_regression", "ts_corr", "ts_covariance", "ts_backfill",
        "ts_count_nans", "ts_product", "ts_delay", "ts_scale", "ts_step",
        "ts_av_diff", "ts_std_dev", "ts_quantile",
        "kth_element", "if_else", "is_nan", "sign", "log", "inverse",
        "abs", "sqrt", "power", "signed_power", "hump", "bucket",
        "not", "min", "max", "multiply", "divide", "subtract", "add",
        "last_diff_value", "days_from_last_change", "reverse",
        "trade_when", "rank_by_side",
        "sector", "subindustry", "industry", "market",
    }
    for token in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr):
        if token.lower() not in _OPERATORS:
            return token
    return expr[:20]


def _make_field_signature(record: dict) -> str:
    """Return a field signature for dedup.

    Single-field expressions → stripped field name (e.g. "anl4_eps_flag").
    Multi-field expressions  → sorted stripped names joined with '+' (e.g. "field_a+field_b").
    Trailing numeric/type suffixes (_ft, _low, _high, _mean, _median, _digits) are stripped.
    """
    fields = record.get("fields") or []
    if not fields:
        fields = [_extract_first_field(record.get("expression", ""))]
    roots = []
    for f in fields:
        root = re.sub(r'(_ft|_low|_high|_mean|_median|_[0-9]+)$', '', f.lower())
        roots.append(root)
    return "+".join(sorted(set(roots)))


def _resolved_route(record: dict, *, bypass_self_corr: bool = False) -> str | None:
    """Return cached route when present, else evaluate via the shared router."""
    cached = record.get("_route")
    if isinstance(cached, str) and not bypass_self_corr:
        return cached
    return route_record(record, bypass_self_corr=bypass_self_corr)


def _without_admission_scores(record: dict) -> dict:
    stripped = dict(record)
    for key in (
        "admission_brain_score_delta",
        "brain_score_delta",
        "admission_delta_pool_sharpe",
    ):
        stripped.pop(key, None)
    admission = stripped.get("admission")
    if isinstance(admission, dict):
        admission = dict(admission)
        admission.pop("brain_score_delta", None)
        admission.pop("delta_pool_sharpe", None)
        stripped["admission"] = admission
    return stripped


def _selection_route(
    record: dict,
    *,
    bypass_self_corr: bool = False,
    ignore_admission_scores: bool = False,
) -> str | None:
    if ignore_admission_scores:
        record = _without_admission_scores(record)
    return route_record(record, bypass_self_corr=bypass_self_corr)


def _tier(route: str, sharpe: float) -> int:
    if route == "negate":
        return 1
    if route == "repair":
        return 2
    if route == "enhance":
        return 3 if sharpe <= 1.5 else 4
    return 99


# ---------------------------------------------------------------------------
# Main selection logic
# ---------------------------------------------------------------------------

def select_candidates(
    all_records: list[dict],
    workshopped_exprs: set[str],
    excluded_exprs: set[str],
    *,
    lane_filter: list[str] | None = None,
    max_candidates: int = 200,
    verbose: bool = False,
    bypass_self_corr: bool = False,
    ignore_admission_scores: bool = False,
) -> list[dict]:
    """Select best representative per field, filtered and prioritised."""

    # 1. Exact-dedup: keep highest-|Sharpe| done record per expression+regime
    by_expr: dict[str, dict] = {}
    for r in all_records:
        if r.get("status") != "done":
            continue
        if r.get("skip_workshop") and not _is_seedable_promoted_semantic_robustness(r):
            continue
        if r.get("search_lane") == "regime_check":
            continue
        expr_key = record_workshop_key(r)
        if not r.get("expression", ""):
            continue
        existing = by_expr.get(expr_key)
        if existing is None or abs(r.get("is_sharpe") or 0) > abs(existing.get("is_sharpe") or 0):
            by_expr[expr_key] = r

    unique = list(by_expr.values())
    if verbose:
        print(f"  Unique done expressions: {len(unique)}")

    # 2. Filter: exclude already workshopped + externally submitted
    skipped_done = sum(1 for r in unique if record_workshop_key(r) in workshopped_exprs)
    unique = [r for r in unique if record_workshop_key(r) not in workshopped_exprs and
              r.get("expression", "") not in excluded_exprs]
    if verbose:
        print(f"  After excluding already-workshopped: {len(unique)}  (skipped {skipped_done})")

    # 3. Filter: skip excluded fields (by signature or by token in expression)
    _excl_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(f) for f in EXCLUDED_FIELDS) + r')\b'
    )

    def _has_excluded_field(r: dict) -> bool:
        if _make_field_signature(r) in EXCLUDED_FIELDS:
            return True
        expr = r.get("expression", "")
        return bool(_excl_pattern.search(expr))

    skipped_field = sum(1 for r in unique if _has_excluded_field(r))
    unique = [r for r in unique if not _has_excluded_field(r)]
    if verbose:
        print(f"  After excluded-fields filter: {len(unique)}  (skipped {skipped_field} excluded-field)")

    # 4. Route each record
    routed: list[tuple[str, dict]] = []
    for r in unique:
        action = _selection_route(
            r,
            bypass_self_corr=bypass_self_corr,
            ignore_admission_scores=ignore_admission_scores,
        )
        if action == "negate" and _is_negate_recovery_record(r):
            continue
        if action and (lane_filter is None or action in lane_filter):
            routed.append((action, r))

    if verbose:
        from collections import Counter
        counts = Counter(a for a, _ in routed)
        print(f"  Routed: {dict(counts)}")

    # 5. Field-based dedup: 1 representative per sig (highest IS Fitness).
    #    context_records = all other variants for that sig, sorted best→worst Fitness
    #    (including failures), so the LLM sees the full intra-family gradient.
    from collections import defaultdict
    field_buckets: dict[tuple[str, str], list[tuple[float, dict]]] = defaultdict(list)

    for action, r in routed:
        sig = (_make_field_signature(r), record_regime(r))
        fitness = r.get("is_fitness") or 0.0
        field_buckets[sig].append((fitness, r))

    selected: list[dict] = []
    for sig, entries in field_buckets.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        winner = dict(entries[0][1])
        winner["_route"] = _selection_route(
            winner,
            bypass_self_corr=bypass_self_corr,
            ignore_admission_scores=ignore_admission_scores,
        )
        others = [r for _, r in entries[1:9]]   # up to 8, best→worst fitness
        if others:
            winner["context_records"] = others
        selected.append(winner)

    if verbose:
        print(f"  After field dedup: {len(selected)} unique fields")

    # 6. Sort by tier then |Sharpe| desc
    def sort_key(r: dict) -> tuple[int, float]:
        action = r.get("_route") or "skip"
        sharpe = r.get("is_sharpe") or 0.0
        return (_tier(action, sharpe), -abs(sharpe))

    selected.sort(key=sort_key)

    # 7. Apply max cap
    if max_candidates > 0:
        selected = selected[:max_candidates]

    return selected


def _candidate_cap(value: int) -> int:
    return value if value > 0 else 999999


def _select_candidates_for_args(
    all_records: list[dict],
    workshopped_exprs: set[str],
    excluded_exprs: set[str],
    args,
) -> list[dict]:
    """Select candidates with an independent negate budget."""
    requested = list(args.lane or ["negate", "repair", "enhance"])
    selected: list[dict] = []

    non_negate_lanes = [lane for lane in requested if lane in {"repair", "enhance"}]
    if non_negate_lanes:
        print("\nSelecting repair/enhance candidates...")
        selected.extend(select_candidates(
            all_records,
            workshopped_exprs,
            excluded_exprs,
            lane_filter=non_negate_lanes,
            max_candidates=_candidate_cap(args.max_candidates),
            verbose=True,
            bypass_self_corr=args.bypass_correlation_check,
            ignore_admission_scores=bool(args.score_seeds),
        ))

    if "negate" in requested:
        print("\nSelecting negate candidates...")
        selected.extend(select_candidates(
            all_records,
            workshopped_exprs,
            excluded_exprs,
            lane_filter=["negate"],
            max_candidates=_candidate_cap(args.max_negate_candidates),
            verbose=True,
            bypass_self_corr=args.bypass_correlation_check,
            ignore_admission_scores=bool(args.score_seeds),
        ))

    deduped: list[dict] = []
    seen: set[str] = set()
    for record in selected:
        key = record_workshop_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


# ---------------------------------------------------------------------------
# Audit file helpers
# ---------------------------------------------------------------------------

def _write_audit(
    selected: list[dict],
    txt_path: Path,
    jsonl_path: Path,
) -> None:
    """Write human-readable txt and machine-readable JSONL audit side by side."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"  {'tier':<6} {'action':<8} {'regime':<26} {'sharpe':>7} {'fitness':>7} {'23fit':>7} "
        f"{'corr':>7} {'field':<30} {'failing':<30}  expression\n"
    )
    divider = "  " + "─" * 166 + "\n"
    lines = [header, divider]
    jsonl_rows = []

    for r in selected:
        action = _resolved_route(r) or "?"
        sharpe = r.get("is_sharpe") or 0.0
        fitness = r.get("is_fitness") or 0.0
        regime = record_regime(r)
        oos23_fit = r.get("oos_fitness_2023")
        oos23_str = f"{oos23_fit:>7.3f}" if oos23_fit is not None else f"{'N/A':>7}"
        corr = r.get("max_self_correlation")
        corr_str = f"{corr:>7.3f}" if corr is not None else f"{'N/A':>7}"
        field_disp = _display_fields(r)
        failing = ",".join(failing_checks(r))
        expr = r.get("expression") or ""
        tier = _tier(action, sharpe)
        lines.append(
            f"  {tier:<6} {action:<8} {regime:<26} {sharpe:>7.3f} {fitness:>7.3f} {oos23_str} "
            f"{corr_str} {field_disp:<30} {failing:<30}  {expr}\n"
        )
        jsonl_rows.append({
            "tier": tier,
            "action": action,
            "regime": regime,
            "sharpe": sharpe,
            "fitness": fitness,
            "oos_fitness_2023": oos23_fit,
            "max_self_correlation": corr,
            "field": field_disp,
            "failing_checks": failing_checks(r),
            "expression": r.get("expression", ""),
            "context_records_meta": [
                {
                    "expression": c.get("expression", ""),
                    "regime": record_regime(c),
                }
                for c in (r.get("context_records") or [])
            ],
            "context_expressions": [
                c.get("expression", "") for c in (r.get("context_records") or [])
            ],
        })

    txt_path.write_text("".join(lines))
    with jsonl_path.open("w") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n  Audit written to:\n    {txt_path}\n    {jsonl_path}")


def _read_from_audit(txt_path: Path, jsonl_path: Path) -> list[dict]:
    """Parse human-edited txt audit + companion JSONL → list of full candidate dicts.

    Any row deleted from the txt will be absent from the result.
    Matching is by (action, regime, field, sharpe).
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Audit txt not found: {txt_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Audit JSONL not found: {jsonl_path}\n"
            f"Re-run --dry-run to regenerate."
        )

    # Parse surviving rows from txt
    kept: set[tuple[str, str, str, float]] = set()
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        # Formats:
        #   new: tier action regime sharpe fitness 23fit corr field ...
        #   old: tier action sharpe fitness 23fit corr field ...
        if len(parts) < 7:
            continue
        try:
            _tier_val = int(parts[0])
            action = parts[1]
            if parts[2].startswith("delay"):
                regime = parts[2]
                sharpe = float(parts[3])
                field = parts[7]
            else:
                regime = regime_from_settings(None)
                sharpe = float(parts[2])
                field = parts[6]
            kept.add((action, regime, field, round(sharpe, 3)))
        except (ValueError, IndexError):
            continue

    # Load JSONL and filter to kept rows
    selected = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (
                row["action"],
                str(row.get("regime") or regime_from_settings(row.get("settings"))),
                row["field"],
                round(row["sharpe"], 3),
            )
            if key in kept:
                selected.append(row)

    return selected


def _audit_row_key(row: dict) -> str:
    regime = str(row.get("regime") or regime_from_settings(row.get("settings")))
    return workshop_expr_key(row.get("expression", ""), regime)


def _audit_context_keys(row: dict) -> list[str]:
    meta = row.get("context_records_meta") or []
    if meta:
        out: list[str] = []
        for entry in meta:
            expr = entry.get("expression", "")
            if not expr:
                continue
            regime = str(entry.get("regime") or row.get("regime") or "")
            out.append(workshop_expr_key(expr, regime))
        return out

    regime = str(row.get("regime") or "")
    return [
        workshop_expr_key(expr, regime)
        for expr in row.get("context_expressions", [])
        if expr
    ]


def _score_float(value, default=-999.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _record_delay_for_admission(record: dict) -> int:
    settings = record.get("settings") or record.get("settings_override") or {}
    try:
        return int(settings.get("delay", 1))
    except (TypeError, ValueError):
        return 1


def _admission_delta_sort(record: dict) -> float:
    value = record.get("admission_brain_score_delta")
    if value is None:
        value = record.get("admission_delta_pool_sharpe")
    return _score_float(value)


def _root_seed_expression(record: dict) -> str:
    params = record.get("params") or {}
    return str(
        params.get("root_expression")
        or params.get("seed_expression")
        or params.get("original_expression")
        or record.get("expression")
        or ""
    )


def _rank_key(record: dict) -> tuple[float, float, float]:
    fitness = _score_float(record.get("is_fitness"), default=-999.0)
    if fitness > -999.0:
        fitness *= reward_multiplier(record_robustness_score(record))
    return (
        _admission_delta_sort(record),
        fitness,
        _score_float(record.get("is_sharpe"), default=-999.0),
    )


def _standalone_passes(record: dict) -> bool:
    return (
        record.get("status") == "done"
        and (record.get("is_sharpe") or 0.0) >= 1.25
        and (record.get("is_fitness") or 0.0) >= 1.0
        and is_passing(record)
    )


def _top_per_seed(records: list[dict], top_n: int) -> list[dict]:
    records = [record for record in records if _standalone_passes(record)]
    if top_n <= 0:
        return list(records)
    counts: dict[str, int] = {}
    selected: list[dict] = []
    for record in sorted(records, key=_rank_key, reverse=True):
        seed = _root_seed_expression(record)
        rank = counts.get(seed, 0) + 1
        if rank > top_n:
            continue
        counts[seed] = rank
        enriched = dict(record)
        enriched["seed_expression"] = seed
        enriched["seed_rank"] = rank
        selected.append(enriched)
    return selected


def _promote_to_templates(
    workshop_sim_log: Path,
    new_templates_path: Path = NEW_TEMPLATES_JSONL,
    summary_path: Path = WORKSHOP_SUMMARY_TXT,
    submittable_path: Path = SUBMITTABLE_JSONL,
    admission_path: Path = WORKSHOP_ADMISSION_JSONL,
    records: list[dict] | None = None,
    session=None,
    portfolio_alpha_data: dict | None = None,
    competition_id: str = "IQC2026S1",
    admission_gate: bool = True,
    min_brain_score_delta: float = 0.0,
    min_fallback_delta_sharpe: float = 0.0,
    pnl_sleep: float = 1.0,
    admission_pnl_attempts: int = 3,
    admission_scorer=None,
    per_seed_top_n: int = 5,
) -> int:
    """Promote passing workshop results to review artifacts.

    Returns number of promoted rows written.
    Writes workshop_submittable.jsonl as the primary Stage 3 seed file, keeps
    new_templates.jsonl as a compatibility/audit artifact, and also writes
    workshop_admission.jsonl plus workshop_summary.txt.

    If `records` is provided, uses those instead of reading the full sim log —
    so the summary/templates/submittable reflect only the current run.
    """
    if records is not None:
        workshop_records = records
    else:
        if not workshop_sim_log.exists():
            return 0
        workshop_records = _load_records([str(workshop_sim_log)])
    if not workshop_records:
        return 0

    # Identify passing records for Stage 3 outputs.
    passing = [r for r in workshop_records if _standalone_passes(r)]

    def _admission_delay(record: dict) -> int:
        settings = record.get("settings") or record.get("settings_override") or {}
        try:
            return int(settings.get("delay", 1))
        except (TypeError, ValueError):
            return 1

    # Admission gate: only marginally positive candidates become templates.
    if admission_gate:
        if admission_scorer is not None:
            admitted_records = admission_scorer(passing)
        elif not session or not portfolio_alpha_data:
            admitted_records = []
            for r in passing:
                enriched = dict(r)
                enriched["admission"] = {
                    "decision": "WATCH",
                    "reason": "PORTFOLIO_PNL_UNAVAILABLE",
                    "delay": _admission_delay(r),
                }
                enriched["admission_decision"] = "WATCH"
                enriched["admission_reason"] = "PORTFOLIO_PNL_UNAVAILABLE"
                enriched["admission_delay"] = _admission_delay(r)
                admitted_records.append(enriched)
        else:
            from core.brain.api import fetch_pnl_series_with_retry
            from core.portfolio.admission import (
                AdmissionConfig,
                annotate_records_with_admission,
                fetch_brain_performance_comparison,
            )

            def _fetch_candidate_pnl(alpha_id: str) -> dict[str, float]:
                if not alpha_id:
                    return {}
                return fetch_pnl_series_with_retry(
                    session,
                    alpha_id,
                    max_attempts=admission_pnl_attempts,
                    sleep_seconds=pnl_sleep,
                    timeout=20.0,
                )

            def _fetch_performance(alpha_id: str) -> dict | None:
                if not alpha_id:
                    return None
                return fetch_brain_performance_comparison(
                    session,
                    alpha_id,
                    competition_id,
                    max_attempts=5,
                    sleep_seconds=pnl_sleep,
                    timeout=20.0,
                )

            admitted_records = annotate_records_with_admission(
                passing,
                portfolio_alpha_data,
                config=AdmissionConfig(
                    min_brain_score_delta=min_brain_score_delta,
                    min_fallback_delta_sharpe=min_fallback_delta_sharpe,
                ),
                fetch_candidate_pnl=_fetch_candidate_pnl,
                fetch_performance_comparison=_fetch_performance,
            )
    else:
        admitted_records = []
        for r in passing:
            enriched = dict(r)
            enriched["admission"] = {
                "decision": "PROMOTE",
                "reason": "ADMISSION_GATE_DISABLED",
                "delay": _admission_delay(r),
            }
            enriched["admission_decision"] = "PROMOTE"
            enriched["admission_reason"] = "ADMISSION_GATE_DISABLED"
            enriched["admission_delay"] = _admission_delay(r)
            enriched["admission_brain_score_delta"] = None
            enriched["admission_brain_score_before"] = None
            enriched["admission_brain_score_after"] = None
            enriched["admission_delta_pool_sharpe"] = None
            enriched["admission_signal"] = "disabled"
            enriched["admission_global_max_corr"] = r.get("max_self_correlation")
            admitted_records.append(enriched)

    promoted = [
        r for r in admitted_records
        if r.get("admission_decision") == "PROMOTE"
    ]
    promoted = sorted(
        promoted,
        key=lambda r: _admission_delta_sort(r)
        if _admission_delta_sort(r) > -999.0
        else r.get("is_fitness") or 0.0,
        reverse=True,
    )

    # Write admission audit — all standalone passing records with gate labels.
    admission_path.parent.mkdir(parents=True, exist_ok=True)
    with admission_path.open("w") as f:
        for r in sorted(
            admitted_records,
            key=lambda x: (
                {"PROMOTE": 2, "WATCH": 1, "REJECT": 0}.get(x.get("admission_decision"), -1),
                _admission_delta_sort(x),
            ),
            reverse=True,
        ):
            f.write(json.dumps(r) + "\n")

    # Write new_templates.jsonl — compatibility / audit output.
    new_templates_path.parent.mkdir(parents=True, exist_ok=True)
    with new_templates_path.open("w") as f:
        for r in promoted:
            settings = r.get("settings") or r.get("settings_override") or {}
            f.write(json.dumps({
                "expression": r.get("expression", ""),
                "is_sharpe": r.get("is_sharpe"),
                "is_fitness": r.get("is_fitness"),
                "oos_fitness_2023": r.get("oos_fitness_2023"),
                "dataset_id": r.get("dataset_id", ""),
                "category": r.get("category", ""),
                "fields": r.get("fields") or [],
                "group": r.get("group", ""),
                "params": r.get("params") or {},
                "settings": settings,
                "settings_override": settings,
                "admission_decision": r.get("admission_decision"),
                "admission_reason": r.get("admission_reason"),
                "admission_delay": r.get("admission_delay"),
                "admission_brain_score_delta": r.get("admission_brain_score_delta"),
                "admission_brain_score_before": r.get("admission_brain_score_before"),
                "admission_brain_score_after": r.get("admission_brain_score_after"),
                "admission_delta_pool_sharpe": r.get("admission_delta_pool_sharpe"),
                "admission_fallback_delta_used": r.get("admission_fallback_delta_used"),
                "admission_signal": r.get("admission_signal"),
                "admission_global_max_corr": r.get("admission_global_max_corr"),
                "source": "workshop",
            }) + "\n")

    # Write workshop_submittable.jsonl — admission-ranked, ready for direct submission
    submittable = _top_per_seed(promoted, per_seed_top_n)
    with submittable_path.open("w") as f:
        for rank, r in enumerate(submittable, start=1):
            settings = r.get("settings") or r.get("settings_override") or {}
            f.write(json.dumps({
                "rank": rank,
                "alpha_id": r.get("alpha_id"),
                "expression": r.get("expression", ""),
                "is_sharpe": r.get("is_sharpe"),
                "is_fitness": r.get("is_fitness"),
                "oos_fitness_2023": r.get("oos_fitness_2023"),
                "grade": r.get("grade"),
                "dataset_id": r.get("dataset_id", ""),
                "category": r.get("category", ""),
                "fields": r.get("fields") or [],
                "group": r.get("group", ""),
                "params": r.get("params") or {},
                "settings": settings,
                "settings_override": settings,
                "is_concentrated_weight_ok": r.get("is_concentrated_weight_ok", True),
                "is_sub_universe_sharpe_ok": r.get("is_sub_universe_sharpe_ok", True),
                "seed_expression": r.get("seed_expression") or _root_seed_expression(r),
                "seed_rank": r.get("seed_rank"),
                "admission_decision": r.get("admission_decision"),
                "admission_reason": r.get("admission_reason"),
                "admission_delay": r.get("admission_delay"),
                "admission_brain_score_delta": r.get("admission_brain_score_delta"),
                "admission_brain_score_before": r.get("admission_brain_score_before"),
                "admission_brain_score_after": r.get("admission_brain_score_after"),
                "admission_delta_pool_sharpe": r.get("admission_delta_pool_sharpe"),
                "admission_fallback_delta_used": r.get("admission_fallback_delta_used"),
                "admission_signal": r.get("admission_signal"),
                "admission_global_max_corr": r.get("admission_global_max_corr"),
            }) + "\n")

    # Write human-readable summary
    n_total = len(workshop_records)
    n_pass = len(passing)
    n_promote = sum(1 for r in admitted_records if r.get("admission_decision") == "PROMOTE")
    n_watch = sum(1 for r in admitted_records if r.get("admission_decision") == "WATCH")
    n_reject = sum(1 for r in admitted_records if r.get("admission_decision") == "REJECT")

    def _metric(value) -> str:
        return "N/A" if value is None else f"{float(value):.3f}"

    lines = [
        f"Workshop complete: {n_total} simulated → {n_pass} standalone-passing → "
        f"{n_promote} admission-promoted\n",
        f"Admission decisions: PROMOTE={n_promote} WATCH={n_watch} REJECT={n_reject}\n",
        "\n",
        f"  Promoted alphas ranked by Brain score delta when available "
        f"({submittable_path.name}, top {per_seed_top_n if per_seed_top_n > 0 else 'all'} per seed):\n",
    ]
    for rank, r in enumerate(submittable, start=1):
        orig = _root_seed_expression(r) or "?"
        oos23f = r.get("oos_fitness_2023")
        oos23_tag = f"  2023-fit={oos23f:.2f}" if oos23f is not None else ""
        alpha_id = r.get("alpha_id") or "?"
        lines.append(
            f"  [{rank}] PROMOTE id={alpha_id}  delay={r.get('admission_delay')}  "
            f"seed_rank={r.get('seed_rank') or '?'}  "
            f"sharpe={r.get('is_sharpe', 0):.2f}  fit={r.get('is_fitness', 0):.2f}{oos23_tag}  "
            f"score_delta={_metric(r.get('admission_brain_score_delta'))}  "
            f"fallback_delta={_metric(r.get('admission_delta_pool_sharpe'))}  "
            f"signal={r.get('admission_signal') or 'N/A'}  "
            f"corr={_metric(r.get('admission_global_max_corr'))}  "
            f"{r.get('expression', '')}\n"
            f"       from: {orig[:80]}\n"
        )
    if not submittable:
        lines.append("  (none)\n")

    non_promoted = [
        r for r in admitted_records
        if r.get("admission_decision") != "PROMOTE"
    ]
    if non_promoted:
        lines.extend([
            "\n",
            f"  Watch/reject standalone passers ({admission_path.name} has full rows):\n",
        ])
        for rank, r in enumerate(non_promoted, start=1):
            alpha_id = r.get("alpha_id") or "?"
            lines.append(
                f"  [{rank}] {r.get('admission_decision')} {r.get('admission_reason')} "
                f"id={alpha_id}  delay={r.get('admission_delay')}  "
                f"sharpe={r.get('is_sharpe', 0):.2f}  fit={r.get('is_fitness', 0):.2f}  "
                f"score_delta={_metric(r.get('admission_brain_score_delta'))}  "
                f"fallback_delta={_metric(r.get('admission_delta_pool_sharpe'))}  "
                f"signal={r.get('admission_signal') or 'N/A'}  "
                f"corr={_metric(r.get('admission_global_max_corr'))}  "
                f"{r.get('expression', '')[:120]}\n"
            )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("".join(lines))
    print(f"\n  Summary written to: {summary_path}")
    if admission_path.exists():
        print(f"  Admission audit:    {admission_path}  ({len(admitted_records)} entries)")
    if new_templates_path.stat().st_size > 0:
        print(f"  New templates:      {new_templates_path}  ({len(promoted)} entries, compatibility copy)")
        print(f"  Submittable alphas: {submittable_path}  ({len(submittable)} entries, review/submit individually)")
    else:
        print(f"  New templates:      {new_templates_path}  (0 entries)")
        print(f"  Submittable alphas: {submittable_path}  (0 entries)")

    return len(promoted)


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(selected: list[dict]) -> None:
    has_score_delta = any(r.get("admission_brain_score_delta") is not None for r in selected)
    has_fallback_delta = any(r.get("admission_delta_pool_sharpe") is not None for r in selected)
    print()
    print(
        f"  {'tier':<6} {'action':<8} {'regime':<26} {'sharpe':>7} {'fitness':>7} {'23fit':>7} "
        f"{'corr':>7}"
        + (f" {'dScore':>7}" if has_score_delta else "")
        + (f" {'dFb':>7}" if has_fallback_delta else "")
        + f" {'field':<30} {'failing':<30}  expression"
    )
    print("  " + "─" * (166 + (8 if has_score_delta else 0) + (8 if has_fallback_delta else 0)))
    for r in selected:
        action = _resolved_route(r) or "?"
        sharpe = r.get("is_sharpe") or 0.0
        fitness = r.get("is_fitness") or 0.0
        regime = record_regime(r)
        oos23_fit = r.get("oos_fitness_2023")
        oos23_str = f"{oos23_fit:>7.3f}" if oos23_fit is not None else f"{'N/A':>7}"
        corr = r.get("max_self_correlation")
        corr_str = f"{corr:>7.3f}" if corr is not None else f"{'N/A':>7}"
        field_disp = _display_fields(r)
        failing = ",".join(failing_checks(r))[:28]
        expr = (r.get("expression") or "")[:60]
        tier = _tier(action, sharpe)
        score_delta_col = ""
        if has_score_delta:
            score_delta = r.get("admission_brain_score_delta")
            score_delta_col = f" {score_delta:>+7.0f}" if score_delta is not None else f" {'N/A':>7}"
        fallback_delta_col = ""
        if has_fallback_delta:
            fallback_delta = r.get("admission_delta_pool_sharpe")
            fallback_delta_col = f" {fallback_delta:>+7.3f}" if fallback_delta is not None else f" {'N/A':>7}"
        print(
            f"  {tier:<6} {action:<8} {regime:<26} {sharpe:>7.3f} {fitness:>7.3f} {oos23_str} "
            f"{corr_str}{score_delta_col}{fallback_delta_col} {field_disp:<30} {failing:<30}  {expr}"
        )
    print()


# ---------------------------------------------------------------------------
# Per-alpha results file
# ---------------------------------------------------------------------------

def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _seed_metadata(record: dict) -> dict:
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    score_delta = _first_non_none(
        record.get("admission_brain_score_delta"),
        record.get("brain_score_delta"),
        admission.get("brain_score_delta"),
    )
    fallback_delta = _first_non_none(
        record.get("admission_delta_pool_sharpe"),
        admission.get("delta_pool_sharpe"),
    )
    return {
        "seed_alpha_id": record.get("alpha_id"),
        "seed_expression": record.get("expression"),
        "seed_sharpe": record.get("is_sharpe"),
        "seed_fitness": record.get("is_fitness"),
        "seed_admission_decision": _first_non_none(
            record.get("admission_decision"),
            admission.get("decision"),
        ),
        "seed_admission_reason": _first_non_none(
            record.get("admission_reason"),
            admission.get("reason"),
        ),
        "seed_admission_signal": _first_non_none(
            record.get("admission_signal"),
            admission.get("signal"),
        ),
        "seed_admission_brain_score_before": _first_non_none(
            record.get("admission_brain_score_before"),
            admission.get("brain_score_before"),
        ),
        "seed_admission_brain_score_after": _first_non_none(
            record.get("admission_brain_score_after"),
            admission.get("brain_score_after"),
        ),
        "seed_admission_brain_score_delta": score_delta,
        "seed_admission_delta_pool_sharpe": fallback_delta,
    }


def _non_empty_seed_metadata(record: dict) -> dict:
    return {k: v for k, v in _seed_metadata(record).items() if v is not None}


def _seed_metadata_from_params(record: dict) -> dict:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    return {
        key: params.get(key)
        for key in (
            "seed_alpha_id",
            "seed_expression",
            "seed_sharpe",
            "seed_fitness",
            "seed_admission_decision",
            "seed_admission_reason",
            "seed_admission_signal",
            "seed_admission_brain_score_before",
            "seed_admission_brain_score_after",
            "seed_admission_brain_score_delta",
            "seed_admission_delta_pool_sharpe",
        )
        if params.get(key) is not None
    }


def _format_seed_score_tag(meta: dict) -> str:
    parts: list[str] = []
    alpha_id = meta.get("seed_alpha_id")
    if alpha_id:
        parts.append(f"seed_id={alpha_id}")
    sharpe = meta.get("seed_sharpe")
    if sharpe is not None:
        parts.append(f"seed_sharpe={float(sharpe):.3f}")
    fitness = meta.get("seed_fitness")
    if fitness is not None:
        parts.append(f"seed_fit={float(fitness):.3f}")
    score_delta = meta.get("seed_admission_brain_score_delta")
    if score_delta is not None:
        parts.append(f"seed_score_delta={float(score_delta):+.0f}")
    fallback_delta = meta.get("seed_admission_delta_pool_sharpe")
    if score_delta is None and fallback_delta is not None:
        parts.append(f"seed_fallback_delta={float(fallback_delta):+.3f}")
    before = meta.get("seed_admission_brain_score_before")
    after = meta.get("seed_admission_brain_score_after")
    if before is not None and after is not None:
        parts.append(f"seed_score={float(before):.0f}->{float(after):.0f}")
    decision = meta.get("seed_admission_decision")
    reason = meta.get("seed_admission_reason")
    if decision:
        tag = str(decision)
        if reason:
            tag += f"/{reason}"
        parts.append(f"seed_admission={tag}")
    return "  ".join(parts)


def _write_per_alpha_results(
    run_records: list[dict],
    out_path: Path,
    top_n: int = 5,
    seed_records: list[dict] | None = None,
) -> None:
    """Write a human-readable file showing top N passing results per seed alpha.

    Groups run_records by root seed, filters to standalone-passing simulations,
    sorts each group by admission score delta when available, and writes the
    top_n for each seed.
    """
    from collections import defaultdict

    by_seed: dict[str, list[dict]] = defaultdict(list)
    for r in run_records:
        orig = _root_seed_expression(r)
        if orig:
            by_seed[orig].append(r)

    if not by_seed:
        return

    seed_meta_by_expr: dict[str, dict] = {}
    for seed in seed_records or []:
        seed_meta = _non_empty_seed_metadata(seed)
        for seed_expr in {
            _root_seed_expression(seed),
            str(seed.get("expression") or ""),
        }:
            if seed_expr:
                seed_meta_by_expr[seed_expr] = seed_meta

    for seed_expr, records in by_seed.items():
        if seed_expr in seed_meta_by_expr:
            continue
        for record in records:
            meta = _seed_metadata_from_params(record)
            if meta:
                seed_meta_by_expr[seed_expr] = meta
                break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    n_seeds = len(by_seed)
    lines.append(f"Per-alpha top-{top_n} standalone-passing workshop results (ranked by admission score) — {n_seeds} seed(s)\n")
    lines.append("=" * 110 + "\n\n")

    for seed_expr, records in sorted(by_seed.items(), key=lambda kv: kv[0]):
        done = [r for r in records if r.get("status") == "done"]
        passing = [r for r in done if _standalone_passes(r)]
        passing.sort(key=_rank_key, reverse=True)
        lane = ((passing or done)[0].get("search_lane") or "?") if (passing or done) else "?"
        lines.append(f"[{lane}] seed: {seed_expr}\n")
        seed_score_tag = _format_seed_score_tag(seed_meta_by_expr.get(seed_expr, {}))
        if seed_score_tag:
            lines.append(f"       {seed_score_tag}\n")
        lines.append(
            f"       {len(done)} completed, {len(passing)} standalone-passing, "
            f"showing top {min(top_n, len(passing))}\n"
        )

        if not done:
            lines.append("       (no completed simulations)\n")
        elif not passing:
            lines.append("       (no standalone-passing simulations)\n")
        for rank, r in enumerate(passing[:top_n], 1):
            alpha_id = r.get("alpha_id") or "?"
            sharpe  = r.get("is_sharpe")
            fitness = r.get("is_fitness")
            oos23f  = r.get("oos_fitness_2023")
            corr    = _first_non_none(r.get("admission_global_max_corr"), r.get("max_self_correlation"))
            score_delta = r.get("admission_brain_score_delta")
            fallback_delta = r.get("admission_delta_pool_sharpe")
            admission = r.get("admission") if isinstance(r.get("admission"), dict) else {}
            admission_decision = _first_non_none(
                r.get("admission_decision"),
                admission.get("decision"),
            )
            admission_reason = _first_non_none(
                r.get("admission_reason"),
                admission.get("reason"),
            )
            checks  = r.get("checks") or []
            failing = [
                c["name"] for c in checks
                if c.get("result") != "PASS" and c.get("name") != "SELF_CORRELATION"
            ]
            status  = f"STANDALONE_FAIL({','.join(failing)})" if failing else "STANDALONE_PASS"
            if admission_decision:
                status += f" ADMIT_{admission_decision}"
                if admission_reason:
                    status += f"({admission_reason})"
            sharpe_s  = f"{sharpe:.3f}"  if sharpe  is not None else " N/A"
            fitness_s = f"{fitness:.3f}" if fitness is not None else " N/A"
            oos_tag   = f"  2023f={oos23f:.2f}" if oos23f is not None else ""
            corr_tag  = f"  corr={corr:.2f}"    if corr   is not None else ""
            score_tag = f"  score_delta={score_delta:+.0f}" if score_delta is not None else ""
            fallback_tag = f"  fallback_delta={fallback_delta:+.3f}" if fallback_delta is not None else ""
            expr = r.get("expression", "")
            lines.append(
                f"  {rank}. id={alpha_id}  sharpe={sharpe_s}  fit={fitness_s}{score_tag}{fallback_tag}{oos_tag}{corr_tag}"
                f"  {status}\n"
                f"     {expr}\n"
            )
        lines.append("\n")

    out_path.write_text("".join(lines))
    print(f"  Per-alpha results: {out_path}  ({n_seeds} seeds)")


# ---------------------------------------------------------------------------
# Per-alpha iterative workshop
# ---------------------------------------------------------------------------

def _is_transient_llm_error(exc: Exception) -> bool:
    """True for retryable LLM transport failures (connection drops, timeouts, rate limits).

    These mean the seed was never actually workshopped — the request never
    completed — so the caller should leave it un-marked and retry it next run,
    rather than burning it as 'done'. Permanent errors (bad request, auth) are
    NOT transient: retrying them every run would loop forever.
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "connection" in name or "timeout" in name or "ratelimit" in name
        or "connection error" in msg or "timed out" in msg or "timeout" in msg
        or "temporarily unavailable" in msg or "rate limit" in msg
        or "remotedisconnected" in msg or "service unavailable" in msg
    )


def _workshop_alpha_iterative(
    record: dict,
    lane: str,
    agent,
    session,
    out_sim_log: Path,
    settings: dict,
    args,
    *,
    n_rounds: int,
    known_exprs: set[str],
    all_records: list[dict],
    history=None,
    origin_note: str = "",
    portfolio_alpha_data: dict | None = None,
) -> tuple[list[dict], bool]:
    """Run N rounds of (LLM generate → simulate → feedback) for one alpha.

    Round 1: Initial LLM call with alpha info + history context.
    Round 2+: Feed round N sim results back into the conversation;
              LLM sees which of its suggestions worked and which didn't.

    Returns (all_simulation_records, retry_seed). retry_seed is True when a
    transient LLM error left the seed with zero results — the caller should NOT
    mark it workshopped so it is re-attempted on the next run.
    """
    record = with_portfolio_prompt_feedback(
        record,
        session,
        portfolio_alpha_data,
        competition_id=args.competition_id,
        pnl_sleep=args.pnl_sleep,
        max_attempts=args.admission_pnl_attempts,
        performance_max_attempts=5,
        performance_sleep_seconds=args.pnl_sleep,
    )
    original = record.get("expression", "")
    regime = record_regime(record)
    root_expression = (record.get("params") or {}).get("root_expression") or (record.get("params") or {}).get("original_expression") or original
    seed_params = _non_empty_seed_metadata(record)
    passing_examples = find_passing_examples(all_records, record)

    # History context across all prior sessions (WorkshopHistory)
    if history is not None:
        orig_from_params = (record.get("params") or {}).get("original_expression") or original
        root = history.root_for(
            workshop_expr_key(original, regime),
            workshop_expr_key(orig_from_params, regime),
        )
        history_ctx = (
            history.context_for_repair(root) if lane == "repair"
            else history.context_for_enhance(root)
        )
    else:
        history_ctx = []

    # Per-alpha prior repair attempts from sim log
    log_trials = collect_trial_history(record, all_records) if lane == "repair" else []
    combined_trials = history_ctx + [
        t for t in log_trials
        if t.get("expression") not in {h.get("expression") for h in history_ctx}
    ]
    context_records = list(record.get("context_records") or [])

    all_new_records: list[dict] = []
    messages: list[dict] = []
    seen_this: set[str] = {original}
    round_sim_results: list[dict] = []
    transient_error = False

    for round_idx in range(n_rounds):
        # ── Generate candidates ────────────────────────────────────────────
        try:
            if round_idx == 0:
                if lane == "repair":
                    result = agent.repair(
                        record,
                        passing_examples=passing_examples,
                        trial_history=combined_trials or None,
                        context_records=context_records or None,
                        origin_note=origin_note,
                    )
                else:
                    ctx = history_ctx + context_records
                    result = agent.enhance(
                        record,
                        passing_examples=passing_examples,
                        context_records=ctx or None,
                        origin_note=origin_note,
                    )
            else:
                if not round_sim_results:
                    break
                if lane == "repair":
                    result = agent.repair_round(messages, round_sim_results)
                else:
                    result = agent.enhance_round(messages, round_sim_results)
        except Exception as e:
            print(f"  [{lane}] round {round_idx + 1} LLM error for {original[:50]}: {e}")
            if _is_transient_llm_error(e):
                transient_error = True
            break

        messages = result.messages

        # ── Filter and build candidate list ───────────────────────────────
        round_candidates = []
        for expr in result.repair_expressions[:6]:
            expr_key = workshop_expr_key(expr, regime)
            if not expr or expr in seen_this or expr_key in known_exprs:
                continue
            if not _looks_like_fastexpr(expr):
                continue
            seen_this.add(expr)
            known_exprs.add(expr_key)
            if lane == "repair":
                params: dict = {
                    "original_expression": original,
                    "root_expression": root_expression,
                    "original_sharpe": record.get("is_sharpe"),
                    "original_failing_checks": failing_checks(record),
                }
            else:
                params = {
                    "original_expression": original,
                    "root_expression": root_expression,
                    "original_sharpe": record.get("is_sharpe"),
                    "original_fitness": record.get("is_fitness"),
                }
            params.update(seed_params)
            round_candidates.append(_make_candidate(expr, record, lane, params))
            print(f"  [{lane}] r{round_idx + 1} → {expr[:80]}")

        if not round_candidates:
            print(f"  [{lane}] round {round_idx + 1}: no new candidates — stopping")
            break

        # ── Simulate this round ────────────────────────────────────────────
        print(f"\n  [{lane}] round {round_idx + 1}: submitting {len(round_candidates)} candidates...")
        round_sim_results = submit_candidate_batch(
            round_candidates, session, out_sim_log, settings,
            args.credentials, args.batch_size,
            portfolio_alpha_data=portfolio_alpha_data,
            pnl_sleep=args.pnl_sleep,
            yearly_stats_attempts=args.yearly_stats_attempts,
            yearly_stats_sleep=args.yearly_stats_sleep,
        )
        all_records.extend(round_sim_results)
        all_new_records.extend(round_sim_results)
        known_exprs.update(record_workshop_key(r) for r in round_sim_results if r.get("expression", ""))
        if history is not None:
            history.update(round_sim_results, iteration=round_idx)

    if all_new_records:
        done = [r for r in all_new_records if r.get("status") == "done"]
        done.sort(key=lambda r: r.get("is_sharpe") or 0.0, reverse=True)
        print(f"\n  [{lane}] {original[:60]} — {len(done)} simulated:")
        for r in done[:5]:
            alpha_id = r.get("alpha_id") or "?"
            sharpe = r.get("is_sharpe")
            fitness = r.get("is_fitness")
            oos23f = r.get("oos_fitness_2023")
            failing = [c["name"] for c in (r.get("checks") or []) if c.get("result") != "PASS" and c.get("name") != "SELF_CORRELATION"]
            status = f"FAIL({','.join(failing)})" if failing else "PASS"
            oos_tag = f"  2023f={oos23f:.2f}" if oos23f is not None else ""
            print(f"    id={alpha_id}  sharpe={sharpe:.3f}  fit={fitness:.3f}{oos_tag}  {status}  {r.get('expression','')[:65]}")

    # A transient LLM error that produced no sims means this seed never got
    # workshopped — signal the caller to leave it for retry.
    retry_seed = transient_error and not all_new_records
    return all_new_records, retry_seed


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _workshop_one(
    records: list[dict],
    all_records: list[dict],
    known_exprs: set[str],
    session,
    settings: dict,
    out_sim_log: Path,
    args,
    label: str = "",
    history=None,
    portfolio_alpha_data: dict | None = None,
) -> list[dict]:
    """Run generate+submit+iterate on a set of records. Returns all new sim records."""
    records = [
        with_portfolio_prompt_feedback(
            r,
            session,
            portfolio_alpha_data,
            competition_id=args.competition_id,
            pnl_sleep=args.pnl_sleep,
            max_attempts=args.admission_pnl_attempts,
            performance_max_attempts=5,
            performance_sleep_seconds=args.pnl_sleep,
        )
        for r in records
    ]
    source_exprs = {record_workshop_key(r) for r in records}

    candidates = generate_candidates(
        records,
        all_records,
        history=history,
        api_key=args.api_key,
        known_exprs=known_exprs,
        verbose=True,
        bypass_self_corr=args.bypass_correlation_check,
    )
    if not candidates:
        print(f"  {label}No candidates generated — skipping.")
        _save_workshopped_sources(source_exprs)
        return []

    print(f"\nSubmitting {len(candidates)} candidates...")
    new_records = submit_candidate_batch(
        candidates,
        session,
        out_sim_log,
        settings,
        args.credentials,
        args.batch_size,
        portfolio_alpha_data=portfolio_alpha_data,
        pnl_sleep=args.pnl_sleep,
        yearly_stats_attempts=args.yearly_stats_attempts,
        yearly_stats_sleep=args.yearly_stats_sleep,
    )
    all_records.extend(new_records)
    known_exprs.update(record_workshop_key(r) for r in new_records if r.get("expression", ""))
    if history is not None:
        history.update(new_records, iteration=0)
    _save_workshopped_sources(source_exprs)

    for iteration in range(1, args.workshop_iterations + 1):
        if not new_records:
            break
        iter_src = {record_workshop_key(r) for r in new_records if r.get("expression", "")}
        iter_candidates = generate_candidates(
            new_records, all_records,
            history=history,
            api_key=args.api_key,
            known_exprs=known_exprs,
            verbose=False,
            bypass_self_corr=args.bypass_correlation_check,
        )
        if not iter_candidates:
            break
        print(f"\n  iteration {iteration}: submitting {len(iter_candidates)} candidates...")
        new_records = submit_candidate_batch(
            iter_candidates,
            session,
            out_sim_log,
            settings,
            args.credentials,
            args.batch_size,
            portfolio_alpha_data=portfolio_alpha_data,
            pnl_sleep=args.pnl_sleep,
            yearly_stats_attempts=args.yearly_stats_attempts,
            yearly_stats_sleep=args.yearly_stats_sleep,
        )
        all_records.extend(new_records)
        known_exprs.update(record_workshop_key(r) for r in new_records if r.get("expression", ""))
        if history is not None:
            history.update(new_records, iteration=iteration)
        _save_workshopped_sources(iter_src)

    return new_records


def _route_negate_results(
    negate_sim_records: list[dict],
    verbose: bool = True,
    *,
    bypass_self_corr: bool = False,
) -> tuple[list[tuple[dict, str]], list[tuple[dict, str]]]:
    """Route negate simulation results into repair and enhance queues.

    Each element in the returned queues is (sim_record, origin_note) where
    origin_note is passed to the LLM so it knows the alpha was negated.
    """
    repair_q: list[tuple[dict, str]] = []
    enhance_q: list[tuple[dict, str]] = []

    for r in negate_sim_records:
        if r.get("status") != "done":
            continue
        sharpe = r.get("is_sharpe") or 0.0
        orig_expr = (r.get("params") or {}).get("original_expression", "")
        note = f"this alpha is a negated version of: {orig_expr}" if orig_expr else ""

        action = route_record(r, bypass_self_corr=bypass_self_corr)

        if action == "enhance":
            enhance_q.append((r, note))
            if verbose:
                print(f"  [negate→enhance]  sharpe={sharpe:.3f}  {r.get('expression','')[:60]}")
        elif action == "repair":
            repair_q.append((r, note))
            if verbose:
                print(f"  [negate→repair]   sharpe={sharpe:.3f}  {r.get('expression','')[:60]}")
        else:
            if verbose:
                print(f"  [negate→discard]  sharpe={sharpe:.3f}  {r.get('expression','')[:60]}")

    return repair_q, enhance_q


def _run_batch(
    selected: list[dict],
    all_records: list[dict],
    known_exprs: set[str],
    session,
    settings: dict,
    out_sim_log: Path,
    args,
    portfolio_alpha_data: dict | None = None,
) -> None:
    """Sequential 3-path workshop pipeline.

    Stage 1 — Negate (deterministic, no LLM):
        Sign-flip all eligible records, submit, collect results.
    Stage 2 — Route negate results:
        Passing → enhance queue. Near-miss → repair queue. Both carry origin_note.
    Stage 3 — Route original non-negate records:
        Repair and enhance originals are added to their respective queues.
    Stage 4 — LLM repair + enhance:
        One combined LLM submission with skip_negate=True and origin_notes passed.
    Iterations — workshop-iterations loop runs on LLM results as before.

    A WorkshopHistory instance is created at the start and threaded through
    every stage so each LLM call receives the full gradient of prior attempts
    on its root seed expression, across all lanes and iterations.
    """
    from alpha_enhancement.workshop.repair import _sign_flip
    from alpha_enhancement.workshop.history import WorkshopHistory

    # Create history keyed on all selected seed expressions
    seed_exprs = [record_workshop_key(r) for r in selected]
    history = WorkshopHistory(seed_exprs)

    lane_filter: list[str] | None = getattr(args, "lane", None)

    print(f"\n{'='*60}")
    print(f"SEQUENTIAL WORKSHOP — {len(selected)} records")
    print(f"{'='*60}")

    # ── Stage 1: Negate ──────────────────────────────────────────────────────
    negate_records = [r for r in selected if r.get("_route") == "negate"]
    negate_source_exprs: set[str] = {record_workshop_key(r) for r in negate_records}
    negate_sim_records: list[dict] = []

    if negate_records and (lane_filter is None or "negate" in lane_filter):
        print(f"\n{'='*60}")
        print(f"STAGE 1 — NEGATE ({len(negate_records)} records)")
        print(f"{'='*60}")
        negate_candidates: list[dict] = []
        for r in negate_records:
            original = r.get("expression", "")
            regime = record_regime(r)
            negated = _sign_flip(original)
            negated_key = workshop_expr_key(negated, regime)
            if negated == original or negated_key in known_exprs:
                continue
            known_exprs.add(negated_key)
            _h = __import__("hashlib")
            _cid = _h.sha1(f"{regime}\n{negated}".encode()).hexdigest()[:16]
            _tid = _h.sha1(f"{regime}\n{original}".encode()).hexdigest()[:16]
            negate_candidates.append({
                "candidate_id": _cid,
                "template_id": f"negate:{_tid}",
                "dataset_id": r.get("dataset_id", ""),
                "category": r.get("category", ""),
                "search_lane": "negate",
                "expression": negated,
                "settings_override": source_settings_override(r),
                "fields": r.get("fields") or [],
                "cluster_ids": r.get("cluster_ids") or [],
                "group": r.get("group", ""),
                "params": {
                    "original_expression": original,
                    "root_expression": (r.get("params") or {}).get("root_expression") or original,
                    "original_sharpe": r.get("is_sharpe"),
                    **_non_empty_seed_metadata(r),
                },
            })
            print(f"  [negate]  {original[:55]}  →  {negated[:55]}")

        if negate_candidates:
            # Register negate lineages in history before submission
            for nc in negate_candidates:
                regime = regime_from_settings(nc.get("settings_override"))
                original = (nc.get("params") or {}).get("original_expression", nc["expression"])
                history.register_lineage(
                    workshop_expr_key(nc["expression"], regime),
                    workshop_expr_key(original, regime),
                )
            print(f"\nSubmitting {len(negate_candidates)} negate candidates...")
            negate_sim_records = submit_candidate_batch(
                negate_candidates,
                session,
                out_sim_log,
                settings,
                args.credentials,
                args.batch_size,
                portfolio_alpha_data=portfolio_alpha_data,
                pnl_sleep=args.pnl_sleep,
                yearly_stats_attempts=args.yearly_stats_attempts,
                yearly_stats_sleep=args.yearly_stats_sleep,
            )
            all_records.extend(negate_sim_records)
            known_exprs.update(record_workshop_key(r) for r in negate_sim_records if r.get("expression", ""))
            history.update(negate_sim_records, iteration=0)
        _save_workshopped_sources(negate_source_exprs)
    else:
        print(f"\nStage 1 skipped (no negate records or lane filter).")

    # Early exit for --lane negate
    if lane_filter is not None and set(lane_filter) == {"negate"}:
        print(f"\n{'='*60}")
        n_promoted = _promote_to_templates(
            out_sim_log,
            session=session,
            portfolio_alpha_data=portfolio_alpha_data,
            competition_id=args.competition_id,
            admission_gate=not args.skip_admission_gate,
            min_brain_score_delta=args.admission_min_brain_score_delta,
            min_fallback_delta_sharpe=args.admission_min_fallback_delta_sharpe,
            pnl_sleep=args.pnl_sleep,
            admission_pnl_attempts=args.admission_pnl_attempts,
            per_seed_top_n=args.per_seed_top_n,
        )
        workshop_records = _load_records([str(out_sim_log)])
        n_pass = sum(1 for r in workshop_records if is_passing(r))
        print(f"Negate-only workshop complete.")
        print(f"Sim log:         {out_sim_log}")
        print(f"Total simulated: {len(workshop_records)}")
        print(f"Passing:         {n_pass}  ({n_promoted} promoted to templates)")
        return

    # ── Stage 2: Route negate results ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 2 — ROUTE NEGATE RESULTS ({len(negate_sim_records)} negate sims)")
    print(f"{'='*60}")
    negate_repair_q, negate_enhance_q = _route_negate_results(
        negate_sim_records,
        verbose=True,
        bypass_self_corr=args.bypass_correlation_check,
    )

    # ── Stage 3: Route original non-negate records ────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 3 — ROUTE ORIGINAL RECORDS")
    print(f"{'='*60}")
    orig_repair_q: list[tuple[dict, str]] = []
    orig_enhance_q: list[tuple[dict, str]] = []
    for r in selected:
        if record_workshop_key(r) in negate_source_exprs:
            continue
        action = _resolved_route(r, bypass_self_corr=args.bypass_correlation_check)
        if action == "repair":
            orig_repair_q.append((r, ""))
        elif action == "enhance":
            orig_enhance_q.append((r, ""))
    print(f"  [originals] {len(orig_repair_q)} repair | {len(orig_enhance_q)} enhance")

    # Merge and apply lane filter
    merged_repair = negate_repair_q + orig_repair_q
    merged_enhance = negate_enhance_q + orig_enhance_q
    if lane_filter is not None:
        if "repair" not in lane_filter:
            merged_repair = []
        if "enhance" not in lane_filter:
            merged_enhance = []

    # ── Stage 4: LLM repair + enhance (per-alpha iterative) ──────────────────
    n_rounds = max(1, args.workshop_iterations)
    print(f"\n{'='*60}")
    print(
        f"STAGE 4 — LLM ITERATIVE  "
        f"({len(merged_repair)} repair | {len(merged_enhance)} enhance | {n_rounds} rounds each)"
    )
    print(f"{'='*60}")

    run_records: list[dict] = []

    if merged_repair or merged_enhance:
        from core import llm as _llm
        resolved_key = _llm.load_api_key(args.api_key)

        if not resolved_key:
            print("  No Anthropic API key — skipping LLM repair/enhance")
        else:
            try:
                from alpha_enhancement.agents.repair import AlphaRepairAgent
                agent = AlphaRepairAgent(api_key=resolved_key)
            except Exception as e:
                print(f"  AlphaRepairAgent unavailable: {e}")
                agent = None

            if agent:
                source_exprs: set[str] = set()

                for r, note in merged_repair:
                    print(f"\n{'─'*55}")
                    print(f"[repair] {r.get('expression', '')[:70]}")
                    alpha_results, retry_seed = _workshop_alpha_iterative(
                        r, "repair", agent, session, out_sim_log, settings, args,
                        n_rounds=n_rounds,
                        known_exprs=known_exprs,
                        all_records=all_records,
                        history=history,
                        origin_note=note,
                        portfolio_alpha_data=portfolio_alpha_data,
                    )
                    run_records.extend(alpha_results)
                    if retry_seed:
                        print(f"  [repair] transient LLM error — leaving seed un-workshopped for retry")
                    else:
                        source_exprs.add(record_workshop_key(r))

                for r, note in merged_enhance:
                    print(f"\n{'─'*55}")
                    print(f"[enhance] {r.get('expression', '')[:70]}")
                    alpha_results, retry_seed = _workshop_alpha_iterative(
                        r, "enhance", agent, session, out_sim_log, settings, args,
                        n_rounds=n_rounds,
                        known_exprs=known_exprs,
                        all_records=all_records,
                        history=history,
                        origin_note=note,
                        portfolio_alpha_data=portfolio_alpha_data,
                    )
                    run_records.extend(alpha_results)
                    if retry_seed:
                        print(f"  [enhance] transient LLM error — leaving seed un-workshopped for retry")
                    else:
                        source_exprs.add(record_workshop_key(r))

                _save_workshopped_sources(source_exprs)

    # Also include negate sim records from this run
    run_records.extend(negate_sim_records)

    _write_per_alpha_results(
        run_records,
        WORKSHOP_DIR / "per_alpha_results.txt",
        top_n=args.per_seed_top_n if args.per_seed_top_n > 0 else 999999,
        seed_records=selected,
    )

    print(f"\n{'='*60}")
    n_promoted = _promote_to_templates(
        out_sim_log,
        records=run_records if run_records else None,
        session=session,
        portfolio_alpha_data=portfolio_alpha_data,
        competition_id=args.competition_id,
        admission_gate=not args.skip_admission_gate,
        min_brain_score_delta=args.admission_min_brain_score_delta,
        min_fallback_delta_sharpe=args.admission_min_fallback_delta_sharpe,
        pnl_sleep=args.pnl_sleep,
        admission_pnl_attempts=args.admission_pnl_attempts,
        per_seed_top_n=args.per_seed_top_n,
    )
    workshop_records = run_records if run_records else []
    n_pass = sum(1 for r in workshop_records if is_passing(r))
    print(f"Sequential workshop complete.")
    print(f"Sim log:         {out_sim_log}")
    print(f"Total simulated: {len(workshop_records)}")
    print(f"Passing:         {n_pass}  ({n_promoted} promoted to templates)")
    print(f"\nTo review: python tools/show_results.py --sim-log {out_sim_log}")


def _run_per_field(
    selected: list[dict],
    all_records: list[dict],
    known_exprs: set[str],
    session,
    settings: dict,
    out_sim_log: Path,
    args,
    portfolio_alpha_data: dict | None = None,
) -> None:
    """Process each candidate field one at a time through the full workshop+template loop."""
    total = len(selected)
    field_results: list[tuple[str, str, int, int]] = []  # (field, action, n_sim, n_pass)

    for i, record in enumerate(selected):
        field = _make_field_signature(record)
        action = _resolved_route(record, bypass_self_corr=args.bypass_correlation_check) or "?"
        sharpe = record.get("is_sharpe") or 0.0
        expr = record.get("expression", "")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}]  {field}  |  {action}  |  sharpe={sharpe:.3f}")
        print(f"  {expr[:100]}")
        print(f"{'='*60}")

        # Track sim log size before this field to measure what was added
        pre_count = len(_load_records([str(out_sim_log)])) if out_sim_log.exists() else 0

        _workshop_one(
            [record], all_records, known_exprs,
            session, settings, out_sim_log, args,
            label=f"[{field}] ",
            portfolio_alpha_data=portfolio_alpha_data,
        )

        post_records = _load_records([str(out_sim_log)])
        field_new = post_records[pre_count:]
        n_pass = sum(1 for r in field_new if is_passing(r))
        field_results.append((field, action, len(field_new), n_pass))
        print(f"\n  → {field}: {len(field_new)} simulated, {n_pass} passing")

    # Final template promotion from all workshop results
    print(f"\n{'='*60}")
    n_promoted = _promote_to_templates(
        out_sim_log,
        session=session,
        portfolio_alpha_data=portfolio_alpha_data,
        competition_id=args.competition_id,
        admission_gate=not args.skip_admission_gate,
        min_brain_score_delta=args.admission_min_brain_score_delta,
        min_fallback_delta_sharpe=args.admission_min_fallback_delta_sharpe,
        pnl_sleep=args.pnl_sleep,
        admission_pnl_attempts=args.admission_pnl_attempts,
        per_seed_top_n=args.per_seed_top_n,
    )
    workshop_records = _load_records([str(out_sim_log)])
    n_pass_total = sum(1 for r in workshop_records if is_passing(r))

    print(f"Per-field workshop complete — {total} fields processed")
    print(f"{'─'*60}")
    print(f"  {'field':<35} {'action':<8} {'sim':>5} {'pass':>5}")
    print(f"  {'─'*35} {'─'*8} {'─'*5} {'─'*5}")
    for field, action, n_sim, n_pass in field_results:
        marker = " ✓" if n_pass else ""
        print(f"  {field:<35} {action:<8} {n_sim:>5} {n_pass:>5}{marker}")
    print(f"{'─'*60}")
    print(f"Total: {len(workshop_records)} simulated, {n_pass_total} passing, {n_promoted} templates promoted")
    print(f"\nSim log:  {out_sim_log}")
    print(f"Submittable alphas: {SUBMITTABLE_JSONL}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show tier counts and candidate list, write audit files, then exit.",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Print the candidate table before running (still runs unless --dry-run).",
    )
    parser.add_argument(
        "--from-audit", metavar="TXT",
        help="Run only candidates from a previously-written (and edited) audit txt file.",
    )
    parser.add_argument(
        "--ignore-workshopped", action="store_true",
        help="Do not skip already-workshopped sources (re-run the full selection/audit even "
             "if some seeds already have results). Useful to force-retry seeds whose expression "
             "appears in the sim-log scan and so cannot be freed by editing workshopped_sources.txt.",
    )
    parser.add_argument(
        "--source", choices=["all", "grammar"], default="all",
        help="Limit input sim logs: 'grammar' reads only alpha_discovery/grammar_method/artifacts/all_sim.jsonl.",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete all ephemeral workshop files (audit, summary, new_templates). Keeps workshopped_sources.txt.",
    )
    parser.add_argument(
        "--lane", nargs="*", choices=["negate", "repair", "enhance"],
        help="Process only specified lanes (default: all).",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=200,
        help="Max repair/enhance candidates to process per run (default 200, 0 = unlimited).",
    )
    parser.add_argument(
        "--max-negate-candidates", type=int, default=50,
        help="Max negate candidates to process per run, independent from --max-candidates (default 50, 0 = unlimited).",
    )
    parser.add_argument(
        "--workshop-iterations", type=int, default=3,
        help="Auto-workshop rounds on each batch's results (default 3).",
    )
    parser.add_argument(
        "--exclude-expr", nargs="*", default=[],
        help="Expressions to exclude (externally submitted alphas).",
    )
    parser.add_argument(
        "--fake-cw-threshold", type=float, default=None, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--spectacular-threshold", type=float, default=None, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sim-log", nargs="*",
        help="Override sim log paths/globs to scan.",
    )
    parser.add_argument(
        "--out-sim-log", default=str(DEFAULT_WORKSHOP_SIM),
        help=f"Output sim log for workshop results (default: {DEFAULT_WORKSHOP_SIM}).",
    )
    parser.add_argument(
        "--per-field", action="store_true",
        help=(
            "Process each field/signal one at a time through the full workshop+template loop, "
            "then move to the next. Good for audited lists from grammar."
        ),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument(
        "--competition-id", default="IQC2026S1",
        help="Competition ID for loading portfolio PnL used in inline correlation backfill "
             "(default: IQC2026S1).",
    )
    parser.set_defaults(score_seeds=True)
    parser.add_argument(
        "--score-seeds", dest="score_seeds", action="store_true",
        help=(
            "Fetch official Brain before/after score for seed alphas before routing (default). "
            "Repair/enhance seeds continue only when the fetched score delta is positive. "
            "Negate seeds are scored for reporting only and are not gated by score. "
            "Requires Brain login (uses --credentials). Works with --dry-run."
        ),
    )
    parser.add_argument(
        "--no-score-seeds", dest="score_seeds", action="store_false",
        help="Disable seed Brain score fetching before routing.",
    )
    parser.add_argument(
        "--skip-portfolio-pnl", action="store_true",
        help=(
            "Skip loading portfolio PnL at startup. This disables inline self-correlation "
            "computation, LLM portfolio-impact feedback, and the local fallback used by "
            "the final admission gate. Seed score fetching is still controlled by --score-seeds."
        ),
    )
    parser.add_argument(
        "--pnl-sleep", type=float, default=1.0, metavar="SECONDS",
        help="Sleep between Brain score/PnL fetch retries (default: 1.0).",
    )
    parser.add_argument(
        "--yearly-stats-attempts",
        type=int,
        default=1,
        help="Retry yearly-stats fetch this many times after each completed workshop simulation.",
    )
    parser.add_argument(
        "--yearly-stats-sleep",
        type=float,
        default=30.0,
        help="Seconds to wait between yearly-stats retry attempts.",
    )
    parser.add_argument(
        "--bypass-correlation-check", action="store_true",
        help="Bypass the workshop self-correlation gate for repair/enhance routing.",
    )
    parser.add_argument(
        "--skip-admission-gate", action="store_true",
        help="Promote all standalone-passing workshop alphas to templates without portfolio admission scoring.",
    )
    parser.add_argument(
        "--admission-min-brain-score-delta", type=float, default=0.0,
        help="Minimum official Brain score lift for template promotion when available (default 0).",
    )
    parser.add_argument(
        "--admission-min-fallback-delta-sharpe", type=float, default=0.0,
        help="Minimum local pool Sharpe delta when Brain score comparison is unavailable after retries (default 0).",
    )
    parser.add_argument(
        "--admission-pnl-attempts", type=int, default=3,
        help="Max PnL fetch attempts per candidate during admission scoring (default 3).",
    )
    parser.add_argument(
        "--per-seed-top-n", type=int, default=5,
        help="Keep the top N admission-ranked promoted variants per root seed in workshop_submittable.jsonl (default 5, 0 = all).",
    )
    args = parser.parse_args()
    if args.fake_cw_threshold is not None or args.spectacular_threshold is not None:
        print("  [workshop] --fake-cw-threshold/--spectacular-threshold are deprecated and ignored; routing is now shared.")

    # ── Cleanup mode ─────────────────────────────────────────────────────
    if args.cleanup:
        removed = []
        for p in _CLEANUP_FILES:
            if p.exists():
                p.unlink()
                removed.append(str(p))
        if removed:
            print("Removed ephemeral files:")
            for p in removed:
                print(f"  {p}")
        else:
            print("Nothing to clean up.")
        print(f"Kept: {WORKSHOPPED_SOURCES_FILE}")
        return

    # ── Determine audit file paths (grammar prefix when --source grammar) ─
    if args.source == "grammar":
        audit_txt  = WORKSHOP_DIR / "grammar_workshop_audit.txt"
        audit_jsonl = WORKSHOP_DIR / "grammar_workshop_audit.jsonl"
    else:
        audit_txt  = WORKSHOP_AUDIT_TXT
        audit_jsonl = WORKSHOP_AUDIT_JSONL

    # ── From-audit mode: skip selection, load pre-approved list ──────────
    if args.from_audit:
        from_txt = Path(args.from_audit)
        from_jsonl = from_txt.with_suffix(".jsonl")
        print(f"Loading candidates from audit: {from_txt}")
        audit_rows = _read_from_audit(from_txt, from_jsonl)
        # audit rows are lightweight dicts — we need full expressions
        # The 'expression' field is already stored in the JSONL
        # Re-wrap as minimal records for generate_candidates
        selected_keys = {_audit_row_key(row) for row in audit_rows}
        # Load all records to build context
        ctx_paths = _resolve_input_paths(args)
        all_records = _filter_records_for_source(_load_records(ctx_paths, tag_source=True), args)
        # Find the full sim records matching the audit selection
        selected = [
            r for r in all_records
            if record_workshop_key(r) in selected_keys and r.get("status") == "done"
        ]
        # Attach context_records from audit rows.
        # Only include context expressions whose parent row survived the audit edit —
        # expressions belonging to deleted rows (and their own context) are blocked.
        full_jsonl = from_txt.with_suffix(".jsonl")
        all_audit_by_key: dict[str, dict] = {}
        with full_jsonl.open() as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        _row = json.loads(_line)
                        all_audit_by_key[_audit_row_key(_row)] = _row
                    except (json.JSONDecodeError, KeyError):
                        pass

        deleted_keys = {
            key for key in all_audit_by_key if key not in selected_keys
        }
        blocked_keys = set(deleted_keys)
        reviewed_entries: list[dict] = []
        for key, row in all_audit_by_key.items():
            if key in deleted_keys:
                blocked_keys.update(_audit_context_keys(row))
                entry = _review_entry_from_audit_row(
                    row,
                    decision="REJECTED_BY_AUDIT",
                    reason="removed from audit before workshop run",
                )
                if entry:
                    reviewed_entries.append(entry)
                reviewed_entries.extend(_review_entries_for_context(
                    row,
                    decision="REJECTED_AUDIT_CONTEXT",
                    reason="context belonged to an audit row removed before workshop run",
                ))
        _append_reviewed_entries(reviewed_entries)

        key_to_ctx = {
            _audit_row_key(row): _audit_context_keys(row)
            for row in audit_rows
        }
        ctx_lookup = {record_workshop_key(r): r for r in all_records}
        audit_order = {_audit_row_key(row): idx for idx, row in enumerate(audit_rows)}
        updated = []
        for r in selected:
            r = dict(r)
            r["_route"] = route_record(r, bypass_self_corr=args.bypass_correlation_check)
            expr_key = record_workshop_key(r)
            ctx_keys = [
                key for key in key_to_ctx.get(expr_key, [])
                if key not in blocked_keys
            ]
            if ctx_keys:
                r["context_records"] = [
                    ctx_lookup[key] for key in ctx_keys if key in ctx_lookup
                ]
            updated.append(r)
        selected = sorted(updated, key=lambda r: audit_order.get(record_workshop_key(r), 10**9))
        print(f"  {len(selected)} candidates from audit ({len(audit_rows)} rows in txt)")
        workshopped_exprs = set() if args.ignore_workshopped else _load_workshopped_sources()
        selected = [r for r in selected if record_workshop_key(r) not in workshopped_exprs]
        print(f"  {len(selected)} after skipping already-workshopped")
    else:
        # ── Load all sim records ──────────────────────────────────────────
        paths = _resolve_input_paths(args)

        print(f"Loading from {len(paths)} sim log(s)...")
        all_records = _filter_records_for_source(_load_records(paths, tag_source=True), args)
        print(f"  {len(all_records)} total records")

        # ── Already-workshopped tracking ──────────────────────────────────
        workshopped_exprs = set() if args.ignore_workshopped else _load_workshopped_sources()
        if workshopped_exprs:
            print(f"  Already workshopped: {len(workshopped_exprs)} source key(s) — will skip")

        excluded_exprs: set[str] = DEFAULT_EXCLUDED_EXPRS | set(args.exclude_expr or [])

        # ── Select candidates ─────────────────────────────────────────────
        print("\nSelecting candidates...")
        selected = _select_candidates_for_args(
            all_records,
            workshopped_exprs,
            excluded_exprs,
            args,
        )

    # ── Optional/default: fetch official Brain score deltas for source seeds ─
    if args.score_seeds:
        from core.brain.api import sign_in

        # Fetch all seed scores for reporting. Negate seeds are still routed
        # independently by reroute_after_seed_scoring(); their score is not a gate.
        score_targets = list(selected)
        if score_targets:
            print("\nFetching official Brain score deltas for source seeds...")
            _score_session = sign_in(args.credentials)
            scored = prescore_seeds(
                score_targets,
                _score_session,
                competition_id=args.competition_id,
                pnl_sleep=args.pnl_sleep,
            )
            scored_by_key = {record_workshop_key(r): r for r in scored}
            selected = [scored_by_key.get(record_workshop_key(r), r) for r in selected]
        selected = reroute_after_seed_scoring(selected, args, route_fn=route_record)

    n_negate = sum(1 for r in selected if _resolved_route(r, bypass_self_corr=args.bypass_correlation_check) == "negate")
    n_repair = sum(1 for r in selected if _resolved_route(r, bypass_self_corr=args.bypass_correlation_check) == "repair")
    n_enhance = sum(1 for r in selected if _resolved_route(r, bypass_self_corr=args.bypass_correlation_check) == "enhance")
    print(f"\nSelected: {len(selected)} candidates — {n_negate} negate | {n_repair} repair | {n_enhance} enhance")

    if args.list_only or args.dry_run:
        print_table(selected)

    if args.dry_run:
        _write_audit(selected, audit_txt, audit_jsonl)
        return

    if not selected:
        print("Nothing to workshop.")
        return

    # ── Full auto-workshop pipeline ───────────────────────────────────────
    from core.brain.api import sign_in
    from core.brain.submit import DEFAULT_SETTINGS

    session = sign_in(args.credentials)
    settings = dict(DEFAULT_SETTINGS)
    out_sim_log = Path(args.out_sim_log)
    portfolio_alpha_data: dict | None = None
    if args.score_seeds:
        # Normally already handled before dry-run/full-run branching. Keep this
        # here for programmatic callers or audit files containing unscored rows.
        unscored = [
            r for r in selected
            if r.get("admission_brain_score_delta") is None
        ]
        if unscored:
            print(f"\nFetching official Brain score deltas for {len(unscored)} unscored source seed(s)...")
            scored = prescore_seeds(
                unscored,
                session,
                competition_id=args.competition_id,
                pnl_sleep=args.pnl_sleep,
            )
            scored_by_key = {record_workshop_key(r): r for r in scored}
            selected = [scored_by_key.get(record_workshop_key(r), r) for r in selected]
            selected = reroute_after_seed_scoring(selected, args, route_fn=route_record)
            n_neg = sum(1 for r in selected if r.get("_route") == "negate")
            n_rep = sum(1 for r in selected if r.get("_route") == "repair")
            n_enh = sum(1 for r in selected if r.get("_route") == "enhance")
            print(f"After seed score fetch: {len(selected)} candidates — {n_neg} negate | {n_rep} repair | {n_enh} enhance")
            if not selected:
                print("Nothing to workshop after seed score fetch.")
                return

    if not args.skip_portfolio_pnl:
        from core.selfcorr import load_portfolio_alpha_data

        portfolio_alpha_data = load_portfolio_alpha_data(
            session,
            competition_id=args.competition_id,
        )

    known_exprs: set[str] = {record_workshop_key(r) for r in all_records if r.get("expression", "")}

    if args.per_field:
        _run_per_field(
            selected, all_records, known_exprs,
            session, settings, out_sim_log, args,
            portfolio_alpha_data=portfolio_alpha_data,
        )
    else:
        _run_batch(
            selected, all_records, known_exprs,
            session, settings, out_sim_log, args,
            portfolio_alpha_data=portfolio_alpha_data,
        )

    if out_sim_log.exists():
        n_new, n_updated = merge_session_to_master(out_sim_log)
        from core.logs import MASTER
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")

        # Post-workshop knowledge compounding: refresh the wiki structured layer
        # and changed deterministic pages (no LLM cost; narrative pages are
        # regenerated separately via tools/build_wiki.py --llm).
        try:
            from core.wiki.generate import generate_wiki
            counters = generate_wiki(use_llm=False)
            print(f"Wiki refreshed: {counters}")
        except Exception as exc:
            print(f"Wiki refresh skipped: {exc}")


if __name__ == "__main__":
    main()
