"""Auto-workshop: route sim records to negate/repair/enhance and generate candidates.

Core logic shared by:
  - tools/alpha_workshop.py  (manual, per-expression)
  - alpha_search/submit.py   (auto, post-submission)
  - tools/batch_workshop.py  (history/audit-driven workshop)
"""

from __future__ import annotations
from core.quality import failing_checks, is_passing, resolve_checks

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from core.robustness import record_robustness_score

# Checks that are structurally fixable by wrapping operators.
# Any other failing check indicates fundamental signal weakness — abandon, don't repair.
REPAIRABLE_CHECKS: frozenset[str] = frozenset({"CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE"})
NEGATE_FITNESS_THRESHOLD = -1.2
REPAIR_CONTRIBUTION_DELTA_MIN = 0.0
REPAIR_SECONDARY_MIN_FITNESS = 0.9
ENHANCE_SECONDARY_MIN_FITNESS = 1.0
ENHANCE_ROBUSTNESS_MIN = 0.30
WORKSHOP_CONCENTRATION_LIMIT = 0.40
WORKSHOP_SELF_CORR_LIMIT = 0.70
WORKSHOP_SELF_CORR_SHARPE_MULTIPLIER = 1.10
DEFAULT_WORKSHOP_SETTINGS: dict[str, Any] = {
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.01,
    "universe": "TOP3000",
}


def _expr_tokens(expr: str) -> frozenset[str]:
    """Extract all identifiers (operators + field names) from a FASTEXPR string."""
    return frozenset(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr))


def _expr_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two FASTEXPR strings."""
    ta, tb = _expr_tokens(a), _expr_tokens(b)
    union = ta | tb
    if not union:
        return 1.0
    return len(ta & tb) / len(union)


def _looks_like_fastexpr(expr: str) -> bool:
    """Return True only if expr looks like a FASTEXPR expression, not prose."""
    s = expr.strip()
    if not s:
        return False
    # Must start with an identifier (optionally unary-negated) followed by ( or be a plain field
    if re.match(r'^-?\(', s):
        return True  # -(expr) form is valid
    if not re.match(r'^-?[a-zA-Z_][a-zA-Z0-9_]*', s):
        return False
    # If it has spaces but no parens, it's prose
    if ' ' in s and '(' not in s:
        return False
    # Reject lines that are clearly sentences (contain common prose words)
    lower = s.lower()
    if any(lower.startswith(w) for w in ('the ', 'this ', 'looking ', 'key ', 'note ', 'here ', 'use ')):
        return False
    return True


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def regime_from_settings(settings: dict | None) -> str:
    """Canonical regime string for a settings dict."""
    from core.brain.submit import _regime_key

    merged = {**DEFAULT_WORKSHOP_SETTINGS, **(settings or {})}
    return _regime_key(merged)


def record_regime(record: dict) -> str:
    """Return the canonical regime string for a record."""
    settings = record.get("settings")
    if not isinstance(settings, dict):
        settings = record.get("settings_override")
    if isinstance(settings, dict):
        return regime_from_settings(settings)
    return str(record.get("regime") or regime_from_settings(None))


def workshop_expr_key(expression: str, regime: str) -> str:
    """Stable key for an expression scoped to a regime."""
    return f"{regime}\t{expression}"


def record_workshop_key(record: dict) -> str:
    """Stable regime-aware key for a record."""
    return workshop_expr_key(record.get("expression", ""), record_regime(record))


def source_settings_override(record: dict) -> dict[str, Any]:
    """Effective settings to preserve when workshopping a source record."""
    settings = record.get("settings")
    return dict(settings) if isinstance(settings, dict) else {}


def _candidate_id(expr: str, regime: str = "") -> str:
    return hashlib.sha1(f"{regime}\n{expr}".encode()).hexdigest()[:16]




# Checks that are warnings only and never block submission or workshop routing.
IGNORED_CHECKS: frozenset[str] = frozenset({"SELF_CORRELATION", "UNITS"})






def _admission_contribution_delta(record: dict) -> float | None:
    """Return Brain score delta, or fallback pool Sharpe delta when Brain is unavailable."""
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    for value in (
        record.get("admission_brain_score_delta"),
        record.get("brain_score_delta"),
        admission.get("brain_score_delta"),
    ):
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    for value in (
        record.get("admission_delta_pool_sharpe"),
        admission.get("delta_pool_sharpe"),
    ):
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def route_record(
    record: dict,
    neg_threshold: float | None = None,
    min_repair_sharpe: float | None = None,
    *,
    bypass_self_corr: bool = False,
) -> str | None:
    """Return 'negate', 'repair', 'enhance', or None (skip).

    Deprecated args (`neg_threshold`, `min_repair_sharpe`) are accepted for
    compatibility but ignored. Routing is now shared across batch/manual
    workshop paths and uses Brain score delta when admission data is present,
    falling back to local same-delay Sharpe delta only when Brain is unavailable.
    """
    if record.get("status") != "done":
        return None
    fitness = record.get("is_fitness")
    if fitness is not None and fitness <= NEGATE_FITNESS_THRESHOLD:
        if _has_extreme_concentration(record):
            return None
        return "negate"

    score_delta = _admission_contribution_delta(record)
    failing = failing_checks(record)

    # ── Repair routing ────────────────────────────────────────────────────
    if failing:
        # Repairable failures are worth workshopping when concentration is not
        # extreme and the alpha is either known positive-contribution or not yet scored.
        if (
            fitness is not None and fitness >= REPAIR_SECONDARY_MIN_FITNESS
            and _is_repair_candidate(record, failing, bypass_self_corr=bypass_self_corr)
        ):
            return "repair"
        return None

    # ── Enhance routing ───────────────────────────────────────────────────
    if not (is_passing(record) and workshop_self_corr_eligible(record, bypass_self_corr=bypass_self_corr)):
        return None
    if fitness is None or fitness < ENHANCE_SECONDARY_MIN_FITNESS:
        return None

    # Passing alphas are provisional enhance seeds until scored. Once scored,
    # only strictly positive Brain score delta, or fallback delta when Brain is
    # unavailable, remains eligible.
    if score_delta is not None and score_delta <= 0.0:
        return None
    # Year-consistency gate: a known-fragile alpha (yearly stats present and
    # robustness below the floor) is an IS spike — enhancing it polishes noise.
    # Missing robustness data does not block, same convention as self-corr.
    rob = record_robustness_score(record)
    if rob is not None and rob < ENHANCE_ROBUSTNESS_MIN:
        return None
    return "enhance"


def workshop_self_corr_eligible(record: dict, *, bypass_self_corr: bool = False) -> bool:
    """Return True when a record is eligible for repair/enhance by self-corr rules.

    Records with no correlation data (max_self_correlation is None) are treated
    as eligible — absence of data is not a reason to block workshop processing.
    """
    if bypass_self_corr:
        return True
    corr = record.get("max_self_correlation")
    if corr is None:
        return True  # no correlation data → assume eligible
    if corr < WORKSHOP_SELF_CORR_LIMIT:
        return True

    benchmark_sharpe = record.get("max_self_correlation_alpha_sharpe")
    sharpe = record.get("is_sharpe")
    if sharpe is None or benchmark_sharpe is None or benchmark_sharpe <= 0.0:
        return False
    return sharpe + 1e-12 >= WORKSHOP_SELF_CORR_SHARPE_MULTIPLIER * benchmark_sharpe


def _is_repair_candidate(
    record: dict,
    failing: list[str] | None = None,
    *,
    bypass_self_corr: bool = False,
    skip_delta_check: bool = False,
) -> bool:
    """Return True when a record qualifies for the repair lane."""
    if not workshop_self_corr_eligible(record, bypass_self_corr=bypass_self_corr):
        return False

    if not skip_delta_check:
        score_delta = _admission_contribution_delta(record)
        if score_delta is not None and score_delta <= REPAIR_CONTRIBUTION_DELTA_MIN:
            return False

    failing_checks_list = failing if failing is not None else failing_checks(record)
    if not failing_checks_list or not all(c in REPAIRABLE_CHECKS for c in failing_checks_list):
        return False

    if "CONCENTRATED_WEIGHT" in failing_checks_list:
        max_concentration = _resolved_concentration_value(record)
        if max_concentration is None or max_concentration >= WORKSHOP_CONCENTRATION_LIMIT:
            return False

    return True


def _resolved_concentration_value(record: dict) -> float | None:
    """Return the best available concentration value for workshop gating."""
    max_concentration = record.get("max_concentration")
    if max_concentration is not None:
        return float(max_concentration)

    for check in resolve_checks(record):
        if check.get("name") != "CONCENTRATED_WEIGHT":
            continue
        value = check.get("value")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        break
    return None


def _has_extreme_concentration(record: dict) -> bool:
    """Return True for fake-signal concentration at or above the workshop limit."""
    max_concentration = _resolved_concentration_value(record)
    return max_concentration is not None and max_concentration >= WORKSHOP_CONCENTRATION_LIMIT


def collect_trial_history(source_record: dict, all_records: list[dict]) -> list[dict]:
    """Return all previously simulated repair variants of source_record.

    Searches all_records for entries whose params.original_expression matches
    the source expression in the same regime.
    Sorted best→worst Sharpe so the LLM sees the gradient clearly.
    """
    original_expr = source_record.get("expression", "")
    target_regime = record_regime(source_record)
    trials = []
    for r in all_records:
        if r.get("status") != "done":
            continue
        params = r.get("params") or {}
        if params.get("original_expression") == original_expr and record_regime(r) == target_regime:
            trials.append(r)
    trials.sort(key=lambda r: r.get("is_sharpe") or 0.0, reverse=True)
    return trials


def find_passing_examples(all_records: list[dict], target: dict, n: int = 8) -> list[dict]:
    """Return up to n passing alphas from the same dataset, sorted by Sharpe desc."""
    dataset = target.get("dataset_id", "")
    target_regime = record_regime(target)
    out = [
        r for r in all_records
        if r.get("status") == "done"
        and r.get("dataset_id") == dataset
        and record_regime(r) == target_regime
        and (r.get("is_sharpe") or 0.0) >= 1.25
        and (r.get("is_fitness") or 0.0) >= 1.0
        and is_passing(r)
    ]
    out.sort(key=lambda r: r.get("is_sharpe") or 0.0, reverse=True)
    return out[:n]


def _make_candidate(expr: str, source: dict, lane: str, params: dict) -> dict:
    regime = record_regime(source)
    return {
        "candidate_id": _candidate_id(expr, regime),
        "template_id": f"{lane}:{_candidate_id(source.get('expression', ''), regime)}",
        "dataset_id": source.get("dataset_id", ""),
        "category": source.get("category", ""),
        "search_lane": lane,
        "expression": expr,
        "settings_override": source_settings_override(source),
        "fields": source.get("fields") or [],
        "cluster_ids": source.get("cluster_ids") or [],
        "group": source.get("group", ""),
        "params": params,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_candidates(
    new_records: list[dict],
    all_records: list[dict],
    *,
    history: "Any | None" = None,
    neg_threshold: float | None = None,
    api_key: str | None = None,
    known_exprs: set[str] | None = None,
    max_variants: int = 5,
    verbose: bool = True,
    skip_negate: bool = False,
    origin_notes: dict[str, str] | None = None,
    bypass_self_corr: bool = False,
) -> list[dict]:
    """Route new_records and generate negate/repair/enhance candidates.

    Args:
        new_records:    Records from the most recent submission batch.
        all_records:    Full sim history (for passing-examples LLM context).
        history:        Optional WorkshopHistory instance.  When provided,
                        prior attempts on the root seed are prepended to
                        every LLM call (repair trial_history + enhance
                        context_records) so the model sees the full gradient
                        across all lanes and iterations.  Lineage for each
                        emitted candidate expression is registered in-place.
        neg_threshold:  Deprecated compatibility argument; ignored.
        api_key:        OpenAI API key (for repair/enhance LLM calls).
        known_exprs:    Set of already-seen regime-aware expression keys;
                        updated in-place to prevent duplicates across iterations.
        max_variants:   Max LLM variants per alpha.
        verbose:        Print routing/generation status.
        skip_negate:    When True, skip the negate lane entirely (used by the
                        sequential pipeline after negate has already run).
        origin_notes:   Optional mapping of regime-aware expression key → note
                        string prepended to the LLM user message (e.g. for
                        negate-originated records: "this alpha is a negated
                        version of: <orig>").
        bypass_self_corr:
                        When True, ignore the workshop self-correlation gate
                        for repair/enhance routing. Intended for explicit CLI
                        override workflows only.

    Returns:
        List of candidate dicts ready for submission via alpha_search.submit.
    """
    from alpha_enhancement.workshop.repair import _sign_flip

    if known_exprs is None:
        known_exprs = {record_workshop_key(r) for r in all_records if r.get("expression", "")}

    notes = origin_notes or {}

    negate_q: list[dict] = []
    repair_q: list[dict] = []
    enhance_q: list[dict] = []

    for r in new_records:
        action = route_record(r, bypass_self_corr=bypass_self_corr)
        if action == "negate":
            if not skip_negate:
                negate_q.append(r)
        elif action == "repair":
            repair_q.append(r)
        elif action == "enhance":
            enhance_q.append(r)

    if verbose:
        print(
            f"  [workshop] {len(negate_q)} negate | "
            f"{len(repair_q)} repair | {len(enhance_q)} enhance"
        )

    candidates: list[dict] = []

    # ── Negations (deterministic, no LLM) ────────────────────────────────
    for r in negate_q:
        original = r.get("expression", "")
        regime = record_regime(r)
        negated = _sign_flip(original)
        negated_key = workshop_expr_key(negated, regime)
        if negated == original or negated_key in known_exprs:
            continue
        known_exprs.add(negated_key)
        if history is not None:
            history.register_lineage(negated_key, workshop_expr_key(original, regime))
        candidates.append(_make_candidate(negated, r, "negate", {
            "original_expression": original,
            "root_expression": (r.get("params") or {}).get("root_expression") or original,
            "original_sharpe": r.get("is_sharpe"),
        }))
        if verbose:
            print(f"  [negate]  {original[:55]}  →  {negated[:55]}")

    # ── Structural diversity guard (enhance queue only) ───────────────────
    # Reject enhancement variants that are token-Jaccard similar (>= 0.70) to
    # any already-emitted enhance candidate this session. Prevents generating
    # many structurally near-identical variants (e.g. 6x group_backfill(anl4_ptp_flag,...))
    # regardless of which source alpha they came from.
    _enhance_emitted_exprs: dict[str, list[str]] = {}

    def _is_structurally_novel_enhance(expr: str, regime: str) -> bool:
        return all(_expr_jaccard(expr, seen) < 0.70 for seen in _enhance_emitted_exprs.get(regime, []))

    # ── Repairs + Enhancements (LLM) ─────────────────────────────────────
    llm_queue = [("repair", r) for r in repair_q] + [("enhance", r) for r in enhance_q]
    if not llm_queue:
        return candidates

    from core import llm as _llm
    resolved_key = _llm.load_api_key(api_key)
    if not resolved_key:
        print("  [workshop] No Anthropic API key — skipping LLM repair/enhance")
        return candidates

    try:
        from alpha_enhancement.agents.repair import AlphaRepairAgent
        agent = AlphaRepairAgent(api_key=resolved_key)
    except Exception as e:
        print(f"  [workshop] AlphaRepairAgent unavailable: {e}")
        return candidates

    for lane, r in llm_queue:
        original = r.get("expression", "")
        regime = record_regime(r)
        orig_from_params = (r.get("params") or {}).get("original_expression") or original
        root_expression = (r.get("params") or {}).get("root_expression") or orig_from_params
        passing_examples = find_passing_examples(all_records, r)
        note = notes.get(workshop_expr_key(original, regime), "")

        # ── History context injection ──────────────────────────────────
        if history is not None:
            root = history.root_for(
                workshop_expr_key(original, regime),
                workshop_expr_key(orig_from_params, regime),
            )
            history_repair_ctx = history.context_for_repair(root)
            history_enhance_ctx = history.context_for_enhance(root)
            if verbose and (history_repair_ctx or history_enhance_ctx):
                n = len(history_repair_ctx) if lane == "repair" else len(history_enhance_ctx)
                print(f"  [{lane}]   history: {n} prior attempts for root seed")
        else:
            root = original
            history_repair_ctx = []
            history_enhance_ctx = []

        # Natural context_records from the record (field-family variants)
        natural_ctx = list(r.get("context_records") or [])

        try:
            if lane == "repair":
                log_trials = collect_trial_history(r, all_records)
                # Merge: history (cross-lane prior attempts) + log trials (same-expr prior repairs)
                combined_trials = history_repair_ctx + [
                    t for t in log_trials
                    if t.get("expression") not in {h.get("expression") for h in history_repair_ctx}
                ]
                result = agent.repair(
                    r,
                    passing_examples=passing_examples,
                    trial_history=combined_trials or None,
                    context_records=natural_ctx or None,
                    origin_note=note,
                )
            else:
                # Enhance: prepend history context to natural field-family context
                ctx_records = history_enhance_ctx + natural_ctx
                result = agent.enhance(
                    r,
                    passing_examples=passing_examples,
                    context_records=ctx_records or None,
                    origin_note=note,
                )
        except Exception as e:
            if verbose:
                print(f"  [{lane}] LLM error for {original[:50]}: {e}")
            continue

        seen_this: set[str] = {original}
        for expr in result.repair_expressions[:max_variants]:
            expr_key = workshop_expr_key(expr, regime)
            if not expr or expr in seen_this or expr_key in known_exprs:
                continue
            if not _looks_like_fastexpr(expr):
                if verbose:
                    print(f"  [{lane}]   SKIP (not FASTEXPR): {expr[:60]}")
                continue
            if lane == "enhance" and not _is_structurally_novel_enhance(expr, regime):
                if verbose:
                    print(f"  [enhance]  SKIP (structurally similar to prior enhance): {expr[:60]}")
                continue
            seen_this.add(expr)
            known_exprs.add(expr_key)
            if lane == "enhance":
                _enhance_emitted_exprs.setdefault(regime, []).append(expr)
            if history is not None:
                history.register_lineage(expr_key, workshop_expr_key(original, regime))
            if lane == "repair":
                params: dict[str, Any] = {
                    "original_expression": original,
                    "root_expression": root_expression,
                    "original_sharpe": r.get("is_sharpe"),
                    "original_failing_checks": failing_checks(r),
                }
            else:
                params = {
                    "original_expression": original,
                    "root_expression": root_expression,
                    "original_sharpe": r.get("is_sharpe"),
                    "original_fitness": r.get("is_fitness"),
                }
            candidates.append(_make_candidate(expr, r, lane, params))
            if verbose:
                print(f"  [{lane}]   → {expr[:80]}")

    return candidates
