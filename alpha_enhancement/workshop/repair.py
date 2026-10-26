"""Near-miss repair: scan simulation logs for high-Sharpe failing alphas and
generate LLM-based repair variants.

The LLM receives:
  - The failing expression + check results with measured values
  - Similar passing alphas from the same dataset (few-shot positive examples)
  - Previously successful repair pairs from repair history (if available)

This feedback-loop approach lets the LLM derive repairs from evidence rather
than from hard-coded heuristics.

Special case: sign-flip (sharpe <= FLIP_THRESHOLD) is handled structurally
before the LLM call — it's a deterministic negation, not a heuristic.

Usage:
    python -m alpha_search.repair
        --sim-log artifacts/model16_sim.jsonl
        --output artifacts/repair_batch.jsonl
        --min-sharpe 1.0
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Minimum Sharpe to consider a near-miss worth repairing.
DEFAULT_MIN_SHARPE = 1.0

# Sharpe at or below this triggers a sign-flip repair (strong inverse signal).
DEFAULT_FLIP_THRESHOLD = -1.5

# ---------------------------------------------------------------------------
# Expression structure helpers (depth-based, not regex)
# ---------------------------------------------------------------------------

def _outermost_call(expr: str) -> tuple[str, list[str]] | None:
    """Parse the outermost function call in expr using a paren-depth scanner.

    Returns (operator_name, [arg0, arg1, ...]) where each arg is a trimmed
    substring, or None if expr is not a function call (leaf node or infix).

    Works correctly for any nesting depth and formatting — does not rely on
    regex patterns tied to specific operator names or argument counts.
    """
    expr = expr.strip()
    paren_pos = expr.find("(")
    if paren_pos <= 0:
        return None  # leaf node or starts with '(' (parenthesised sub-expr)

    op = expr[:paren_pos].strip()
    # The argument list runs from paren_pos+1 to the matching closing paren.
    depth = 0
    args: list[str] = []
    current: list[str] = []
    for i, ch in enumerate(expr[paren_pos:], start=paren_pos):
        if ch == "(":
            if depth > 0:
                current.append(ch)
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                args.append("".join(current).strip())
                break
            else:
                current.append(ch)
        elif ch == "," and depth == 1:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    return op, args


def _reconstruct(op: str, args: list[str]) -> str:
    """Reconstruct a function call string from operator name and arg list."""
    return f"{op}({', '.join(args)})"


# _TRANSPARENT_OPS: f(-x) == -f(x), so negation passes through unchanged.
# Includes symmetric wrappers AND group ops that are odd functions:
#   group_neutralize(x, g) = x - mean_g(x)  →  group_neutralize(-x, g) = -group_neutralize(x, g)
#   group_zscore, group_scale: same odd-function property.
# We recurse into the first arg of these — pushing the negation deeper.
_TRANSPARENT_OPS = frozenset({
    "winsorize", "scale", "zscore", "hump",
    "group_neutralize", "group_zscore", "group_scale",
})

# _RANK_OPS: negating the first arg reverses rank order — this is the semantic
# sign flip we want. Stop recursing here.
#   group_rank(-x, g)  →  reversed ranking within group
#   rank(-x)           →  reversed universe ranking
_RANK_OPS = frozenset({
    "group_rank", "rank",
})


def _sign_flip(expr: str) -> str:
    """Return the sign-flipped expression, pushing negation as deep as possible.

    Uses a paren-depth scanner to parse the outermost call — handles any
    expression format or nesting depth without regex pattern matching.

    Traversal:
      - Transparent wrappers (winsorize, scale, rank, zscore, hump): recurse
        into first arg, keep remaining args unchanged.
      - Group ops (group_rank, group_zscore, group_neutralize, group_scale):
        negate the first arg and stop — reversing the signal at the source.
      - Anything else (ts_*, binary ops, leaf): wrap outer with -(...).
    """
    parsed = _outermost_call(expr)
    if parsed is None:
        return f"-({expr})"

    op, args = parsed
    if not args:
        return f"-({expr})"

    if op in _TRANSPARENT_OPS:
        # Recurse into first arg; all other args (e.g. std=4) stay untouched.
        flipped_first = _sign_flip(args[0])
        return _reconstruct(op, [flipped_first] + args[1:])

    if op in _RANK_OPS:
        # Negate first arg (the signal); last arg is the group string — leave it.
        return _reconstruct(op, [f"-({args[0]})"] + args[1:])

    # Fallback: wrap the whole expression.
    return f"-({expr})"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _resolve_checks(record: dict) -> list[dict]:
    """Get checks list, falling back to raw_alpha_detail.is.checks if top-level is absent."""
    checks = record.get("checks")
    if checks:
        return checks
    raw = record.get("raw_alpha_detail") or {}
    return (raw.get("is") or {}).get("checks") or []


def read_sim_log(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("status") == "done":
                    records.append(r)
            except json.JSONDecodeError:
                pass
    return records


def _candidate_id(expression: str) -> str:
    return hashlib.sha1(expression.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Near-miss selection
# ---------------------------------------------------------------------------

def find_near_misses(
    records: list[dict],
    min_sharpe: float = DEFAULT_MIN_SHARPE,
    flip_threshold: float = DEFAULT_FLIP_THRESHOLD,
) -> list[dict]:
    """Return records eligible for repair.

    Includes:
    - High-Sharpe failing alphas (sharpe >= min_sharpe, at least one failing check).
    - Strong inverse signals (sharpe <= flip_threshold) regardless of checks —
      these are candidates for sign-flip repair.
    """
    near_misses = []
    seen_exprs: set[str] = set()

    for r in records:
        sharpe = r.get("is_sharpe")
        if sharpe is None:
            continue
        expr = r.get("expression", "")
        if not expr or expr in seen_exprs:
            continue

        is_flip_candidate = sharpe <= flip_threshold
        is_near_miss = sharpe >= min_sharpe

        if not is_flip_candidate and not is_near_miss:
            continue

        # Near-miss (positive) must have at least one failing check.
        # Flip candidates don't need to — the signal itself is the issue.
        if is_near_miss and not is_flip_candidate:
            checks = _resolve_checks(r)
            failing = [c["name"] for c in checks if c.get("result") != "PASS"]
            if not failing:
                continue

        seen_exprs.add(expr)
        near_misses.append(r)

    # Sort: positive near-misses first (highest Sharpe), then flip candidates (most negative last).
    near_misses.sort(key=lambda r: r.get("is_sharpe") or 0, reverse=True)
    return near_misses


def _failing_checks(record: dict) -> list[str]:
    return [c["name"] for c in _resolve_checks(record) if c.get("result") != "PASS"]


# ---------------------------------------------------------------------------
# Feedback-loop context builders
# ---------------------------------------------------------------------------

def _find_passing_examples(
    all_records: list[dict],
    target_record: dict,
    n: int = 6,
) -> list[dict]:
    """Return up to n passing alphas from the same dataset, sorted by Sharpe desc.

    Filters for records that pass all submission-relevant checks (excluding
    SELF_CORRELATION). Used as few-shot context for the LLM repair prompt.
    """
    target_dataset = target_record.get("dataset_id", "")
    passing = []
    for r in all_records:
        if r.get("status") != "done":
            continue
        if r.get("dataset_id") != target_dataset:
            continue
        sharpe = r.get("is_sharpe") or 0.0
        fitness = r.get("is_fitness") or 0.0
        if sharpe < 1.25 or fitness < 1.0:
            continue
        if not r.get("is_concentrated_weight_ok", True):
            continue
        if not r.get("is_sub_universe_sharpe_ok", True):
            continue
        passing.append(r)
    passing.sort(key=lambda r: r.get("is_sharpe") or 0, reverse=True)
    return passing[:n]


def generate_repairs_llm(
    record: dict,
    all_records: list[dict],
    flip_threshold: float = DEFAULT_FLIP_THRESHOLD,
    api_key: str | None = None,
) -> list[str]:
    """Generate LLM-based repair expressions for a near-miss record.

    Sign-flip is handled deterministically before the LLM call (it's structural).
    For all other failures, the LLM receives feedback-loop context:
      - Similar passing alphas from the same dataset (few-shot positive examples)
      - Previously successful repair pairs (if repair history is available)

    Returns [] if the LLM is unavailable (API error / missing key).
    """
    expr = record.get("expression", "").strip()
    if not expr:
        return []

    sharpe = record.get("is_sharpe") or 0.0

    # Sign-flip: strong inverse signal — deterministic structural negation
    if sharpe <= flip_threshold:
        flipped = _sign_flip(expr)
        return [flipped] if flipped != expr else []

    failing = _failing_checks(record)
    if not failing:
        return []

    # Build feedback-loop context
    passing_examples = _find_passing_examples(all_records, record)

    try:
        import os
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from alpha_enhancement.agents.repair import AlphaRepairAgent

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        agent = AlphaRepairAgent(api_key=resolved_key)
        result = agent.repair(
            record,
            passing_examples=passing_examples,
        )
        # Deduplicate, exclude original
        seen: set[str] = {expr}
        unique: list[str] = []
        for r in result.repair_expressions:
            if r and r not in seen:
                seen.add(r)
                unique.append(r)
        return unique
    except Exception as e:
        print(f"  [REPAIR] LLM unavailable: {e}")
        return []


# ---------------------------------------------------------------------------
# Build repair plan
# ---------------------------------------------------------------------------


def build_repair_plan(
    sim_log_paths: list[str],
    output_path: str,
    min_sharpe: float = DEFAULT_MIN_SHARPE,
    max_repairs_per_alpha: int = 5,
    flip_threshold: float = DEFAULT_FLIP_THRESHOLD,
    api_key: str | None = None,
) -> list[dict]:
    """Scan sim logs, find near-misses, generate LLM repair variants, write JSONL plan.

    The LLM receives passing examples and trial history as feedback-loop context.
    Sign-flip candidates (sharpe <= flip_threshold) are handled deterministically.

    Returns the list of repair candidates.
    """
    all_records: list[dict] = []
    for path in sim_log_paths:
        all_records.extend(read_sim_log(path))

    near_misses = find_near_misses(all_records, min_sharpe, flip_threshold)
    n_flip = sum(1 for r in near_misses if (r.get("is_sharpe") or 0) <= flip_threshold)
    n_pos = len(near_misses) - n_flip
    print(
        f"Found {len(near_misses)} repair candidates across {len(sim_log_paths)} log(s): "
        f"{n_pos} near-misses (sharpe >= {min_sharpe}), "
        f"{n_flip} sign-flip candidates (sharpe <= {flip_threshold})"
    )

    known_exprs: set[str] = {r.get("expression", "") for r in all_records}

    candidates: list[dict] = []
    for nm in near_misses:
        original_expr = nm.get("expression", "")
        failing = _failing_checks(nm)
        sharpe = nm.get("is_sharpe", 0)
        template_id = nm.get("template_id", "repair")

        repair_exprs = generate_repairs_llm(
            nm,
            all_records=all_records,
            flip_threshold=flip_threshold,
            api_key=api_key,
        )[:max_repairs_per_alpha]

        if not repair_exprs:
            print(f"  No repairs generated for {original_expr[:60]}  failing={failing}")
            continue

        print(f"  Near-miss sharpe={sharpe:.3f}  failing={failing}  →  {len(repair_exprs)} repairs")

        for r_expr in repair_exprs:
            if r_expr in known_exprs:
                continue
            known_exprs.add(r_expr)
            candidates.append({
                "candidate_id": _candidate_id(r_expr),
                "template_id": f"repair:{template_id}",
                "dataset_id": nm.get("dataset_id", ""),
                "category": nm.get("category", ""),
                "search_lane": "repair",
                "expression": r_expr,
                "fields": nm.get("fields", []),
                "group": nm.get("group", ""),
                "params": {
                    "original_expression": original_expr,
                    "original_sharpe": sharpe,
                    "original_failing_checks": failing,
                },
                "priority_score": sharpe,
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    print(f"\nWrote {len(candidates)} repair candidates to {output_path}")
    return candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate repair variants for near-miss alphas.")
    p.add_argument("--sim-log", nargs="+", required=True, help="One or more simulation log JSONL paths.")
    p.add_argument("--output", required=True, help="Output repair plan JSONL path.")
    p.add_argument("--min-sharpe", type=float, default=DEFAULT_MIN_SHARPE,
                   help=f"Minimum Sharpe to consider a near-miss (default {DEFAULT_MIN_SHARPE}).")
    p.add_argument("--flip-threshold", type=float, default=DEFAULT_FLIP_THRESHOLD,
                   help=f"Sharpe at or below this triggers sign-flip repair (default {DEFAULT_FLIP_THRESHOLD}).")
    p.add_argument("--max-repairs", type=int, default=5, help="Max repair variants per near-miss.")
    p.add_argument("--api-key", default=None, help="OpenAI API key (default: OPENAI_API_KEY env var).")
    args = p.parse_args()

    build_repair_plan(
        sim_log_paths=args.sim_log,
        output_path=args.output,
        min_sharpe=args.min_sharpe,
        max_repairs_per_alpha=args.max_repairs,
        flip_threshold=args.flip_threshold,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
