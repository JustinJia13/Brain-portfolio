"""Compact Brain FASTEXPR grammar context for LLM expression generation."""

from __future__ import annotations

from typing import Any


BANNED_GENERATION_OPERATORS = {"group_mean"}

SAFE_EXAMPLES = [
    "group_neutralize(rank(ts_zscore(field_a, 20)), subindustry)",
    "trade_when(rank(ts_zscore(condition_field, 20)), group_neutralize(rank(ts_delta(signal_field, 5)), subindustry), 0)",
    "group_rank(ts_zscore(signal_field, 60), bucket(rank(condition_field), buckets=5))",
    "group_neutralize(multiply(rank(ts_delta(signal_field, 10)), rank(ts_zscore(condition_field, 20))), subindustry)",
    "group_neutralize(rank(ts_corr(signal_field, condition_field, 20)), subindustry)",
]

INVALID_PATTERN_WARNINGS = [
    "Do not invent field IDs; use only the supplied delay-specific field catalog.",
    "Do not output prose, equations, or templates; output concrete FASTEXPR strings.",
    "Do not generate a batch where candidates only change windows or scalar constants.",
    "Do not use unsupported operators or Python-style functions.",
    "Do not use group_mean; local history shows it usually creates flat zero-turnover UNKNOWN alphas.",
    "Do not rely on template names such as confirm_product or residual_spread.",
    "Use group labels such as subindustry, industry, sector, market, or bucket(...); do not quote group names.",
]


def operator_context() -> dict[str, Any]:
    """Return a JSON-serializable operator guide from the local grammar registry."""
    try:
        from core.expr.operators import OPERATOR_REGISTRY
    except Exception:
        OPERATOR_REGISTRY = {}

    by_category: dict[str, list[dict[str, Any]]] = {}
    for spec in OPERATOR_REGISTRY.values():
        if spec.name in BANNED_GENERATION_OPERATORS:
            continue
        by_category.setdefault(spec.category, []).append(
            {
                "name": spec.name,
                "arity": spec.delta_k,
                "commutative": bool(spec.commutative),
                "variadic": bool(spec.variadic),
                "max_variadic_arity": spec.max_variadic_arity if spec.variadic else None,
                "requires_proxy_rhs": bool(spec.requires_proxy_rhs),
                "group_arg_index": spec.group_arg_index,
                "default_enabled": bool(spec.default_enabled),
            }
        )

    return {
        "operators_by_category": by_category,
        "allowed_function_operators": sorted(
            name for name in OPERATOR_REGISTRY
            if name not in BANNED_GENERATION_OPERATORS
            and name not in {"+", "-", "*", "/", "<", "<=", "==", ">", ">=", "!="}
        ),
        "infix_operators": ["+", "-", "*", "/", "<", "<=", "==", ">", ">=", "!="],
        "common_groups": ["subindustry", "industry", "sector", "market"],
    }


def build_brain_grammar_context() -> dict[str, Any]:
    return {
        "language": "FASTEXPR",
        "operator_context": operator_context(),
        "safe_examples": SAFE_EXAMPLES,
        "invalid_pattern_warnings": INVALID_PATTERN_WARNINGS,
        "batch_diversity_rule": (
            "Generate different realizations of the same economic theory: vary operator structure, "
            "field-role mapping, conditioning style, and normalization. Do not merely sweep windows."
        ),
    }
