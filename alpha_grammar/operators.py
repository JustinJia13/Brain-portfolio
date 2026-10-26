"""Single source of truth for all Brain FASTEXPR operator definitions.

Every operator used by the grammar is defined here with its metadata.
``grammar.py`` derives its module-level frozensets (``_COMMUTATIVE``,
``_GROUP_OPS``, ``_GROUP_BACKFILL_OPS``, ``DIVISION_OPS``) and
``GrammarConfig`` defaults directly from this registry.

To add a new operator
---------------------
1. Add one ``OperatorSpec`` entry to ``OPERATOR_REGISTRY``.
2. Set ``category`` to the grammar production it participates in.
3. Set ``commutative=True`` if the first two args are order-independent
   (enables automatic arg-sorting in canonicalization).
4. Set ``group_arg_index`` if the op takes a group string argument:
   ``-1`` = last arg, ``1`` = index 1 (backfill-style).
5. Set ``requires_proxy_rhs=True`` for division-like operators
   (RHS must be a size-proxy field to avoid near-zero denominators).
6. Set ``default_enabled=False`` for operators controlled by probability
   or value config fields in ``GrammarConfig`` (``trade_when_prob``, etc.)
   rather than by ``allowed_*`` tuples — they are documented here but not
   included in the default operator sets.
7. Set ``variadic=True`` and ``max_variadic_arity`` for named operators that
   accept 2 or more arguments in Brain FASTEXPR (e.g. ``add(a, b, c)``).
   Infix operators (``+``, ``-``, ``*``, ``/``) are always strictly binary
   and must NOT be marked variadic.  The grammar will sample arity uniformly
   from [2, min(max_variadic_arity, remaining_budget)]; Δk scales with arity
   (each additional arg costs +1 Δk vs. the base binary cost of 2).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSpec:
    """Metadata for one Brain FASTEXPR operator as used by the grammar.

    Attributes:
        name:               Operator string as it appears in
                            ``ExpressionNode.operator``.
        category:           Grammar production category — determines which
                            ``allowed_*_ops`` tuple this operator belongs to.
        delta_k:            Δk budget consumed by this production rule.
        commutative:        First two args are order-independent; they will be
                            sorted alphabetically during canonicalization.
        group_arg_index:    Position of the group-string argument, or ``None``
                            if the operator takes no group argument.
                            ``-1`` means the last positional arg;
                            ``1`` means index 1 (used by ``group_backfill``).
                            ``_collect_field_ids`` uses this to skip group
                            literals so they are never mistaken for field IDs.
        requires_proxy_rhs: Right operand must be drawn from size-proxy fields
                            (prevents near-zero denominators in division).
        default_enabled:    ``False`` for operators governed by separate
                            probability/value config fields (``trade_when_prob``,
                            ``bucket_group_prob``, etc.) — they are excluded
                            from ``ops_by_category`` and therefore from
                            ``GrammarConfig`` defaults.
        variadic:           ``True`` for named operators that accept 2 or more
                            arguments in Brain FASTEXPR (e.g. ``add(a, b, c)``).
                            Infix operators (``+``, ``-``, ``*``, ``/``) are
                            strictly binary and must not be marked variadic.
                            When ``True``, the grammar samples arity from
                            ``[2, min(max_variadic_arity, remaining_budget)]``
                            and Δk consumed equals the sampled arity.
        max_variadic_arity: Upper bound on the number of arguments when
                            ``variadic=True``; ignored when ``variadic=False``.
    """

    name: str
    category: str
    delta_k: int
    commutative: bool = False
    group_arg_index: int | None = None
    requires_proxy_rhs: bool = False
    default_enabled: bool = True
    variadic: bool = False
    max_variadic_arity: int = 2  # only meaningful when variadic=True


# ---------------------------------------------------------------------------
# Operator registry — one entry per operator, ordered by category
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY: dict[str, OperatorSpec] = {op.name: op for op in [

    # ── Time-series rolling  RollingOp(Expr, window)  Δk=2 ────────────────
    OperatorSpec("ts_rank",          "ts_rolling", 2),
    OperatorSpec("ts_zscore",        "ts_rolling", 2),
    OperatorSpec("ts_mean",          "ts_rolling", 2),
    OperatorSpec("ts_std_dev",       "ts_rolling", 2),
    OperatorSpec("ts_decay_linear",  "ts_rolling", 2),
    OperatorSpec("ts_av_diff",       "ts_rolling", 2),
    OperatorSpec("ts_scale",         "ts_rolling", 2),
    OperatorSpec("ts_quantile",      "ts_rolling", 2),
    OperatorSpec("ts_sum",           "ts_rolling", 2),
    OperatorSpec("ts_arg_max",       "ts_rolling", 2),
    OperatorSpec("ts_arg_min",       "ts_rolling", 2),
    OperatorSpec("ts_backfill",      "ts_rolling", 2),
    # Disabled for new generation: local sim history is already saturated with
    # NaN-presence signals and they keep producing repetitive sparse alphas.
    OperatorSpec("ts_count_nans",    "ts_rolling", 2, default_enabled=False),
    OperatorSpec("ts_product",       "ts_rolling", 2),

    # ── Time-series delta  DeltaOp(Expr, delta_window)  Δk=2 ──────────────
    OperatorSpec("ts_delta",         "ts_delta", 2),
    OperatorSpec("ts_delay",         "ts_delta", 2),
    OperatorSpec("last_diff_value",  "ts_delta", 2),

    # ── Time-series paired  PairedOp(ExprA, ExprB, window)  Δk=3  A≠B ─────
    OperatorSpec("ts_corr",       "ts_paired", 3, commutative=True),
    OperatorSpec("ts_covariance", "ts_paired", 3, commutative=True),
    OperatorSpec("ts_regression", "ts_paired", 3),  # 5-arg: (y, x, window, lag=0, rettype)

    # ── Group ops  GroupOp(TSExpr, group)  Δk=1  group arg = last ──────────
    OperatorSpec("group_rank",       "group", 1, group_arg_index=-1),
    OperatorSpec("group_zscore",     "group", 1, group_arg_index=-1),
    OperatorSpec("group_neutralize", "group", 1, group_arg_index=-1),
    OperatorSpec("group_scale",      "group", 1, group_arg_index=-1),

    # ── Group window ops  group_backfill(TSExpr, group, d)  Δk=2 ───────────
    # group arg is at index 1 — NOT the last arg.
    OperatorSpec("group_backfill", "group_window", 2, group_arg_index=1),

    # ── Group weighted ops  group_mean(TSExpr, weight_field, group)  Δk=2 ──
    # Kept in the registry so historical expressions can still be parsed and
    # scored, but disabled for new grammar sampling. Local sim history shows it
    # overwhelmingly creates flat zero-turnover UNKNOWN alphas.
    OperatorSpec("group_mean", "group_weighted", 2, group_arg_index=-1, default_enabled=False),

    # ── Cross-sectional  CSOp(TSExpr)  Δk=1  no group arg ─────────────────
    # Used when group="none"; operates on the full stock universe.
    # normalize is NOT a standard Brain FASTEXPR operator — disabled to prevent NaN alphas.
    OperatorSpec("normalize", "cs", 1, default_enabled=False),
    OperatorSpec("rank",      "cs", 1),
    OperatorSpec("zscore",    "cs", 1),
    OperatorSpec("winsorize", "cs", 1),
    OperatorSpec("scale",     "cs", 1),
    # quantile as a 1-arg CS op is not confirmed in Brain FASTEXPR — disabled.
    OperatorSpec("quantile",  "cs", 1, default_enabled=False),

    # ── Binary field-level  BinaryOp(FieldL, FieldR)  Δk=2 ────────────────
    # Infix arithmetic — strictly binary (syntax requires exactly 2 operands).
    OperatorSpec("+", "binary", 2, commutative=True),
    OperatorSpec("-", "binary", 2),
    OperatorSpec("*", "binary", 2, commutative=True),
    OperatorSpec("/", "binary", 2, requires_proxy_rhs=True),
    # Named function forms — accept 2 or more arguments in Brain FASTEXPR.
    # variadic=True enables n-ary sampling; Δk consumed = sampled arity.
    OperatorSpec("add",      "binary", 2, commutative=True, variadic=True, max_variadic_arity=5),
    OperatorSpec("subtract", "binary", 2),
    OperatorSpec("multiply", "binary", 2, commutative=True, variadic=True, max_variadic_arity=2),
    OperatorSpec("divide",   "binary", 2, requires_proxy_rhs=True),  # division: always binary
    # Extended arithmetic — strictly binary
    OperatorSpec("power",        "binary", 2),
    OperatorSpec("signed_power", "binary", 2),
    # max/min — variadic named forms, e.g. max(a, b, c)
    OperatorSpec("max", "binary", 2, commutative=True, variadic=True, max_variadic_arity=5),
    OperatorSpec("min", "binary", 2, commutative=True, variadic=True, max_variadic_arity=5),
    # Comparison ops — return 0/1 bool series; kept in "comparison" category so
    # they are NOT included in allowed_binary_ops and cannot appear as arbitrary
    # Expr nodes.  Brain rejects expressions like subtract(a == b, c) because a
    # bool series cannot be passed to arithmetic operators.
    OperatorSpec("<",  "comparison", 2),
    OperatorSpec("<=", "comparison", 2),
    OperatorSpec("==", "comparison", 2),
    OperatorSpec(">",  "comparison", 2),
    OperatorSpec(">=", "comparison", 2),
    OperatorSpec("!=", "comparison", 2),
    # Logical ops — same reason: and/or return 0/1 series; excluded from binary.
    OperatorSpec("and", "comparison", 2, commutative=True, variadic=True, max_variadic_arity=4),
    OperatorSpec("or",  "comparison", 2, commutative=True, variadic=True, max_variadic_arity=4),

    # ── Unary field-level  UnaryOp(Expr)  Δk=1 ────────────────────────────
    OperatorSpec("abs",                   "unary", 1),
    OperatorSpec("sign",                  "unary", 1),
    OperatorSpec("log",                   "unary", 1),
    OperatorSpec("sqrt",                  "unary", 1),
    OperatorSpec("reverse",               "unary", 1),
    OperatorSpec("inverse",               "unary", 1),  # 1/x
    # Disabled for new generation for the same reason as ts_count_nans.
    OperatorSpec("is_nan",                "unary", 1, default_enabled=False),
    OperatorSpec("not",                   "unary", 1, default_enabled=False),  # returns 0/1 boolean; produces all-zero on continuous signals
    OperatorSpec("hump",                  "unary", 1),  # limits change magnitude
    OperatorSpec("days_from_last_change", "unary", 1),
    OperatorSpec("vec_avg",               "unary", 1, default_enabled=False),  # vector fields only; produces NaN on scalar pv/fundamental fields
    OperatorSpec("vec_sum",               "unary", 1, default_enabled=False),  # vector fields only; produces NaN on scalar pv/fundamental fields
    OperatorSpec("densify",               "unary", 1, default_enabled=False),  # requires group-typed input (Unit[Group:1]); unsafe in generic unary position

    # ── Special operators  (default_enabled=False) ─────────────────────────
    # Hardcoded in grammar.py sampling logic; governed by probability or value
    # config fields (kth_element_k_values, trade_when_prob, etc.) — NOT by
    # allowed_* tuples.  Documented here for completeness and so that metadata
    # (commutative, group_arg_index) is still derived from the registry.
    OperatorSpec("kth_element", "special", 2, default_enabled=False),
    OperatorSpec("ts_step",     "special", 0, default_enabled=False),
    OperatorSpec("trade_when",  "special", 3, default_enabled=False),
    OperatorSpec("if_else",     "special", 3, default_enabled=False),
    # bucket(rank(field), range_str): used as the group argument in group_ops.
    # group_arg_index=-1 causes _collect_field_ids to skip the range_str
    # literal so it is never mistaken for a field ID.
    OperatorSpec("bucket", "special", 0, group_arg_index=-1, default_enabled=False),
]}

BANNED_GENERATION_OPS: frozenset[str] = frozenset({
    "group_mean",
    "is_nan",
    "ts_count_nans",
})


# ---------------------------------------------------------------------------
# Derived sets — imported by grammar.py
# ---------------------------------------------------------------------------

def ops_by_category(category: str) -> tuple[str, ...]:
    """Return names of all default-enabled operators for a grammar category."""
    return tuple(
        spec.name for spec in OPERATOR_REGISTRY.values()
        if (
            spec.category == category
            and spec.default_enabled
            and spec.name not in BANNED_GENERATION_OPS
        )
    )


COMMUTATIVE_OPS: frozenset[str] = frozenset(
    spec.name for spec in OPERATOR_REGISTRY.values() if spec.commutative
)

GROUP_OPS: frozenset[str] = frozenset(
    spec.name for spec in OPERATOR_REGISTRY.values() if spec.group_arg_index is not None
)

GROUP_BACKFILL_OPS: frozenset[str] = frozenset(
    spec.name for spec in OPERATOR_REGISTRY.values() if spec.group_arg_index == 1
)

DIVISION_OPS: frozenset[str] = frozenset(
    spec.name for spec in OPERATOR_REGISTRY.values() if spec.requires_proxy_rhs
)

VARIADIC_OPS: frozenset[str] = frozenset(
    spec.name for spec in OPERATOR_REGISTRY.values() if spec.variadic
)


__all__ = [
    "OperatorSpec",
    "OPERATOR_REGISTRY",
    "ops_by_category",
    "COMMUTATIVE_OPS",
    "GROUP_OPS",
    "GROUP_BACKFILL_OPS",
    "DIVISION_OPS",
    "VARIADIC_OPS",
    "BANNED_GENERATION_OPS",
]
