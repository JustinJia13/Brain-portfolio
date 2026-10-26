"""Context-free grammar for syntactically and semantically valid Brain alpha expressions.

Production rules use a Δk budget (α-Sem-k style) to bound expression depth.
All semantic constraints (division safety, paired-field distinctness) are enforced
during derivation, not as a post-filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

import numpy as np

from core.expr.models import DataProfile, ExpressionNode
from core.expr.operators import (
    BANNED_GENERATION_OPS,
    COMMUTATIVE_OPS,
    DIVISION_OPS,
    GROUP_BACKFILL_OPS,
    GROUP_OPS,
    OPERATOR_REGISTRY,
    ops_by_category,
)


# Derived from OPERATOR_REGISTRY — no need to maintain these manually.
_COMMUTATIVE        = COMMUTATIVE_OPS
_GROUP_OPS          = GROUP_OPS
_GROUP_BACKFILL_OPS = GROUP_BACKFILL_OPS

# Registry-derived defaults for GrammarConfig operator fields.
_D_TS_ROLLING  = ops_by_category("ts_rolling")
_D_TS_DELTA    = ops_by_category("ts_delta")
_D_TS_PAIRED   = ops_by_category("ts_paired")
_D_GROUP       = ops_by_category("group")
_D_GROUP_WIN   = ops_by_category("group_window")
_D_GROUP_WGT   = ops_by_category("group_weighted")
_D_CS          = ops_by_category("cs")
_D_BINARY      = ops_by_category("binary")
_D_UNARY       = ops_by_category("unary")


@dataclass(frozen=True)
class GrammarConfig:
    """Knobs for the CFG sampler.  Only two free parameters matter for search quality:
    exploration_fraction and ucb_c — everything else is an operator-set decision."""

    budget: int = 10                   # max Δk budget K (inclusive); paper (α-Sem-k) optimal K=10
    n_samples: int = 500               # candidates to generate per batch
    diversity_threshold: float = 0.80  # Jaccard ceiling; above → reject as duplicate
    exploration_fraction: float = 0.30 # retained for API compatibility; sampler is uniform
    min_coverage: float = 0.0          # champion.coverage minimum for a cluster to qualify.
    # Set to 0.0 to include event-driven / sector-specific fields that are
    # sparse by design (earnings surprises, short interest, option flow).
    # The coverage-seeded cold-start prior already down-weights low-coverage
    # clusters (exploit term ≈ coverage at cold start), so the bandit self-
    # corrects: sparse-but-productive fields surface quickly; useless ones stay
    # suppressed.  Use 0.50 to restrict to broad market fields only.
    ucb_c: float = 1.4                 # UCB exploration constant
    skeleton_oversample_factor: int = 3  # pool oversampling multiplier for novelty ranking

    # Lookback windows (days)
    windows: tuple[int, ...] = (20, 40, 60, 120, 200, 252)
    delta_windows: tuple[int, ...] = (5, 10, 20)
    groups: tuple[str, ...] = ("market", "sector", "industry", "subindustry", "none")
    # "none" is a sentinel: the grammar generates a full-universe CS op (rank/zscore
    # with no group argument) instead of a group_op.  "sector" is GICS sector level,
    # between market and industry in granularity.

    # ── Time-series operators ──────────────────────────────────────────────
    # Defaults sourced from OPERATOR_REGISTRY via ops_by_category().
    # To restrict to a subset, pass an explicit tuple to GrammarConfig().
    # RollingOp(Expr, window): Δk=2  (Table 7)
    allowed_ts_rolling_ops: tuple[str, ...] = _D_TS_ROLLING
    # DeltaOp(Expr, delta_window): ts_delta, ts_delay, last_diff_value → Δk=2
    allowed_ts_delta_ops: tuple[str, ...] = _D_TS_DELTA
    # PairedRollingOp(Expr, Expr, window): Δk=3  (Table 7)
    allowed_ts_paired_ops: tuple[str, ...] = _D_TS_PAIRED
    # ts_regression rettype constants: 0=residuals, 1=fitted, 2=alpha, 3=beta
    # Emitted as 4th positional arg: ts_regression(y, x, d, rettype).
    # Default (0,) keeps 3-arg form until 4-arg form is verified with Brain.
    ts_regression_rettypes: tuple[int, ...] = (0,)
    # ts_quantile fraction: the percentile to query (Brain requires 3rd positional arg).
    # ts_quantile(x, d, fraction) where fraction ∈ (0, 1).
    ts_quantile_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    # kth_element(x, d, k) — rolling with two integer params → Δk=2
    kth_element_k_values: tuple[int, ...] = (1, 2, 3)

    # ── Group operators ───────────────────────────────────────────────────
    # Standard group_op(TSExpr, group): Δk=1 at Alpha level
    allowed_group_ops: tuple[str, ...] = _D_GROUP
    # group_backfill(TSExpr, group, d): Δk=2 — group at index 1, not last
    allowed_group_window_ops: tuple[str, ...] = _D_GROUP_WIN
    # group_mean(TSExpr, weight_field, group): Δk=2 — weight field is arg 1
    allowed_group_weighted_ops: tuple[str, ...] = _D_GROUP_WGT

    # ── Cross-sectional ops ───────────────────────────────────────────────
    # cs_op(TSExpr): Δk=1 at Alpha level — full universe, no group argument
    allowed_cs_ops: tuple[str, ...] = _D_CS

    # ── Field-level operators ─────────────────────────────────────────────
    # BinaryOp(Expr, Expr): Δk=2  (Table 7)
    # Includes infix (+,-,*,/) and named forms (add,subtract,multiply,divide) —
    # semantically identical in FASTEXPR.  Division ops (/ and divide) require
    # a size-proxy right operand (DIVISION_OPS).  Comparison ops return 0/1.
    allowed_binary_ops: tuple[str, ...] = _D_BINARY
    # UnaryOp(Expr): Δk=1  (Table 7)
    allowed_unary_ops: tuple[str, ...] = _D_UNARY

    # ── Unified Expr nesting depth ────────────────────────────────────────
    # max_expr_depth: absolute tree depth ceiling for _expr().  Each recursive
    # call decrements the counter; reaching 0 forces a field leaf.  depth=8
    # allows e.g. group(ts(group(ts(binary(unary(field), field))))).
    max_expr_depth: int = 8
    # max_inner_normalizer_depth: how many group/cs normalization ops may appear
    # *below* the outermost Alpha wrapper.  Prevents rank(rank(rank(...))) chains.
    # Default=1 matches the old double_group_op limit.  Set to 0 to restrict
    # inner expressions to pure field/ts/arithmetic.
    max_inner_normalizer_depth: int = 1

    # ── trade_when / bucket / ts_step config ──────────────────────────────
    # trade_when(signal, cond, 0): ternary at Alpha level, Δk=3.
    # Feasible only at budget≥6 (needs 3 for trade_when + ≥3 for inner alpha).
    # trade_when_prob: fraction of eligible Alpha samples that attempt trade_when.
    trade_when_prob: float = 0.15
    # bucket(rank(field), buckets=N): creates custom decile/quintile group partitions.
    # Used as the group argument in group_op instead of a literal ("market" etc).
    # bucket_n_values: choices for number of equal-width buckets (5→quintiles, etc.)
    bucket_n_values: tuple[int, ...] = (5, 10, 20)
    # bucket_group_prob: probability any group_op uses bucket() for the group arg.
    bucket_group_prob: float = 0.15
    # group_field_prob: probability a group argument is a GROUP-typed data field.
    # GROUP fields are never used as numeric leaves; this is their only path.
    group_field_prob: float = 0.10
    # ts_step_prob: probability that the secondary series in a paired TS op is
    # ts_step(1) (linear day counter) rather than a real data field.
    ts_step_prob: float = 0.08

    # ── Alpha-level arithmetic (composite alpha patterns) ─────────────────
    # alpha_binary_ops: operators for combining two full Alpha expressions at
    # the top level.  Produces patterns like alpha_a - alpha_b (long/short
    # composite) or add(alpha_a, alpha_b) (ensemble blend).
    # Enabled only when budget >= 9 (each sub-alpha needs K≥3).
    # Probs controlled by alpha_binary_prob (per-sample gate, like trade_when_prob).
    allowed_alpha_binary_ops: tuple[str, ...] = ("+", "-")
    alpha_binary_prob: float = 0.5      # fraction of eligible samples that attempt alpha_binary

    # ── Scalar constant operands (deferred — disabled by default) ──────────
    # When scalar_constants is non-empty and scalar_constant_prob > 0, the
    # grammar may substitute the RHS of a binary op with a scalar literal
    # (e.g. ts_mean(pe_ratio, 20) * 0.5).  Currently unused; set both fields
    # to enable once structural coverage is satisfactory.
    scalar_constants: tuple[float, ...] = ()  # e.g. (-0.1, -0.05, 0.05, 0.1)
    scalar_constant_prob: float = 0.0         # fraction of binary ops to apply this to

    # ── Boolean / comparison ops for if_else conditions ───────────────────
    # BoolExpr nonterminal: comparison_op(Expr, Expr) | logical_op(BoolExpr, BoolExpr)
    # These operators are ONLY reachable as the condition of if_else; they cannot
    # appear in arbitrary Expr positions (Brain rejects bool series in arithmetic).
    allowed_comparison_ops: tuple[str, ...] = ("<", "<=", "==", ">", ">=", "!=")
    allowed_logical_ops: tuple[str, ...] = ("and", "or")
    bool_logical_prob: float = 0.20   # probability the condition is a logical(cmp, cmp)

    def __post_init__(self) -> None:
        """Enforce generation-wide operator bans even for explicit overrides."""
        operator_fields = (
            "allowed_ts_rolling_ops",
            "allowed_ts_delta_ops",
            "allowed_ts_paired_ops",
            "allowed_group_ops",
            "allowed_group_window_ops",
            "allowed_group_weighted_ops",
            "allowed_cs_ops",
            "allowed_binary_ops",
            "allowed_unary_ops",
            "allowed_alpha_binary_ops",
            "allowed_comparison_ops",
            "allowed_logical_ops",
        )
        for field_name in operator_fields:
            values = tuple(
                op for op in getattr(self, field_name)
                if op not in BANNED_GENERATION_OPS
            )
            object.__setattr__(self, field_name, values)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lit(value: str | int | float) -> ExpressionNode:
    return ExpressionNode(literal=value)


def _horizon_from_window(window: int) -> str:
    if window <= 40:
        return "short"
    if window <= 120:
        return "medium"
    return "long"


def _extract_field_ids(tree: ExpressionNode) -> list[str]:
    """Walk the tree and collect field_id string literals.

    String literals that are the group-name argument of a group_op are excluded.
    group_backfill(x, group, d): group is at index 1.
    All other group ops: group is the last arg.
    group_mean(x, weight_field, group): weight_field at index 1 is a real field → included.
    """
    result: list[str] = []
    _collect_field_ids(tree, result, is_group_arg=False)
    return result


def _collect_field_ids(
    tree: ExpressionNode, out: list[str], is_group_arg: bool
) -> None:
    if tree.literal is not None:
        if isinstance(tree.literal, str) and not is_group_arg:
            out.append(tree.literal)
        return
    for i, arg in enumerate(tree.args):
        if isinstance(arg, ExpressionNode):
            if tree.operator in _GROUP_BACKFILL_OPS:
                arg_is_group = (i == 1)
            elif tree.operator in _GROUP_OPS:
                arg_is_group = (i == len(tree.args) - 1)
            else:
                arg_is_group = False
            _collect_field_ids(arg, out, is_group_arg=arg_is_group)


def _top_window(tree: ExpressionNode) -> int | None:
    """Return the outermost rolling window integer in the tree, if any."""
    if tree.literal is not None:
        return None
    # The first integer literal child of a ts_* operator is the window.
    if tree.operator and tree.operator.startswith("ts_"):
        for arg in tree.args:
            if isinstance(arg, ExpressionNode) and isinstance(arg.literal, int):
                return arg.literal
    for arg in tree.args:
        if isinstance(arg, ExpressionNode):
            w = _top_window(arg)
            if w is not None:
                return w
    return None


# ---------------------------------------------------------------------------
# BrainGrammar
# ---------------------------------------------------------------------------

class BrainGrammar:
    """Bounded CFG sampler for Brain alpha expressions.

    The grammar has two non-terminals: Alpha (outermost normalizer wrapper) and
    Expr (unified inner expression, as general as Brain FASTEXPR itself).  Every
    production consumes Δk budget; derivation halts when budget is exhausted
    or depth limits are reached.

    Complete BNF with Δk costs
    --------------------------
    Alpha   ::= group_op(Expr, GroupExpr)                      Δk=1
              | cs_op(Expr)                                    Δk=1
              | group_backfill(Expr, GroupExpr, d)             Δk=2
              | group_mean(Expr, field_id, GroupExpr)          Δk=2
              | trade_when(Alpha, Expr, 0)                     Δk=3  [prob-gated: trade_when_prob]
              | group_op(group_op(Expr, GroupExpr), GroupExpr) Δk=2  [double_group_op; top-level only]
              | op(Alpha, Alpha)                               Δk=3  [alpha_binary; budget≥9; top-level only]
                op ∈ allowed_alpha_binary_ops  (default: +, -)

    Expr    ::= field_id                                       Δk=0
              | unary_op(Expr)                                 Δk=1
              | binary_op(Expr, Expr)                         Δk=2  [/ → proxy RHS]
              | if_else(Expr, Expr, Expr)                     Δk=3
              | ts_rolling(Expr, w)                           Δk=2
              | ts_delta_op(Expr, dw)                         Δk=2
              | kth_element(Expr, d, k)                       Δk=2
              | ts_paired(Expr, Expr, w)                      Δk=3  [A≠B profiles]
              | ts_regression(Expr, Expr, w)                  Δk=3  [A≠B profiles]
              | group_op(Expr, GroupExpr)                     Δk=1  [norm_depth_remaining≥1]
              | cs_op(Expr)                                   Δk=1  [norm_depth_remaining≥1]
              | group_backfill(Expr, GroupExpr, d)            Δk=2  [norm_depth_remaining≥1]
              | group_mean(Expr, WeightExpr, GroupExpr)       Δk=2  [norm_depth_remaining≥1]

    GroupExpr ::= group_name_literal                          Δk=0
                | bucket(rank(field_id), buckets=N)           Δk=0  [prob-gated: bucket_group_prob]

    Depth guards:
      depth_remaining (max_expr_depth=8): absolute Expr tree depth ceiling.
        Each recursive _expr() call decrements this; 0 forces a field leaf.
      norm_depth_remaining (max_inner_normalizer_depth=1): how many inner
        group/cs ops may appear below the outermost Alpha wrapper.
        Prevents rank(rank(rank(...))) chains.

    Newly reachable expressions (vs old 4-level hierarchy):
      group_rank(ts_corr(rank(f_a), rank(f_b), 60), g)   # cs inside ts
      ts_mean(group_zscore(ts_delta(x,5), "sector"), 20) # ts(group(ts(field)))
      group_rank(ts_delta(group_rank(ts_mean(x,20),"sector"),5), "market")

    Minimum viable K=3:  group_op(ts_rolling(field, w), g)
    TS-in-TS K=5:        group_op(ts_mean(ts_rank(field, 20), 40), "sector")
    CS inside TS K=5:    group_rank(ts_corr(rank(f_a), rank(f_b), 60), g)
    double_group_op K=5: group_rank(group_zscore(ts_mean(field, 20), "subindustry"), "sector")
    alpha_binary K=9:    group_rank(ts_mean(f_a, 20), "sector") - group_rank(ts_mean(f_b, 20), "sector")

    Use ``feasible_alpha_productions`` and ``feasible_expr_productions`` to
    audit which productions are reachable at any given budget.
    """

    def __init__(self, config: GrammarConfig) -> None:
        self._cfg = config
        self._group_profiles: tuple[DataProfile, ...] = ()
        self._field_sampler: Callable[[np.random.Generator, list[FieldCluster]], DataProfile] | None = None
        self._choice_sampler: Callable[[np.random.Generator, str, Sequence[Any]], Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    #
    # Production rules (Δk costs from α-Sem-k Table 7 extended):
    #   Alpha  → group_op(Expr, group)                          Δk=1
    #          → cs_op(Expr)                                    Δk=1
    #          → group_backfill(Expr, group, d)                 Δk=2
    #          → group_mean(Expr, wt, group)                    Δk=2
    #          → trade_when(Alpha, Expr, 0)                     Δk=3  [prob-gated]
    #          → group_op(group_op(Expr, g), g)                 Δk=2  [double_group_op; top-level only]
    #          → op(Alpha, Alpha)  op ∈ {+,-}                   Δk=3  [alpha_binary; budget≥9; top-level only]
    #
    #   Expr   → field                                          Δk=0
    #          → unary_op(Expr)                                 Δk=1
    #          → binary_op(Expr, Expr)                          Δk=2  [/ → R.is_size_proxy]
    #          → if_else(Expr, Expr, Expr)                      Δk=3
    #          → ts_rolling(Expr, window)                       Δk=2
    #          → ts_delta_op(Expr, dw)                          Δk=2
    #          → kth_element(Expr, d, k)                        Δk=2
    #          → ts_paired(Expr, Expr, window)                  Δk=3  [A≠B]
    #          → ts_regression(Expr, Expr, window)              Δk=3  [A≠B]
    #          → group_op(Expr, group)                          Δk=1  [norm_depth≥1]
    #          → cs_op(Expr)                                    Δk=1  [norm_depth≥1]
    #          → group_backfill(Expr, group, d)                 Δk=2  [norm_depth≥1]
    #          → group_mean(Expr, wt, group)                    Δk=2  [norm_depth≥1]
    #
    # Minimum viable K=3: group_op(ts_rolling(field, w), g)
    # CS inside TS K=5: group_rank(ts_corr(rank(f_a), rank(f_b), 60), g)

    def sample(
        self,
        budget: int,
        rng: np.random.Generator,
        all_clusters: list[FieldCluster],
        size_proxy_profiles: list[DataProfile],
        group_profiles: Sequence[DataProfile] | None = None,
        group: str | None = None,
        field_sampler: Callable[[np.random.Generator, list[FieldCluster]], DataProfile] | None = None,
        choice_sampler: Callable[[np.random.Generator, str, Sequence[Any]], Any] | None = None,
    ) -> ExpressionNode | None:
        """Sample one alpha expression tree using at most `budget` Δk.

        TSL-MDP entry point: starts from the Alpha nonterminal and expands
        production rules uniformly from the feasible set at each nonterminal.
        Field terminals are sampled uniformly by dataset, then cluster, then field.

        all_clusters: eligible FieldCluster objects (one per semantic concept).
        size_proxy_profiles: profiles where is_size_proxy=True (division RHS).
        group: the group string to use (e.g. "market").  If None, sampled uniformly.
        Returns None if no valid derivation exists under the budget.
        """
        previous_group_profiles = self._group_profiles
        previous_field_sampler = self._field_sampler
        previous_choice_sampler = self._choice_sampler
        self._group_profiles = tuple(group_profiles or ())
        self._field_sampler = field_sampler
        self._choice_sampler = choice_sampler
        try:
            return self._alpha(budget, rng, all_clusters, size_proxy_profiles, group)
        finally:
            self._group_profiles = previous_group_profiles
            self._field_sampler = previous_field_sampler
            self._choice_sampler = previous_choice_sampler

    def canonicalize(self, tree: ExpressionNode) -> ExpressionNode:
        """Sort args of commutative operators alphabetically by rendered form.

        For variadic commutative ops (e.g. add, max, min with 3+ args) ALL
        arguments are sorted.  For binary commutative ops only the first two
        args are sorted (the remainder are non-expression literals like windows).
        """
        if tree.literal is not None:
            return tree
        new_args = tuple(
            self.canonicalize(arg) if isinstance(arg, ExpressionNode) else arg
            for arg in tree.args
        )
        if tree.operator in _COMMUTATIVE and len(new_args) >= 2:
            _key = lambda a: a.render() if isinstance(a, ExpressionNode) else str(a)
            spec = OPERATOR_REGISTRY.get(tree.operator)
            if spec and spec.variadic and len(new_args) > 2:
                # Variadic commutative: sort all args so add(c,a,b) == add(a,b,c).
                new_args = tuple(sorted(new_args, key=_key))
            else:
                # Binary commutative: sort first two only; later args are literals.
                head = sorted(new_args[:2], key=_key)
                new_args = tuple(head) + new_args[2:]
        return ExpressionNode(operator=tree.operator, args=new_args)

    def structure_fingerprint(self, tree: ExpressionNode) -> tuple[str, ...]:
        """Pre-order sequence of operator names (omits literals)."""
        ops: list[str] = []
        self._collect_ops(tree, ops)
        return tuple(ops)

    # ------------------------------------------------------------------
    # Feasibility queries — use these to audit the reachable search space
    # without tracing through sampling logic.
    # ------------------------------------------------------------------

    def feasible_alpha_productions(
        self,
        budget: int,
        n_profiles: int = 2,
        compound_depth: int = 0,
    ) -> list[str]:
        """Return which Alpha-level productions are available at this budget.

        Args:
            budget:         Δk budget remaining when entering the Alpha non-terminal.
            n_profiles:     number of distinct field profiles available (affects
                            whether group_weighted_op is offered).
            compound_depth: nesting depth of compound Alpha productions
                            (double_group_op, alpha_binary).  When > 0 these are
                            excluded to prevent exponential recursion.
        """
        if budget < 1:
            return []
        # raw_expr: Δk=0 — Alpha expands directly to Expr with no outer normalizer.
        # This is a valid Brain alpha (outer neutralization is in simulation settings).
        # No probability gate: treated as any other production; probability = 1/|feasible|.
        feasible: list[str] = ["raw_expr", "group_op"]  # Δk=0 and Δk=1 always
        if self._cfg.allowed_cs_ops:
            feasible.append("cs_op")        # Δk=1
        if budget >= 2:
            if self._cfg.allowed_group_window_ops:
                feasible.append("group_window_op")   # Δk=2
        if budget >= 4:
            if n_profiles >= 2 and self._cfg.allowed_group_weighted_ops:
                feasible.append("group_weighted_op") # Δk=2; needs ≥4 so inner Expr has budget≥2 for TS op
        if budget >= 6 and self._cfg.trade_when_prob > 0:
            feasible.append("trade_when")            # Δk=3 (needs budget≥6 for inner alpha≥3)
        # Compound Alpha patterns — only at top level (compound_depth=0) to prevent
        # exponential blowup. At compound_depth=0 the inner _alpha calls receive
        # compound_depth=1 and will NOT generate further compound productions.
        if compound_depth == 0:
            if budget >= 4 and self._cfg.allowed_group_ops:
                feasible.append("double_group_op")   # Δk=2: group_op(group_op(ts_expr,g),g)
            if (
                budget >= 9
                and n_profiles >= 2
                and self._cfg.allowed_alpha_binary_ops
                and self._cfg.alpha_binary_prob > 0
            ):
                feasible.append("alpha_binary")      # Δk=3: op(alpha_a, alpha_b)
        return feasible

    def feasible_expr_productions(
        self,
        budget: int,
        depth_remaining: int,
        norm_depth_remaining: int,
        n_profiles: int = 2,
    ) -> list[str]:
        """Return which Expr productions are available at this budget/depth.

        Args:
            budget:               Δk budget remaining.
            depth_remaining:      remaining tree depth (0 = field leaf only).
            norm_depth_remaining: remaining inner normalizer levels (0 = no group/cs).
            n_profiles:           number of distinct profiles in all_clusters
                                  (affects whether ts_paired/ts_regression are offered).
        """
        feasible: list[str] = ["field"]  # Δk=0, always
        if depth_remaining <= 0:
            return feasible

        # Field-arithmetic productions
        if budget >= 1 and self._cfg.allowed_unary_ops:
            feasible.append("unary")      # Δk=1
        if budget >= 2 and self._cfg.allowed_binary_ops:
            feasible.append("binary")     # Δk=2
        # if_else gated by bool_logical_prob > 0 (set to 0.0 to disable entirely)
        if budget >= 3 and self._cfg.bool_logical_prob > 0:
            feasible.append("if_else")    # Δk=3

        # Time-series productions
        if budget >= 2:
            if self._cfg.allowed_ts_rolling_ops:
                feasible.append("ts_rolling")    # Δk=2
            if self._cfg.allowed_ts_delta_ops:
                feasible.append("ts_delta_op")   # Δk=2
            # kth_element gated by non-empty k_values tuple
            if self._cfg.kth_element_k_values:
                feasible.append("kth_element")   # Δk=2
        if budget >= 3 and n_profiles >= 2:
            if self._cfg.allowed_ts_paired_ops:
                feasible.append("ts_paired")     # Δk=3 [A≠B]
            # ts_regression gated by presence in allowed_ts_paired_ops
            if "ts_regression" in self._cfg.allowed_ts_paired_ops:
                feasible.append("ts_regression") # Δk=3 [A≠B]

        # Inner normalizer productions (only when norm_depth_remaining ≥ 1)
        if norm_depth_remaining >= 1:
            if budget >= 1 and self._cfg.allowed_group_ops:
                feasible.append("expr_group_op")          # Δk=1
            if budget >= 1 and self._cfg.allowed_cs_ops:
                feasible.append("expr_cs_op")             # Δk=1
            if budget >= 2 and self._cfg.allowed_group_window_ops:
                feasible.append("expr_group_window_op")   # Δk=2
            if budget >= 4 and n_profiles >= 2 and self._cfg.allowed_group_weighted_ops:
                feasible.append("expr_group_weighted_op") # Δk=2; needs ≥4 so inner Expr has budget≥2 for TS op

        return feasible

    def feasible_ts_productions(
        self, budget: int, ts_depth: int, n_profiles: int = 2
    ) -> list[str]:
        """Return which TSExpr productions are available at this budget.

        Deprecated: the unified grammar uses feasible_expr_productions().
        Retained for backward compatibility with tests and external audit code.
        ts_depth is no longer used internally (depth is now a unified counter).

        Args:
            budget:     Δk budget remaining when entering the TSExpr non-terminal.
            ts_depth:   retained for API compatibility; ignored.
            n_profiles: number of distinct profiles (paired ops need ≥2).
        """
        if budget < 2:
            return []
        feasible: list[str] = ["ts_rolling", "ts_delta_op"]  # Δk=2 each
        if budget >= 3 and n_profiles >= 2:
            feasible.append("ts_paired")     # Δk=3
            feasible.append("ts_regression") # Δk=3
        if budget >= 4:
            feasible.append("ts_delta_rolling")  # Δk=4 (compound)
        if budget >= 4 and ts_depth >= 1:
            feasible.append("ts_rolling_of_ts")  # Δk=2 outer + inner TSExpr
        if budget >= 5 and ts_depth >= 1 and n_profiles >= 2:
            feasible.append("ts_paired_of_ts")   # Δk=3 outer + two inner TSExprs
        return feasible

    def feasible_field_productions(
        self, budget: int, depth_remaining: int = 2
    ) -> list[str]:
        """Return which FieldExpr productions are available at this budget.

        Deprecated: the unified grammar uses feasible_expr_productions().
        Retained for backward compatibility with tests and external audit code.

        Args:
            budget:          Δk budget remaining when entering the FieldExpr non-terminal.
            depth_remaining: remaining FieldExpr nesting depth (0 = field literals only).
        """
        if budget < 0:
            return []
        feasible: list[str] = ["field"]  # Δk=0, always
        if budget >= 1 and depth_remaining >= 1:
            feasible.append("unary")   # Δk=1
        if budget >= 2 and depth_remaining >= 1:
            feasible.append("binary")  # Δk=arity (2..max_variadic_arity)
        if budget >= 3 and depth_remaining >= 1:
            feasible.append("if_else") # Δk=3
        return feasible

    # ------------------------------------------------------------------
    # Production rules
    # ------------------------------------------------------------------

    def _group_expr(
        self,
        rng: np.random.Generator,
        all_clusters: list[FieldCluster],
        chosen_group: str,
    ) -> ExpressionNode:
        """Return a group expression: literal, GROUP field, or bucket(rank(field), range).

        bucket(rank(field), buckets=N) Δk=0 at the group-argument position.
        Creates custom quantile-based groups from any data field, e.g.:
            group_zscore(ts_rank(pe_ratio,20), bucket(rank(roa), buckets=10))
        The buckets arg is skipped by _collect_field_ids (bucket in _GROUP_OPS).
        """
        if self._group_profiles and rng.random() < self._cfg.group_field_prob:
            return _lit(_uniform_dataset_profile(rng, self._group_profiles).field_id)
        if all_clusters and rng.random() < self._cfg.bucket_group_prob:
            fp = self._sample_field(rng, all_clusters)
            rank_node = ExpressionNode(operator="rank", args=(_field_node(fp, rng),))
            n = int(_rng_choice(rng, self._cfg.bucket_n_values))
            return ExpressionNode(operator="bucket", args=(rank_node, _lit(n)))
        return _lit(chosen_group)

    def _alpha(
        self,
        budget: int,
        rng: np.random.Generator,
        all_clusters: list[FieldCluster],
        proxies: list[DataProfile],
        group: str | None = None,
        _compound_depth: int = 0,
    ) -> ExpressionNode | None:
        """Alpha nonterminal expansion (TSL-MDP state transition).

        Samples uniformly from feasible Alpha productions given budget K:
          raw_expr                         Δk=0  bare Expr, no outer normalizer
          group_op(Expr, GroupExpr)        Δk=1  cross-sectional group normalization
          cs_op(Expr)                      Δk=1  full-universe CS normalization
          group_backfill(Expr, group, d)   Δk=2  group op with rolling backfill window
          group_mean(Expr, wt, group)      Δk=2  group op with weight field
          trade_when(Alpha, BoolExpr, 0)   Δk=3  conditional trading gate [prob-gated]
          double_group_op(ts, g_in, g_out) Δk=2  group_op(group_op(ts,g),g) [compound=0]
          op(Alpha, Alpha)                 Δk=3  alpha_binary [compound=0, budget≥9]

        Field terminals inside Expr use uniform dataset → cluster → field sampling.
        """
        if budget < 1:
            return None

        chosen_group = group if group is not None else str(_rng_choice(rng, self._cfg.groups))

        # "none" → full-universe CS op (no group argument).
        # Falls back to market-wide group_op if no cs_ops are configured.
        if chosen_group == "none":
            if self._cfg.allowed_cs_ops:
                ts = self._expr(budget - 1, rng, all_clusters, proxies,
                                self._cfg.max_expr_depth, self._cfg.max_inner_normalizer_depth)
                if ts is None:
                    return None
                op = self._choice("cs_op", rng, self._cfg.allowed_cs_ops)
                return ExpressionNode(operator=op, args=(ts,))
            chosen_group = "market"  # safe fallback when cs_ops not configured

        # Build feasible alpha-level productions (uniform TSL-MDP action set).
        # trade_when and alpha_binary remain prob-gated (compound patterns with
        # high branching cost; pure uniform would over-generate them at high K).
        feasible = self.feasible_alpha_productions(
            budget, n_profiles=len(all_clusters), compound_depth=_compound_depth
        )
        if "trade_when" in feasible and rng.random() >= self._cfg.trade_when_prob:
            feasible.remove("trade_when")
        if "alpha_binary" in feasible and rng.random() >= self._cfg.alpha_binary_prob:
            feasible.remove("alpha_binary")

        production = self._choice("alpha_production", rng, feasible)

        if production == "raw_expr":
            # Alpha → Expr: no outer normalizer.  Valid Brain alpha (simulation settings
            # handle neutralization).  Enables patterns like ts_mean(field_a,20)-field_b.
            return self._expr(budget, rng, all_clusters, proxies,
                              self._cfg.max_expr_depth, self._cfg.max_inner_normalizer_depth)

        if production == "group_op":
            ts = self._expr(budget - 1, rng, all_clusters, proxies,
                            self._cfg.max_expr_depth, self._cfg.max_inner_normalizer_depth)
            if ts is None:
                return None
            op = self._choice("group_op", rng, self._cfg.allowed_group_ops)
            grp = self._group_expr(rng, all_clusters, chosen_group)
            return ExpressionNode(operator=op, args=(ts, grp))

        if production == "cs_op":
            # cs_op(Expr): Δk=1 — full-universe cross-sectional normalization.
            ts = self._expr(budget - 1, rng, all_clusters, proxies,
                            self._cfg.max_expr_depth, self._cfg.max_inner_normalizer_depth)
            if ts is None:
                return None
            op = self._choice("cs_op", rng, self._cfg.allowed_cs_ops)
            return ExpressionNode(operator=op, args=(ts,))

        if production == "group_window_op":
            # group_backfill(Expr, group, d): Δk=2
            ts = self._expr(budget - 2, rng, all_clusters, proxies,
                            self._cfg.max_expr_depth, self._cfg.max_inner_normalizer_depth)
            if ts is None:
                return None
            op = self._choice("group_window_op", rng, self._cfg.allowed_group_window_ops)
            d = int(self._choice("delta_window", rng, self._cfg.delta_windows))
            grp = self._group_expr(rng, all_clusters, chosen_group)
            return ExpressionNode(operator=op, args=(ts, grp, _lit(d)))

        if production == "group_weighted_op":
            # group_mean(Expr, weight_field, group): Δk=2
            ts = self._expr(budget - 2, rng, all_clusters, proxies,
                            self._cfg.max_expr_depth, self._cfg.max_inner_normalizer_depth)
            if ts is None:
                return None
            # Pick two distinct profiles: signal field and weight field.
            wt_primary = self._sample_field(rng, all_clusters)
            weight = _pick_secondary(rng, wt_primary, all_clusters, self._sample_field)
            if weight is None:
                op = self._choice("group_op", rng, self._cfg.allowed_group_ops)
                grp = self._group_expr(rng, all_clusters, chosen_group)
                return ExpressionNode(operator=op, args=(ts, grp))
            op = self._choice("group_weighted_op", rng, self._cfg.allowed_group_weighted_ops)
            grp = self._group_expr(rng, all_clusters, chosen_group)
            return ExpressionNode(operator=op, args=(ts, _field_node(weight, rng), grp))

        if production == "trade_when":
            # trade_when(signal, condition, 0): Δk=3
            sub_budget = budget - 3
            signal = self._alpha(
                sub_budget, rng, all_clusters, proxies,
                group=chosen_group, _compound_depth=_compound_depth + 1,
            )
            if signal is None:
                return None
            cond = self._bool_expr(rng, all_clusters, proxies)
            return ExpressionNode(operator="trade_when", args=(signal, cond, _lit(0)))

        if production == "double_group_op":
            # group_op_outer(group_op_inner(Expr, g_inner), g_outer)  Δk=2
            ts = self._expr(budget - 2, rng, all_clusters, proxies,
                            self._cfg.max_expr_depth, self._cfg.max_inner_normalizer_depth)
            if ts is None:
                return None
            op_inner = self._choice("group_op", rng, self._cfg.allowed_group_ops)
            op_outer = self._choice("group_op", rng, self._cfg.allowed_group_ops)
            non_none_groups = [g for g in self._cfg.groups if g != "none"]
            if len(non_none_groups) < 2:
                grp = self._group_expr(rng, all_clusters, chosen_group)
                return ExpressionNode(operator=op_inner, args=(ts, grp))
            g_inner = str(_rng_choice(rng, non_none_groups))
            g_outer = str(_rng_choice(rng, [g for g in non_none_groups if g != g_inner]))
            inner_grp = self._group_expr(rng, all_clusters, g_inner)
            outer_grp = self._group_expr(rng, all_clusters, g_outer)
            inner_alpha = ExpressionNode(operator=op_inner, args=(ts, inner_grp))
            return ExpressionNode(operator=op_outer, args=(inner_alpha, outer_grp))

        if production == "alpha_binary":
            # op(alpha_a, alpha_b)  Δk=3 — composite long/short signal
            # Each sub-alpha gets (budget-3)//2 budget.
            # Both sub-alphas sample independently from all_clusters.
            sub_budget = (budget - 3) // 2
            alpha_a = self._alpha(
                sub_budget, rng, all_clusters, proxies,
                group=chosen_group, _compound_depth=_compound_depth + 1,
            )
            alpha_b = self._alpha(
                sub_budget, rng, all_clusters, proxies,
                group=chosen_group, _compound_depth=_compound_depth + 1,
            )
            if alpha_a is None or alpha_b is None:
                return None
            op = self._choice("alpha_binary_op", rng, self._cfg.allowed_alpha_binary_ops)
            return ExpressionNode(operator=op, args=(alpha_a, alpha_b))

        return None

    def _expr(
        self,
        budget: int,
        rng: np.random.Generator,
        all_clusters: list[FieldCluster],
        proxies: list[DataProfile],
        depth_remaining: int,
        norm_depth_remaining: int,
    ) -> ExpressionNode | None:
        """Expr nonterminal expansion (TSL-MDP state transition).

        Samples uniformly from feasible Expr productions given (budget, depth, norm_depth).
        Field terminals are sampled uniformly by dataset → cluster → field at
        every leaf — no pre-fixed primary field.  This means binary(field_a,
        field_b) where both operands independently pick from all_clusters,
        enabling cross-field arithmetic.

        Args:
            all_clusters:         all coverage-eligible profiles (uniform terminal pool).
            proxies:              size-proxy profiles (required for division RHS).
            depth_remaining:      remaining tree depth (0 forces a field leaf).
            norm_depth_remaining: remaining inner normalizer levels (0 = no group/cs).
        """
        if not all_clusters:
            return None
        if budget < 0:
            return None

        # Force leaf when depth or budget exhausted.
        if depth_remaining == 0 or budget == 0:
            return _field_node(self._sample_field(rng, all_clusters), rng)

        feasible = self.feasible_expr_productions(
            budget, depth_remaining, norm_depth_remaining, n_profiles=len(all_clusters)
        )
        production = self._choice("expr_production", rng, feasible)

        # ── Field literal ──────────────────────────────────────────────────
        if production == "field":
            # Terminal: sample uniformly by dataset → cluster → field.
            return _field_node(self._sample_field(rng, all_clusters), rng)

        # ── Unary op ──────────────────────────────────────────────────────
        if production == "unary":
            inner = self._expr(budget - 1, rng, all_clusters, proxies,
                               depth_remaining - 1, norm_depth_remaining)
            if inner is None:
                return _field_node(self._sample_field(rng, all_clusters), rng)
            op = self._choice("unary_op", rng, self._cfg.allowed_unary_ops)
            return ExpressionNode(operator=op, args=(inner,))

        # ── Binary op ─────────────────────────────────────────────────────
        if production == "binary":
            op = self._choice("binary_op", rng, self._cfg.allowed_binary_ops)
            # Division safety: operators in DIVISION_OPS require a size-proxy right operand.
            if op in DIVISION_OPS and not proxies:
                non_div = [o for o in self._cfg.allowed_binary_ops if o not in DIVISION_OPS]
                op = self._choice("binary_op", rng, non_div or ["+"])

            # Variadic arity: named ops like add/max/min accept 2+ args.
            # Δk consumed = arity (each arg costs 1 Δk, matching the binary base of 2).
            spec = OPERATOR_REGISTRY.get(op)
            if spec and spec.variadic and spec.max_variadic_arity > 2 and budget >= 3:
                max_arity = min(spec.max_variadic_arity, budget)
                arity = int(rng.integers(2, max_arity + 1))
            else:
                arity = 2

            inner_budget = budget - arity

            # Optional scalar constant RHS (disabled when scalar_constants is empty).
            if (
                arity == 2
                and op not in DIVISION_OPS
                and self._cfg.scalar_constants
                and rng.random() < self._cfg.scalar_constant_prob
            ):
                left = self._expr(inner_budget, rng, all_clusters, proxies,
                                  depth_remaining - 1, norm_depth_remaining)
                if left is None:
                    return _field_node(self._sample_field(rng, all_clusters), rng)
                const = float(_rng_choice(rng, self._cfg.scalar_constants))
                return ExpressionNode(operator=op, args=(left, _lit(const)))

            # Build arity expression children.
            # Each child independently uses dataset-uniform field sampling,
            # enabling cross-field arithmetic (e.g. implied_vol30 - implied_vol40).
            # Division: last arg must be a size-proxy field (drawn directly from proxies).
            args: list[ExpressionNode] = []
            for i in range(arity):
                use_proxies = op in DIVISION_OPS and i == arity - 1
                if use_proxies:
                    child: ExpressionNode = _field_node(_uniform_dataset_profile(rng, proxies), rng)
                else:
                    child = self._expr(
                        inner_budget, rng, all_clusters, proxies,
                        depth_remaining - 1, norm_depth_remaining,
                    )
                    if child is None:
                        child = _field_node(self._sample_field(rng, all_clusters), rng)
                args.append(child)
            return ExpressionNode(operator=op, args=tuple(args))

        # ── if_else ───────────────────────────────────────────────────────
        if production == "if_else":
            child_budget = budget - 3
            # BoolExpr nonterminal for condition: comparison_op(Expr, Expr).
            # This is the ONLY place comparison/logical operators are reachable,
            # preventing bool series from appearing in arithmetic positions.
            cond = self._bool_expr(rng, all_clusters, proxies)
            then_branch = self._expr(child_budget, rng, all_clusters, proxies,
                                     depth_remaining - 1, norm_depth_remaining)
            else_branch = self._expr(child_budget, rng, all_clusters, proxies,
                                     depth_remaining - 1, norm_depth_remaining)
            if cond is None or then_branch is None or else_branch is None:
                return _field_node(self._sample_field(rng, all_clusters), rng)
            return ExpressionNode(operator="if_else", args=(cond, then_branch, else_branch))

        # ── ts_rolling ────────────────────────────────────────────────────
        if production == "ts_rolling":
            inner = self._expr(budget - 2, rng, all_clusters, proxies,
                               depth_remaining - 1, norm_depth_remaining)
            if inner is None:
                return _field_node(self._sample_field(rng, all_clusters), rng)
            op = self._choice("ts_rolling_op", rng, self._cfg.allowed_ts_rolling_ops)
            window = int(self._choice("window", rng, self._cfg.windows))
            # Brain signature: ts_quantile(x, fraction, lookback) — fraction comes before window.
            if op == "ts_quantile":
                fraction = float(_rng_choice(rng, self._cfg.ts_quantile_fractions))
                return ExpressionNode(operator=op, args=(inner, _lit(fraction), _lit(window)))
            return ExpressionNode(operator=op, args=(inner, _lit(window)))

        # ── ts_delta_op ───────────────────────────────────────────────────
        if production == "ts_delta_op":
            inner = self._expr(budget - 2, rng, all_clusters, proxies,
                               depth_remaining - 1, norm_depth_remaining)
            if inner is None:
                return _field_node(self._sample_field(rng, all_clusters), rng)
            op = self._choice("ts_delta_op", rng, self._cfg.allowed_ts_delta_ops)
            dw = int(self._choice("delta_window", rng, self._cfg.delta_windows))
            return ExpressionNode(operator=op, args=(inner, _lit(dw)))

        # ── kth_element ───────────────────────────────────────────────────
        if production == "kth_element":
            inner = self._expr(budget - 2, rng, all_clusters, proxies,
                               depth_remaining - 1, norm_depth_remaining)
            if inner is None:
                return _field_node(self._sample_field(rng, all_clusters), rng)
            d = int(self._choice("window", rng, self._cfg.windows))
            k = int(_rng_choice(rng, self._cfg.kth_element_k_values))
            return ExpressionNode(operator="kth_element", args=(inner, _lit(d), _lit(k)))

        # ── ts_paired ─────────────────────────────────────────────────────
        if production == "ts_paired":
            # PairedRollingOp(Expr, Expr, Num)  Δk=3  [A≠B semantic constraint]
            # field_a samples freely from all_clusters.
            # field_b uses _pick_secondary to enforce A≠B (ts_corr(x,x,...) = 1 always).
            primary_profile = self._sample_field(rng, all_clusters)
            field_a = self._expr(budget - 3, rng, all_clusters, proxies,
                                 depth_remaining - 1, norm_depth_remaining)
            if field_a is None:
                return _field_node(primary_profile, rng)
            if rng.random() < self._cfg.ts_step_prob:
                field_b: ExpressionNode = ExpressionNode(operator="ts_step", args=(_lit(1),))
            else:
                secondary = _pick_secondary(rng, primary_profile, all_clusters, self._sample_field)
                if secondary is None:
                    op_r = self._choice("ts_rolling_op", rng, self._cfg.allowed_ts_rolling_ops)
                    window_r = int(self._choice("window", rng, self._cfg.windows))
                    return ExpressionNode(operator=op_r, args=(field_a, _lit(window_r)))
                field_b_node = self._expr(budget - 3, rng, all_clusters, proxies,
                                          depth_remaining - 1, norm_depth_remaining)
                if field_b_node is None:
                    return _field_node(primary_profile, rng)
                field_b = field_b_node
            op = self._choice("ts_paired_op", rng, self._cfg.allowed_ts_paired_ops)
            window = int(self._choice("window", rng, self._cfg.windows))
            return ExpressionNode(operator=op, args=(field_a, field_b, _lit(window)))

        # ── ts_regression ─────────────────────────────────────────────────
        if production == "ts_regression":
            # ts_regression(y, x, lookback [, rettype])  Δk=3
            # Both y and x sample independently from all_clusters.
            primary_profile = self._sample_field(rng, all_clusters)
            secondary = _pick_secondary(rng, primary_profile, all_clusters, self._sample_field)
            if secondary is None:
                inner = self._expr(budget - 2, rng, all_clusters, proxies,
                                   depth_remaining - 1, norm_depth_remaining)
                if inner is None:
                    return _field_node(primary_profile, rng)
                op_r = self._choice("ts_rolling_op", rng, self._cfg.allowed_ts_rolling_ops)
                window_r = int(self._choice("window", rng, self._cfg.windows))
                return ExpressionNode(operator=op_r, args=(inner, _lit(window_r)))
            y_expr = self._expr(budget - 3, rng, all_clusters, proxies,
                                depth_remaining - 1, norm_depth_remaining)
            x_expr = self._expr(budget - 3, rng, all_clusters, proxies,
                                depth_remaining - 1, norm_depth_remaining)
            if y_expr is None or x_expr is None:
                return _field_node(primary_profile, rng)
            window = int(self._choice("window", rng, self._cfg.windows))
            rettype = int(_rng_choice(rng, self._cfg.ts_regression_rettypes))
            # 3-arg form when rettype=0 (Brain default); 4-arg form otherwise.
            # NOTE: 4-arg form ts_regression(y,x,d,rettype) requires Brain verification.
            if rettype == 0:
                return ExpressionNode(operator="ts_regression",
                                      args=(y_expr, x_expr, _lit(window)))
            return ExpressionNode(operator="ts_regression",
                                  args=(y_expr, x_expr, _lit(window), _lit(rettype)))

        # ── Inner group_op ────────────────────────────────────────────────
        if production == "expr_group_op":
            inner = self._expr(budget - 1, rng, all_clusters, proxies,
                               depth_remaining - 1, norm_depth_remaining - 1)
            if inner is None:
                return _field_node(self._sample_field(rng, all_clusters), rng)
            op = self._choice("group_op", rng, self._cfg.allowed_group_ops)
            non_none = [g for g in self._cfg.groups if g != "none"]
            chosen = str(_rng_choice(rng, non_none)) if non_none else "market"
            grp = self._group_expr(rng, all_clusters, chosen)
            return ExpressionNode(operator=op, args=(inner, grp))

        # ── Inner cs_op ───────────────────────────────────────────────────
        if production == "expr_cs_op":
            inner = self._expr(budget - 1, rng, all_clusters, proxies,
                               depth_remaining - 1, norm_depth_remaining - 1)
            if inner is None:
                return _field_node(self._sample_field(rng, all_clusters), rng)
            op = self._choice("cs_op", rng, self._cfg.allowed_cs_ops)
            return ExpressionNode(operator=op, args=(inner,))

        # ── Inner group_backfill ──────────────────────────────────────────
        if production == "expr_group_window_op":
            inner = self._expr(budget - 2, rng, all_clusters, proxies,
                               depth_remaining - 1, norm_depth_remaining - 1)
            if inner is None:
                return _field_node(self._sample_field(rng, all_clusters), rng)
            op = self._choice("group_window_op", rng, self._cfg.allowed_group_window_ops)
            non_none = [g for g in self._cfg.groups if g != "none"]
            chosen = str(_rng_choice(rng, non_none)) if non_none else "market"
            grp = self._group_expr(rng, all_clusters, chosen)
            d = int(self._choice("delta_window", rng, self._cfg.delta_windows))
            return ExpressionNode(operator=op, args=(inner, grp, _lit(d)))

        # ── Inner group_mean ──────────────────────────────────────────────
        if production == "expr_group_weighted_op":
            # weight_expr is a cheap Expr (budget capped at 2, no norm ops).
            inner = self._expr(budget - 2, rng, all_clusters, proxies,
                               depth_remaining - 1, norm_depth_remaining - 1)
            if inner is None:
                return _field_node(self._sample_field(rng, all_clusters), rng)
            wt_primary = self._sample_field(rng, all_clusters)
            weight_profile = _pick_secondary(rng, wt_primary, all_clusters, self._sample_field)
            if weight_profile is None:
                op = self._choice("group_op", rng, self._cfg.allowed_group_ops)
                non_none = [g for g in self._cfg.groups if g != "none"]
                chosen = str(_rng_choice(rng, non_none)) if non_none else "market"
                grp = self._group_expr(rng, all_clusters, chosen)
                return ExpressionNode(operator=op, args=(inner, grp))
            weight_expr = self._expr(min(budget - 2, 2), rng, all_clusters, proxies,
                                     depth_remaining - 1, 0)
            if weight_expr is None:
                weight_expr = _field_node(weight_profile, rng)
            op = self._choice("group_weighted_op", rng, self._cfg.allowed_group_weighted_ops)
            non_none = [g for g in self._cfg.groups if g != "none"]
            chosen = str(_rng_choice(rng, non_none)) if non_none else "market"
            grp = self._group_expr(rng, all_clusters, chosen)
            return ExpressionNode(operator=op, args=(inner, weight_expr, grp))

        # Unreachable — feasible_expr_productions guarantees a valid production.
        return _field_node(self._sample_field(rng, all_clusters), rng)

    # ------------------------------------------------------------------
    # BoolExpr nonterminal — comparison / logical ops for if_else conditions
    # ------------------------------------------------------------------

    def _bool_expr(
        self,
        rng: np.random.Generator,
        all_clusters: list[FieldCluster],
        proxies: list[DataProfile],
    ) -> ExpressionNode:
        """BoolExpr → comparison_op(Field, Field) | logical_op(BoolExpr, BoolExpr).

        Comparison and logical operators are ONLY reachable through this method
        (called from the if_else production).  They cannot appear in arbitrary Expr
        positions, preventing bool series from being passed to arithmetic operators.

        Uses plain field leaves (not recursive Expr) to keep the condition cheap
        and interpretable.
        """
        if (
            self._cfg.allowed_logical_ops
            and rng.random() < self._cfg.bool_logical_prob
        ):
            lhs = self._single_cmp(rng, all_clusters)
            rhs = self._single_cmp(rng, all_clusters)
            op = _rng_choice(rng, self._cfg.allowed_logical_ops)
            return ExpressionNode(operator=op, args=(lhs, rhs))
        return self._single_cmp(rng, all_clusters)

    def _single_cmp(
        self,
        rng: np.random.Generator,
        all_clusters: list[FieldCluster],
    ) -> ExpressionNode:
        """comparison_op(field_a, field_b) — both fields drawn uniformly from all_clusters."""
        fa = _field_node(self._sample_field(rng, all_clusters), rng)
        fb = _field_node(self._sample_field(rng, all_clusters), rng)
        op = _rng_choice(rng, self._cfg.allowed_comparison_ops)
        return ExpressionNode(operator=op, args=(fa, fb))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_ops(self, tree: ExpressionNode, ops: list[str]) -> None:
        if tree.literal is not None:
            return
        if tree.operator:
            ops.append(tree.operator)
        for arg in tree.args:
            if isinstance(arg, ExpressionNode):
                self._collect_ops(arg, ops)

    def _sample_field(
        self,
        rng: np.random.Generator,
        all_clusters: list[FieldCluster],
    ) -> DataProfile:
        if self._field_sampler is not None:
            return self._field_sampler(rng, all_clusters)
        return _stratified_field(rng, all_clusters)

    def _choice(
        self,
        kind: str,
        rng: np.random.Generator,
        seq: Sequence[_T],
    ) -> _T:
        if self._choice_sampler is not None:
            chosen = self._choice_sampler(rng, kind, seq)
            if chosen in seq:
                return chosen
        return _rng_choice(rng, seq)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Field sampling.
# Numeric leaves are sampled uniformly by dataset, then by cluster within that
# dataset, then by member field within that cluster. Dataset weights are kept
# only for backward compatibility with external imports; they are not used by
# the grammar sampler.
# ---------------------------------------------------------------------------

DATASET_WEIGHTS: dict[str, float] = {
    "model":       1.0,
    "analyst":     1.0,
    "fundamental": 1.0,
    "news":        1.0,
    "option":      1.0,
    "pv":          1.0,
    "socialmedia": 1.0,
}
_DATASET_WEIGHT_DEFAULT = 1.0


def _dataset_weight(dataset_id: str) -> float:
    """Return sampling weight for a field based on its dataset prefix."""
    ds = dataset_id.lower()
    for prefix, w in DATASET_WEIGHTS.items():
        if ds.startswith(prefix):
            return w
    return _DATASET_WEIGHT_DEFAULT


def _rng_choice(rng: np.random.Generator, seq: Sequence[_T]) -> _T:
    """Pick one element from seq uniformly at random."""
    idx = int(rng.integers(len(seq)))
    return seq[idx]


def _uniform_all_fields(rng: np.random.Generator, clusters: list) -> DataProfile:
    """Flat-uniform over EVERY usable member field across all clusters.

    Every MATRIX/VECTOR field has equal probability — no dataset/cluster
    stratification and no prior weighting. This is the "open in all directions"
    sampler: with an uncapped cluster catalog its pool is the full field
    universe, so the grammar can reach any field×operator combination.
    """
    pool = [
        member
        for cluster in clusters
        for member in cluster.members
        if member.field_type.upper() in {"MATRIX", "VECTOR"}
    ]
    if not pool:
        raise ValueError("No MATRIX/VECTOR profiles available for numeric expression sampling")
    return pool[int(rng.integers(len(pool)))]


def _stratified_field(rng: np.random.Generator, clusters: list) -> DataProfile:
    """Sample a numeric field uniformly by dataset → cluster → field."""
    by_dataset: dict[str, list[tuple[object, tuple[DataProfile, ...]]]] = {}
    for cluster in clusters:
        members = tuple(
            member for member in cluster.members
            if member.field_type.upper() in {"MATRIX", "VECTOR"}
        )
        if not members:
            continue
        dataset_id = getattr(cluster, "dataset_id", "") or members[0].dataset_id
        by_dataset.setdefault(dataset_id, []).append((cluster, members))
    if not by_dataset:
        raise ValueError("No MATRIX/VECTOR profiles available for numeric expression sampling")
    dataset_id = _rng_choice(rng, tuple(sorted(by_dataset)))
    _cluster, members = _rng_choice(rng, by_dataset[dataset_id])
    return _rng_choice(rng, members)


def _uniform_dataset_profile(
    rng: np.random.Generator,
    profiles: Sequence[DataProfile],
) -> DataProfile:
    """Sample a profile uniformly by dataset, then uniformly within dataset."""
    by_dataset: dict[str, list[DataProfile]] = {}
    for profile in profiles:
        by_dataset.setdefault(profile.dataset_id, []).append(profile)
    if not by_dataset:
        raise ValueError("No profiles available for uniform dataset sampling")
    dataset_id = _rng_choice(rng, tuple(sorted(by_dataset)))
    return _rng_choice(rng, by_dataset[dataset_id])


def _field_node(profile: DataProfile, rng: np.random.Generator) -> ExpressionNode:
    """Return a leaf ExpressionNode for a field.

    MATRIX fields → bare literal.
    VECTOR fields → vec_avg(field) or vec_sum(field), chosen randomly.
    This is a mandatory type-conversion wrapper: vector fields are multi-valued
    per stock per day and must be collapsed to a scalar before any TS/CS op.
    The wrap is treated as Δk=0 (it doesn't consume budget) because it's not a
    grammar choice — it's forced by the field's type.
    """
    lit = _lit(profile.field_id)
    if profile.is_vector:
        op = _rng_choice(rng, ("vec_avg", "vec_sum"))
        return ExpressionNode(operator=op, args=(lit,))
    return lit


def _pick_secondary(
    rng: np.random.Generator,
    primary: DataProfile,
    all_clusters: list[FieldCluster],
    field_sampler: Callable[[np.random.Generator, list[FieldCluster]], DataProfile] | None = None,
) -> DataProfile | None:
    """Pick a secondary profile different from primary using dataset-uniform sampling."""
    for _ in range(20):
        sampler = field_sampler or _stratified_field
        candidate = sampler(rng, all_clusters)
        if candidate.field_id != primary.field_id:
            return candidate
    return None


# Public re-exports for convenience
__all__ = [
    "GrammarConfig",
    "BrainGrammar",
    "_GROUP_OPS",
    "_GROUP_BACKFILL_OPS",
    "_extract_field_ids",
    "_top_window",
    "_horizon_from_window",
]
