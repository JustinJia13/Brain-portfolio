"""Relationship template registry for operator-first alpha discovery.

Each template encodes one economic hypothesis via a fixed operator skeleton.
Field slots ({a}, {b}, optional {c}) are filled at sampling time.
Window ({w}) is swept over a fixed grid when needs_window=True.

Slot constraints:
    slot_a_datasets / slot_b_datasets: frozenset of allowed dataset_id values.
    slot_a_roles / slot_b_roles: frozenset of allowed semantic_role values
        (defined in field_sampler.py).  None on any constraint means "any".

Role-constrained templates restrict which economic concepts fill each slot —
e.g., rank({a} / {b}) is most meaningful when a=income-statement field and
b=market-size or balance-sheet field (classic fundamental ratios).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RelationshipTemplate:
    template_id: str
    description: str
    expression: str                        # FASTEXPR with {a}, {b}, optional {c}, {w}
    needs_window: bool
    arity: int = 2                         # number of field slots (2 or 3)
    windows: tuple[int, ...] = ()
    # Dataset constraints (None = any dataset)
    slot_a_datasets: frozenset[str] | None = None
    slot_b_datasets: frozenset[str] | None = None
    slot_c_datasets: frozenset[str] | None = None
    # Semantic role constraints (None = any role); role constants live in field_sampler.py
    slot_a_roles: frozenset[str] | None = None
    slot_b_roles: frozenset[str] | None = None
    slot_c_roles: frozenset[str] | None = None

    def render(self, a: str, b: str, w: int | None = None, c: str | None = None) -> str:
        expr = self.expression.replace("{a}", a).replace("{b}", b)
        if c is not None:
            expr = expr.replace("{c}", c)
        if w is not None:
            expr = expr.replace("{w}", str(w))
        return expr

    def candidate_windows(self) -> tuple[int, ...]:
        """Windows to sweep: defined list when windowed, single None when not."""
        return self.windows if self.needs_window else ()


# ---------------------------------------------------------------------------
# Screening templates — submitted first to identify promising field pairs
# ---------------------------------------------------------------------------

SCREENING_TEMPLATES: list[RelationshipTemplate] = [
    # --- Windowless: ratio / spread ---
    RelationshipTemplate(
        template_id="ratio_raw",
        description="Cross-sectional ratio — long stocks with high a/b",
        expression="rank({a} / ({b} + 1e-8))",
        needs_window=False,
    ),
    RelationshipTemplate(
        template_id="ratio_group",
        description="Sector-relative ratio",
        expression="group_rank(rank({a} / ({b} + 1e-8)), sector)",
        needs_window=False,
    ),
    RelationshipTemplate(
        template_id="rank_spread",
        description="Rank difference — long high-a, short high-b",
        expression="rank({a}) - rank({b})",
        needs_window=False,
    ),
    RelationshipTemplate(
        template_id="winsorized_ratio",
        description="Outlier-robust ratio",
        expression="rank(winsorize({a}) / (winsorize({b}) + 1e-8))",
        needs_window=False,
    ),
    # --- Windowed: correlation / covariance / regression ---
    RelationshipTemplate(
        template_id="corr_ranked",
        description="Rolling correlation of CS-ranked inputs",
        expression="rank(ts_corr(rank({a}), rank({b}), {w}))",
        needs_window=True,
        windows=(20, 60, 120),
    ),
    RelationshipTemplate(
        template_id="corr_sector_neutral",
        description="Sector-neutralised rolling correlation",
        expression="group_neutralize(rank(ts_corr({a}, {b}, {w})), sector)",
        needs_window=True,
        windows=(20, 60, 120),
    ),
    RelationshipTemplate(
        template_id="covariance_ranked",
        description="Rolling covariance — magnitude-sensitive",
        expression="rank(ts_covariance(rank({a}), rank({b}), {w}))",
        needs_window=True,
        windows=(20, 60, 120),
    ),
    RelationshipTemplate(
        template_id="regression_slope",
        description="Rolling OLS slope of a on b",
        expression="rank(ts_regression(rank({a}), rank({b}), {w}))",
        needs_window=True,
        windows=(20, 60, 120),
    ),
    RelationshipTemplate(
        template_id="regression_resid",
        description="Rolling OLS residual — a after removing b's influence",
        expression="rank(ts_regression(rank({a}), rank({b}), {w}, rettype=1))",
        needs_window=True,
        windows=(20, 60, 120),
    ),
    RelationshipTemplate(
        template_id="rolling_spread",
        description="Rolling mean of raw spread",
        expression="rank(ts_mean({a} - {b}, {w}))",
        needs_window=True,
        windows=(20, 60, 120),
    ),
    RelationshipTemplate(
        template_id="rolling_ratio_mean",
        description="Smoothed rolling ratio",
        expression="rank(ts_mean({a} / ({b} + 1e-8), {w}))",
        needs_window=True,
        windows=(20, 60, 120),
    ),
]

# ---------------------------------------------------------------------------
# Promotion templates — applied to field pairs that pass screening
# ---------------------------------------------------------------------------

PROMOTION_TEMPLATES: list[RelationshipTemplate] = [
    # Triples
    RelationshipTemplate(
        template_id="corr_of_spread",
        description="Correlation of intra-pair spread with third signal",
        expression="rank(ts_corr({a} - {b}, {c}, {w}))",
        needs_window=True,
        arity=3,
        windows=(20, 60, 120),
        slot_c_datasets=frozenset({"pv1", "news18", "sentiment1"}),
    ),
    RelationshipTemplate(
        template_id="ratio_times_signal",
        description="Ratio scaled by third signal",
        expression="rank({a} / ({b} + 1e-8) * rank({c}))",
        needs_window=False,
        arity=3,
        slot_c_datasets=frozenset({"pv1", "news18", "sentiment1", "model16", "model51"}),
    ),
    # Neutralization variants of the winning screening templates
    RelationshipTemplate(
        template_id="ratio_raw_industry",
        description="Industry-neutralised ratio",
        expression="group_rank(rank({a} / ({b} + 1e-8)), industry)",
        needs_window=False,
    ),
    RelationshipTemplate(
        template_id="ratio_raw_market",
        description="Market-neutralised ratio",
        expression="group_rank(rank({a} / ({b} + 1e-8)), market)",
        needs_window=False,
    ),
    RelationshipTemplate(
        template_id="corr_industry_neutral",
        description="Industry-neutralised rolling correlation",
        expression="group_neutralize(rank(ts_corr({a}, {b}, {w})), industry)",
        needs_window=True,
        windows=(20, 60, 120),
    ),
]

# ---------------------------------------------------------------------------
# Role-constrained screening templates
# These apply the same operators but restrict slots by semantic role.
# Import role constants lazily to avoid circular import with field_sampler.
# ---------------------------------------------------------------------------

def _make_role_constrained_templates() -> list[RelationshipTemplate]:
    """Build role-constrained template variants using semantic role constants."""
    from alpha_discovery.legacy.operator_skeleton_method.roles import (
        ROLE_INCOME_STMT, ROLE_CASHFLOW, ROLE_BALANCE_SHEET,
        ROLE_MARKET_SIZE, ROLE_VALUATION,
        ROLE_ANALYST, ROLE_FACTOR,
        ROLE_PRICE, ROLE_VOLUME, ROLE_RETURN, ROLE_VOLATILITY,
        ROLE_SENTIMENT, ROLE_OPTION,
    )
    _FUNDAMENTAL = frozenset({ROLE_INCOME_STMT, ROLE_CASHFLOW, ROLE_BALANCE_SHEET, ROLE_VALUATION})
    _SIZE = frozenset({ROLE_MARKET_SIZE, ROLE_BALANCE_SHEET})
    _SIGNAL = frozenset({ROLE_ANALYST, ROLE_SENTIMENT, ROLE_FACTOR, ROLE_OPTION})
    _MOMENTUM = frozenset({ROLE_PRICE, ROLE_RETURN, ROLE_VOLUME})

    return [
        # Fundamental ratio: income/cashflow field divided by market-size/balance-sheet
        # e.g., rank(ebitda / (cap + 1e-8))  — earnings yield style
        RelationshipTemplate(
            template_id="fundamental_ratio",
            description="Fundamental yield: income/cashflow numerator scaled by size/balance-sheet denominator",
            expression="rank({a} / ({b} + 1e-8))",
            needs_window=False,
            slot_a_roles=_FUNDAMENTAL,
            slot_b_roles=_SIZE,
        ),
        RelationshipTemplate(
            template_id="fundamental_ratio_sector",
            description="Sector-relative fundamental yield",
            expression="group_rank(rank({a} / ({b} + 1e-8)), sector)",
            needs_window=False,
            slot_a_roles=_FUNDAMENTAL,
            slot_b_roles=_SIZE,
        ),
        # Analyst vs. fundamental: how does analyst estimate relate to what the income statement shows?
        RelationshipTemplate(
            template_id="analyst_vs_fundamental_corr",
            description="Rolling corr between analyst estimate and fundamental data",
            expression="rank(ts_corr(rank({a}), rank({b}), {w}))",
            needs_window=True,
            windows=(20, 60, 120),
            slot_a_roles=frozenset({ROLE_ANALYST}),
            slot_b_roles=_FUNDAMENTAL,
        ),
        # Momentum vs. fundamental: does recent return correlate with a fundamental signal?
        RelationshipTemplate(
            template_id="momentum_vs_fundamental_corr",
            description="Rolling corr between price/return and fundamental signal",
            expression="rank(ts_corr(rank({a}), rank({b}), {w}))",
            needs_window=True,
            windows=(20, 60, 120),
            slot_a_roles=_MOMENTUM,
            slot_b_roles=_FUNDAMENTAL,
        ),
        # Signal normalization: any alpha signal scaled by market size
        RelationshipTemplate(
            template_id="signal_per_size",
            description="Soft signal scaled by market size",
            expression="rank({a} / ({b} + 1e-8))",
            needs_window=False,
            slot_a_roles=_SIGNAL,
            slot_b_roles=frozenset({ROLE_MARKET_SIZE}),
        ),
        # Rolling spread between two fundamental fields (e.g., assets - liabilities)
        RelationshipTemplate(
            template_id="fundamental_spread",
            description="Rolling mean of spread between two fundamental fields",
            expression="rank(ts_mean({a} - {b}, {w}))",
            needs_window=True,
            windows=(20, 60, 120),
            slot_a_roles=_FUNDAMENTAL,
            slot_b_roles=_FUNDAMENTAL,
        ),
    ]


ROLE_CONSTRAINED_TEMPLATES: list[RelationshipTemplate] = _make_role_constrained_templates()

SCREENING_TEMPLATE_IDS: frozenset[str] = frozenset(t.template_id for t in SCREENING_TEMPLATES)
PROMOTION_TEMPLATE_IDS: frozenset[str] = frozenset(t.template_id for t in PROMOTION_TEMPLATES)
ROLE_CONSTRAINED_TEMPLATE_IDS: frozenset[str] = frozenset(
    t.template_id for t in ROLE_CONSTRAINED_TEMPLATES
)

ALL_TEMPLATES: dict[str, RelationshipTemplate] = {
    t.template_id: t
    for t in SCREENING_TEMPLATES + PROMOTION_TEMPLATES + ROLE_CONSTRAINED_TEMPLATES
}
