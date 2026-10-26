"""Proven alpha template library for targeted window/settings sweeps.

Each AlphaTemplate holds a concrete, verified FASTEXPR alpha. The pipeline sweeps
only over window values and Brain submission settings — no field or group substitution.

Placeholders in expressions:
    {w}   — primary window (days); values from .windows list
    {w2}  — secondary window; values from .windows2 list (cross-product with {w})
    {c1}  — numeric constant; values from .constants["c1"]
    {c2}  — numeric constant; values from .constants["c2"]

If no {w} placeholder is present but .windows is non-empty, all integers in [2, 250]
in the expression are swept one-at-a-time (same integer replaced uniformly per pass).

Usage:
    from wq_alpha.search.templates import AlphaTemplate, ALL_SETTINGS_VARIANTS

    ALPHA_TEMPLATES.append(
        AlphaTemplate(
            alpha_id="my_alpha_v1",
            expression="group_rank(ts_zscore(my_field, {w}), sector)",
            description="...",
            windows=[10, 20, 40, 60, 120],
            settings_overrides=ALL_SETTINGS_VARIANTS,
            dataset_id="model16",
            category="model",
            priority=3.0,
        )
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from itertools import product
from typing import Iterator

# ---------------------------------------------------------------------------
# Settings sweep presets
# ---------------------------------------------------------------------------

#: Each entry is a delta dict overlaid on DEFAULT_SETTINGS for one additional variant.
#: The baseline (empty override) is always included implicitly by variants().
ALL_SETTINGS_VARIANTS: list[dict] = [
    {"neutralization": "INDUSTRY"},
    {"neutralization": "SECTOR"},
    {"neutralization": "MARKET"},
    {"universe": "TOP1000"},
    {"delay": 0},
    {"decay": 1},
    {"decay": 4},
    {"decay": 8},
    {"truncation": 0.05},
]

NEUTRALIZATION_VARIANTS: list[dict] = [
    {"neutralization": "INDUSTRY"},
    {"neutralization": "SECTOR"},
    {"neutralization": "MARKET"},
]

DECAY_VARIANTS: list[dict] = [
    {"decay": 1},
    {"decay": 4},
    {"decay": 8},
]


# ---------------------------------------------------------------------------
# AlphaTemplate
# ---------------------------------------------------------------------------

@dataclass
class AlphaTemplate:
    """A proven alpha expression with targeted window/settings sweep config."""

    alpha_id: str
    """Unique slug, e.g. 'm16_zscore_analyst_rev_sector'."""

    expression: str
    """Concrete FASTEXPR string. May contain {w}, {w2}, {c1}, {c2} placeholders.
    Literal integers are swept when windows != [] and no {w} is present."""

    description: str
    """One-sentence economic rationale."""

    windows: list[int] = dc_field(default_factory=list)
    """Values for {w}. If expression has no {w}, auto-replaces all literal integers
    in [2, 250] in the expression one-at-a-time (same value → same replacement)."""

    windows2: list[int] = dc_field(default_factory=list)
    """Values for {w2} (secondary window). Cross-product with windows."""

    constants: dict[str, list[int | float]] = dc_field(default_factory=dict)
    """Named constant sweep: {"c1": [5, 10, 20]}.
    Each key must match a {key} placeholder in expression.
    Cross-product across all constant axes."""

    settings_overrides: list[dict] = dc_field(default_factory=list)
    """Settings delta dicts to overlay on DEFAULT_SETTINGS, one variant each.
    The baseline (no override) is always yielded first."""

    dataset_id: str = ""
    category: str = ""
    priority: float = 1.0

    # ------------------------------------------------------------------
    def _window_expressions(self) -> list[tuple[str, dict]]:
        """Return (expression, params) pairs after window substitution."""
        uses_w = "{w}" in self.expression
        uses_w2 = "{w2}" in self.expression

        if uses_w:
            w_values: list = self.windows if self.windows else [None]
            w2_values: list = self.windows2 if (uses_w2 and self.windows2) else [None]
            results = []
            for w, w2 in product(w_values, w2_values):
                expr = self.expression
                params: dict = {}
                if w is not None:
                    expr = expr.replace("{w}", str(w))
                    params["window"] = w
                if w2 is not None:
                    expr = expr.replace("{w2}", str(w2))
                    params["window2"] = w2
                results.append((expr, params))
            return results

        elif self.windows:
            # Auto-detect literal integers and sweep them one axis at a time.
            raw_ints = sorted(
                {int(m) for m in re.findall(r'\b(\d+)\b', self.expression)
                 if 2 <= int(m) <= 250}
            )
            seen_exprs: set[str] = {self.expression}
            results: list[tuple[str, dict]] = [(self.expression, {})]
            for orig in raw_ints:
                for repl in self.windows:
                    if repl == orig:
                        continue
                    new_expr = re.sub(r'\b' + str(orig) + r'\b', str(repl), self.expression)
                    if new_expr in seen_exprs:
                        continue
                    seen_exprs.add(new_expr)
                    results.append((new_expr, {"original_window": orig, "window": repl}))
            return results

        else:
            return [(self.expression, {})]

    def _constant_combos(self) -> list[dict]:
        """Return list of {placeholder: value} dicts from constants cross-product."""
        if not self.constants:
            return [{}]
        keys = sorted(self.constants.keys())
        combos = []
        for vals in product(*[self.constants[k] for k in keys]):
            combos.append(dict(zip(keys, vals)))
        return combos

    def variants(self) -> Iterator[tuple[str, dict, dict]]:
        """Yield (expression, params, settings_override) for every variant.

        - expression:       fully substituted FASTEXPR string
        - params:           parameter values used (window, constants) for logging
        - settings_override: dict to overlay on DEFAULT_SETTINGS ({} = baseline)
        """
        settings_list: list[dict] = [{}] + list(self.settings_overrides)
        for win_expr, win_params in self._window_expressions():
            for const_combo in self._constant_combos():
                expr = win_expr
                for placeholder, value in const_combo.items():
                    expr = expr.replace(f"{{{placeholder}}}", str(value))
                params = {**win_params, **const_combo}
                for settings_override in settings_list:
                    yield expr, params, settings_override


# ---------------------------------------------------------------------------
# Template registry — populate with your proven alphas
# ---------------------------------------------------------------------------

ALPHA_TEMPLATES: list[AlphaTemplate] = [
    # Add proven alpha templates here. Each entry should be a concrete working
    # alpha you want to enhance by sweeping windows and Brain settings. Example:
    #
    # AlphaTemplate(
    #     alpha_id="m16_zscore_analyst_rev_sector",
    #     expression="group_rank(ts_zscore(analyst_revision_rank_derivative, {w}), sector)",
    #     description="Analyst revision momentum ranked within sector.",
    #     windows=[10, 20, 40, 60, 120],
    #     settings_overrides=ALL_SETTINGS_VARIANTS,
    #     dataset_id="model16",
    #     category="model",
    #     priority=3.0,
    # ),

    # ---------------------------------------------------------------------------
    # EBIT/OpEx covariance alpha — coefficient + window sweep
    #
    # Sweep axes:
    #   {w}   covariance window & capx std-dev window  (orig: 120)
    #   {w2}  anchor short-term momentum window         (orig: 5)
    #   {c1}  base weight in final blend                (orig: 0.87; anchor = 1-c1)
    #   {c2}  size fixed-offset weight                  (orig: 0.85; rank = 1-c2)
    #
    # Total variants: 4w × 3w2 × 4c1 × 4c2 × 4 settings = 768
    # ---------------------------------------------------------------------------
    AlphaTemplate(
        alpha_id="ebit_opex_cov_size_capx_v1",
        expression="\n".join([
            "e = winsorize(inverse(ts_backfill(ebit, 252)));",
            "o = winsorize(log(ts_backfill(operating_expense, 252)));",
            "size = {c2} + (1.0 - {c2}) * group_rank(ts_delay(ts_backfill(assets, 252), 10), industry);",
            "capx = 0.25 + group_rank(group_backfill(ts_std_dev(anl4_capex_low, {w}), industry, 5), industry);",
            "raw = -divide(ts_covariance(e, o, {w}), multiply(size, capx));",
            "base = group_backfill(raw, industry, 5);",
            "anchor = 0.5 * group_rank(-ts_sum(returns, {w2}), industry)"
            "       + 0.5 * group_rank(-(close / vwap - 1), industry);",
            "normalize({c1} * winsorize(base) + (1.0 - {c1}) * anchor)",
        ]),
        description=(
            "EBIT/OpEx covariance scaled by size+capx, blended with short-term reversal anchor."
        ),
        windows=[60, 90, 120, 180],
        windows2=[3, 5, 10],
        constants={
            "c1": [0.75, 0.80, 0.87, 0.93],   # base weight (anchor = 1 - c1)
            "c2": [0.75, 0.80, 0.85, 0.90],   # size fixed-offset (rank = 1 - c2)
        },
        settings_overrides=NEUTRALIZATION_VARIANTS,
        dataset_id="analyst4",
        category="fundamental",
        priority=4.0,
    ),
]


def get_alpha_templates() -> list[AlphaTemplate]:
    """Return the registered alpha template list."""
    return list(ALPHA_TEMPLATES)
