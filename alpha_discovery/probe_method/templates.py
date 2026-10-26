"""Probe templates for fast field-level signal discovery.

These templates are intentionally simple. They are meant to answer whether a
field has a direction, horizon, or normalization that is worth mining further.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTemplate:
    template_id: str
    family: str
    expression: str
    group: str
    horizon: str
    direction: str
    description: str
    concentration_risk: str = "normal"

    def render(self, x: str) -> str:
        return self.expression.format(x=x)


CORE_TEMPLATES: tuple[ProbeTemplate, ...] = (
    ProbeTemplate(
        "cs_rank_level",
        "level",
        "rank({x})",
        "none",
        "cross_sectional",
        "positive",
        "Plain cross-sectional level.",
    ),
    ProbeTemplate(
        "cs_rank_level_neg",
        "level",
        "rank(-{x})",
        "none",
        "cross_sectional",
        "negative",
        "Plain cross-sectional level with sign flipped.",
    ),
    ProbeTemplate(
        "grp_rank_industry",
        "relative_value",
        "group_rank({x}, industry)",
        "industry",
        "cross_sectional",
        "positive",
        "Industry-relative level rank.",
    ),
    ProbeTemplate(
        "grp_rank_industry_neg",
        "relative_value",
        "group_rank(-{x}, industry)",
        "industry",
        "cross_sectional",
        "negative",
        "Industry-relative level rank with sign flipped.",
    ),
    ProbeTemplate(
        "grp_zscore_sector",
        "relative_value",
        "group_zscore({x}, sector)",
        "sector",
        "cross_sectional",
        "positive",
        "Sector-relative standardized level.",
    ),
    ProbeTemplate(
        "ts_rank_60_industry",
        "time_series_rank",
        "group_rank(ts_rank({x}, 60), industry)",
        "industry",
        "medium",
        "positive",
        "Medium-horizon time-series rank, then industry-relative rank.",
    ),
    ProbeTemplate(
        "ts_rank_120_industry_neg",
        "time_series_rank",
        "group_rank(-ts_rank({x}, 120), industry)",
        "industry",
        "long",
        "negative",
        "Long-horizon time-series rank with sign flipped.",
    ),
    ProbeTemplate(
        "ts_zscore_120_sector",
        "time_series_zscore",
        "group_zscore(ts_zscore({x}, 120), sector)",
        "sector",
        "long",
        "positive",
        "Long-horizon standardized abnormal level.",
    ),
    ProbeTemplate(
        "delta_rev_5_industry",
        "delta_reversal",
        "group_rank(-ts_delta({x}, 5), industry)",
        "industry",
        "short",
        "negative_delta",
        "Short-horizon reversal probe.",
    ),
    ProbeTemplate(
        "delta_rev_20_industry",
        "delta_reversal",
        "group_rank(-ts_delta({x}, 20), industry)",
        "industry",
        "medium",
        "negative_delta",
        "Medium-horizon reversal probe.",
    ),
    ProbeTemplate(
        "self_mean_revert_60",
        "self_residual",
        "group_rank(-(({x}) - ts_mean({x}, 60)), industry)",
        "industry",
        "medium",
        "mean_reversion",
        "Distance from own medium-term mean, sign flipped.",
    ),
    ProbeTemplate(
        "smooth_level_20",
        "smooth_level",
        "group_rank(ts_mean({x}, 20), industry)",
        "industry",
        "short",
        "positive",
        "Smoothed short-horizon level.",
    ),
)


BROAD_TEMPLATES: tuple[ProbeTemplate, ...] = CORE_TEMPLATES + (
    ProbeTemplate(
        "winsor_rank_level",
        "distribution_repair",
        "rank(winsorize({x}))",
        "none",
        "cross_sectional",
        "positive",
        "Clipped level rank, useful for concentration-sensitive fields.",
    ),
    ProbeTemplate(
        "scale_winsor_level",
        "distribution_repair",
        "scale(winsorize({x}))",
        "none",
        "cross_sectional",
        "positive",
        "Clipped and scaled level.",
    ),
    ProbeTemplate(
        "vol_stability_60",
        "stability",
        "group_rank(-ts_std_dev({x}, 60), industry)",
        "industry",
        "medium",
        "low_vol",
        "Low own-volatility or stable-field probe.",
    ),
    ProbeTemplate(
        "decayed_delta_rev_20",
        "delta_reversal",
        "group_rank(ts_decay_linear(-ts_delta({x}, 20), 10), industry)",
        "industry",
        "medium",
        "negative_delta",
        "Smoothed medium-horizon reversal.",
    ),
)


EVENT_TEMPLATES: tuple[ProbeTemplate, ...] = (
    ProbeTemplate(
        "days_from_change_industry",
        "event_staleness",
        "group_rank(-days_from_last_change({x}), industry)",
        "industry",
        "event",
        "recent_change",
        "Recent-change event probe; can concentrate on sparse fields.",
        concentration_risk="high",
    ),
    ProbeTemplate(
        "kth_recent_60_sector",
        "event_staleness",
        "group_zscore(kth_element({x}, 60, k=1), sector)",
        "sector",
        "event",
        "recent_value",
        "Recent non-NaN value probe; can concentrate on sparse fields.",
        concentration_risk="high",
    ),
)


def default_templates(pack: str = "core") -> tuple[ProbeTemplate, ...]:
    """Return templates for a named probe pack."""
    pack = pack.lower().strip()
    if pack == "core":
        return CORE_TEMPLATES
    if pack == "broad":
        return BROAD_TEMPLATES
    if pack == "event":
        return EVENT_TEMPLATES
    if pack == "all":
        return BROAD_TEMPLATES + EVENT_TEMPLATES
    raise ValueError(f"Unknown template pack: {pack}")
