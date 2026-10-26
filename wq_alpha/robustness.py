"""Year-consistency robustness scoring for sim records.

The IQC competition scores forward out-of-sample performance, so selection
must reward alphas whose performance is consistent year over year, not IS
spikes. This module turns a record's ``yearly_stats`` into a single
``robustness_score`` in [0, 1] plus its components, and provides the reward
multiplier used by the Q-table, workshop routing, and ranking.

Conventions shared with the rest of the pipeline:
- Missing yearly stats is not evidence of fragility: scores are None and the
  reward multiplier is 1.0 (same convention as workshop_self_corr_eligible).
- yearly_stats keys are ints at fetch time but strings after a JSON
  round-trip; everything here goes through normalize_yearly_stats.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, fields as dataclass_fields

# ---------------------------------------------------------------------------
# Tunables (validated offline by tools/replay_robustness.py before changing)
# ---------------------------------------------------------------------------

SHARPE_TARGET: float = 1.25       # yearly sharpe at which a component saturates
MIN_SCORED_YEARS: int = 3         # fewer full years -> no score (treated as missing)
WEIGHT_SIGN_CONSISTENCY: float = 0.45
WEIGHT_MEDIAN: float = 0.30
WEIGHT_RECENT2: float = 0.25
BAD_YEAR_SHARPE: float = -1.0     # any oriented year below this -> haircut
BAD_YEAR_HAIRCUT: float = 0.5
REWARD_FLOOR: float = 0.25        # multiplier = floor + (1-floor) * score


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalize_yearly_stats(raw: dict | None) -> dict[int, dict]:
    """Return {int_year: stats} keeping only IS-stage years with a sharpe.

    Handles both int keys (fresh from fetch_yearly_stats) and string keys
    (after JSON round-trip through sim logs).
    """
    out: dict[int, dict] = {}
    if not isinstance(raw, dict):
        return out
    for key, stats in raw.items():
        if not isinstance(stats, dict):
            continue
        try:
            year = int(key)
        except (TypeError, ValueError):
            continue
        stage = str(stats.get("stage") or "IS").upper()
        if stage != "IS":
            continue
        if stats.get("sharpe") is None:
            continue
        out[year] = stats
    return out


@dataclass
class RobustnessScore:
    n_years: int
    min_year_sharpe: float
    median_year_sharpe: float
    sign_consistency: float
    recent2_sharpe: float
    score: float

    def as_record_fields(self) -> dict:
        return {f"robustness_{f.name}": getattr(self, f.name) for f in dataclass_fields(self)}


def compute_robustness(
    yearly_stats: dict | None,
    *,
    direction_hint: float | None = None,
    partial_year: int | None = None,
) -> RobustnessScore | None:
    """Score year-over-year consistency; None when there is too little data.

    direction_hint: overall is_sharpe of the record. Negative-direction alphas
    (negate-lane material) are oriented by their overall sign so a consistently
    negative signal scores as consistent, matching the |fitness| convention in
    compute_q_reward.
    partial_year: calendar year the simulation was run; that year's stats only
    cover a fraction of the year, so it is dropped when enough full years remain.
    """
    stats = normalize_yearly_stats(yearly_stats)
    if not stats:
        return None

    years = sorted(stats)
    if partial_year is not None and partial_year in stats and len(years) - 1 >= MIN_SCORED_YEARS:
        years.remove(partial_year)
    if len(years) < MIN_SCORED_YEARS:
        return None

    sharpes = [float(stats[y].get("sharpe") or 0.0) for y in years]

    direction = 0.0
    if direction_hint is not None:
        direction = float(direction_hint)
    if direction == 0.0:
        direction = sum(sharpes)
    orient = -1.0 if direction < 0 else 1.0
    oriented = [orient * s for s in sharpes]

    sign_consistency = sum(1 for s in oriented if s > 0.0) / len(oriented)
    median_sharpe = statistics.median(oriented)
    min_sharpe = min(oriented)
    recent2 = statistics.mean(oriented[-2:])

    score = (
        WEIGHT_SIGN_CONSISTENCY * sign_consistency
        + WEIGHT_MEDIAN * _clamp01(median_sharpe / SHARPE_TARGET)
        + WEIGHT_RECENT2 * _clamp01(recent2 / SHARPE_TARGET)
    )
    if min_sharpe < BAD_YEAR_SHARPE:
        score *= BAD_YEAR_HAIRCUT

    return RobustnessScore(
        n_years=len(years),
        min_year_sharpe=min_sharpe,
        median_year_sharpe=median_sharpe,
        sign_consistency=sign_consistency,
        recent2_sharpe=recent2,
        score=round(_clamp01(score), 4),
    )


def _record_partial_year(record: dict) -> int | None:
    for key in ("logged_at", "date_created"):
        value = record.get(key)
        if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


def attach_robustness(record: dict) -> dict:
    """Enrich a sim record in place with flat robustness_* fields."""
    rs = compute_robustness(
        record.get("yearly_stats"),
        direction_hint=record.get("is_sharpe"),
        partial_year=_record_partial_year(record),
    )
    if rs is not None:
        record.update(rs.as_record_fields())
    return record


def record_robustness_score(record: dict) -> float | None:
    """Robustness score of a record: stored field first, else computed."""
    value = record.get("robustness_score")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    rs = compute_robustness(
        record.get("yearly_stats"),
        direction_hint=record.get("is_sharpe"),
        partial_year=_record_partial_year(record),
    )
    return rs.score if rs is not None else None


def reward_multiplier(score: float | None) -> float:
    """Reward multiplier in [REWARD_FLOOR, 1.0]; missing data is not punished."""
    if score is None:
        return 1.0
    return REWARD_FLOOR + (1.0 - REWARD_FLOOR) * _clamp01(float(score))


__all__ = [
    "RobustnessScore",
    "attach_robustness",
    "compute_robustness",
    "normalize_yearly_stats",
    "record_robustness_score",
    "reward_multiplier",
]
