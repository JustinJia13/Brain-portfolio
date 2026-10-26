"""Canonical alpha quality ordering: submittable > score > fitness.

Fitness alone never ranks an alpha as "good". The lexicographic key is:
  1. submittable — sharpe >= 1.25, fitness >= 1.0, turnover < 0.70, and all
     submission-relevant checks pass
  2. robustness score — year-consistency (wq_alpha.robustness)
  3. |fitness| — tiebreaker only

Use quality_rank_key / is_submittable everywhere alphas are ranked or
selected (wiki exemplars, archaeology, theory promotion, review).
"""

from __future__ import annotations

from wq_alpha.robustness import record_robustness_score
from wq_alpha.workshop.core import is_passing

PASS_SHARPE = 1.25
PASS_FITNESS = 1.0
PASS_TURNOVER = 0.70


def is_submittable(rec: dict) -> bool:
    """Brain-submittable: thresholds AND all submission-relevant checks pass."""
    sharpe = rec.get("is_sharpe")
    fitness = rec.get("is_fitness")
    turnover = rec.get("is_turnover")
    if sharpe is None or fitness is None:
        return False
    if abs(float(sharpe)) < PASS_SHARPE or abs(float(fitness)) < PASS_FITNESS:
        return False
    if turnover is not None and float(turnover) >= PASS_TURNOVER:
        return False
    return is_passing(rec)


def quality_rank_key(rec: dict) -> tuple[int, float, float]:
    """Lexicographic alpha quality: (submittable, robustness score, |fitness|)."""
    rob = record_robustness_score(rec)
    return (
        int(is_submittable(rec)),
        rob if rob is not None else -1.0,
        abs(float(rec.get("is_fitness") or 0.0)),
    )


__all__ = ["is_submittable", "quality_rank_key", "PASS_SHARPE", "PASS_FITNESS", "PASS_TURNOVER"]
