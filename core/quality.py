"""Canonical alpha quality ordering: submittable > score > fitness.

Fitness alone never ranks an alpha as "good". The lexicographic key is:
  1. submittable — sharpe >= 1.25, fitness >= 1.0, turnover < 0.70, and all
     submission-relevant checks pass
  2. robustness score — year-consistency (core.robustness)
  3. |fitness| — tiebreaker only

Use quality_rank_key / is_submittable everywhere alphas are ranked or
selected (wiki exemplars, archaeology, theory promotion, review).
"""

from __future__ import annotations

from core.robustness import record_robustness_score

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

# ---------------------------------------------------------------------------
# The submittable predicate. A simulated alpha is submittable only when EVERY
# BRAIN check passes -- including concentrated-weight and sub-universe, not just
# sharpe/fitness/turnover. This lives here, not in the enhancement engine, because
# every lane needs it and the shared layer must not depend upward.
# ---------------------------------------------------------------------------

def resolve_checks(record: dict) -> list[dict]:
    """Get checks list, falling back to raw_alpha_detail.is.checks."""
    checks = record.get("checks")
    if checks:
        return checks
    raw = record.get("raw_alpha_detail") or {}
    return (raw.get("is") or {}).get("checks") or []


def failing_checks(record: dict) -> list[str]:
    """Names of failing checks, excluding warning-only checks (SELF_CORRELATION, UNITS)."""
    return [
        c["name"] for c in resolve_checks(record)
        if c.get("name") not in IGNORED_CHECKS and c.get("result") != "PASS"
    ]


def is_passing(record: dict) -> bool:
    """True if all submission-relevant checks pass (warning-only checks excluded)."""
    checks = resolve_checks(record)
    if not checks:
        grade = (record.get("grade") or "").upper()
        return grade in ("PASS", "GOOD", "AVERAGE")
    for c in checks:
        if c.get("name") in IGNORED_CHECKS:
            continue
        if c.get("result") != "PASS":
            return False
    return True
