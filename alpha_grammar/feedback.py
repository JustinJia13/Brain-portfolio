"""Quality gate and simulation result type for the grammar pipeline.

Cluster selection is handled by Q-table pruning in qtable.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from wq_alpha.robustness import compute_robustness, reward_multiplier


# ---------------------------------------------------------------------------
# Brain minimum thresholds (used by compute_quality_weight and template extractor)
# ---------------------------------------------------------------------------

SHARPE_MIN: float = 1.25    # Brain IS Sharpe minimum
FITNESS_MIN: float = 1.0    # Brain IS Fitness minimum
TURNOVER_MAX: float = 0.70  # Brain turnover cap (70%)
QUALITY_CAP: float = 4.0    # max quality weight


def compute_quality_weight(
    is_sharpe: float,
    is_fitness: float,
    is_turnover: float | None = None,
    concentrated_weight_ok: bool = True,
    sub_universe_sharpe_ok: bool = True,
    yearly_stats: dict | None = None,
) -> float:
    """Quality weight for legacy compatibility.

    Returns 0.0 if the alpha fails any Brain minimum threshold.
    Returns a value in [0.0, QUALITY_CAP] if it passes, scaled by fitness.

    Note: this function is kept for callers that need a binary pass/fail
    quality gate. For Q-table reward computation use qtable.compute_q_reward()
    instead.
    """
    if not concentrated_weight_ok:
        return 0.0
    if is_sharpe < SHARPE_MIN:
        return 0.0
    if is_fitness < FITNESS_MIN:
        return 0.0
    if is_turnover is not None and is_turnover >= TURNOVER_MAX:
        return 0.0
    if not sub_universe_sharpe_ok:
        return 0.0

    base = min(is_fitness, QUALITY_CAP)

    if yearly_stats:
        rs = compute_robustness(yearly_stats, direction_hint=is_sharpe)
        base = base * reward_multiplier(rs.score if rs is not None else None)

    return base


# ---------------------------------------------------------------------------
# SimResult
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Outcome of one Brain simulation."""
    candidate_id: str
    expression: str
    cluster_ids: tuple[str, ...]
    is_sharpe: float
    is_fitness: float
    passed: bool
    is_turnover: float | None = None
    concentrated_weight_ok: bool = True
    sub_universe_sharpe_ok: bool = True
    yearly_stats: dict | None = None
    max_self_correlation: float | None = None   # from Brain SELF_CORRELATION check value
    max_self_correlation_alpha_id: str | None = None
    max_self_correlation_alpha_sharpe: float | None = None
    max_self_correlation_alpha_fitness: float | None = None
    max_concentration: float | None = None      # from Brain CONCENTRATED_WEIGHT check value


__all__ = [
    "SHARPE_MIN", "FITNESS_MIN", "TURNOVER_MAX", "QUALITY_CAP",
    "compute_quality_weight",
    "SimResult",
]
