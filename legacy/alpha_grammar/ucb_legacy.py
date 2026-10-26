"""Archived UCB bandit implementation — no longer used by the main pipeline.

Replaced by Q-table pruning (qtable.py).  Kept here for reference.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .feedback import SimResult, compute_quality_weight


@dataclass
class BanditArm:
    """One UCB arm for a field cluster (archived)."""
    arm_id: str
    quality_sum: float
    count: float

    def update(self, quality: float) -> None:
        self.quality_sum += quality
        self.count += 1.0

    def ucb_score(self, total_pulls: int, c: float = 1.4) -> float:
        exploit = self.quality_sum / self.count
        explore = c * math.sqrt(math.log(max(total_pulls, 1)) / self.count)
        return exploit + explore

    def quality_rate(self) -> float:
        return self.quality_sum / self.count

    def pass_rate(self) -> float:
        return self.quality_rate()


class FeedbackStore:
    """Archived UCB FeedbackStore — no longer wired into the pipeline."""

    def __init__(self) -> None:
        self._arms: dict[str, BanditArm] = {}
        self._total_pulls: int = 0

    def register_cluster(self, cluster_id: str, coverage: float) -> None:
        if cluster_id not in self._arms:
            cov = max(0.0, min(1.0, coverage))
            k = 2.0
            self._arms[cluster_id] = BanditArm(
                arm_id=cluster_id,
                quality_sum=cov * k,
                count=k,
            )

    def update(self, result: SimResult) -> None:
        quality = compute_quality_weight(
            result.is_sharpe, result.is_fitness, result.is_turnover,
            result.concentrated_weight_ok, result.sub_universe_sharpe_ok,
            result.yearly_stats,
        )
        for cluster_id in result.cluster_ids:
            if cluster_id in self._arms:
                self._arms[cluster_id].update(quality)
                self._total_pulls += 1

    def end_of_batch(self) -> None:
        pass

    def ucb_scores(self, cluster_ids: list[str], c: float = 1.4) -> list[float]:
        return [
            self._arms[cid].ucb_score(self._total_pulls, c)
            if cid in self._arms else 0.0
            for cid in cluster_ids
        ]

    def sampling_probs(self, cluster_ids: list[str], c: float = 1.4) -> list[float]:
        scores = self.ucb_scores(cluster_ids, c)
        if not scores:
            return []
        min_s = min(scores)
        shifted = [s - min_s for s in scores]
        total_s = sum(shifted)
        if total_s == 0.0:
            n = len(shifted)
            return [1.0 / n] * n
        return [s / total_s for s in shifted]

    def top_clusters(self, n: int = 10) -> list[tuple[str, float, float]]:
        ranked = sorted(
            self._arms.values(),
            key=lambda arm: arm.quality_rate(),
            reverse=True,
        )
        return [(arm.arm_id, arm.quality_rate(), arm.count) for arm in ranked[:n]]

    @property
    def total_pulls(self) -> int:
        return self._total_pulls

    @property
    def n_arms(self) -> int:
        return len(self._arms)

    def save(self, path: str) -> None:
        data: dict[str, Any] = {
            "total_pulls": self._total_pulls,
            "arms": {
                arm_id: {"quality_sum": arm.quality_sum, "count": arm.count}
                for arm_id, arm in self._arms.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        self._total_pulls = int(data["total_pulls"])
        self._arms = {
            arm_id: BanditArm(
                arm_id=arm_id,
                quality_sum=float(vals["quality_sum"]),
                count=float(vals["count"]),
            )
            for arm_id, vals in data["arms"].items()
        }


__all__ = ["BanditArm", "FeedbackStore"]
