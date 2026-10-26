"""Grammar-coverage awareness for the theory enumerator.

Reuses the grammar pipeline's learning (Q-table over 78k sims) so the theory
lane stops re-simulating (field, operator) regions already explored — and
especially the ones the grammar found dead. Without this the lane re-covers
~90% of structures already in master.

Two uses:
- is_dead(field, op, cluster): hard-skip a variant whose (field, op) is pruned
  or has many observations with consistently near-zero reward.
- strength(field, op): novelty/quality score used to prefer untested or strong
  (field, op) variants when sampling a family down to size.
- saturated_summary(...): a compact map of dead/over-explored operator families
  per cluster, fed to the economist so it proposes structurally novel motifs.
"""

from __future__ import annotations

from alpha_grammar.qtable import QTable, extract_primary_op

DEAD_MIN_OBS = 8      # enough tries to trust the verdict
DEAD_MEAN_Q = 0.10    # reward this low after DEAD_MIN_OBS tries = effectively dead


class GrammarCoverage:
    def __init__(self, qtable: QTable) -> None:
        self.q = qtable
        self._pruned2 = qtable.pruned_level2()
        self._pruned1 = qtable.pruned_level1()

    @classmethod
    def load(cls, path=None) -> "GrammarCoverage":
        from alpha_grammar.qtable import DEFAULT_QTABLE_PATH
        return cls(QTable.load(path or DEFAULT_QTABLE_PATH))

    def primary_op(self, expression: str, field: str) -> str | None:
        return extract_primary_op(expression, field)

    def is_dead(self, field: str, op: str | None, cluster: str | None = None) -> bool:
        if op is None:
            return False
        if (field, op) in self._pruned2:
            return True
        if cluster is not None and (cluster, op) in self._pruned1:
            return True
        entry = self.q.level2.get((field, op))
        if entry is not None and entry.n_obs >= DEAD_MIN_OBS and entry.mean_q < DEAD_MEAN_Q:
            return True
        return False

    def strength(self, field: str, op: str | None) -> float:
        """Higher = better candidate. Untested (field, op) are novel (=1.0);
        tested ones score by their mean reward; dead ones score negative."""
        if op is None:
            return 0.5
        entry = self.q.level2.get((field, op))
        if entry is None:
            return 1.0  # never tried — novel, prefer it
        if entry.n_obs >= DEAD_MIN_OBS and entry.mean_q < DEAD_MEAN_Q:
            return -1.0
        return min(1.0, entry.mean_q)

    def saturated_summary(self, cluster_ids: list[str], *, max_lines: int = 25) -> str:
        """Per-cluster: which operator families are dead or already heavy.

        Matches level-1 (cluster, op) entries to the batch's clusters.
        """
        lines: list[str] = []
        for cid in cluster_ids:
            dead, heavy = [], []
            for (cluster, op), entry in self.q.level1.items():
                if cluster != cid:
                    continue
                if entry.n_obs >= DEAD_MIN_OBS and entry.mean_q < DEAD_MEAN_Q:
                    dead.append(op)
                elif entry.n_obs >= DEAD_MIN_OBS:
                    heavy.append(op)
            if dead or heavy:
                lines.append(
                    f"- {cid}: dead ops [{', '.join(sorted(set(dead))[:6])}] "
                    f"already-heavy ops [{', '.join(sorted(set(heavy))[:6])}]"
                )
            if len(lines) >= max_lines:
                break
        return "\n".join(lines) if lines else "(no prior grammar coverage for these clusters)"


__all__ = ["GrammarCoverage", "DEAD_MIN_OBS", "DEAD_MEAN_Q"]
