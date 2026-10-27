"""Q-table pruning system for the grammar discovery pipeline.

Two-level Q-table that learns which (cluster, operator) and (field, operator)
combinations are dead ends, then permanently prunes them from the search space.

Design:
- Q(cluster_id, specific_op)   — Level 1: cluster × operator  (MIN_OBS_L1=8)
- Q(field_id,   specific_op)   — Level 2: field × operator    (MIN_OBS_L2=12)
- Both updated from every simulation result with the same reward.
- Prune when mean_q < DEAD_THRESHOLD after the level's MIN_OBS observations.
- Pruning is one-way: entries are never un-pruned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from core.field_catalogs import derive_fields
from core.robustness import compute_robustness, reward_multiplier
from core.quality import CONCENTRATION_LIMIT, SELF_CORR_LIMIT, SELF_CORR_SHARPE_MULTIPLIER

if TYPE_CHECKING:
    from .feedback import SimResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_OBS_L1: int = 8       # minimum observations for Level 1 (cluster × op) pruning
MIN_OBS_L2: int = 12      # minimum observations for Level 2 (field × op) pruning
DEAD_THRESHOLD: float = 0.05  # mean_q below this → prune
MIN_PRUNE_SKELETONS: int = 3  # require structural diversity before pruning
MIN_PRUNE_CONTEXTS_L1: int = 2  # require group/window diversity for cluster-level pruning

# All signal/structural operators tracked by the Q-table.
# Pure mathematical unary ops (abs, log, sign, sqrt, inverse, hump, reverse,
# densify, not, vec_avg, vec_sum) are intentionally excluded — they are
# preprocessing transforms, not signal operators, and the new extraction logic
# will look past them to find the enclosing signal op.
SPECIFIC_OPS: list[str] = [
    # time-series rolling
    "ts_autocorr",
    "ts_av_diff",
    "ts_arg_max",
    "ts_arg_min",
    "ts_backfill",
    "ts_corr",
    "ts_covariance",
    "ts_count_nans",
    "ts_decay_linear",
    "ts_delta",
    "ts_mean",
    "ts_product",
    "ts_quantile",
    "ts_rank",
    "ts_regression",
    "ts_scale",
    "ts_std_dev",
    "ts_std",
    "ts_sum",
    "ts_zscore",
    "ts_delay",
    "ts_max",
    "ts_min",
    # lookback / event
    "days_from_last_change",
    "last_diff_value",
    "kth_element",
    "is_nan",
    # group
    "group_neutralize",
    "group_backfill",
    "group_zscore",
    "group_scale",
    "group_rank",
    "group_mean",
    # cross-sectional
    "normalize",
    "quantile",
    "winsorize",
    "zscore",
    "scale",
    "rank",
    # binary / special
    "signed_power",
    "trade_when",
    "if_else",
    "divide",
    "multiply",
    "subtract",
    "power",
    "add",
    "sub",
    "max",
    "min",
]


# ---------------------------------------------------------------------------
# Degenerate-result filtering
# ---------------------------------------------------------------------------

def _is_zeroish(value: object, tol: float = 1e-12) -> bool:
    """True when value is None or numerically indistinguishable from zero."""
    if value is None:
        return True
    try:
        return abs(float(value)) <= tol
    except (TypeError, ValueError):
        return False


def is_degenerate_sim_record(record: dict) -> bool:
    """Return True for zero-signal rows that should not drive learning/backfill.

    These are the effectively empty alphas where Brain leaves top-level
    fitness unavailable while sharpe and turnover are zero. They do not carry
    useful signal for the Q-table and cannot produce meaningful correlation.
    """
    return (
        record.get("status") == "done"
        and record.get("is_fitness") is None
        and _is_zeroish(record.get("is_sharpe"))
        and _is_zeroish(record.get("is_turnover"))
    )


# ---------------------------------------------------------------------------
# Operator extraction
# ---------------------------------------------------------------------------

def extract_primary_op(expression: str, field_id: str) -> str | None:
    """Return the innermost signal op that is an ancestor of field_id.

    Scans the expression string for all SPECIFIC_OPS occurrences that open
    (i.e. 'op(') before the position of field_id.  In prefix notation the
    outermost op is leftmost, so the LAST such occurrence before the field
    is the innermost enclosing signal op.

    Pure mathematical unary ops (abs, log, sign, sqrt, …) are not in
    SPECIFIC_OPS, so the search automatically looks past them to the first
    semantically meaningful ancestor.

    Examples:
        group_rank(ts_mean(pe_ratio, 20), sector)  → "ts_mean"
        rank(abs(receivable))                       → "rank"   (abs skipped)
        ts_covariance(cashflow_fin, ts_step(1), 60) → "ts_covariance"
        scale(ts_regression(debt, other, 20))       → "ts_regression"
        rank(pe_ratio)                              → "rank"
    """
    field_pos = expression.find(field_id)
    if field_pos == -1:
        return None

    prefix = expression[:field_pos]
    best_op: str | None = None
    best_pos: int = -1

    for op in SPECIFIC_OPS:
        pattern = rf"\b{re.escape(op)}\s*\("
        for m in re.finditer(pattern, prefix):
            if m.start() > best_pos:
                best_pos = m.start()
                best_op = op

    return best_op


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_q_reward(result: "SimResult") -> float:
    """Compute Q-table reward for one simulation result.

    Fitness-driven (no hard Sharpe/turnover cutoffs — fitness already
    reflects signal quality).  Hard zeros:
    - Extreme concentration >= 40% (genuinely fake signal)
    - Self-correlation >= 0.70 (structurally correlated, not submittable)

    Soft 0.5x penalties for borderline concentration and low sub-universe
    Sharpe (both potentially fixable by the workshop).

    Note: max_self_correlation is the signed maximum Pearson correlation with the
    competition portfolio (computed from PnL daily returns). Brain uses the signed
    max (not abs): negative correlation = diversifying, not penalised.
    Gate:
    - < 0.70: no self-correlation penalty
    - ≥ 0.70: keep reward only if candidate IS Sharpe is at least 1.10x the
      most-correlated portfolio alpha's IS Sharpe; otherwise hard zero
    - missing/non-positive benchmark Sharpe fails closed to hard zero

    Year-consistency: when yearly_stats are available the reward is scaled by
    the robustness multiplier (core.robustness), so IS spikes with bad
    year-by-year profiles stop dominating the Q-table. Missing yearly stats
    leave the reward untouched.
    """
    base = min(abs(result.is_fitness), 4.0)

    rs = compute_robustness(result.yearly_stats, direction_hint=result.is_sharpe)
    base *= reward_multiplier(rs.score if rs is not None else None)

    # Concentration: >= 40% = fake signal (hard zero); < 40% = workshop-fixable (soft)
    if not result.concentrated_weight_ok:
        if result.max_concentration is not None and result.max_concentration >= CONCENTRATION_LIMIT:
            return 0.0
        base *= 0.75

    # Sub-universe Sharpe: soft penalty only
    if not result.sub_universe_sharpe_ok:
        base *= 0.5

    # Self-correlation: signed max correlation with competition portfolio.
    # Negative correlation = good diversification, no penalty.
    if result.max_self_correlation is not None:
        if result.max_self_correlation >= SELF_CORR_LIMIT:
            benchmark_sharpe = result.max_self_correlation_alpha_sharpe
            if benchmark_sharpe is None or benchmark_sharpe <= 0.0:
                return 0.0
            if result.is_sharpe + 1e-12 < SELF_CORR_SHARPE_MULTIPLIER * benchmark_sharpe:
                return 0.0

    return base


def apply_pool_delta_reward_veto(record: dict, reward: float, portfolio_score: dict | None = None) -> float:
    """Return zero reward when admission data says contribution delta is non-positive.

    Brain score delta is preferred when present. If Brain comparison was unavailable,
    local fallback delta_pool_sharpe is used. Missing admission data leaves the existing
    Q reward untouched. Positive delta does not add a bonus; it only avoids this hard veto.
    """
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    source = portfolio_score or {}

    delta = _first_float(
        record.get("admission_brain_score_delta"),
        admission.get("brain_score_delta"),
        source.get("admission_brain_score_delta"),
        source.get("brain_score_delta"),
        record.get("admission_delta_pool_sharpe"),
        admission.get("delta_pool_sharpe"),
        source.get("admission_delta_pool_sharpe"),
        source.get("delta_pool_sharpe"),
    )
    if delta is not None and delta <= 0.0:
        return 0.0

    decision = _first_str(
        record.get("admission_decision"),
        admission.get("decision"),
        source.get("admission_decision"),
        source.get("decision"),
    ).upper()
    reason = _first_str(
        record.get("admission_reason"),
        admission.get("reason"),
        source.get("admission_reason"),
        source.get("reason"),
    ).upper()
    if decision == "REJECT" and reason in {"NON_POSITIVE_DELTA", "NON_POSITIVE_BRAIN_SCORE_DELTA"}:
        return 0.0

    return reward


def _first_float(*values: object) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_str(*values: object) -> str:
    for value in values:
        if value in (None, ""):
            continue
        return str(value)
    return ""


def _q_context(*, group: str | None = None, window: int | str | None = None) -> str:
    """Compact structural context used to prevent one-shape Q pruning.

    The Q-table still learns at (cluster, op) and (field, op), but pruning now
    requires that the pair was tried across more than one group/window context.
    This keeps probe-guided sampling from killing a promising pair after only a
    narrow expression family failed.
    """
    group_key = str(group or "none")
    window_key = str(window if window not in (None, "") else "none")
    return f"group={group_key}|window={window_key}"


# ---------------------------------------------------------------------------
# Q-table entry
# ---------------------------------------------------------------------------

@dataclass
class QEntry:
    """Running sum and observation count for one (state, action) pair."""
    value_sum: float = 0.0
    n_obs: int = 0
    skeletons: set[str] = field(default_factory=set)
    contexts: set[str] = field(default_factory=set)

    def update(
        self,
        reward: float,
        *,
        skeleton: str | None = None,
        context: str | None = None,
    ) -> None:
        self.value_sum += reward
        self.n_obs += 1
        if skeleton:
            self.skeletons.add(str(skeleton))
        if context:
            self.contexts.add(str(context))

    @property
    def mean_q(self) -> float:
        return self.value_sum / self.n_obs if self.n_obs > 0 else 0.0

    def should_prune(
        self,
        min_obs: int = MIN_OBS_L1,
        dead_threshold: float = DEAD_THRESHOLD,
        min_skeletons: int = MIN_PRUNE_SKELETONS,
        min_contexts: int = 1,
    ) -> bool:
        if self.n_obs < min_obs or self.mean_q >= dead_threshold:
            return False
        if len(self.skeletons) < min_skeletons:
            return False
        if len(self.contexts) < min_contexts:
            return False
        return True


# ---------------------------------------------------------------------------
# Q-table
# ---------------------------------------------------------------------------

class QTable:
    """Two-level Q-table tracking (cluster, op) and (field, op) rewards.

    level1[(cluster_id, specific_op)] → QEntry  pruned after MIN_OBS_L1 obs
    level2[(field_id,   specific_op)] → QEntry  pruned after MIN_OBS_L2 obs
    """

    def __init__(
        self,
        min_obs_l1: int = MIN_OBS_L1,
        min_obs_l2: int = MIN_OBS_L2,
        dead_threshold: float = DEAD_THRESHOLD,
        min_prune_skeletons: int = MIN_PRUNE_SKELETONS,
        min_prune_contexts_l1: int = MIN_PRUNE_CONTEXTS_L1,
    ) -> None:
        self.level1: dict[tuple[str, str], QEntry] = {}
        self.level2: dict[tuple[str, str], QEntry] = {}
        self.min_obs_l1 = min_obs_l1
        self.min_obs_l2 = min_obs_l2
        self.dead_threshold = dead_threshold
        self.min_prune_skeletons = min_prune_skeletons
        self.min_prune_contexts_l1 = min_prune_contexts_l1

    def update(
        self,
        cluster_id: str,
        field_id: str,
        specific_op: str | None,
        reward: float,
        *,
        skeleton: str | None = None,
        group: str | None = None,
        window: int | str | None = None,
    ) -> None:
        """Update both levels with the same reward."""
        if specific_op is None:
            return
        context = _q_context(group=group, window=window)
        k2 = (field_id, specific_op)
        if k2 not in self.level2:
            self.level2[k2] = QEntry()
        self.level2[k2].update(reward, skeleton=skeleton, context=context)
        if not cluster_id:  # skip empty cluster_id — old records without metadata
            return
        k1 = (cluster_id, specific_op)
        if k1 not in self.level1:
            self.level1[k1] = QEntry()
        self.level1[k1].update(reward, skeleton=skeleton, context=context)

    def pruned_level1(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            k for k, e in self.level1.items()
            if e.should_prune(
                self.min_obs_l1,
                self.dead_threshold,
                self.min_prune_skeletons,
                self.min_prune_contexts_l1,
            )
        )

    def pruned_level2(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            k for k, e in self.level2.items()
            if e.should_prune(
                self.min_obs_l2,
                self.dead_threshold,
                self.min_prune_skeletons,
                1,
            )
        )

    def save(self, path: str | Path) -> None:
        data = {
            "min_obs_l1": self.min_obs_l1,
            "min_obs_l2": self.min_obs_l2,
            "dead_threshold": self.dead_threshold,
            "min_prune_skeletons": self.min_prune_skeletons,
            "min_prune_contexts_l1": self.min_prune_contexts_l1,
            "level1": {
                f"{k[0]}|||{k[1]}": {
                    "value_sum": e.value_sum,
                    "n_obs": e.n_obs,
                    "skeletons": sorted(e.skeletons),
                    "contexts": sorted(e.contexts),
                }
                for k, e in self.level1.items()
            },
            "level2": {
                f"{k[0]}|||{k[1]}": {
                    "value_sum": e.value_sum,
                    "n_obs": e.n_obs,
                    "skeletons": sorted(e.skeletons),
                    "contexts": sorted(e.contexts),
                }
                for k, e in self.level2.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "QTable":
        with open(path) as f:
            data = json.load(f)
        qt = cls(
            min_obs_l1=int(data.get("min_obs_l1", MIN_OBS_L1)),
            min_obs_l2=int(data.get("min_obs_l2", MIN_OBS_L2)),
            dead_threshold=float(data.get("dead_threshold", DEAD_THRESHOLD)),
            min_prune_skeletons=int(data.get("min_prune_skeletons", MIN_PRUNE_SKELETONS)),
            min_prune_contexts_l1=int(data.get("min_prune_contexts_l1", MIN_PRUNE_CONTEXTS_L1)),
        )
        for raw_key, vals in data.get("level1", {}).items():
            parts = raw_key.split("|||", 1)
            if len(parts) == 2:
                k = (parts[0], parts[1])
                qt.level1[k] = QEntry(
                    value_sum=float(vals["value_sum"]),
                    n_obs=int(vals["n_obs"]),
                    skeletons={str(v) for v in vals.get("skeletons", []) if v},
                    contexts={str(v) for v in vals.get("contexts", []) if v},
                )
        for raw_key, vals in data.get("level2", {}).items():
            parts = raw_key.split("|||", 1)
            if len(parts) == 2:
                k = (parts[0], parts[1])
                qt.level2[k] = QEntry(
                    value_sum=float(vals["value_sum"]),
                    n_obs=int(vals["n_obs"]),
                    skeletons={str(v) for v in vals.get("skeletons", []) if v},
                    contexts={str(v) for v in vals.get("contexts", []) if v},
                )
        return qt

    def summary(self, top_n: int = 20, bottom_n: int = 0) -> str:
        """Human-readable summary of Q-table entries.

        top_n:    show top-N entries by mean_q (highest first)
        bottom_n: show bottom-N entries by mean_q (lowest first, with n_obs)
                  useful for identifying pruning candidates before threshold is reached
        """
        lines: list[str] = []
        pruned1 = self.pruned_level1()
        pruned2 = self.pruned_level2()

        def _fmt(k: tuple[str, str], e: QEntry, pruned: frozenset) -> str:
            flag = " [PRUNED]" if k in pruned else ""
            return (
                f"  {k[0][:40]:<40}  {k[1]:<18}  "
                f"mean_q={e.mean_q:.3f}  n={e.n_obs}  "
                f"skel={len(e.skeletons)} ctx={len(e.contexts)}{flag}"
            )

        l1_sorted = sorted(self.level1.items(), key=lambda kv: kv[1].mean_q, reverse=True)
        l2_sorted = sorted(self.level2.items(), key=lambda kv: kv[1].mean_q, reverse=True)

        lines.append("=== Q-Table Level 1 (cluster × op) — TOP ===")
        for k, e in l1_sorted[:top_n]:
            lines.append(_fmt(k, e, pruned1))

        if bottom_n:
            lines.append(f"\n=== Q-Table Level 1 (cluster × op) — BOTTOM {bottom_n} ===")
            for k, e in l1_sorted[-bottom_n:]:
                lines.append(_fmt(k, e, pruned1))

        lines.append("\n=== Q-Table Level 2 (field × op) — TOP ===")
        for k, e in l2_sorted[:top_n]:
            lines.append(_fmt(k, e, pruned2))

        if bottom_n:
            lines.append(f"\n=== Q-Table Level 2 (field × op) — BOTTOM {bottom_n} ===")
            for k, e in l2_sorted[-bottom_n:]:
                lines.append(_fmt(k, e, pruned2))

        n_pruned1 = len(pruned1)
        n_pruned2 = len(pruned2)
        lines.append(
            f"\nLevel 1: {len(self.level1)} entries, {n_pruned1} pruned  |  "
            f"Level 2: {len(self.level2)} entries, {n_pruned2} pruned"
        )
        return "\n".join(lines)

    def exploration_summary(
        self,
        clusters: "list | None" = None,
        saturation_threshold: int = 15,
        max_unexplored: int = 12,
    ) -> str:
        """Exploration-oriented summary for LLM prompt.

        Unlike summary() (which shows top rewards / exploitation), this shows:
          1. Dead ends: pruned (cluster, op) combos grouped by operator — avoid entirely
          2. Saturated: high n_obs combos — already well-explored, move on
          3. Unexplored clusters: cluster_ids with zero Level-1 observations — prioritise these

        clusters: optional list of FieldCluster (or objects with .cluster_id attr) to
                  identify which clusters have never been observed yet.
        saturation_threshold: n_obs >= this → show as saturated.
        max_unexplored: cap on number of unexplored cluster_ids to show.
        """
        from collections import defaultdict

        pruned1 = self.pruned_level1()
        lines: list[str] = []

        # --- 1. Dead ends: pruned combos by operator ---
        pruned_by_op: dict[str, list[str]] = defaultdict(list)
        for k in pruned1:
            pruned_by_op[k[1]].append(k[0])

        if pruned_by_op:
            lines.append("=== DEAD ENDS (pruned — do not revisit) ===")
            for op in sorted(pruned_by_op, key=lambda o: -len(pruned_by_op[o])):
                n = len(pruned_by_op[op])
                lines.append(f"  {op:<25}  {n} cluster(s) pruned")
        else:
            lines.append("=== DEAD ENDS ===  (none yet)")

        # --- 2. Saturated combos ---
        saturated = [
            (k, e) for k, e in self.level1.items()
            if e.n_obs >= saturation_threshold and k not in pruned1
        ]
        saturated.sort(key=lambda kv: -kv[1].n_obs)

        if saturated:
            lines.append(f"\n=== SATURATED (n_obs >= {saturation_threshold}, move on) ===")
            for k, e in saturated[:20]:
                lines.append(
                    f"  {k[0][:38]:<38}  {k[1]:<18}  n={e.n_obs}  mean_q={e.mean_q:.3f}"
                    f"  skel={len(e.skeletons)} ctx={len(e.contexts)}"
                )
        else:
            lines.append(f"\n=== SATURATED (n_obs >= {saturation_threshold}) ===  (none yet)")

        # --- 3. Unexplored clusters ---
        if clusters is not None:
            observed_clusters = {k[0] for k in self.level1}
            unexplored = [
                c for c in clusters
                if getattr(c, "cluster_id", None) not in observed_clusters
            ]
            if unexplored:
                lines.append(f"\n=== UNEXPLORED CLUSTERS (0 obs — prioritise) ===")
                for c in unexplored[:max_unexplored]:
                    cid = getattr(c, "cluster_id", str(c))
                    lines.append(f"  {cid}")
                remainder = len(unexplored) - max_unexplored
                if remainder > 0:
                    lines.append(f"  ... and {remainder} more unexplored clusters")
            else:
                lines.append("\n=== UNEXPLORED CLUSTERS ===  (all clusters have been visited)")

        n_pruned1 = len(pruned1)
        lines.append(
            f"\nTotal Level-1 entries: {len(self.level1)}  pruned: {n_pruned1}  "
            f"saturated: {len(saturated)}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tested registry
# ---------------------------------------------------------------------------

class TestedRegistry:
    """Tracks (field_id, specific_op) pairs already submitted to Brain.

    Once a combination is registered it is never re-submitted, regardless
    of the Q-table reward.  Prevents re-asking the same question in a
    different expression shape.
    """

    def __init__(self) -> None:
        self._tested: set[tuple[str, str]] = set()

    def add(self, field_id: str, specific_op: str | None) -> None:
        if specific_op is not None:
            self._tested.add((field_id, specific_op))

    def contains(self, field_id: str, specific_op: str | None) -> bool:
        if specific_op is None:
            return False
        return (field_id, specific_op) in self._tested

    def __len__(self) -> int:
        return len(self._tested)

    def save(self, path: str | Path) -> None:
        data = {"tested": [list(t) for t in self._tested]}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "TestedRegistry":
        tr = cls()
        with open(path) as f:
            data = json.load(f)
        for pair in data.get("tested", []):
            if len(pair) == 2:
                tr._tested.add((pair[0], pair[1]))
        return tr


# ---------------------------------------------------------------------------
# Skeleton counter (novelty layer)
# ---------------------------------------------------------------------------

class SkeletonCounter:
    """Tracks how many times each operator skeleton has been submitted.

    A skeleton is the pre-order operator sequence of an expression tree with
    all literals (field IDs, window values, group names) wildcarded, joined
    by '|'.  For example::

        group_rank(ts_mean(pe_ratio, 20), "sector") → "group_rank|ts_mean"

    novelty_score = 1 / (n_obs + 1): unseen skeletons score 1.0, heavily
    tested skeletons approach 0.  Used to bias candidate selection toward
    structurally novel expressions.  No reward tracking — quality is handled
    entirely by the Q-table.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def increment(self, skeleton: str) -> None:
        self._counts[skeleton] = self._counts.get(skeleton, 0) + 1

    def novelty_score(self, skeleton: str) -> float:
        return 1.0 / (self._counts.get(skeleton, 0) + 1)

    def count(self, skeleton: str) -> int:
        return self._counts.get(skeleton, 0)

    def __len__(self) -> int:
        return len(self._counts)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({"counts": self._counts}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SkeletonCounter":
        sc = cls()
        with open(path) as f:
            data = json.load(f)
        for skeleton, n in data.get("counts", {}).items():
            sc._counts[str(skeleton)] = int(n)
        return sc

    def summary(self, top_n: int = 20) -> str:
        """Show the most-seen skeletons (highest n_obs first)."""
        lines = ["=== SkeletonCounter — most tested structural templates ==="]
        sorted_entries = sorted(self._counts.items(), key=lambda kv: kv[1], reverse=True)
        for skeleton, n in sorted_entries[:top_n]:
            lines.append(f"  n={n:<5}  {skeleton}")
        lines.append(f"\nTotal unique skeletons tracked: {len(self._counts)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bootstrap from existing sim log
# ---------------------------------------------------------------------------

def bootstrap_qtable(
    sim_log_path: str | Path | list[str | Path],
    qtable: QTable,
    *,
    skeleton_counter: "SkeletonCounter | None" = None,
    lanes: tuple[str, ...] | None = None,
    portfolio_scores: "dict[str, dict] | None" = None,
    dedupe: bool = False,
) -> int:
    """Replay existing sim.jsonl records into the Q-table.

    sim_log_path: single path or list of paths (e.g. include workshop_sim.jsonl too).
    lanes: filter by search_lane if provided; pass None to include all lanes.
    portfolio_scores: optional dict mapping expression → admission metrics dict
        (keys: brain_score_delta or fallback delta_pool_sharpe, decision, ...).
        When provided, any expression with non-positive contribution delta has
        its reward hard-zeroed regardless of individual alpha quality.
    dedupe: when True, replay each alpha/candidate/expression-regime key only once
        across all provided logs. This is useful when rebuilding from merged logs,
        archives, and scored sidecars.

    Reads all completed (status=="done") records, extracts
    (cluster_id, field_id, specific_op) from each expression, computes the
    Q-reward (with correlation term skipped if unavailable in old records),
    and updates both Q-table levels.

    Returns the number of records processed.
    """
    from .feedback import SimResult  # local import to avoid circular

    paths: list[Path]
    if isinstance(sim_log_path, (str, Path)):
        paths = [Path(sim_log_path)]
    else:
        paths = [Path(p) for p in sim_log_path]

    count = 0
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") != "done":
                    continue
                if rec.get("skip_qtable_update") or rec.get("search_lane") == "regime_check":
                    continue
                if is_degenerate_sim_record(rec):
                    continue
                if lanes is not None and rec.get("search_lane") not in lanes:
                    continue

                expression = rec.get("expression") or ""
                field_ids = rec.get("fields") or []
                cluster_ids = rec.get("cluster_ids") or []
                if not field_ids and expression:
                    # api_backfill / submitted records carry an expression but no
                    # parsed fields — derive them so they feed the Q-table too.
                    field_ids = derive_fields(expression)
                if not field_ids or not expression:
                    continue
                if dedupe:
                    key = _bootstrap_dedupe_key(rec, expression)
                    if key in seen:
                        continue
                    seen.add(key)

                primary_field = str(field_ids[0])
                primary_cluster = str(cluster_ids[0]) if cluster_ids else ""
                specific_op = extract_primary_op(expression, primary_field)

                result = SimResult(
                    candidate_id=rec.get("candidate_id", ""),
                    expression=expression,
                    cluster_ids=tuple(str(c) for c in cluster_ids),
                    is_sharpe=float(rec.get("is_sharpe") or 0.0),
                    is_fitness=float(rec.get("is_fitness") or 0.0),
                    passed=bool((rec.get("grade") or "").upper() == "PASS"),
                    is_turnover=float(rec["is_turnover"]) if rec.get("is_turnover") is not None else None,
                    concentrated_weight_ok=bool(rec.get("is_concentrated_weight_ok", True)),
                    sub_universe_sharpe_ok=bool(rec.get("is_sub_universe_sharpe_ok", True)),
                    max_self_correlation=float(rec["max_self_correlation"]) if rec.get("max_self_correlation") is not None else None,
                    max_self_correlation_alpha_id=str(rec["max_self_correlation_alpha_id"]) if rec.get("max_self_correlation_alpha_id") is not None else None,
                    max_self_correlation_alpha_sharpe=float(rec["max_self_correlation_alpha_sharpe"]) if rec.get("max_self_correlation_alpha_sharpe") is not None else None,
                    max_self_correlation_alpha_fitness=float(rec["max_self_correlation_alpha_fitness"]) if rec.get("max_self_correlation_alpha_fitness") is not None else None,
                    max_concentration=float(rec["max_concentration"]) if rec.get("max_concentration") is not None else None,
                    yearly_stats=rec.get("yearly_stats") or None,
                )

                reward = compute_q_reward(result)
                ps = None
                if portfolio_scores:
                    ps = portfolio_scores.get(expression)
                reward = apply_pool_delta_reward_veto(rec, reward, ps)
                template_id = rec.get("template_id") or ""
                qtable.update(
                    primary_cluster,
                    primary_field,
                    specific_op,
                    reward,
                    skeleton=template_id,
                    group=str(rec.get("group") or ""),
                    window=_record_window(rec),
                )

                if skeleton_counter is not None:
                    if template_id:
                        skeleton_counter.increment(template_id)

                count += 1

    return count


def _bootstrap_dedupe_key(rec: dict, expression: str) -> tuple[str, str]:
    alpha_id = rec.get("alpha_id")
    if alpha_id:
        return ("alpha_id", str(alpha_id))
    candidate_id = rec.get("candidate_id")
    if candidate_id:
        return ("candidate_id", str(candidate_id))
    settings = rec.get("settings") or rec.get("settings_override") or {}
    regime = "|".join(
        str(settings.get(key, ""))
        for key in ("delay", "neutralization", "universe", "decay", "truncation")
    )
    return ("expression", f"{expression}\t{regime}\t{rec.get('search_lane') or ''}")


# ---------------------------------------------------------------------------
# Incremental on-disk Q-table update
# ---------------------------------------------------------------------------

_GRAMMAR_ARTIFACTS = Path(__file__).resolve().parents[1] / "alpha_discovery" / "grammar_method" / "artifacts"
DEFAULT_QTABLE_PATH = _GRAMMAR_ARTIFACTS / "all_qtable.json"


def update_qtable_from_records(
    records: list[dict],
    qtable_path: "str | Path | None" = None,
) -> int:
    """Incrementally update the on-disk Q-table from a list of sim records.

    Loads the existing qtable (or creates empty), updates it with the new
    records, and saves back.  Skips records with no fields or degenerate signal.

    Used by the workshop and the template sweep pipeline so that every simulation
    — not just grammar pipeline runs — feeds back into the Q-table.

    Returns the number of records that contributed to the update.
    """
    from .feedback import SimResult  # local import to avoid circular

    qtable_path = Path(qtable_path) if qtable_path else DEFAULT_QTABLE_PATH

    qtable = QTable.load(qtable_path) if qtable_path.exists() else QTable()

    count = 0
    for rec in records:
        if rec.get("status") != "done":
            continue
        if rec.get("skip_qtable_update") or rec.get("search_lane") == "regime_check":
            continue
        if is_degenerate_sim_record(rec):
            continue
        expression = rec.get("expression") or ""
        field_ids = rec.get("fields") or []
        cluster_ids = rec.get("cluster_ids") or []
        if not field_ids and expression:
            field_ids = derive_fields(expression)
        if not field_ids or not expression:
            continue

        primary_field = str(field_ids[0])
        primary_cluster = str(cluster_ids[0]) if cluster_ids else ""
        specific_op = extract_primary_op(expression, primary_field)

        result = SimResult(
            candidate_id=rec.get("candidate_id", ""),
            expression=expression,
            cluster_ids=tuple(str(c) for c in cluster_ids),
            is_sharpe=float(rec.get("is_sharpe") or 0.0),
            is_fitness=float(rec.get("is_fitness") or 0.0),
            passed=bool((rec.get("grade") or "").upper() == "PASS"),
            is_turnover=float(rec["is_turnover"]) if rec.get("is_turnover") is not None else None,
            concentrated_weight_ok=bool(rec.get("is_concentrated_weight_ok", True)),
            sub_universe_sharpe_ok=bool(rec.get("is_sub_universe_sharpe_ok", True)),
            max_self_correlation=float(rec["max_self_correlation"]) if rec.get("max_self_correlation") is not None else None,
            max_self_correlation_alpha_id=str(rec["max_self_correlation_alpha_id"]) if rec.get("max_self_correlation_alpha_id") is not None else None,
            max_self_correlation_alpha_sharpe=float(rec["max_self_correlation_alpha_sharpe"]) if rec.get("max_self_correlation_alpha_sharpe") is not None else None,
            max_self_correlation_alpha_fitness=float(rec["max_self_correlation_alpha_fitness"]) if rec.get("max_self_correlation_alpha_fitness") is not None else None,
            max_concentration=float(rec["max_concentration"]) if rec.get("max_concentration") is not None else None,
            yearly_stats=rec.get("yearly_stats") or None,
        )

        reward = compute_q_reward(result)
        reward = apply_pool_delta_reward_veto(rec, reward)
        qtable.update(
            primary_cluster,
            primary_field,
            specific_op,
            reward,
            skeleton=rec.get("template_id") or "",
            group=str(rec.get("group") or ""),
            window=_record_window(rec),
        )
        count += 1

    if count:
        qtable_path.parent.mkdir(parents=True, exist_ok=True)
        qtable.save(qtable_path)

    return count


def _record_window(record: dict) -> int | str | None:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    for key in ("window", "window1", "window2", "days", "lookback"):
        value = params.get(key)
        if value not in (None, ""):
            return value
    return None


__all__ = [
    "SPECIFIC_OPS", "MIN_OBS_L1", "MIN_OBS_L2", "DEAD_THRESHOLD",
    "extract_primary_op", "compute_q_reward", "apply_pool_delta_reward_veto", "is_degenerate_sim_record",
    "QEntry", "QTable", "SkeletonCounter",
    "bootstrap_qtable", "update_qtable_from_records",
    "DEFAULT_QTABLE_PATH",
]
