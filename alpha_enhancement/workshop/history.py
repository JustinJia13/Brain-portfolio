"""WorkshopHistory — shared trial history across all workshop lanes and iterations.

Tracks every proposed and simulated alpha, keyed by its root seed expression.
Resolves multi-hop lineage chains so that A → (negate) → B → (enhance) → C
all trace back to root A, and the LLM always sees the full gradient of what
has been tried on that signal regardless of which lane generated it.

Used by:
  - alpha_enhancement/workshop/core.py   generate_candidates()
  - tools/batch_workshop.py    _run_batch(), _workshop_one()
  - tools/alpha_workshop.py    main loop
  - alpha_enhancement.workshop.targeted targeted explicit-alpha runs
"""

from __future__ import annotations

from collections import defaultdict


def _regime_key(settings: dict | None) -> str:
    from core.brain.submit import _regime_key as _canonical_regime_key

    return _canonical_regime_key(settings or {})


def _record_regime(record: dict) -> str:
    settings = record.get("settings")
    if not isinstance(settings, dict):
        settings = record.get("settings_override")
    if isinstance(settings, dict):
        return _regime_key(settings)
    return str(record.get("regime") or _regime_key(None))


def _record_key(record: dict, expr: str | None = None) -> str:
    expression = expr if expr is not None else record.get("expression", "")
    return f"{_record_regime(record)}\t{expression}"

# Fields extracted from each sim record for the history store.
# These are exactly the fields that repair_agent.py reads from
# trial_history and context_records entries.
_KEEP_FIELDS = frozenset({
    "expression", "is_sharpe", "is_fitness", "is_turnover",
    "is_returns", "is_drawdown", "oos_sharpe_2023", "oos_fitness_2023",
    "grade", "search_lane", "checks", "status", "fields", "cluster_ids",
    "dataset_id", "params", "admission_brain_score_delta",
    "admission_delta_pool_sharpe", "admission_global_max_corr",
    "max_self_correlation", "alpha_id", "candidate_id",
})

# How many history entries to inject per LLM call (repair trial_history cap
# is 10 in repair_agent.py; context_records cap is 8).
HISTORY_CAP_REPAIR = 10
HISTORY_CAP_ENHANCE = 8


def _slim(record: dict) -> dict:
    """Return a trimmed copy of a sim record with only the fields agents use."""
    return {k: record[k] for k in _KEEP_FIELDS if k in record}


class WorkshopHistory:
    """Per-seed trial history for the full workshop pipeline.

    Attributes
    ----------
    _seeds : set[str]
        Root seed keys registered at construction time.
    _history : dict[str, list[dict]]
        root_key → list of slim sim records, sorted best→worst sharpe.
    _lineage : dict[str, str]
        any_key → root_key, for multi-hop chain resolution.
    """

    def __init__(self, seed_exprs: list[str] | None = None) -> None:
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._lineage: dict[str, str] = {}
        self._seeds: set[str] = set()
        for expr in (seed_exprs or []):
            self.register_seed(expr)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_seed(self, expr_key: str) -> None:
        """Declare expr_key as a root seed (creates an empty history bucket)."""
        self._seeds.add(expr_key)
        if expr_key not in self._history:
            self._history[expr_key] = []
        self._lineage[expr_key] = expr_key

    def register_lineage(self, child_key: str, parent_key: str) -> None:
        """Record that child_key was derived from parent_key.

        Resolves transitively: if parent is itself a child, the root of
        parent becomes the root of child.
        """
        root = self._lineage.get(parent_key, parent_key)
        self._lineage[child_key] = root
        # Ensure a bucket exists for the root
        if root not in self._history:
            self._history[root] = []

    def root_for(self, expr_key: str, fallback_original: str | None = None) -> str:
        """Return the root seed key for any expression key.

        Resolution order:
          1. Direct lineage table hit.
          2. fallback_original → resolve that, then cache for expr_key.
          3. expr_key itself (treat as its own root if unknown).
        """
        if expr_key in self._lineage:
            return self._lineage[expr_key]
        if fallback_original and fallback_original in self._lineage:
            root = self._lineage[fallback_original]
            self._lineage[expr_key] = root
            return root
        if fallback_original:
            self._lineage[expr_key] = fallback_original
            return fallback_original
        return expr_key

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, new_records: list[dict], iteration: int = 0) -> None:
        """Absorb simulation results from one batch into the history.

        Records whose root cannot be resolved are skipped silently (they
        are from a different pipeline run unrelated to the current seeds).
        """
        for r in new_records:
            if r.get("status") != "done":
                continue
            expr_key = _record_key(r)
            orig_expr = (r.get("params") or {}).get("original_expression") or r.get("expression", "")
            orig_key = _record_key(r, orig_expr)
            root = self.root_for(expr_key, orig_key)

            # Skip records that don't trace to any known seed bucket
            if root not in self._history:
                continue

            entry = _slim(r)
            entry["_iteration"] = iteration
            self._history[root].append(entry)
            # Keep sorted best → worst sharpe
            self._history[root].sort(
                key=lambda e: e.get("is_sharpe") or 0.0, reverse=True
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def context_for_enhance(self, root_key: str) -> list[dict]:
        """Return prior attempts for the enhance lane (injected as context_records).

        Sorted best → worst sharpe, capped at HISTORY_CAP_ENHANCE.
        """
        return list(self._history.get(root_key, []))[:HISTORY_CAP_ENHANCE]

    def context_for_repair(self, root_key: str) -> list[dict]:
        """Return prior attempts for the repair lane (injected as trial_history).

        Sorted best → worst sharpe, capped at HISTORY_CAP_REPAIR.
        """
        return list(self._history.get(root_key, []))[:HISTORY_CAP_REPAIR]

    def context_for_root(self, root_key: str) -> list[dict]:
        """Return full prior attempts for a root without lane-specific caps."""
        return list(self._history.get(root_key, []))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, root_key: str) -> str:
        entries = self._history.get(root_key, [])
        n_pass = sum(
            1 for e in entries
            if (e.get("is_sharpe") or 0) >= 1.25
            and (e.get("is_fitness") or 0) >= 1.0
        )
        lanes: dict[str, int] = defaultdict(int)
        for e in entries:
            lanes[e.get("search_lane") or "?"] += 1
        lane_str = "  ".join(f"{l}={c}" for l, c in sorted(lanes.items()))
        return f"{len(entries)} tried ({lane_str}), {n_pass} passing"

    def print_summary(self) -> None:
        print(f"\nWorkshopHistory ({len(self._seeds)} seeds):")
        for seed in sorted(self._seeds):
            print(f"  {seed[:80]}")
            print(f"    {self.summary(seed)}")
