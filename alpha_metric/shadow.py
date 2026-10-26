"""shadow.py — log-only shadow gate for run_pipeline (non-invasive rollout).

Shadow mode runs `should_simulate` ALONGSIDE the existing Q-table pruning and
records what the new gate WOULD have decided, without changing which candidates
actually get simulated.  After a run, join the shadow log with realised sim
outcomes (by candidate_id / expression) to measure the live saved-vs-killed
frontier before cutting over.

Hard rule: shadow mode must NEVER break the live pipeline.  Every entry point is
wrapped so any failure (missing model, import error, bad expr) degrades to a
logged warning and the pipeline proceeds untouched.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_SHADOW_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "gate_shadow.jsonl",
)


class ShadowGate:
    """Lazy, fail-safe wrapper around AlphaMetric for log-only shadow runs."""

    def __init__(self, log_path: str = _SHADOW_LOG, qtable=None, beta=None):
        self.log_path = log_path
        self._am = None
        self._ok = None  # None=untried, True/False=loaded/failed
        self._qtable = qtable  # optional incumbent, for dual-shadow paired logging
        self._beta = beta      # optional quality-UCB override (else deployed default 2.5)

    def _qtable_skip(self, cand: dict) -> bool | None:
        """Incumbent q-table keep/skip for a candidate (for the live paired analysis)."""
        from alpha_grammar.qtable import extract_primary_op
        fields = cand.get("fields") or []
        if not fields:
            return False
        field = str(fields[0])
        op = extract_primary_op(cand.get("expression", ""), field)
        if not op:
            return False
        clusters = cand.get("cluster_ids") or []
        cluster = str(clusters[0]) if clusters else ""
        l1 = self._qtable.level1.get((cluster, op)) if hasattr(self._qtable, "level1") else None
        l2 = self._qtable.level2.get((field, op)) if hasattr(self._qtable, "level2") else None
        return bool((l2 and l2.should_prune()) or (l1 and l1.should_prune()))

    def _ensure(self) -> bool:
        if self._ok is not None:
            return self._ok
        try:
            from .metric import AlphaMetric
            from .gate import GateParams
            params = GateParams(beta=self._beta) if self._beta is not None else None
            self._am = AlphaMetric.load(params=params)
            self._ok = True
            if self._beta is not None:
                print(f"[shadow-gate] quality-UCB beta override = {self._beta}")
        except Exception as exc:  # never propagate into the live pipeline
            print(f"[shadow-gate] disabled (load failed: {exc})")
            self._ok = False
        return self._ok

    def log_round(self, candidates: list[dict[str, Any]], round_num: int,
                  mode: str = "shadow") -> dict:
        """Decide on each candidate, append to the shadow log, and (if mode != shadow)
        return the survivors the pipeline should actually simulate.

        mode: shadow=log only; reliable=live dedup+turnover kills; full=quality too.
        """
        if not self._ensure():
            return {"enabled": False, "survivors": candidates}
        from .cutover import should_live_skip
        n = sim = kill = 0
        reasons: dict[str, int] = {}
        survivors: list[dict[str, Any]] = []
        live_skipped = 0
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a") as fh:
                for c in candidates:
                    expr = c.get("expression")
                    if not isinstance(expr, str):
                        survivors.append(c)  # fail-safe: keep for simulation
                        continue
                    try:
                        d = self._am.decide(expr)
                    except Exception:
                        survivors.append(c)  # fail-safe: simulate on any gate error
                        continue
                    n += 1
                    sim += int(d.simulate)
                    kill += int(not d.simulate)
                    reasons[d.reason] = reasons.get(d.reason, 0) + 1
                    dd = d.to_dict()
                    if mode != "shadow" and should_live_skip(dd, mode):
                        live_skipped += 1   # actually drop from this round's sims
                    else:
                        survivors.append(c)
                    row = {
                        "ts": time.time(),
                        "round": round_num,
                        "candidate_id": c.get("candidate_id"),
                        "expression": expr,
                        "decision": dd,
                    }
                    # Task 4: also log the incumbent q-table's decision when a
                    # qtable is supplied, so the live paired (champion vs challenger)
                    # analysis can be re-run on fresh, unselected labels.
                    if self._qtable is not None:
                        try:
                            row["qtable_skip"] = self._qtable_skip(c)
                        except Exception:
                            row["qtable_skip"] = None
                    fh.write(json.dumps(row) + "\n")
        except Exception as exc:
            print(f"[shadow-gate] log error (ignored): {exc}")
            return {"enabled": True, "error": str(exc), "survivors": candidates}
        summary = {"enabled": True, "mode": mode, "n": n, "would_simulate": sim,
                   "would_kill": kill, "reasons": reasons,
                   "live_skipped": live_skipped, "survivors": survivors}
        if n:
            extra = f"  LIVE-skipped {live_skipped} (mode={mode})" if mode != "shadow" else ""
            print(f"[shadow-gate] round {round_num}: of {n} candidates, "
                  f"would simulate {sim} / kill {kill} "
                  f"(slots saved {kill / n:.0%}); reasons={reasons}{extra}")
        return summary


_SINGLETON: ShadowGate | None = None


def get_shadow_gate(qtable=None, beta=None) -> ShadowGate:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ShadowGate(qtable=qtable, beta=beta)
    elif qtable is not None and _SINGLETON._qtable is None:
        _SINGLETON._qtable = qtable
    return _SINGLETON
