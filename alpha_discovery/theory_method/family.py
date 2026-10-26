"""Family-level verdicts and the mechanism Beta posterior.

The unit of evidence is the family, not the member: a hypothesis is VALIDATED
when a supermajority of its pre-registered variants point the declared way,
and the promoted members are the median-robustness ones — never the max-sharpe
spike, which is the multiple-testing artifact this lane exists to kill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.quality import is_submittable
from core.robustness import record_robustness_score

# Verdict thresholds (pre-registered; change only between batches, never after
# seeing a family's results)
SUPPORT_SHARPE_MIN = 0.5      # member supports the theory when oriented sharpe exceeds this
SUPPORT_RATE_VALIDATED = 0.60
SUPPORT_RATE_FALSIFIED = 0.30
MIN_MEMBERS_FOR_VERDICT = 5
EARLY_STOP_FIRST = 5          # simulate this many members first
PROMOTE_COUNT = 3             # median-robustness members promoted per validated family

# Retro (archaeology) cards are prior, not posterior: cap their pseudo-counts.
RETRO_PSEUDO_CAP = 3.0

VALIDATED = "VALIDATED"
FALSIFIED = "FALSIFIED"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class FamilyVerdict:
    theory_id: str
    verdict: str
    n_done: int
    n_support: int
    support_rate: float
    median_robustness: float | None
    promoted: list[dict] = field(default_factory=list)

    def submittable_promoted(self) -> list[dict]:
        """Promoted members that are Brain-submittable (eligible for brain re-sim)."""
        return [r for r in self.promoted if is_submittable(r)]

    def as_dict(self) -> dict:
        return {
            "theory_id": self.theory_id,
            "verdict": self.verdict,
            "n_done": self.n_done,
            "n_support": self.n_support,
            "support_rate": round(self.support_rate, 4),
            "median_robustness": self.median_robustness,
            "promoted_expressions": [r.get("expression") for r in self.promoted],
            "submittable_promoted": [r.get("expression") for r in self.submittable_promoted()],
        }


def member_supports(record: dict, direction: str) -> bool:
    sharpe = record.get("is_sharpe")
    if sharpe is None:
        return False
    oriented = -float(sharpe) if direction == "-" else float(sharpe)
    if oriented <= SUPPORT_SHARPE_MIN:
        return False
    # When yearly data exists, an inconsistent profile does not support the
    # theory even if the headline sharpe does. Missing data does not block.
    sign_consistency = record.get("robustness_sign_consistency")
    if sign_consistency is not None and float(sign_consistency) < 0.5:
        return False
    return True


def evaluate_family(
    theory_id: str,
    records: list[dict],
    *,
    direction: str = "+",
) -> FamilyVerdict:
    done = [r for r in records if r.get("status") == "done" and r.get("is_sharpe") is not None]
    supporters = [r for r in done if member_supports(r, direction)]
    n_done, n_support = len(done), len(supporters)
    rate = n_support / n_done if n_done else 0.0

    if n_done < MIN_MEMBERS_FOR_VERDICT:
        verdict = INCONCLUSIVE
    elif rate >= SUPPORT_RATE_VALIDATED:
        verdict = VALIDATED
    elif rate <= SUPPORT_RATE_FALSIFIED:
        verdict = FALSIFIED
    else:
        verdict = INCONCLUSIVE

    promoted: list[dict] = []
    median_rob: float | None = None
    if verdict == VALIDATED and supporters:
        # Promotion follows the canonical quality ordering: prefer the
        # submittable supporters, and within them take the MEDIAN-robustness
        # members (never the max-sharpe spike). If none are submittable yet,
        # fall back to all supporters so the mechanism still seeds the workshop.
        submittable = [r for r in supporters if is_submittable(r)]
        pool = submittable or supporters
        pool = sorted(
            pool,
            key=lambda r: (record_robustness_score(r) if record_robustness_score(r) is not None else -1.0),
        )
        mid = len(pool) // 2
        half = PROMOTE_COUNT // 2
        lo = max(0, mid - half)
        promoted = pool[lo:lo + PROMOTE_COUNT]
        scores = [record_robustness_score(r) for r in pool if record_robustness_score(r) is not None]
        if scores:
            median_rob = scores[len(scores) // 2]

    return FamilyVerdict(
        theory_id=theory_id,
        verdict=verdict,
        n_done=n_done,
        n_support=n_support,
        support_rate=rate,
        median_robustness=median_rob,
        promoted=promoted,
    )


def early_stop_after_first(records: list[dict], direction: str = "+") -> bool:
    """True when the first-wave results already condemn the family (0 supporters)."""
    done = [r for r in records if r.get("status") == "done" and r.get("is_sharpe") is not None]
    if len(done) < EARLY_STOP_FIRST:
        return False
    return not any(member_supports(r, direction) for r in done)


def update_beta(card: dict, verdict: FamilyVerdict) -> dict:
    """Update the card's Beta posterior in place from a family outcome."""
    alpha = float(card.get("beta_alpha") or 1.0)
    beta = float(card.get("beta_beta") or 1.0)
    alpha += verdict.n_support
    beta += verdict.n_done - verdict.n_support
    if card.get("retro") and not card.get("live_confirmed"):
        # retro evidence is prior, not posterior: cap total pseudo-counts
        total = alpha + beta
        cap = 2.0 + RETRO_PSEUDO_CAP
        if total > cap:
            scale = cap / total
            alpha *= scale
            beta *= scale
    card["beta_alpha"] = round(alpha, 4)
    card["beta_beta"] = round(beta, 4)
    return card


# ---------------------------------------------------------------------------
# Master-as-cache: historical sims are free family evidence
# ---------------------------------------------------------------------------

def load_expression_cache(master_path: str | Path = "artifacts/sim/master.jsonl") -> dict[str, dict]:
    """{normalized_expression: best record} from master for cache lookups."""
    cache: dict[str, dict] = {}
    path = Path(master_path)
    if not path.exists():
        return cache
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "done" or rec.get("is_sharpe") is None:
                continue
            key = " ".join(str(rec.get("expression") or "").split())
            if not key:
                continue
            old = cache.get(key)
            if old is None or bool(rec.get("yearly_stats")) > bool(old.get("yearly_stats")):
                cache[key] = rec
    return cache


__all__ = [
    "FALSIFIED", "INCONCLUSIVE", "VALIDATED",
    "FamilyVerdict",
    "early_stop_after_first",
    "evaluate_family",
    "load_expression_cache",
    "member_supports",
    "update_beta",
]
