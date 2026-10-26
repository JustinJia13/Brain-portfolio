"""Theory-first hypothesis lane.

Mechanism-posterior inference loop: the LLM proposes pre-registered hypothesis
cards (prior), a constrained enumerator renders expression families from the
grammar vocabulary (designed experiment), Brain sims provide the likelihood,
and the mechanism registry + wiki store the posterior.

Modules:
- context:    field/cluster catalogs + proposer context assembly
- economist:  TheoryEconomistAgent (hypothesis cards via prompts/theory_propose)
- dedup:      embedding-only mechanism dedup with LLM adjudication
- enumerate_: card -> FASTEXPR family rendering (slot expansion, stratified cap)
- family:     Beta posterior, family verdicts, member promotion
- runner:     orchestration + Brain submission
"""

from alpha_discovery.theory_method.family import FamilyVerdict, evaluate_family
from alpha_discovery.theory_method.enumerate_ import enumerate_family

__all__ = ["FamilyVerdict", "evaluate_family", "enumerate_family"]
