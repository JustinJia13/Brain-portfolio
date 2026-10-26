"""alpha_grammar — Grammar-guided alpha discovery with Q-table pruning.

Typical usage::

    from alpha_grammar import GrammarPlanner, GrammarConfig
    from alpha_grammar.feedback import SimResult
    from alpha_grammar.qtable import QTable, bootstrap_qtable

    config = GrammarConfig(n_samples=300)
    qtable = QTable()

    # Bootstrap from existing sim log (zero new simulations):
    bootstrap_qtable("artifacts/sim.jsonl", qtable)

    planner = GrammarPlanner(config, qtable=qtable)

    for batch_num in range(n_batches):
        candidates = planner.build_plan(clusters, seed=batch_num)
        # Submit candidates to Brain and collect results ...

    # Persist state for next session:
    qtable.save("artifacts/qtable.json")
"""

from .feedback import SimResult, compute_quality_weight
from .grammar import GrammarConfig
from .operators import OPERATOR_REGISTRY, OperatorSpec
from .planner import GrammarPlanner
from .qtable import QTable, bootstrap_qtable
from .novelty import TestedFingerprintRegistry, bootstrap_tested_fingerprints

__all__ = [
    "GrammarConfig", "GrammarPlanner",
    "SimResult", "compute_quality_weight",
    "QTable", "bootstrap_qtable",
    "TestedFingerprintRegistry", "bootstrap_tested_fingerprints",
    "OperatorSpec", "OPERATOR_REGISTRY",
]
