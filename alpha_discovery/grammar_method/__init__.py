"""alpha_discovery/grammar_method — Grammar-guided alpha discovery with Q-table pruning.

Typical usage::

    from alpha_discovery/grammar_method import GrammarPlanner, GrammarConfig
    from core.feedback import SimResult
    from core.qtable import QTable, bootstrap_qtable

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

from core.feedback import SimResult, compute_quality_weight
from .grammar import GrammarConfig
from core.expr.operators import OPERATOR_REGISTRY, OperatorSpec
from .planner import GrammarPlanner
from core.qtable import QTable, bootstrap_qtable
from .novelty import TestedFingerprintRegistry, bootstrap_tested_fingerprints

__all__ = [
    "GrammarConfig", "GrammarPlanner",
    "SimResult", "compute_quality_weight",
    "QTable", "bootstrap_qtable",
    "TestedFingerprintRegistry", "bootstrap_tested_fingerprints",
    "OperatorSpec", "OPERATOR_REGISTRY",
]
