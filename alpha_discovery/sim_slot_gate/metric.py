"""metric.py — top-level API wiring Object A + Object B into the sim-slot gate.

The kept/killed decision is the only deliverable; predictions are a throwaway
filter (never reported as alpha quality, never optimised against).

Typical use (shadow mode in run_pipeline):
    am = AlphaMetric.load()                 # loads psi + surrogate + builds index
    decision = am.decide(expr)              # GateDecision (simulate?, reason, ...)
    if decision.simulate: run_brain_sim(expr)

Also exposes the two Object-A primitives for inspection/tests:
    corr_model(X, Y)   structural signal-corr estimate, [-1,1]
    d_red(X, Y)        1 - |psi(corr_model)|   redundancy distance, [0,1]
"""

from __future__ import annotations

import os

from . import db_adapter
from .aggregate import CorrModel, corr_model as _corr_model_default
from .calibrate import Isotonic, load_params
from .features import default_featurizer
from .gate import Gate, GateParams, GateDecision
from .redundancy import RedundancyIndex, CoverageIndex, build_signatures
from .surrogate import Surrogate, MODEL_PATH

corr_model = _corr_model_default

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COVERAGE_CACHE = os.path.join(_HERE, "alpha_discovery/sim_slot_gate", "coverage_index.npz")


class AlphaMetric:
    def __init__(self, surrogate: Surrogate, model: CorrModel, psi: Isotonic,
                 redundancy: RedundancyIndex, coverage: CoverageIndex | None = None,
                 params: GateParams | None = None):
        self.surrogate = surrogate
        self.model = model
        self.psi = psi
        self.redundancy = redundancy
        self.coverage = coverage
        self.params = params or GateParams()
        self.featurizer = default_featurizer()
        self.gate = Gate(surrogate, redundancy, coverage, self.featurizer, self.params)

    # ---- factory --------------------------------------------------------- #
    @classmethod
    def load(cls, sim_index_size: int = 8000, params: GateParams | None = None,
             require_surrogate: bool = True, use_coverage: bool = True,
             coverage_cache: str = _COVERAGE_CACHE) -> "AlphaMetric":
        psi, sigma_corr = load_params()
        model = CorrModel()
        if require_surrogate and os.path.exists(MODEL_PATH):
            surrogate = Surrogate.load()
        else:
            surrogate = Surrogate()  # untrained -> infinite sigma -> simulates
        # pool signatures (submitted book) for c_max
        pool_sigs = build_signatures(db_adapter.load_pool_exprs())
        red = RedundancyIndex(model, psi, pool_sigs, [])
        # coverage index in phi-space (cached to disk)
        coverage = None
        if use_coverage:
            if coverage_cache and os.path.exists(coverage_cache):
                coverage = CoverageIndex.load(coverage_cache)
            else:
                sims = db_adapter.load_sims()
                if sim_index_size and len(sims) > sim_index_size:
                    step = len(sims) / sim_index_size
                    sims = [sims[int(i * step)] for i in range(sim_index_size)]
                coverage = CoverageIndex.from_exprs([r.expr for r in sims])
                if coverage_cache:
                    coverage.save(coverage_cache)
        p = params or GateParams()
        p.sigma_corr = sigma_corr
        return cls(surrogate, model, psi, red, coverage, p)

    # ---- decisions ------------------------------------------------------- #
    def decide(self, expr, **kw) -> GateDecision:
        return self.gate.decide(expr, **kw)

    def should_simulate(self, expr, **kw) -> bool:
        return self.gate.should_simulate(expr, **kw)

    def d_red(self, x_expr, y_expr) -> float:
        cm = abs(self.model.corr_model(x_expr, y_expr))
        return 1.0 - float(self.psi.predict(cm))


def d_red(x_expr, y_expr) -> float:
    """1 - |psi(corr_model(X, Y))| using persisted calibration."""
    psi, _ = load_params()
    cm = abs(_corr_model_default(x_expr, y_expr))
    return 1.0 - float(psi.predict(cm))
