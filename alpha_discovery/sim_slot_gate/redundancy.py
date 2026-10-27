"""redundancy.py — pool redundancy c_max and coverage density n_eff (§8).

Object A / coverage enter the gate in exactly two places:
  * c_max  — max |psi(corr_model)| of a candidate against the submitted pool
             (corr_model + psi over the ~31 book members; cheap, exact). A
             reliable, no-epsilon KILL when confidently high (near-duplicate of
             something already submitted). Validation (Checkpoint 1) showed
             corr_model is trustworthy in the HIGH range, so the kill uses a high
             tau_corr.
  * n_eff  — kernel density of the candidate over the sim index, in the SPACE THE
             SURROGATE EXTRAPOLATES IN: phi (feature) space.  Low n_eff = B is
             off-support -> SIMULATE (locality for COVERAGE, not prediction).

Why phi-space and not d_red space for coverage (deviation from §8, documented in
metric.md §7): d_red = 1-|psi(corr_model)| compresses all distances into roughly
[0.32, 0.91] because psi caps near 0.68 (high structural corr only predicts ~0.68
PnL-corr), so even structural twins get negligible kernel weight -- coverage would
fire everywhere.  Coverage is about whether B has nearby training data, and B
operates on phi; a standardised-phi Gaussian density is both correct for that and
fully vectorised (fast enough for the budget backtest's millions of evaluations).
"""

from __future__ import annotations


import numpy as np

from .aggregate import CorrModel
from .calibrate import Isotonic
from .features import default_featurizer
from .signature import Signature, build_signature


class RedundancyIndex:
    """Holds the pool signatures and a sim-history index for c_max / n_eff."""

    def __init__(self, model: CorrModel, psi: Isotonic,
                 pool_sigs: list[Signature] | None = None,
                 sim_sigs: list[Signature] | None = None):
        self.model = model
        self.psi = psi
        self.pool_sigs = pool_sigs or []
        self.sim_sigs = sim_sigs or []

    # ---- pool redundancy -------------------------------------------------- #
    def c_max(self, cand: Signature) -> float:
        """max_P |psi(corr_model(cand, P))| over the submitted pool, in [0,1]."""
        if cand is None or not self.pool_sigs:
            return 0.0
        best = 0.0
        for P in self.pool_sigs:
            cm = abs(self.model.corr_model_sig(cand, P))
            val = float(self.psi.predict(cm))
            if val > best:
                best = val
        return best


class CoverageIndex:
    """Vectorised phi-space density: n_eff = how much training data is near X.

    Standardises phi by the sim-index mean/std, then a Gaussian kernel over
    Euclidean distance in standardised-phi space.  Distances are scaled by 1/sqrt(D)
    so the bandwidth `h` is dimension-agnostic and reusable across feature layouts.
    """

    def __init__(self, phi: np.ndarray, h: float = 1.0):
        self.h = h
        self.mean = phi.mean(axis=0)
        self.std = phi.std(axis=0) + 1e-9
        self.Z = (phi - self.mean) / self.std       # n x D standardised
        self.D = phi.shape[1]

    def n_eff(self, phi: np.ndarray, h: float | None = None) -> float:
        h = self.h if h is None else h
        z = (phi - self.mean) / self.std
        d2 = ((self.Z - z) ** 2).sum(axis=1) / self.D   # mean-squared per-dim distance
        return float(np.exp(-0.5 * d2 / (h * h)).sum())

    # persistence (cache the standardised stats + raw phi)
    def save(self, path: str):
        np.savez_compressed(path, phi=(self.Z * self.std + self.mean), h=self.h)

    @classmethod
    def load(cls, path: str, h: float | None = None) -> "CoverageIndex":
        d = np.load(path)
        ci = cls(d["phi"], h=float(d["h"]))
        if h is not None:
            ci.h = h
        return ci

    @classmethod
    def from_exprs(cls, exprs, h: float = 1.0, featurizer=None) -> "CoverageIndex":
        fz = featurizer or default_featurizer()
        rows = [fz.featurize(e) for e in exprs]
        rows = [r for r in rows if r is not None]
        return cls(np.array(rows), h=h)


def build_signatures(exprs, builder=None) -> list[Signature]:
    out = []
    for e in exprs:
        s = build_signature(e, builder)
        if s is not None:
            out.append(s)
    return out
