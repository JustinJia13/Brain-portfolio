"""gate.py — should_simulate: the sim-slot decision (§8).

Skip a candidate only when a RELIABLE signal says so (turnover or pool
redundancy), or when the QUALITY model is CONFIDENTLY below a must-pass threshold;
simulate when under-explored or when anything plausibly clears.  Per-metric UCB is
nudged in the exploration-preserving direction so the binding constraint emerges
per-candidate.

Two distinct uncertainties, both required (§8):
  * sigma(phi)  — how noisy B's prediction is at X (wide where the map is hard)
  * n_eff       — how much data exists near X (low off-support, even where B is
                  wrongly confident)

Returns a GateDecision carrying the kept/killed boolean AND every component value
+ the firing reason, so shadow mode can log why each candidate was kept/killed.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import numpy as np

from . import db_adapter
from .db_adapter import TURNOVER_FLOOR
from .redundancy import RedundancyIndex
from .signature import Signature, build_signature
from .surrogate import Surrogate


@dataclass
class GateParams:
    # quality thresholds (Discovery: submission needs sharpe>=1.25, fitness>=1.0,
    # turnover<=0.7; sub-universe-sharpe limit is low ~0.3). Tuned by the backtest.
    tau_sub: float = 0.30      # min plausible sub-universe Sharpe
    tau_fit: float = 1.00      # min plausible fitness (SUBMIT_MIN_FITNESS)
    tau_to: float = 0.70       # max plausible turnover (SUBMIT_MAX_TURNOVER)
    # psi caps ~0.68 and the optimistic adj subtracts beta*sigma_corr(~0.19), so
    # tau_corr must sit below ~0.49 to ever fire; 0.35 kills corr_model >~0.9
    # (genuine structural duplicates -> predicted PnL-corr ~0.55). Backtest-tuned.
    tau_corr: float = 0.40     # kill if optimistic predicted pool corr stays above this
    # DECOUPLED UCB widths: the reliable kills and the quality gate point in
    # opposite directions and want different conservatism.  A single beta couples
    # them so that raising it to protect good alphas (quality) also disables the
    # redundancy/turnover kills (their threshold runs past psi's ~0.68 cap).
    # Default from the HONEST time-split budget backtest (T-G3): beta=2.5 saves
    # ~45% of slots while keeping ~85% of good alphas in the quality zone. (An
    # earlier random-split backtest overstated recall at beta=2.0; the time split
    # is the authoritative frontier.) Raise toward 3.0 to protect more good alphas.
    beta: float = 2.5          # quality-gate UCB width
    beta_kill: float = 1.0     # reliable-kill optimistic-adjustment width
    n_min: float = 200.0       # min coverage density (phi-space) before trusting B
    h: float = 1.0             # n_eff kernel bandwidth (standardised phi-space)
    eps: float = 0.05          # exploration trickle
    sigma_corr: float = 0.186  # redundancy uncertainty (from calibration)
    # §1 safety: B5 showed ~0.04 upper-tail under-coverage (UCB slightly optimistic).
    # Inflate head sigma to err toward simulate while proving the gate live. 1.0=off.
    sigma_inflate: float = 1.0


@dataclass
class GateDecision:
    simulate: bool
    reason: str
    turnover_lcb: float = float("nan")
    c_max: float = float("nan")
    n_eff: float = float("nan")
    sub_ucb: float = float("nan")
    fit_ucb: float = float("nan")
    parse_ok: bool = True
    # raw per-head predictions (q50 + sigma) so beta stays a POST-HOC knob in the
    # cutover monitor — recompute any decision/frontier offline from these.
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Gate:
    def __init__(self, surrogate: Surrogate, redundancy: RedundancyIndex,
                 coverage=None, featurizer=None, params: GateParams | None = None, rng=None):
        self.model = surrogate
        self.red = redundancy
        self.coverage = coverage  # CoverageIndex (phi-space) or None to disable branch (3)
        self.p = params or GateParams()
        if featurizer is None:
            from .features import default_featurizer
            featurizer = default_featurizer()
        self.featurizer = featurizer
        self.rng = rng if rng is not None else random.Random(0)

    def decide(self, cand, *, use_coverage: bool = True) -> GateDecision:
        sig = cand if isinstance(cand, Signature) else build_signature(cand)
        if sig is None:
            # unparseable -> simulate (safe direction for a sim-slot gate)
            return GateDecision(True, "unparseable_simulate", parse_ok=False)
        phi = self.featurizer.featurize(sig)
        if phi is None:
            return GateDecision(True, "no_features_simulate", parse_ok=False)
        p = self.p
        d = GateDecision(True, "")

        # Raw per-head quantiles [q16,q50,q84] computed ONCE; reused by every branch
        # and logged so the cutover monitor can recompute coverage AND re-derive any
        # beta offline (beta stays a post-hoc knob).
        M = self.model
        infl = p.sigma_inflate

        def q3(t):
            return [M.q(t, 0.16, phi), M.q(t, 0.50, phi), M.q(t, 0.84, phi)]
        Q = {t: q3(t) for t in ("subsharpe", "sharpe", "ret", "turnover")}
        sub_q50, sub_sig = Q["subsharpe"][1], infl * (Q["subsharpe"][2] - Q["subsharpe"][1])
        sh_q50, sh_sig = Q["sharpe"][1], infl * (Q["sharpe"][2] - Q["sharpe"][1])
        rt_q50, rt_sig = Q["ret"][1], infl * (Q["ret"][2] - Q["ret"][1])
        to_q50, to_sig = Q["turnover"][1], infl * (Q["turnover"][2] - Q["turnover"][1])
        c_max = self.red.c_max(sig)
        d.c_max = c_max
        # n_eff computed ALWAYS (beta-independent) so the cutover monitor can measure
        # the live novelty rate over every candidate, not just those reaching branch 3.
        if use_coverage and self.coverage is not None:
            d.n_eff = self.coverage.n_eff(phi, h=p.h)
        d.raw = {**{t: [float(x) for x in v] for t, v in Q.items()},
                 "c_max": float(c_max), "sigma_inflate": infl}

        def _fit_upper(beta):
            if not np.isfinite([sh_q50, rt_q50, to_q50]).all():
                return float("inf")
            return (sh_q50 + beta * sh_sig) * np.sqrt(
                abs(rt_q50 + beta * rt_sig) / max(to_q50 - beta * to_sig, TURNOVER_FLOOR))

        # (1) RELIABLE KILLS — sidestep the hard Sharpe map; no eps-rescue.
        # Use beta_kill (not beta): these point opposite the quality UCB.
        to_lcb = to_q50 - p.beta_kill * to_sig
        d.turnover_lcb = to_lcb
        if np.isfinite(to_lcb) and to_lcb > p.tau_to:
            d.simulate, d.reason = False, "kill_turnover"
            return d
        if c_max - p.beta_kill * p.sigma_corr > p.tau_corr:
            d.simulate, d.reason = False, "kill_redundant"
            return d

        # (2) EPS EXPLORATION — un-calcify the quality model's blind spots.
        if self.rng.random() < p.eps:
            d.reason = "eps_explore"
            return d

        # (3) COVERAGE ABSTENTION — B extrapolating off-support -> simulate.
        if use_coverage and self.coverage is not None and d.n_eff < p.n_min:
            d.reason = "coverage_simulate"
            return d

        # (4) QUALITY CONJUNCTION — skip ONLY if confidently below a must-pass.
        sub_ucb = sub_q50 + p.beta * sub_sig
        fit_ucb = _fit_upper(p.beta)
        d.sub_ucb, d.fit_ucb = sub_ucb, fit_ucb
        sub_ok = (not np.isfinite(sub_ucb)) or (sub_ucb >= p.tau_sub)
        fit_ok = (not np.isfinite(fit_ucb)) or (fit_ucb >= p.tau_fit)
        if sub_ok and fit_ok:
            d.reason = "quality_pass_simulate"
            d.simulate = True
        else:
            d.reason = "kill_quality"
            d.simulate = False
        return d

    def should_simulate(self, cand, **kw) -> bool:
        return self.decide(cand, **kw).simulate
