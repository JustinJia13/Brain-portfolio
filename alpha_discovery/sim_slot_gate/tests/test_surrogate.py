"""Object B acceptance test — quantile CALIBRATION, not accuracy (§7.4, §9).

Loads the persisted surrogate and asserts held-out coverage is near nominal.
Requires a trained alpha_discovery/sim_slot_gate/surrogate.pkl (run surrogate.Surrogate().fit()).

    ${PYTHON:-python3} -m alpha_discovery/sim_slot_gate.tests.test_surrogate
"""

from __future__ import annotations

import os

from alpha_discovery.sim_slot_gate.surrogate import Surrogate, MODEL_PATH, QUANTILES

_TOL = {0.16: 0.06, 0.50: 0.08, 0.84: 0.06}


def test_coverage_near_nominal():
    if not os.path.exists(MODEL_PATH):
        print("  SKIP (no trained surrogate.pkl)")
        return
    s = Surrogate.load()
    assert s.coverage, "no coverage report stored"
    for target, rep in s.coverage.items():
        if not rep.get("trained"):
            continue
        cov = rep["coverage"]
        for q in QUANTILES:
            got = cov[q]
            assert abs(got - q) <= _TOL[q], f"{target} coverage@{q}={got:.3f} off by >{_TOL[q]}"
        print(f"  PASS {target:10s} coverage ~ nominal")


def test_turnover_precision_at_cap():
    if not os.path.exists(MODEL_PATH):
        print("  SKIP (no trained surrogate.pkl)")
        return
    s = Surrogate.load()
    rep = s.coverage.get("turnover", {})
    if "precision_at_cap" in rep:
        assert rep["precision_at_cap"] >= 0.6, rep["precision_at_cap"]
        print(f"  PASS turnover precision@cap = {rep['precision_at_cap']:.2f}")


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print("\nsurrogate acceptance checks done")


if __name__ == "__main__":
    _run()
