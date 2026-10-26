"""corr_model unit tests (§9, Object A channel-level + aggregation).

    ${PYTHON:-python3} -m alpha_discovery/sim_slot_gate.tests.test_corr_model
"""

from __future__ import annotations

from alpha_discovery.sim_slot_gate.aggregate import corr_model
from alpha_discovery.sim_slot_gate.signature import build_signature


def test_identity():
    assert corr_model("rank(ts_mean(close, 5))", "rank(ts_mean(close, 5))") > 0.999


def test_rank_vs_zscore_same_field():
    # rank and zscore are mutually monotone -> ~1
    v = corr_model("rank(close)", "zscore(close)")
    assert v > 0.98, v


def test_negation_is_minus_one():
    v = corr_model("rank(close)", "reverse(rank(close))")
    assert v < -0.98, v
    # |corr_model| ~ 1 : redundancy treats them identical
    assert abs(v) > 0.98


def test_unary_minus_negation():
    v = corr_model("ts_mean(close, 10)", "-ts_mean(close, 10)")
    assert v < -0.95, v


def test_group_neutralize_attenuation():
    # corr_model(f, group_neutralize(f)) ~ kappa_g = sqrt(1-R2) (~0.84 prior)
    v = corr_model("rank(close)", "group_neutralize(rank(close), industry)")
    assert 0.78 <= v <= 0.90, v


def test_window_separation():
    near = corr_model("ts_mean(close, 5)", "ts_mean(close, 6)")
    far = corr_model("ts_mean(close, 5)", "ts_mean(close, 250)")
    assert near > 0.95 and far < 0.5 and near > far, (near, far)


def test_lowpass_highpass_same_field_low():
    v = corr_model("ts_mean(close, 20)", "ts_delta(close, 1)")
    assert v < 0.3, v


def test_different_fields_lower_than_same():
    same = corr_model("rank(ts_mean(close, 20))", "rank(ts_mean(close, 22))")
    diff = corr_model("rank(ts_mean(close, 20))", "rank(ts_mean(sales, 20))")
    assert same > diff, (same, diff)


def test_two_term_alpha_builds():
    s = build_signature("add(rank(close), group_neutralize(zscore(sales), industry))")
    assert s is not None and len(s.primitives) == 2
    fams = {p.warp for p in s.primitives}
    assert "rank" in fams and "group_neutralize" in fams


def test_structural_duplicate_separation():
    # Near-duplicates (same skeleton, swapped field/window) must score high;
    # structurally unrelated alphas must score low. This is the gate's core job.
    near = [
        ("group_rank(ts_mean(assets, 60), industry)", "group_rank(ts_mean(assets, 66), industry)"),
        ("rank(ts_delta(close, 5))", "rank(ts_delta(close, 6))"),
        ("zscore(ts_mean(sales, 20))", "zscore(ts_mean(sales, 22))"),
    ]
    # genuinely far: differ in BOTH field category (fundamental vs price) and
    # filter class (low-pass vs high-pass)
    far = [
        ("group_rank(ts_mean(assets, 60), industry)", "rank(ts_delta(close, 5))"),
        ("zscore(ts_mean(sales, 20))", "ts_zscore(returns, 250)"),
        ("rank(ts_delta(close, 5))", "group_neutralize(ts_mean(assets, 200), sector)"),
    ]
    nv = [corr_model(a, b) for a, b in near]
    fv = [abs(corr_model(a, b)) for a, b in far]
    assert min(nv) > 0.8, nv
    assert max(fv) < 0.5, fv


def test_unparseable_returns_zero():
    assert corr_model("@@@bad@@@", "rank(close)") == 0.0


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} corr_model tests passed")


if __name__ == "__main__":
    _run()
