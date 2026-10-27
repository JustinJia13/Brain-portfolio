"""Channel-level unit tests (§9, Object A).

    ${PYTHON:-python3} -m alpha_discovery.sim_slot_gate.tests.test_channels
"""

from __future__ import annotations

from alpha_discovery.sim_slot_gate import filters, channels


def test_ts_mean_close_windows_similar():
    h5 = filters.response("ts_mean", 5)
    h6 = filters.response("ts_mean", 6)
    val = channels.omega_time(h5, h6)
    assert val > 0.98, f"ts_mean_5 vs ts_mean_6 = {val:.4f}, want >0.98"


def test_ts_mean_far_windows_dissimilar():
    h5 = filters.response("ts_mean", 5)
    h250 = filters.response("ts_mean", 250)
    val = channels.omega_time(h5, h250)
    assert val < 0.2, f"ts_mean_5 vs ts_mean_250 = {val:.4f}, want <0.2"


def test_lowpass_vs_highpass_orthogonal():
    hm = filters.response("ts_mean", 10)
    hd = filters.response("ts_delta", 1)
    val = channels.omega_time(hm, hd)
    assert val < 0.1, f"ts_mean vs ts_delta = {val:.4f}, want <0.1"


def test_whiten_is_highpass():
    # ts_zscore should look more like a high-pass than a low-pass
    hz = filters.response("ts_zscore", 20)
    hd = filters.response("ts_delta", 5)
    hm = filters.response("ts_mean", 20)
    assert channels.omega_time(hz, hd) > channels.omega_time(hz, hm)


def test_delay_is_allpass():
    hd = filters.response("ts_delay", 5)
    assert (hd == 1.0).all()


def test_compose_multiplies():
    import numpy as np
    a = filters.response("ts_mean", 10)
    b = filters.response("ts_delta", 1)
    comp = filters.compose([a, b])
    assert np.allclose(comp, a * b)


def test_omega_cs_monotone_family():
    assert channels.omega_cs("rank", "zscore") == 1.0
    assert channels.omega_cs("rank", "normalize") == 1.0
    assert channels.omega_cs("rank", "scale") == 1.0


def test_omega_cs_winsorize_near_rank():
    val = channels.omega_cs("winsorize", "rank")
    assert 0.95 <= val <= 0.99, val


def test_omega_cs_sign_lower():
    assert channels.omega_cs("sign", "rank") < channels.omega_cs("winsorize", "rank")


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} channel tests passed")


if __name__ == "__main__":
    _run()
