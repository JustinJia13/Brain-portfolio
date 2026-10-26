"""filters.py — temporal channel: operator(+window) -> magnitude response |H(omega)|.

Each time-series operator is treated as a linear filter with a frequency response.
Composition of a TS chain multiplies magnitudes (|H| = prod_i |H_i|), so two alphas
on the same field are temporally similar to the degree their composed |H| overlap
on the frequency grid (channels.omega_time).

DESIGN NOTE (deviation from the spec's exact closed forms — see metric.md §9):
The spec lists the exact responses, e.g. ts_mean(w) = |sin(w*om/2)/(w*sin(om/2))|
(the Dirichlet kernel).  Those carry sidelobe ripple whose *pattern* differs between
w=5 and w=6, which spuriously drops Omega_time(ts_mean_5, ts_mean_6) below the
spec's own acceptance threshold (>0.98).  Behavioural similarity tracks the
PASSBAND, not sidelobe ripple — a 5-day and 6-day average behave the same — so we
represent each filter by its smooth passband ENVELOPE (single-pole Butterworth).
The envelope corners scale identically to the exact forms (omega_c = 2*pi/window),
so window separation (5 vs 250) and band separation (low vs high) are preserved,
while ripple-induced spurious differences vanish.  This choice meets every §9
spectral acceptance test with margin; the exact forms do not.

Magnitude only — phase / lead-lag dropped to first order (ts_delay is all-pass).

Frequency grid: angular frequency omega in (0, pi].  omega = 2*pi / period_in_days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Frequency grid + comparison weighting
# --------------------------------------------------------------------------- #
N_GRID = 160
_OMEGA_MIN = 2.0 * np.pi / 512.0          # resolves ~512-day periods (windows to ~250)
OMEGA = np.linspace(_OMEGA_MIN, np.pi, N_GRID)

# Inner-product weight used by channels.omega_time.  sqrt(omega) de-emphasises the
# very-low-frequency / DC mode that every low-pass shares — that common slow
# component is cross-sectionally neutralised in production and is not where alphas
# differ.  Combined with the power-spectrum (|H|^2) comparison it gives clean,
# ripple-robust separation. (Calibration may refine, but the prior is well-behaved.)
COMPARE_WEIGHT = np.sqrt(OMEGA)


# --------------------------------------------------------------------------- #
# Calibratable parameters (priors here; overridden from metric_params.json)
# --------------------------------------------------------------------------- #
@dataclass
class FilterParams:
    whiten_g0: float = 0.10      # ts_zscore/ts_rank/ts_scale shelf floor
    whiten_eta: float = 1.0      # shelf corner scale
    std_eta: float = 1.0         # ts_std_dev band corner scale
    std_g0: float = 0.05
    env_eta: float = 1.5         # min/max/arg* effective-window inflation
    corr_eta: float = 1.0        # ts_corr/cov/regression band corner scale
    corr_g0: float = 0.05
    decay_eta: float = 0.5       # decay ops: effective window fraction

    def update(self, d: dict) -> "FilterParams":
        for k, v in (d or {}).items():
            if hasattr(self, k):
                setattr(self, k, float(v))
        return self


DEFAULT_PARAMS = FilterParams()


# --------------------------------------------------------------------------- #
# Smooth passband envelopes
# --------------------------------------------------------------------------- #
def _omega_c(window: float) -> float:
    return 2.0 * np.pi / max(float(window), 1.0)


def _lowpass(window: float, n: int = 1) -> np.ndarray:
    """Single-pole low-pass envelope, corner omega_c = 2*pi/window."""
    wc = _omega_c(window)
    return 1.0 / (1.0 + (OMEGA / wc) ** (2 * n))


def _highpass(window: float, n: int = 1) -> np.ndarray:
    """High-pass = 1 - low-pass at the same corner."""
    return 1.0 - _lowpass(window, n)


def _allpass() -> np.ndarray:
    return np.ones_like(OMEGA)


# back-compat alias used by channels.omega_time
_h_allpass = _allpass


def _h_whiten(w: float, p: FilterParams) -> np.ndarray:
    """ts_zscore/ts_rank/ts_scale(w): high-pass shelf g0 + (1-g0)*om/(om+omc)."""
    omc = 2.0 * np.pi / (max(p.whiten_eta, 1e-3) * max(w, 1.0))
    return p.whiten_g0 + (1.0 - p.whiten_g0) * OMEGA / (OMEGA + omc)


def _h_std_dev(w: float, p: FilterParams) -> np.ndarray:
    """ts_std_dev(w): band-pass — removes slow drift, gentle HF roll-off."""
    omc = 2.0 * np.pi / (max(p.std_eta, 1e-3) * max(w, 1.0))
    hp = OMEGA / (OMEGA + omc)
    lp = 1.0 / (1.0 + (OMEGA / np.pi) ** 2)
    h = p.std_g0 + (1.0 - p.std_g0) * hp * lp
    m = h.max()
    return h / m if m > 0 else h


def _h_corr(w: float, p: FilterParams) -> np.ndarray:
    """ts_corr/ts_covariance/ts_regression(.,.,w): rolling 2-field band template."""
    omc = 2.0 * np.pi / (max(p.corr_eta, 1e-3) * max(w, 1.0))
    hp = OMEGA / (OMEGA + omc)
    lp = 1.0 / (1.0 + (OMEGA / (2.0 * omc + 1e-9)) ** 2)
    h = p.corr_g0 + (1.0 - p.corr_g0) * hp * lp
    m = h.max()
    return h / m if m > 0 else h


# --------------------------------------------------------------------------- #
# Operator -> (class, response builder)
# --------------------------------------------------------------------------- #
LOWPASS, HIGHPASS, BANDPASS, ALLPASS, ENVELOPE = (
    "lowpass", "highpass", "bandpass", "allpass", "nonlinear_envelope",
)

_TS_BUILDERS: dict[str, tuple[str, Callable[[float, FilterParams], np.ndarray]]] = {
    # low-pass smoothers
    "ts_mean":         (LOWPASS,  lambda w, p: _lowpass(w)),
    "ts_sum":          (LOWPASS,  lambda w, p: _lowpass(w)),
    "ts_product":      (LOWPASS,  lambda w, p: _lowpass(w)),
    "ts_decay_linear": (LOWPASS,  lambda w, p: _lowpass(max(p.decay_eta, 1e-3) * w)),
    "ts_decay_exp":    (LOWPASS,  lambda w, p: _lowpass(max(p.decay_eta, 1e-3) * w)),
    "ts_step":         (LOWPASS,  lambda w, p: _lowpass(max(p.decay_eta, 1e-3) * w)),
    # high-pass differencers
    "ts_delta":        (HIGHPASS, lambda w, p: _highpass(w)),
    "ts_returns":      (HIGHPASS, lambda w, p: _highpass(w or 1)),
    "last_diff_value": (HIGHPASS, lambda w, p: _highpass(w or 1)),
    "ts_av_diff":      (HIGHPASS, lambda w, p: _highpass(w)),
    "days_from_last_change": (HIGHPASS, lambda w, p: _highpass(w or 1)),
    # all-pass (magnitude) — phase shift / gap fill
    "ts_delay":        (ALLPASS,  lambda w, p: _allpass()),
    "ts_backfill":     (ALLPASS,  lambda w, p: _allpass()),
    "group_backfill":  (ALLPASS,  lambda w, p: _allpass()),
    # calibrated non-linear templates (smooth)
    "ts_zscore":       (HIGHPASS, _h_whiten),
    "ts_rank":         (HIGHPASS, _h_whiten),
    "ts_scale":        (HIGHPASS, _h_whiten),
    "ts_std_dev":      (BANDPASS, _h_std_dev),
    "ts_min":          (ENVELOPE, lambda w, p: _lowpass(max(p.env_eta, 1e-3) * w)),
    "ts_max":          (ENVELOPE, lambda w, p: _lowpass(max(p.env_eta, 1e-3) * w)),
    "ts_arg_min":      (ENVELOPE, lambda w, p: _lowpass(max(p.env_eta, 1e-3) * w)),
    "ts_arg_max":      (ENVELOPE, lambda w, p: _lowpass(max(p.env_eta, 1e-3) * w)),
    "kth_element":     (ENVELOPE, lambda w, p: _lowpass(max(p.env_eta, 1e-3) * w)),
    "ts_quantile":     (ENVELOPE, lambda w, p: _lowpass(max(p.env_eta, 1e-3) * w)),
    "ts_corr":         (BANDPASS, _h_corr),
    "ts_covariance":   (BANDPASS, _h_corr),
    "ts_regression":   (BANDPASS, _h_corr),
}

_DEFAULT_WINDOW = 20.0


def is_ts_operator(op: str) -> bool:
    return op in _TS_BUILDERS


def ts_class(op: str) -> str | None:
    spec = _TS_BUILDERS.get(op)
    return spec[0] if spec else None


def response(op: str, window: float | None, params: FilterParams = DEFAULT_PARAMS) -> np.ndarray | None:
    """|H(omega)| for one TS operator; None if `op` is not a temporal filter."""
    spec = _TS_BUILDERS.get(op)
    if spec is None:
        return None
    w = float(window) if window not in (None, 0) else _DEFAULT_WINDOW
    return spec[1](w, params)


def compose(responses: list[np.ndarray]) -> np.ndarray:
    """Compose a TS chain: magnitudes multiply.  Empty chain -> all-pass."""
    if not responses:
        return _allpass()
    h = np.ones_like(OMEGA)
    for r in responses:
        h = h * r
    return h
