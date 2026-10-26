"""aggregate.py — rho_sig and corr_model: the analytic signal-correlation estimate (§3.5).

Per-primitive behavioural similarity is the product of the four channels:

    rho_sig(tau, sigma) = s_tau*s_sigma
                          * field_sim(f_tau, f_sigma)      (fields.py)
                          * omega_time(H_tau, H_sigma)     (channels.py, spectral)
                          * omega_cs(W_tau, W_sigma)        (channels.py, CS warp)
                          * kappa_tau * kappa_sigma         (group-neut attenuation)

and alphas combine as a generalised cosine over their primitives:

    corr_model(X, Y) = sum_{tau,sigma} c^X_tau c^Y_sigma rho_sig(tau,sigma)
                       / sqrt(G_X * G_Y)
    G_X = sum_{tau,tau'} c^X_tau c^X_tau' rho_sig(tau,tau')

This is a strong *estimator* of signal correlation, not exact — multiplicative /
ts_corr terms are first-order (signature.py composes child |H|, unions fields).
The estimate is mapped to PnL-correlation by the calibrated psi (calibrate.py).
"""

from __future__ import annotations

import numpy as np

from . import channels
from .fields import FieldEmbedder, default_embedder
from .signature import Primitive, Signature, build_signature


class CorrModel:
    """Computes corr_model(X, Y) with a fixed embedder + calibrated CS-warp table."""

    def __init__(self, embedder: FieldEmbedder | None = None, warp_table: dict | None = None):
        self.embedder = embedder or default_embedder()
        self.warp_table = warp_table  # None -> channel priors

    # ---- primitive-pair similarity --------------------------------------- #
    def rho_sig(self, a: Primitive, b: Primitive) -> float:
        sign = 1.0 if a.sign == b.sign else -1.0
        fld = self.embedder.field_sim(list(a.fields), list(b.fields))
        tim = channels.omega_time(a.H, b.H)
        cs = channels.omega_cs(a.warp, b.warp, self.warp_table)
        kap = a.kappa_g * b.kappa_g
        return sign * fld * tim * cs * kap

    # ---- gram helpers ---------------------------------------------------- #
    def _gram(self, prims) -> float:
        g = 0.0
        for i, ti in enumerate(prims):
            for tj in prims:
                g += ti.coef * tj.coef * self.rho_sig(ti, tj)
        return g

    def corr_model_sig(self, sx: Signature, sy: Signature) -> float:
        if sx is None or sy is None:
            return 0.0
        Gx = self._gram(sx.primitives)
        Gy = self._gram(sy.primitives)
        if Gx <= 1e-12 or Gy <= 1e-12:
            return 0.0
        cross = 0.0
        for ti in sx.primitives:
            for tj in sy.primitives:
                cross += ti.coef * tj.coef * self.rho_sig(ti, tj)
        return float(np.clip(cross / np.sqrt(Gx * Gy), -1.0, 1.0))

    def corr_model(self, x_expr, y_expr) -> float:
        sx = x_expr if isinstance(x_expr, Signature) else build_signature(x_expr)
        sy = y_expr if isinstance(y_expr, Signature) else build_signature(y_expr)
        if sx is None or sy is None:
            return 0.0
        return self.corr_model_sig(sx, sy)


_DEFAULT_MODEL = CorrModel()


def corr_model(x_expr, y_expr) -> float:
    """corr_model(X, Y) in [-1, 1] using default embedder + prior CS-warp table."""
    return _DEFAULT_MODEL.corr_model(x_expr, y_expr)
