"""channels.py — behavioural-similarity channels for the redundancy metric.

Three of the four channels live here:
  * omega_time  — spectral overlap of composed TS magnitude responses (filters.py)
  * omega_cs    — cross-sectional warp similarity (rank/zscore/.../neutralize)
  * sign        — orientation product (helper; applied in aggregate.py)
The fourth channel (field similarity) lives in fields.py because it needs the
metadata-seeded embedding.
"""

from __future__ import annotations

import numpy as np

from . import filters


def omega_time(h1: np.ndarray | None, h2: np.ndarray | None) -> float:
    """Spectral overlap of two magnitude responses, in [0,1].

    Generalised cosine on the omega-weighted POWER spectrum:
        Omega = <p1, p2> / (||p1|| ||p2||),   p_i = |H_i|^2 * sqrt(omega)
    The |H|^2 (power) sharpens passband distinctions and the sqrt(omega) weight
    suppresses the cross-sectionally-neutralised DC mode every low-pass shares
    (see filters.COMPARE_WEIGHT).  This is the representation that meets the §9
    spectral acceptance thresholds with margin.

    None means an all-pass (no temporal shaping) — overlaps fully with anything,
    which is correct: a candidate with no TS chain on a field is temporally
    indistinguishable from any other treatment of that field at first order.
    """
    if h1 is None:
        h1 = filters._allpass()
    if h2 is None:
        h2 = filters._allpass()
    p1 = (h1 ** 2) * filters.COMPARE_WEIGHT
    p2 = (h2 ** 2) * filters.COMPARE_WEIGHT
    n1 = float(np.linalg.norm(p1))
    n2 = float(np.linalg.norm(p2))
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(p1, p2) / (n1 * n2), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Cross-sectional warp channel (§3.4)
# --------------------------------------------------------------------------- #
# Map each CS operator to a warp *family*.  Positions live in rank space, so
# mutually monotone warps are ~1; quantisers (bucket/sign) and neutralisation
# attenuate.  group_neutralize additionally carries a kappa_g attenuation that
# is applied as a separate multiplicative factor in the signature (not here).
_WARP_FAMILY = {
    "rank": "mono", "group_rank": "mono",
    "zscore": "mono", "group_zscore": "mono",
    "normalize": "mono", "group_normalize": "mono",
    "scale": "mono", "group_scale": "mono",
    "winsorize": "winsor",
    "bucket": "bucket", "quantile": "bucket",
    "sign": "sign",
    "group_neutralize": "neutralize", "normalize_neut": "neutralize",
    "none": "identity", "": "identity",
}

# Symmetric prior similarity between warp families (diag = 1.0). Calibratable.
_WARP_SIM = {
    frozenset(["mono", "winsor"]): 0.98,
    frozenset(["mono", "bucket"]): 0.95,
    frozenset(["mono", "sign"]): 0.80,
    frozenset(["mono", "neutralize"]): 0.90,
    frozenset(["mono", "identity"]): 0.85,
    frozenset(["winsor", "bucket"]): 0.94,
    frozenset(["winsor", "sign"]): 0.80,
    frozenset(["winsor", "neutralize"]): 0.89,
    frozenset(["winsor", "identity"]): 0.86,
    frozenset(["bucket", "sign"]): 0.82,
    frozenset(["bucket", "neutralize"]): 0.88,
    frozenset(["bucket", "identity"]): 0.83,
    frozenset(["sign", "neutralize"]): 0.78,
    frozenset(["sign", "identity"]): 0.75,
    frozenset(["neutralize", "identity"]): 0.85,
}


def warp_family(cs_op: str | None) -> str:
    if not cs_op:
        return "identity"
    return _WARP_FAMILY.get(cs_op, "mono")  # unknown CS op defaults to monotone


def omega_cs(w1: str | None, w2: str | None, table: dict | None = None) -> float:
    """Cross-sectional warp similarity in [0,1] keyed by warp family."""
    f1, f2 = warp_family(w1), warp_family(w2)
    if f1 == f2:
        return 1.0
    sim = table if table is not None else _WARP_SIM
    return float(sim.get(frozenset([f1, f2]), 0.85))


