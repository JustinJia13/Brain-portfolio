"""features.py — Signature/AST -> flat feature vector phi for the surrogate (§7.2).

phi captures the structure the submission components depend on:
  * curated scalars: #terms, depth, node count, #TS stages, window stats,
    passband centroid/bandwidth/DC-gain (from the composed filters), CS-warp and
    sign summaries, support flags;
  * hashed operator multiset counts;
  * hashed root-to-leaf path n-grams (n=1,2,3) -- captures "reversal-op above
    close" that bag-of-ops misses;
  * field-category fractions (cluster proxy via metadata categories).

Deterministic feature hashing (md5).  Fixed output dimension so the surrogate
sees a stable layout.
"""

from __future__ import annotations

import hashlib

import numpy as np

from alpha_grammar.models import ExpressionNode

from . import filters
from .fields import default_embedder, _load_metadata
from .signature import Signature, build_signature, _node_window

# layout
N_OP_HASH = 64
N_PATH_HASH = 96
_CATEGORIES = [
    "fundamental", "analyst", "price", "volume", "model", "news", "sentiment",
    "option", "institutions", "insider", "macro", "other",
]
N_CAT = len(_CATEGORIES)

# curated scalar slots (order matters; keep stable)
_SCALARS = [
    "n_terms", "depth", "n_nodes", "n_ts_stages", "max_ts_nesting",
    "n_fields", "n_distinct_ops",
    "win_min", "win_med", "win_max", "win_logspread",
    "pass_centroid", "pass_bandwidth", "dc_gain",
    "frac_lowpass", "frac_highpass", "frac_bandpass",
    "n_signflips", "frac_neg_terms",
    "has_trade_when", "has_group_filter", "n_neutralize",
    "frac_mono_warp", "frac_quantize_warp",
]
N_SCALAR = len(_SCALARS)

DIM_FULL = N_SCALAR + N_OP_HASH + N_PATH_HASH + N_CAT
# T-B2 (out-of-time ablation) found the hashed groups (op-hash, path-n-grams) HURT
# transfer — they overfit specific structures. Pruning them lifts held-out IC
# 0.29 -> 0.35. So the default phi = scalars + field-category fractions only.
DIM_PRUNED = N_SCALAR + N_CAT
DIM = DIM_PRUNED


def _bucket(token: str, d: int) -> int:
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % d


def _walk_ops_paths(node: ExpressionNode, path: tuple, ops: list, paths: list,
                    depth_acc: list, ts_nest: int, max_ts_nest: list, sign_flips: list):
    if not isinstance(node, ExpressionNode):
        return
    if node.operator is None:
        # leaf
        leaf = ("field" if isinstance(node.literal, str) else "num")
        for n in (1, 2, 3):
            if len(path) >= n - 1:
                paths.append("/".join(path[-(n - 1):] + (leaf,)) if n > 1 else leaf)
        return
    op = node.operator
    ops.append(op)
    if op in ("reverse", "negate", "inverse"):
        sign_flips.append(1)
    new_ts = ts_nest + (1 if filters.is_ts_operator(op) else 0)
    max_ts_nest[0] = max(max_ts_nest[0], new_ts)
    new_path = path + (op,)
    for n in (1, 2, 3):
        if len(new_path) >= n:
            paths.append("/".join(new_path[-n:]))
    for a in node.args:
        if isinstance(a, ExpressionNode):
            _walk_ops_paths(a, new_path, ops, paths, depth_acc, new_ts, max_ts_nest, sign_flips)


class Featurizer:
    def __init__(self, include_hashed: bool = False):
        # include_hashed=False (default) -> pruned phi = scalars + field_cat (B2).
        self.include_hashed = include_hashed
        self.dim = DIM_FULL if include_hashed else DIM_PRUNED
        meta = _load_metadata()
        self._cat_of = {fid: (m.category or "other").lower() for fid, m in meta.items()}

    def _category(self, field_id: str) -> str:
        c = self._cat_of.get(field_id, "")
        for cat in _CATEGORIES:
            if cat in c:
                return cat
        # crude id-token fallback
        low = field_id.lower()
        for cat in _CATEGORIES:
            if cat in low:
                return cat
        return "other"

    def featurize(self, expr_or_sig) -> np.ndarray | None:
        sig = expr_or_sig if isinstance(expr_or_sig, Signature) else build_signature(expr_or_sig)
        if sig is None:
            return None
        vec = np.zeros(DIM_FULL, dtype=float)  # build full layout, slice on return

        ops: list[str] = []
        paths: list[str] = []
        max_ts_nest = [0]
        sign_flips: list[int] = []
        _walk_ops_paths(sig.tree, (), ops, paths, [], 0, max_ts_nest, sign_flips)

        # ---- curated scalars -------------------------------------------- #
        prims = sig.primitives
        windows: list[int] = []

        def collect_windows(n: ExpressionNode):
            if not isinstance(n, ExpressionNode) or n.operator is None:
                return
            if filters.is_ts_operator(n.operator):
                w = _node_window(n)
                if w:
                    windows.append(int(w))
            for a in n.args:
                if isinstance(a, ExpressionNode):
                    collect_windows(a)
        collect_windows(sig.tree)

        s = {k: 0.0 for k in _SCALARS}
        s["n_terms"] = len(prims)
        s["depth"] = sig.tree.depth()
        s["n_nodes"] = len(ops)
        s["n_ts_stages"] = sum(1 for o in ops if filters.is_ts_operator(o))
        s["max_ts_nesting"] = max_ts_nest[0]
        s["n_fields"] = len(sig.fields)
        s["n_distinct_ops"] = len(set(ops))
        if windows:
            wlog = np.log(np.array(windows, float))
            s["win_min"] = float(min(windows))
            s["win_med"] = float(np.median(windows))
            s["win_max"] = float(max(windows))
            s["win_logspread"] = float(wlog.max() - wlog.min())
        # spectral summary averaged over primitives
        cent, bw, dc = [], [], []
        P = filters.OMEGA
        for p in prims:
            h2 = p.H ** 2
            tot = h2.sum()
            if tot > 1e-12:
                c = float((P * h2).sum() / tot)
                cent.append(c)
                bw.append(float(np.sqrt(((P - c) ** 2 * h2).sum() / tot)))
            dc.append(float(p.H[0]))
        s["pass_centroid"] = float(np.mean(cent)) if cent else 0.0
        s["pass_bandwidth"] = float(np.mean(bw)) if bw else 0.0
        s["dc_gain"] = float(np.mean(dc)) if dc else 1.0
        cls = [filters.ts_class(o) for o in ops if filters.is_ts_operator(o)]
        nts = max(len(cls), 1)
        s["frac_lowpass"] = sum(c == "lowpass" for c in cls) / nts
        s["frac_highpass"] = sum(c == "highpass" for c in cls) / nts
        s["frac_bandpass"] = sum(c == "bandpass" for c in cls) / nts
        s["n_signflips"] = len(sign_flips)
        s["frac_neg_terms"] = float(np.mean([p.sign < 0 for p in prims])) if prims else 0.0
        s["has_trade_when"] = float(any(p.has_trade_when for p in prims))
        s["has_group_filter"] = float(any(p.has_group_filter for p in prims))
        s["n_neutralize"] = sum(1 for o in ops if o == "group_neutralize")
        mono = {"rank", "zscore", "normalize", "scale", "group_rank", "group_zscore", "group_scale"}
        quant = {"bucket", "sign", "quantile"}
        s["frac_mono_warp"] = float(np.mean([p.warp in mono for p in prims])) if prims else 0.0
        s["frac_quantize_warp"] = float(np.mean([p.warp in quant for p in prims])) if prims else 0.0
        for i, k in enumerate(_SCALARS):
            vec[i] = s[k]

        # ---- hashed op counts + path n-grams (only if include_hashed) ---- #
        if self.include_hashed:
            base = N_SCALAR
            for o in ops:
                vec[base + _bucket("op=" + o, N_OP_HASH)] += 1.0
            base2 = N_SCALAR + N_OP_HASH
            for pth in paths:
                vec[base2 + _bucket("p=" + pth, N_PATH_HASH)] += 1.0
        # ---- field-category fractions ----------------------------------- #
        base3 = N_SCALAR + N_OP_HASH + N_PATH_HASH
        flds = sig.fields
        if flds:
            for f in flds:
                cat = self._category(f)
                vec[base3 + _CATEGORIES.index(cat)] += 1.0 / len(flds)
        if self.include_hashed:
            return vec
        # pruned layout: scalars ++ field_cat
        return np.concatenate([vec[:N_SCALAR], vec[base3:base3 + N_CAT]])


_DEFAULT = None


def default_featurizer() -> "Featurizer":
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Featurizer()
    return _DEFAULT


def featurize(expr_or_sig) -> np.ndarray | None:
    return default_featurizer().featurize(expr_or_sig)
