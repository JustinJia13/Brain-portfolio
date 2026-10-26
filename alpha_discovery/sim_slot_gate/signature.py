"""signature.py — AST -> Signature of additive behavioural primitives (§3.1-3.2).

An alpha is decomposed under top-level +/- into additive primitives
    S_X ~= sum_tau c_tau * tau
where a multiplicative / divisive node (multiply, divide, ts_corr, signed_power,
...) is itself ONE primitive (not additively separable, §3.5).  Each primitive
carries the four behavioural channels used by aggregate.corr_model:

    fields   f_tau   data fields touched (group-name args excluded)
    H        |H|     composed TS magnitude response of the chain (all-pass if none)
    warp     W_tau   outermost cross-sectional warp op name ("none" if none)
    kappa_g          group-neutralisation attenuation (1.0 if none)
    sign     s_tau   +/-1 orientation
    coef     c_tau   magnitude (~1; sign carried in `sign`)
    support          has_trade_when / has_group_filter flags

Multiplicative primitives are handled by a single subtree walk that unions fields
and composes all TS responses across both operands -- exactly the §3.5 first-order
"elementwise product of child |H|, field set by union" rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.expr.models import ExpressionNode
from core.expr.operators import OPERATOR_REGISTRY

from . import filters
from .parse import parse_expression, canonicalize

# --------------------------------------------------------------------------- #
# Operator categories derived from the registry (robust to private renames)
# --------------------------------------------------------------------------- #
_GROUP_ARG_INDEX = {
    name: spec.group_arg_index
    for name, spec in OPERATOR_REGISTRY.items()
    if spec.group_arg_index is not None
}

# Cross-sectional warps (the final distribution-shaping op).  group_backfill /
# group_mean are NOT warps (fill / weighted-agg); they are handled elsewhere.
CS_WARP_OPS = frozenset({
    "rank", "zscore", "normalize", "winsorize", "scale",
    "group_rank", "group_zscore", "group_neutralize", "group_scale",
    "bucket", "sign",
})

_SIGN_FLIP_OPS = frozenset({"reverse", "negate", "inverse"})
_NEUTRALIZE_OPS = frozenset({"group_neutralize"})
_GROUP_FILTER_OPS = frozenset(_GROUP_ARG_INDEX)  # any op carrying a group arg


@dataclass
class Primitive:
    fields: tuple[str, ...]
    H: np.ndarray
    warp: str
    kappa_g: float
    sign: int
    coef: float
    has_trade_when: bool
    has_group_filter: bool


@dataclass
class Signature:
    primitives: tuple[Primitive, ...]
    tree: ExpressionNode
    expr: str

    @property
    def fields(self) -> tuple[str, ...]:
        out: list[str] = []
        for p in self.primitives:
            out.extend(p.fields)
        return tuple(dict.fromkeys(out))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _node_window(node: ExpressionNode) -> int | None:
    """First integer literal arg of a node -> its rolling window."""
    for a in node.args:
        if isinstance(a, ExpressionNode) and isinstance(a.literal, int):
            return a.literal
    return None


def _group_arg_position(op: str, n_args: int) -> int | None:
    idx = _GROUP_ARG_INDEX.get(op)
    if idx is None:
        return None
    return (n_args - 1) if idx == -1 else idx


def _additive_terms(node: ExpressionNode, sign: int = 1) -> list[tuple[int, ExpressionNode]]:
    """Peel top-level +/- into (coef_sign, subtree) terms."""
    op = node.operator
    if op == "add":
        out: list[tuple[int, ExpressionNode]] = []
        for a in node.args:
            if isinstance(a, ExpressionNode):
                out.extend(_additive_terms(a, sign))
        return out
    if op == "subtract" and len(node.args) == 2:
        a, b = node.args
        return _additive_terms(a, sign) + _additive_terms(b, -sign)
    if op == "negate" and len(node.args) == 1:
        return _additive_terms(node.args[0], -sign)
    return [(sign, node)]


def _term_sign(node: ExpressionNode) -> int:
    """Orientation of a (multiplicative) term, first-order."""
    if node.literal is not None:
        if not isinstance(node.literal, str):
            return -1 if node.literal < 0 else 1
        return 1
    op = node.operator
    children = [a for a in node.args if isinstance(a, ExpressionNode)]
    if op in _SIGN_FLIP_OPS and children:
        return -_term_sign(children[0])
    if op == "multiply":
        s = 1
        for c in children:
            s *= _term_sign(c)
        return s
    if op == "divide" and len(children) >= 2:
        return _term_sign(children[0]) * _term_sign(children[1])
    if children:
        return _term_sign(children[0])  # monotone ops propagate orientation
    return 1


class SignatureBuilder:
    def __init__(self, params: filters.FilterParams | None = None, kappa_g: float = 0.84):
        self.params = params or filters.DEFAULT_PARAMS
        self.kappa_g_default = kappa_g

    # ---- per-term walk ---------------------------------------------------- #
    def _walk_term(self, node: ExpressionNode):
        """Collect (fields, ts_responses, outermost_warp, kappa, flags) for a term."""
        fields: list[str] = []
        responses: list[np.ndarray] = []
        warp: str | None = None
        kappa = 1.0
        has_trade_when = False
        has_group_filter = False

        def walk(n: ExpressionNode, is_group_arg: bool):
            nonlocal warp, kappa, has_trade_when, has_group_filter
            if n.literal is not None:
                if isinstance(n.literal, str) and not is_group_arg:
                    fields.append(n.literal)
                return
            op = n.operator
            if op in CS_WARP_OPS and warp is None:
                warp = op
            if op in _NEUTRALIZE_OPS:
                kappa *= self.kappa_g_default
            if op == "trade_when":
                has_trade_when = True
            if op in _GROUP_FILTER_OPS:
                has_group_filter = True
            if filters.is_ts_operator(op):
                h = filters.response(op, _node_window(n), self.params)
                if h is not None:
                    responses.append(h)
            gpos = _group_arg_position(op, len(n.args))
            for i, a in enumerate(n.args):
                if isinstance(a, ExpressionNode):
                    walk(a, is_group_arg=(i == gpos))

        # outermost first so warp = topmost CS op
        walk(node, is_group_arg=False)
        H = filters.compose(responses)
        return (
            tuple(dict.fromkeys(fields)),
            H,
            warp or "none",
            kappa,
            has_trade_when,
            has_group_filter,
        )

    def build(self, expr) -> Signature | None:
        if isinstance(expr, ExpressionNode):
            tree = canonicalize(expr)
            expr_str = expr.render()
        else:
            try:
                tree = canonicalize(parse_expression(expr))
            except Exception:
                return None
            expr_str = expr
        prims: list[Primitive] = []
        for csign, term in _additive_terms(tree):
            fields, H, warp, kappa, tw, gf = self._walk_term(term)
            sign = csign * _term_sign(term)
            prims.append(
                Primitive(
                    fields=fields,
                    H=H,
                    warp=warp,
                    kappa_g=kappa,
                    sign=1 if sign >= 0 else -1,
                    coef=1.0,
                    has_trade_when=tw,
                    has_group_filter=gf,
                )
            )
        if not prims:
            return None
        return Signature(primitives=tuple(prims), tree=tree, expr=expr_str)


_DEFAULT_BUILDER = SignatureBuilder()


def build_signature(expr, builder: SignatureBuilder | None = None) -> Signature | None:
    return (builder or _DEFAULT_BUILDER).build(expr)
