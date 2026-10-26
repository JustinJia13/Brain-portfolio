"""parse.py — FastExpr string -> ExpressionNode AST, plus metric-level canonicalize.

The repo's grammar only *generates* ``ExpressionNode`` trees and renders them to
strings; there is no string->AST parser anywhere (see Discovery report).  The
redundancy metric must read the historical corpus, which stores expressions as
strings, so this module supplies a small recursive-descent parser.

Convention (matches alpha_discovery/grammar_method):
  * field / group leaf  -> ExpressionNode(literal=<str>)
  * numeric leaf        -> ExpressionNode(literal=<int|float>)
  * call / operator     -> ExpressionNode(operator=<name>, args=(...))

Normalisation applied while parsing so that structurally identical alphas written
differently collapse to one tree:
  * infix  + - * /                  -> add / subtract / multiply / divide
  * infix  < <= == != >= >          -> less / less_equal / equal / ... (rare; for
                                       trade_when / if_else conditions)
  * unary  -x                       -> negate(x)
  * named kwargs  k=N, buckets=N    -> appended as positional value args, dropping
                                       the keyword (the *value* is what the
                                       signature cares about; this also matches
                                       ExpressionNode.render's positional handling
                                       of kth_element/bucket).

Anything the parser cannot handle (ternary ``?:``, ``[]`` indexing — both <0.2 %
of the corpus and almost all junk) raises ``ParseError``; callers treat an
unparseable candidate as *simulate* (the safe direction for a sim-slot gate).
"""

from __future__ import annotations

import re
from typing import Any

from core.expr.models import ExpressionNode

try:  # operator metadata is reused where available; parser must not hard-fail without it
    from core.expr.operators import COMMUTATIVE_OPS as _COMMUTATIVE_OPS
except Exception:  # pragma: no cover - defensive
    _COMMUTATIVE_OPS = frozenset({"add", "multiply", "max", "min"})


class ParseError(ValueError):
    """Raised when an expression string cannot be parsed into an AST."""


# --------------------------------------------------------------------------- #
# Tokeniser
# --------------------------------------------------------------------------- #

# Order matters: multi-char operators before single-char.
_TOKEN_RE = re.compile(
    r"""
      (?P<WS>\s+)
    | (?P<STRING>"[^"]*"|'[^']*')
    | (?P<NUMBER>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_.]*)
    | (?P<OP><=|>=|==|!=|&&|\|\||[+\-*/<>=(),;?:])
    """,
    re.VERBOSE,
)

# strip `# ...` line comments before tokenising
_COMMENT_RE = re.compile(r"#[^\n]*")

_INFIX_NAME = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
    "<": "less",
    "<=": "less_equal",
    "==": "equal",
    "!=": "not_equal",
    ">=": "greater_equal",
    ">": "greater",
    "&&": "and",
    "||": "or",
}


def _tokenize(text: str) -> list[tuple[str, str]]:
    text = _COMMENT_RE.sub("", text)
    tokens: list[tuple[str, str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ParseError(f"unexpected character {text[pos]!r} at {pos} in {text!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "WS":
            continue
        tokens.append((kind, m.group()))
    tokens.append(("EOF", ""))
    return tokens


def _to_number(tok: str) -> int | float:
    if any(c in tok for c in ".eE"):
        return float(tok)
    return int(tok)


# --------------------------------------------------------------------------- #
# Recursive-descent parser (precedence climbing)
# --------------------------------------------------------------------------- #


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.toks = tokens
        self.i = 0
        self.env: dict[str, ExpressionNode] = {}

    def _peek(self) -> tuple[str, str]:
        return self.toks[self.i]

    def _peek2(self) -> tuple[str, str]:
        j = self.i + 1
        return self.toks[j] if j < len(self.toks) else ("EOF", "")

    def _next(self) -> tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def _expect(self, value: str) -> None:
        kind, val = self._next()
        if val != value:
            raise ParseError(f"expected {value!r} but got {val!r}")

    # program := statement (';' statement)*
    # statement := IDENT '=' expr        (variable binding; inlined on use)
    #            | expr                   (output expression)
    def parse(self) -> ExpressionNode:
        output: ExpressionNode | None = None
        last_binding: ExpressionNode | None = None
        while self._peek()[0] != "EOF":
            if self._peek()[1] == ";":
                self._next()
                continue
            # binding?  IDENT '=' (but not '==')
            if self._peek()[0] == "IDENT" and self._peek2()[1] == "=":
                name = self._next()[1]
                self._next()  # '='
                value = self._ternary()
                self.env[name] = value
                last_binding = value
            else:
                output = self._ternary()
            if self._peek()[1] == ";":
                self._next()
        node = output if output is not None else last_binding
        if node is None:
            raise ParseError("no output expression")
        if self._peek()[0] != "EOF":
            raise ParseError(f"trailing tokens starting at {self._peek()[1]!r}")
        return node

    # ternary := comparison ('?' ternary ':' ternary)?  ->  if_else(cond, a, b)
    def _ternary(self) -> ExpressionNode:
        cond = self._comparison()
        if self._peek()[1] == "?":
            self._next()
            a = self._ternary()
            self._expect(":")
            b = self._ternary()
            return ExpressionNode(operator="if_else", args=(cond, a, b))
        return cond

    def _comparison(self) -> ExpressionNode:
        node = self._additive()
        while self._peek()[1] in ("<", "<=", "==", "!=", ">=", ">"):
            op = self._next()[1]
            rhs = self._additive()
            node = ExpressionNode(operator=_INFIX_NAME[op], args=(node, rhs))
        return node

    def _additive(self) -> ExpressionNode:
        node = self._multiplicative()
        while self._peek()[1] in ("+", "-"):
            op = self._next()[1]
            rhs = self._multiplicative()
            node = ExpressionNode(operator=_INFIX_NAME[op], args=(node, rhs))
        return node

    def _multiplicative(self) -> ExpressionNode:
        node = self._unary()
        while self._peek()[1] in ("*", "/"):
            op = self._next()[1]
            rhs = self._unary()
            node = ExpressionNode(operator=_INFIX_NAME[op], args=(node, rhs))
        return node

    def _unary(self) -> ExpressionNode:
        if self._peek()[1] == "-":
            self._next()
            operand = self._unary()
            # Fold negation of a numeric literal into the literal.
            if operand.literal is not None and not isinstance(operand.literal, str):
                return ExpressionNode(literal=-operand.literal)
            return ExpressionNode(operator="negate", args=(operand,))
        if self._peek()[1] == "+":  # unary plus, harmless
            self._next()
            return self._unary()
        return self._primary()

    def _primary(self) -> ExpressionNode:
        kind, val = self._peek()
        if kind == "NUMBER":
            self._next()
            return ExpressionNode(literal=_to_number(val))
        if kind == "STRING":
            self._next()
            return ExpressionNode(literal=val.strip("\"'"))
        if val == "(":
            self._next()
            node = self._ternary()
            self._expect(")")
            return node
        if kind == "IDENT":
            self._next()
            if self._peek()[1] == "(":
                return self._call(val)
            # previously-bound variable -> inline its expansion
            if val in self.env:
                return self.env[val]
            # bare identifier -> field / group leaf
            return ExpressionNode(literal=val)
        raise ParseError(f"unexpected token {val!r}")

    def _call(self, name: str) -> ExpressionNode:
        self._expect("(")
        args: list[Any] = []
        if self._peek()[1] != ")":
            args.append(self._arg())
            while self._peek()[1] == ",":
                self._next()
                args.append(self._arg())
        self._expect(")")
        return ExpressionNode(operator=name, args=tuple(args))

    def _arg(self) -> ExpressionNode:
        # kwarg form:  IDENT = expr   (keep value only)
        if self._peek()[0] == "IDENT" and self._peek2()[1] == "=":
            self._next()  # name
            self._next()  # '='
            return self._ternary()
        return self._ternary()


def parse_expression(text: str) -> ExpressionNode:
    """Parse a FastExpr string into an ExpressionNode tree. Raises ParseError."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty expression")
    return _Parser(_tokenize(text)).parse()


# --------------------------------------------------------------------------- #
# Canonicalisation (metric-level, §3.1)
# --------------------------------------------------------------------------- #

# Cross-sectional ops under which rank is idempotent: rank(M(x)) == rank(x) when
# M is strictly monotone increasing in cross-section.
_RANK_MONOTONE = frozenset({"rank", "zscore", "normalize", "scale", "winsorize"})


def _is_positive_const(node: ExpressionNode) -> bool:
    return node.literal is not None and not isinstance(node.literal, str) and node.literal > 0


def canonicalize(node: ExpressionNode) -> ExpressionNode:
    """Canonical form for redundancy comparison (§3.1).

    * sort commutative children to a fixed order;
    * collapse rank-idempotent nests: rank(rank(x)) -> rank(x),
      rank(monotone_cs(x)) -> rank(x), rank(c*x)/rank(x*c) -> rank(x) for c>0;
    * collapse double sign flips: negate(negate(x)) -> x, reverse(reverse(x)) -> x.

    Filter-level identities (ts_mean windows etc.) are handled spectrally in the
    signature, not here.
    """
    if node.literal is not None or node.operator is None:
        return node

    new_args = tuple(
        canonicalize(a) if isinstance(a, ExpressionNode) else a for a in node.args
    )
    node = ExpressionNode(operator=node.operator, args=new_args)

    # double sign flip
    if node.operator in ("negate", "reverse") and len(node.args) == 1:
        child = node.args[0]
        if isinstance(child, ExpressionNode) and child.operator in ("negate", "reverse"):
            return child.args[0]

    # rank idempotence
    if node.operator == "rank" and node.args:
        inner = node.args[0]
        if isinstance(inner, ExpressionNode) and inner.operator in _RANK_MONOTONE:
            return canonicalize(ExpressionNode(operator="rank", args=(inner.args[0],)))
        # rank(c*x) / rank(x*c) -> rank(x)
        if isinstance(inner, ExpressionNode) and inner.operator == "multiply" and len(inner.args) == 2:
            a, b = inner.args
            if isinstance(a, ExpressionNode) and _is_positive_const(a):
                return canonicalize(ExpressionNode(operator="rank", args=(b,)))
            if isinstance(b, ExpressionNode) and _is_positive_const(b):
                return canonicalize(ExpressionNode(operator="rank", args=(a,)))

    # sort commutative children
    if node.operator in _COMMUTATIVE_OPS and len(node.args) >= 2:
        key = lambda a: a.render() if isinstance(a, ExpressionNode) else str(a)
        expr_args = [a for a in node.args if isinstance(a, ExpressionNode)]
        lit_args = [a for a in node.args if not isinstance(a, ExpressionNode)]
        node = ExpressionNode(operator=node.operator, args=tuple(sorted(expr_args, key=key)) + tuple(lit_args))

    return node


def parse_canonical(text: str) -> ExpressionNode:
    return canonicalize(parse_expression(text))
