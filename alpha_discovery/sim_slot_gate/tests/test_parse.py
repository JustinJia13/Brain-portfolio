"""Parser + canonicalisation tests (§9 — the parse/canonicalize portion).

Run with the worldquant interpreter:
    ${PYTHON:-python3} -m alpha_discovery/sim_slot_gate.tests.test_parse
"""

from __future__ import annotations

from core.expr.models import ExpressionNode
from alpha_discovery.sim_slot_gate.parse import parse_expression, canonicalize, parse_canonical, ParseError


def _ops(node: ExpressionNode):
    out = []
    def walk(n):
        if isinstance(n, ExpressionNode) and n.operator:
            out.append(n.operator)
            for a in n.args:
                walk(a)
    walk(node)
    return out


def test_basic_call():
    t = parse_expression("ts_mean(close, 5)")
    assert t.operator == "ts_mean"
    assert t.args[0].literal == "close"
    assert t.args[1].literal == 5


def test_infix_normalised_to_call():
    # (a) + (b) and add(a, b) must collapse to the same operator name
    a = parse_expression("(rank(close)) + (rank(open))")
    b = parse_expression("add(rank(close), rank(open))")
    assert a.operator == "add" and b.operator == "add"
    # multiply infix vs function form
    assert parse_expression("(close) * (open)").operator == "multiply"
    assert parse_expression("close / open").operator == "divide"


def test_precedence():
    # a + b * c  ->  add(a, multiply(b, c))
    t = parse_expression("close + open * volume")
    assert t.operator == "add"
    assert t.args[1].operator == "multiply"


def test_unary_minus():
    t = parse_expression("-close")
    assert t.operator == "negate" and t.args[0].literal == "close"
    # negative numeric literal folds into the literal
    t2 = parse_expression("ts_mean(close, -3)")
    assert t2.args[1].literal == -3


def test_kwargs_dropped_to_positional():
    t = parse_expression("kth_element(close, 200, k=2)")
    assert t.operator == "kth_element"
    assert [a.literal for a in t.args] == ["close", 200, 2]
    b = parse_expression("bucket(rank(close), buckets=5)")
    assert b.operator == "bucket" and b.args[1].literal == 5


def test_multistatement_variable_expansion():
    t = parse_expression("x = ts_mean(close, 5); rank(x) + rank(x)")
    # both rank(x) must be expanded to rank(ts_mean(close,5))
    assert t.operator == "add"
    for child in t.args:
        assert child.operator == "rank"
        assert child.args[0].operator == "ts_mean"


def test_program_returns_last_binding_if_no_bare_expr():
    t = parse_expression("a = rank(close); b = group_neutralize(a, industry)")
    assert t.operator == "group_neutralize"


def test_comments_stripped():
    t = parse_expression("# a comment\nrank(close)")
    assert t.operator == "rank"


def test_ternary_to_if_else():
    t = parse_expression("returns > 0 ? 1 : 0")
    assert t.operator == "if_else"
    assert t.args[0].operator == "greater"


def test_canonicalize_rank_idempotence():
    # rank(rank(x)) -> rank(x)
    t = parse_canonical("rank(rank(close))")
    assert t.operator == "rank" and t.args[0].literal == "close"
    # rank(zscore(x)) -> rank(x)
    t2 = parse_canonical("rank(zscore(close))")
    assert t2.operator == "rank" and t2.args[0].literal == "close"
    # rank(2 * x) -> rank(x)
    t3 = parse_canonical("rank(2 * close)")
    assert t3.operator == "rank" and t3.args[0].literal == "close"


def test_canonicalize_double_signflip():
    assert canonicalize(parse_expression("reverse(reverse(close))")).literal == "close"
    assert canonicalize(parse_expression("-(-close)")).literal == "close"


def test_canonicalize_commutative_order():
    a = parse_canonical("add(rank(open), rank(close))")
    b = parse_canonical("add(rank(close), rank(open))")
    assert a.render() == b.render()


def test_parse_error_on_garbage():
    for bad in ["", "   ", "((("]:
        try:
            parse_expression(bad)
            assert False, f"expected ParseError for {bad!r}"
        except ParseError:
            pass


def _run():
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} parse tests passed")


if __name__ == "__main__":
    _run()
