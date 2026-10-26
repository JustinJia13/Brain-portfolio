"""alpha_metric — sim-free pre-screen for FastExpr alpha candidates.

Two independent objects (see metric.md):
  * Object A — analytic redundancy distance ``d_red(X, Y)``  (parse/signature/aggregate)
  * Object B — fitted submission-component surrogate            (features/surrogate)
glued by ``gate.should_simulate``.

The kept/killed decision is the only deliverable; the predictions are a throwaway
filter and must never be reported as alpha quality or optimized against.
"""

from __future__ import annotations

__all__ = ["parse_expression", "canonicalize", "ParseError"]


def __getattr__(name: str):  # lazy re-export to avoid importing numpy at package import
    if name in ("parse_expression", "canonicalize", "ParseError"):
        from . import parse as _parse

        return getattr(_parse, name)
    raise AttributeError(name)
