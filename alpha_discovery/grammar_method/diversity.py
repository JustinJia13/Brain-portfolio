"""Expression tree fingerprint utilities.

DiversityTracker (Jaccard-based per-batch dedup) has been removed.
Production cross-batch novelty is now handled by
alpha_discovery/grammar_method.novelty.TestedFingerprintRegistry. Within-batch dedup uses exact
expression string matching in planner.py.

These fingerprint functions are kept as utilities for diagnostic use.
"""

from __future__ import annotations

from core.expr.models import ExpressionNode


def tree_fingerprint(tree: ExpressionNode) -> frozenset[str]:
    """Return the set of all subtree render() strings."""
    result: set[str] = set()
    _collect_subtrees(tree, result)
    return frozenset(result)


def _collect_subtrees(tree: ExpressionNode, out: set[str]) -> None:
    out.add(tree.render())
    for arg in tree.args:
        if isinstance(arg, ExpressionNode):
            _collect_subtrees(arg, out)


def fingerprint_jaccard(fp1: frozenset[str], fp2: frozenset[str]) -> float:
    """Jaccard similarity between two tree fingerprints."""
    union = fp1 | fp2
    if not union:
        return 1.0
    return len(fp1 & fp2) / len(union)


__all__ = ["tree_fingerprint", "fingerprint_jaccard"]
