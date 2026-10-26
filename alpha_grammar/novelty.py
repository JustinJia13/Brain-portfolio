"""Cluster-scoped structural novelty for generated grammar expressions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence


TESTED_FINGERPRINT_STATUSES = frozenset({"done", "no_alpha_id", "submit_failed"})

_NUMERIC_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_TOKEN_RE = re.compile(r"FIELD\d+|GROUP_LITERAL|GROUP_FIELD|[A-Za-z_]\w*|#|<=|>=|==|!=|[()+\-*/<>,=]")
_GROUP_LITERAL_RE = re.compile(r"\b(?:market|sector|industry|subindustry|none)\b")
_GROUP_OP_RE = re.compile(r"\b(?:group_rank|group_zscore|group_neutralize|group_scale|group_backfill|group_mean)\s*\(")
_NAMED_ARGS = frozenset({"k", "buckets"})


def normalized_expression(expression: str, fields: Sequence[str] | None = None) -> str:
    """Return a structural expression string with fields/windows normalized.

    Field IDs are mapped to FIELD0/FIELD1/... based on their order in the
    record, numeric constants are collapsed to "#", and literal group names are
    collapsed to GROUP_LITERAL. Remaining bare identifiers are treated as
    group-field arguments when the expression contains a group operator and as
    generic field placeholders otherwise.
    """
    expr = " ".join(str(expression or "").split())
    for i, field_id in enumerate(fields or ()):
        fid = str(field_id)
        if not fid:
            continue
        expr = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(fid)}(?![A-Za-z0-9_])", f"FIELD{i}", expr)

    expr = _GROUP_LITERAL_RE.sub("GROUP_LITERAL", expr)
    expr = _NUMERIC_RE.sub("#", expr)
    group_context = bool(_GROUP_OP_RE.search(expr))

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        end = match.end()
        if token.startswith("FIELD") and token[5:].isdigit():
            return token
        if token in {"GROUP_LITERAL", "GROUP_FIELD"}:
            return token
        if token in _NAMED_ARGS:
            return token
        if end < len(expr) and expr[end:].lstrip().startswith("("):
            return token
        if group_context:
            return "GROUP_FIELD"
        return "FIELD"

    return _IDENT_RE.sub(repl, expr)


def normalized_fingerprint(
    expression: str,
    fields: Sequence[str] | None = None,
) -> frozenset[str]:
    """Return a set fingerprint that captures normalized operator structure."""
    normalized = normalized_expression(expression, fields)
    tokens = _TOKEN_RE.findall(normalized)
    features: set[str] = {f"FULL:{normalized}"}

    for token in tokens:
        if token and (token[0].isalpha() or token.startswith("FIELD")):
            features.add(f"TOK:{token}")

    ops = [
        match.group(1)
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", normalized)
        if match.group(1) not in {"FIELD", "GROUP_FIELD"}
    ]
    for op in ops:
        features.add(f"OP:{op}")
    for left, right in zip(ops, ops[1:]):
        features.add(f"OP2:{left}>{right}")

    for n in (3, 4, 5):
        for i in range(0, max(0, len(tokens) - n + 1)):
            features.add(f"N{n}:{' '.join(tokens[i:i + n])}")

    if "bucket" in ops:
        features.add("GROUP_KIND:bucket")
    if "GROUP_LITERAL" in tokens:
        features.add("GROUP_KIND:literal")
    if "GROUP_FIELD" in tokens:
        features.add("GROUP_KIND:field")

    return frozenset(features)


def fingerprint_jaccard(fp1: frozenset[str], fp2: frozenset[str]) -> float:
    """Jaccard similarity between two normalized expression fingerprints."""
    union = fp1 | fp2
    if not union:
        return 1.0
    return len(fp1 & fp2) / len(union)


class TestedFingerprintRegistry:
    """Persisted per-primary-cluster history of tested expression fingerprints."""

    def __init__(self, by_cluster: dict[str, Iterable[frozenset[str]]] | None = None) -> None:
        self._by_cluster: dict[str, set[frozenset[str]]] = {
            str(cluster_id): set(fingerprints)
            for cluster_id, fingerprints in (by_cluster or {}).items()
        }

    def __len__(self) -> int:
        return sum(len(fingerprints) for fingerprints in self._by_cluster.values())

    def cluster_size(self, cluster_id: str) -> int:
        return len(self._by_cluster.get(str(cluster_id), set()))

    def add(self, cluster_id: str, expression: str, fields: Sequence[str] | None = None) -> bool:
        if not cluster_id or not expression:
            return False
        fp = normalized_fingerprint(expression, fields)
        bucket = self._by_cluster.setdefault(str(cluster_id), set())
        before = len(bucket)
        bucket.add(fp)
        return len(bucket) != before

    def add_record(self, record: dict) -> bool:
        if record.get("status") not in TESTED_FINGERPRINT_STATUSES:
            return False
        expression = record.get("expression") or ""
        fields = record.get("fields") or []
        cluster_ids = record.get("cluster_ids") or []
        if not expression or not fields or not cluster_ids:
            return False
        return self.add(str(cluster_ids[0]), str(expression), [str(fid) for fid in fields])

    def max_similarity(
        self,
        cluster_id: str,
        expression: str,
        fields: Sequence[str] | None = None,
    ) -> float:
        candidates = self._by_cluster.get(str(cluster_id), set())
        if not candidates:
            return 0.0
        fp = normalized_fingerprint(expression, fields)
        return max(fingerprint_jaccard(fp, prior) for prior in candidates)

    def save(self, path: str | Path) -> None:
        payload = {
            cluster_id: [sorted(fp) for fp in sorted(fingerprints, key=lambda item: sorted(item))]
            for cluster_id, fingerprints in sorted(self._by_cluster.items())
        }
        with Path(path).open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "TestedFingerprintRegistry":
        p = Path(path)
        with p.open() as f:
            raw = json.load(f)
        by_cluster = {
            str(cluster_id): {frozenset(str(feature) for feature in fp) for fp in fingerprints}
            for cluster_id, fingerprints in raw.items()
        }
        return cls(by_cluster)


def bootstrap_tested_fingerprints(
    sim_log_path: str | Path | Sequence[str | Path],
    registry: TestedFingerprintRegistry,
    *,
    lanes: tuple[str, ...] | None = ("grammar",),
) -> int:
    """Replay simulation JSONL records into ``registry``.

    Returns the number of unique fingerprints added. Only records with statuses
    in TESTED_FINGERPRINT_STATUSES are registered.
    """
    if isinstance(sim_log_path, (str, Path)):
        paths = [Path(sim_log_path)]
    else:
        paths = [Path(p) for p in sim_log_path]

    added = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if lanes is not None and record.get("search_lane") not in lanes:
                    continue
                if registry.add_record(record):
                    added += 1
    return added


__all__ = [
    "TESTED_FINGERPRINT_STATUSES",
    "TestedFingerprintRegistry",
    "bootstrap_tested_fingerprints",
    "fingerprint_jaccard",
    "normalized_expression",
    "normalized_fingerprint",
]
