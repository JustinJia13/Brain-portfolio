"""Core data types for the alpha_grammar pipeline.

Copied from alpha_search/models.py — alpha_grammar has no dependency on alpha_search.
Only the types actually used by this pipeline are included.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


@dataclass(frozen=True)
class DataProfile:
    """Normalized metadata for one Brain data field."""
    field_id: str
    category: str
    dataset_id: str
    dataset_name: str = ""
    subcategory_id: str = ""
    subcategory_name: str = ""
    field_type: str = ""
    description: str = ""
    coverage: float | None = None
    date_coverage: float | None = None
    user_count: int = 0
    alpha_count: int = 0
    economic_type: str = "OTHER"
    is_size_proxy: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_vector(self) -> bool:
        return self.field_type.upper() == "VECTOR"

    @property
    def cluster_label(self) -> str:
        if self.description:
            return self.description
        if self.field_id:
            return self.field_id
        return self.subcategory_name or self.subcategory_id or self.dataset_id

    @property
    def cluster_id(self) -> str:
        base = self.cluster_label or self.field_id
        economic_type = self.economic_type or "other"
        return f"{self.dataset_id}:{slugify(economic_type)}:{slugify(base)}"


@dataclass(frozen=True)
class FieldCluster:
    """A semantic similarity bucket of fields representing one field concept."""
    cluster_id: str
    category: str
    dataset_id: str
    label: str
    champion: DataProfile
    sleeper: DataProfile | None
    members: tuple[DataProfile, ...]

    def representatives(self, include_sleeper: bool = False) -> tuple[DataProfile, ...]:
        if include_sleeper and self.sleeper and self.sleeper.field_id != self.champion.field_id:
            return (self.champion, self.sleeper)
        return (self.champion,)


@dataclass(frozen=True)
class ExpressionNode:
    """Lightweight expression tree that renders to Brain FASTEXPR strings."""
    operator: str | None = None
    args: tuple[Any, ...] = ()
    literal: str | int | float | None = None

    def render(self) -> str:
        if self.literal is not None:
            return str(self.literal)
        rendered_args = []
        for arg in self.args:
            if isinstance(arg, ExpressionNode):
                rendered_args.append(arg.render())
            else:
                rendered_args.append(str(arg))
        _INFIX = {"+", "-", "*", "/", "<", "<=", "==", ">", ">=", "!="}
        if self.operator in _INFIX and len(rendered_args) == 2:
            left, right = rendered_args
            return f"({left}) {self.operator} ({right})"
        # kth_element requires k as a named parameter: kth_element(expr, window, k=N)
        if self.operator == "kth_element" and len(rendered_args) == 3:
            return f"kth_element({rendered_args[0]}, {rendered_args[1]}, k={rendered_args[2]})"
        # bucket requires buckets as a named parameter: bucket(expr, buckets=N)
        if self.operator == "bucket" and len(rendered_args) == 2:
            return f"bucket({rendered_args[0]}, buckets={rendered_args[1]})"
        return f"{self.operator}({', '.join(rendered_args)})"

    def depth(self) -> int:
        if self.literal is not None:
            return 0
        child_depths = [arg.depth() for arg in self.args if isinstance(arg, ExpressionNode)]
        return 1 + max(child_depths, default=0)


@dataclass(frozen=True)
class CandidateSpec:
    """A fully specified alpha candidate ready for submission or logging."""
    candidate_id: str
    family_id: str
    category: str
    dataset_id: str
    intention_id: str
    intention_label: str
    template_id: str
    economic_source: str
    search_lane: str
    expression: str
    fields: tuple[str, ...]
    field_types: tuple[str, ...]
    size_proxy_flags: tuple[bool, ...]
    cluster_ids: tuple[str, ...]
    group: str
    params: Mapping[str, Any]
    horizon_bucket: str
    settings_signature: str
    semantic_signature: str
    rationale_type: str
    tree_depth: int
    priority_score: float

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        for key in ("fields", "field_types", "size_proxy_flags", "cluster_ids"):
            record[key] = list(record[key])
        record["params"] = dict(record["params"])
        record["motif_id"] = self.intention_id
        record["motif_label"] = self.intention_label
        return record


__all__ = ["DataProfile", "FieldCluster", "ExpressionNode", "CandidateSpec", "slugify"]
