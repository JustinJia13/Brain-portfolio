"""Shared dataclasses for the economic hypothesis workflow."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, max_len: int = 56) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    clean = re.sub(r"_+", "_", clean)
    return (clean or "item")[:max_len].strip("_")


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]


def normalize_expression(expression: str) -> str:
    return " ".join((expression or "").split())


def semantic_normalize_expression(expression: str) -> str:
    """Normalize formatting that does not change FASTEXPR semantics."""
    return re.sub(r"\s+", "", (expression or "").lower())


def ensure_theory_id(theory: Mapping[str, Any]) -> str:
    existing = str(theory.get("theory_id") or "").strip()
    if existing:
        return slugify(existing, max_len=72)
    title = str(theory.get("title") or theory.get("mechanism") or "theory")
    return f"{slugify(title, 48)}_{stable_hash(json.dumps(dict(theory), sort_keys=True), 8)}"


@dataclass
class TheorySpec:
    theory_id: str
    title: str
    mechanism: str
    data_roles: Mapping[str, Any]
    expected_sign: str
    horizon_days: list[int]
    template_families: list[str]
    field_cluster_targets: list[str]
    novelty_claim: str
    kill_criteria: list[str]
    expand_criteria: list[str]
    max_probe_candidates: int = 24
    max_iterations: int = 2
    status: str = "PROPOSED"
    raw_experimental: bool = False
    raw_expressions: list[str] = field(default_factory=list)
    eligible_delays: list[int] = field(default_factory=lambda: [0, 1])
    delay_rationale: Mapping[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TheorySpec":
        d = dict(data)
        d["theory_id"] = ensure_theory_id(d)
        d.setdefault("title", d["theory_id"])
        d.setdefault("mechanism", "")
        d.setdefault("data_roles", {})
        d.setdefault("expected_sign", "unknown")
        d.setdefault("horizon_days", [20, 60])
        d.setdefault("template_families", ["confirm_product", "gated_trade_when"])
        d.setdefault("field_cluster_targets", [])
        d.setdefault("novelty_claim", "")
        d.setdefault("kill_criteria", [])
        d.setdefault("expand_criteria", [])
        d.setdefault("max_probe_candidates", 24)
        d.setdefault("max_iterations", 2)
        d.setdefault("status", "PROPOSED")
        d.setdefault("raw_experimental", False)
        d.setdefault("raw_expressions", [])
        d.setdefault("eligible_delays", [0, 1])
        d.setdefault("delay_rationale", {})
        d.setdefault("created_at", utc_now())
        d.setdefault("updated_at", utc_now())

        if isinstance(d["horizon_days"], int):
            d["horizon_days"] = [int(d["horizon_days"])]
        if isinstance(d["template_families"], str):
            d["template_families"] = [d["template_families"]]
        if isinstance(d["field_cluster_targets"], str):
            d["field_cluster_targets"] = [d["field_cluster_targets"]]
        if isinstance(d["kill_criteria"], str):
            d["kill_criteria"] = [d["kill_criteria"]]
        if isinstance(d["expand_criteria"], str):
            d["expand_criteria"] = [d["expand_criteria"]]
        if isinstance(d["raw_expressions"], str):
            d["raw_expressions"] = [d["raw_expressions"]]
        if isinstance(d["eligible_delays"], int):
            d["eligible_delays"] = [int(d["eligible_delays"])]
        elif isinstance(d["eligible_delays"], str):
            d["eligible_delays"] = [int(x) for x in re.findall(r"[01]", d["eligible_delays"])]
        else:
            vals = []
            for x in d["eligible_delays"]:
                try:
                    delay = int(x)
                except (TypeError, ValueError):
                    continue
                if delay in {0, 1}:
                    vals.append(delay)
            d["eligible_delays"] = vals
        if not d["eligible_delays"]:
            d["eligible_delays"] = [0, 1]

        known = set(cls.__dataclass_fields__)
        return cls(**{k: d[k] for k in known if k in d})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_status(self, status: str) -> "TheorySpec":
        data = self.to_dict()
        data["status"] = status
        data["updated_at"] = utc_now()
        return TheorySpec.from_dict(data)


@dataclass
class CandidatePlanRecord:
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
    fields: list[str]
    field_types: list[str]
    size_proxy_flags: list[bool]
    cluster_ids: list[str]
    group: str
    params: Mapping[str, Any]
    horizon_bucket: str
    settings_signature: str
    semantic_signature: str
    rationale_type: str
    tree_depth: int
    priority_score: float
    theory_id: str
    experiment_stage: str
    template_family: str
    field_roles: Mapping[str, str]
    novelty_score: float
    target_delay: int | None = None
    settings_override: Mapping[str, Any] = field(default_factory=dict)
    cross_delay_retest: bool = False

    @classmethod
    def from_expression(
        cls,
        *,
        theory_id: str,
        expression: str,
        fields: list[str],
        cluster_ids: list[str],
        dataset_id: str,
        category: str,
        template_family: str,
        template_id: str,
        field_roles: Mapping[str, str],
        experiment_stage: str,
        horizon_bucket: str,
        novelty_score: float,
        target_delay: int | None = None,
        cross_delay_retest: bool = False,
        group: str = "subindustry",
        params: Mapping[str, Any] | None = None,
    ) -> "CandidatePlanRecord":
        normalized = normalize_expression(expression)
        delay_part = f"delay{target_delay}" if target_delay is not None else "delayna"
        cid = stable_hash(f"{theory_id}|{experiment_stage}|{delay_part}|{normalized}", 16)
        settings_override = {"delay": target_delay} if target_delay is not None else {}
        return cls(
            candidate_id=cid,
            family_id=f"economic|{theory_id}|{template_family}",
            category=category,
            dataset_id=dataset_id,
            intention_id=f"theory:{theory_id}",
            intention_label=f"Economic theory search: {theory_id}",
            template_id=template_id,
            economic_source=theory_id,
            search_lane="economic_theory",
            expression=normalized,
            fields=list(fields),
            field_types=[],
            size_proxy_flags=[],
            cluster_ids=list(cluster_ids),
            group=group,
            params=dict(params or {}),
            horizon_bucket=horizon_bucket,
            settings_signature=delay_part if target_delay is not None else "",
            semantic_signature=f"{theory_id}|{delay_part}|{template_family}|{'_'.join(fields[:3])}",
            rationale_type="economic_hypothesis",
            tree_depth=max(1, normalized.count("(")),
            priority_score=novelty_score,
            theory_id=theory_id,
            experiment_stage=experiment_stage,
            template_family=template_family,
            field_roles=dict(field_roles),
            novelty_score=novelty_score,
            target_delay=target_delay,
            settings_override=settings_override,
            cross_delay_retest=cross_delay_retest,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TheoryEvent:
    theory_id: str
    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
