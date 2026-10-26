"""Deterministic candidate expansion from accepted economic theories."""

from __future__ import annotations

import csv
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .field_catalogs import parse_target_delay
from .history_index import DEFAULT_FIELD_CSV, HistoryIndex
from .schemas import CandidatePlanRecord, TheorySpec, normalize_expression, stable_hash


_FUNC_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

_STOP_TOKENS = {
    "field",
    "fields",
    "signal",
    "condition",
    "data",
    "with",
    "from",
    "over",
    "under",
    "where",
    "when",
    "and",
    "or",
    "the",
    "for",
    "ratio",
    "scaled",
    "scale",
    "using",
    "capturing",
    "captures",
    "proxy",
    "proxies",
    "filter",
    "gate",
    "context",
}

_CORE_ROLE_NEEDLES = (
    "signal",
    "primary",
    "alpha",
    "condition",
    "confirm",
    "confirmation",
    "gate",
    "context",
    "anchor",
)

_SIGNAL_ROLE_NEEDLES = ("signal", "primary", "alpha")

_CONDITION_ROLE_NEEDLES = (
    "condition",
    "confirm",
    "confirmation",
    "gate",
    "context",
    "anchor",
    "timing",
    "filter",
)

_DEFAULT_EXPLORATION_FAMILIES = [
    "confirm_product",
    "gated_trade_when",
    "bucket_condition",
    "residual_spread",
    "reversal_condition",
    "delta_confirmation",
    "delta_spread",
    "ratio_level",
    "ratio_change",
    "volatility_scaled",
    "uncertainty_reversal",
    "rank_blend",
    "correlation_alignment",
]


@dataclass(frozen=True)
class RawCandidate:
    expression: str
    fields: list[str]
    family: str
    roles: dict[str, str]
    audit: dict[str, Any]


@dataclass(frozen=True)
class FieldChoice:
    field_id: str
    dataset_id: str
    category: str
    field_name: str
    coverage: float
    alpha_count: int
    cluster_id: str
    delay: int | None = None
    universe: str = ""


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", clean)[:72] or "unknown"


class FieldCatalog:
    def __init__(
        self,
        csv_path: Path = DEFAULT_FIELD_CSV,
        *,
        target_delay: int | None = None,
        require_delay_metadata: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.target_delay = parse_target_delay(target_delay) if target_delay is not None else None
        self.require_delay_metadata = require_delay_metadata
        self.fields: list[FieldChoice] = []
        self.dataset_ids: set[str] = set()
        self.load()

    def load(self) -> None:
        if not self.csv_path.exists():
            return
        with self.csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            has_delay = "delay" in (reader.fieldnames or [])
            if self.target_delay is not None and self.require_delay_metadata and not has_delay:
                raise ValueError(f"Field catalog lacks delay metadata: {self.csv_path}")
            for row in reader:
                field_id = row.get("field_id", "")
                if not field_id:
                    continue
                row_delay = _safe_int(row.get("delay")) if has_delay else None
                if self.target_delay is not None and has_delay and row_delay != self.target_delay:
                    continue
                dataset_id = row.get("dataset_id", "")
                field_name = row.get("field_name", "")
                category = (row.get("category") or "").lower().replace(" ", "_")
                self.fields.append(
                    FieldChoice(
                        field_id=field_id,
                        dataset_id=dataset_id,
                        category=category,
                        field_name=field_name,
                        coverage=_safe_float(row.get("coverage")),
                        alpha_count=int(_safe_float(row.get("alpha_count"))),
                        cluster_id=f"{dataset_id}:{_slug(field_name or field_id)}",
                        delay=row_delay,
                        universe=str(row.get("universe") or ""),
                    )
                )
        self.dataset_ids = {f.dataset_id.lower() for f in self.fields if f.dataset_id}

    def search(self, query: str, limit: int = 6) -> list[FieldChoice]:
        query = query or ""
        tokens = [
            t
            for t in re.findall(r"[a-zA-Z0-9_]+", query.lower())
            if len(t) >= 3 and t not in _STOP_TOKENS
        ]
        explicit_datasets = {t for t in tokens if t in self.dataset_ids}
        fields = [
            field
            for field in self.fields
            if not explicit_datasets or field.dataset_id.lower() in explicit_datasets
        ]
        if explicit_datasets and not fields:
            return []
        if not tokens:
            return sorted(fields, key=lambda f: (f.alpha_count, f.coverage), reverse=True)[:limit]

        scored: list[tuple[float, FieldChoice]] = []
        for field in fields:
            hay = f"{field.field_id} {field.field_name} {field.dataset_id} {field.category}".lower()
            score = 0.0
            field_id = field.field_id.lower()
            field_name = field.field_name.lower()
            dataset_id = field.dataset_id.lower()
            category = field.category.lower()
            for token in tokens:
                if token == field_id:
                    score += 14.0
                elif token in field_id:
                    score += 7.0
                elif token in field_name:
                    score += 3.0
                elif token == dataset_id:
                    score += 2.0
                elif token == category:
                    score += 1.0
                elif token in hay:
                    score += 0.25
            if score <= 0:
                continue
            score += min(field.alpha_count, 100000) / 400000.0
            score += min(field.coverage, 1.0) * 0.25
            scored.append((score, field))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:limit]]


class CandidateExpander:
    def __init__(
        self,
        *,
        field_csv: Path = DEFAULT_FIELD_CSV,
        history_index: HistoryIndex | None = None,
        target_delay: int | None = None,
        require_delay_metadata: bool = False,
    ) -> None:
        self.target_delay = parse_target_delay(target_delay) if target_delay is not None else None
        self.catalog = FieldCatalog(
            field_csv,
            target_delay=self.target_delay,
            require_delay_metadata=require_delay_metadata,
        )
        self.history_index = history_index

    def generate(
        self,
        theory: TheorySpec,
        *,
        stage: str = "probe",
        max_candidates: int | None = None,
        existing_theory_expressions: set[str] | None = None,
        cross_delay_expressions: set[str] | None = None,
        target_delay: int | None = None,
    ) -> list[CandidatePlanRecord]:
        delay = self.target_delay if target_delay is None else parse_target_delay(target_delay)
        if delay is not None and delay not in theory.eligible_delays:
            return []
        cap = max_candidates or theory.max_probe_candidates
        cap = min(max(cap, 1), 60)
        existing = existing_theory_expressions or set()
        cross_delay = cross_delay_expressions or set()

        role_fields = self._select_role_fields(theory)
        if self._missing_core_role(theory, role_fields):
            return []
        raw_candidates = self._raw_candidates(theory, stage, cap, role_fields)
        records: list[CandidatePlanRecord] = []
        seen: set[str] = set()

        for raw in raw_candidates:
            expr, fields, family, roles = raw.expression, raw.fields, raw.family, raw.roles
            expr = normalize_expression(expr)
            if expr in seen or expr in existing:
                continue
            if self.history_index and delay is not None and self.history_index.expression_exists_for_delay(expr, delay):
                continue
            if self.history_index and delay is None and self.history_index.expression_exists(expr):
                continue
            skeleton = _FUNC_RE.findall(expr)
            if self.history_index and self.history_index.field_skeleton_exists(fields, skeleton, target_delay=delay):
                continue
            is_cross_delay_retest = expr in cross_delay
            if self.history_index and delay is not None and self.history_index.expression_exists_other_delay(expr, delay):
                is_cross_delay_retest = True
            if self.history_index:
                risk = self.history_index.jaccard_risk(fields, skeleton, target_delay=delay)
                novelty = max(0.0, 1.0 - float(risk.get("max_jaccard") or 0.0))
                if risk.get("high"):
                    continue
            else:
                novelty = 1.0
            if is_cross_delay_retest:
                novelty = min(novelty, 0.5)

            field_objs = [self._field_by_id(fid) for fid in fields]
            first = next((f for f in field_objs if f is not None), None)
            clusters = [f.cluster_id for f in field_objs if f is not None]
            dataset_id = first.dataset_id if first else ""
            category = first.category if first else ""
            horizon = _horizon_bucket(theory.horizon_days)
            record = CandidatePlanRecord.from_expression(
                theory_id=theory.theory_id,
                expression=expr,
                fields=fields,
                cluster_ids=clusters,
                dataset_id=dataset_id,
                category=category,
                template_family=family,
                template_id="|".join(skeleton) or family,
                field_roles=roles,
                experiment_stage=stage,
                horizon_bucket=horizon,
                novelty_score=novelty,
                target_delay=delay,
                cross_delay_retest=is_cross_delay_retest,
                params={
                    "theory_title": theory.title,
                    "expected_sign": theory.expected_sign,
                    "template_family": family,
                    "target_delay": delay,
                    "cross_delay_retest": is_cross_delay_retest,
                    "translation_audit": raw.audit,
                },
            )
            records.append(record)
            seen.add(expr)
            if len(records) >= cap:
                break
        return records

    def _field_by_id(self, field_id: str) -> FieldChoice | None:
        for field in self.catalog.fields:
            if field.field_id == field_id:
                return field
        return None

    def _select_role_fields(self, theory: TheorySpec) -> dict[str, list[FieldChoice]]:
        role_fields: dict[str, list[FieldChoice]] = {}
        roles = dict(theory.data_roles or {})
        if not roles:
            roles = {"signal": " ".join(theory.field_cluster_targets), "condition": "price volume returns"}
        for role, desc in roles.items():
            if isinstance(desc, dict):
                query = " ".join(str(v) for v in desc.values())
            elif isinstance(desc, list):
                query = " ".join(str(v) for v in desc)
            else:
                query = str(desc)
            role_fields[str(role)] = self.catalog.search(query, limit=6)

        if len(role_fields) == 1:
            key = next(iter(role_fields))
            role_fields["condition"] = self.catalog.search("price volume returns volatility", limit=6)
            if key == "condition":
                role_fields["signal"] = self.catalog.search("model analyst fundamental signal", limit=6)
        return role_fields

    def _missing_core_role(self, theory: TheorySpec, role_fields: dict[str, list[FieldChoice]]) -> bool:
        for role, desc in (theory.data_roles or {}).items():
            role_name = str(role)
            if not _is_core_role(role_name):
                continue
            explicit = _explicit_dataset_tokens(desc, self.catalog.dataset_ids)
            if explicit and not role_fields.get(role_name):
                return True
        return False

    def _raw_candidates(
        self,
        theory: TheorySpec,
        stage: str,
        cap: int,
        role_fields: dict[str, list[FieldChoice]],
    ) -> Iterable[RawCandidate]:
        windows = _windows(theory.horizon_days)
        families = _expression_families(theory)
        role_pairs = _role_pairs(role_fields)

        yielded = 0
        streams = [
            iter(self._family_candidates(theory, family, role_pairs, windows))
            for family in families
        ]
        while streams and yielded < cap * 6:
            next_streams = []
            for stream in streams:
                try:
                    candidate = next(stream)
                except StopIteration:
                    continue
                yielded += 1
                yield candidate
                if yielded >= cap * 6:
                    return
                next_streams.append(stream)
            streams = next_streams

        if theory.raw_experimental:
            for expr in theory.raw_expressions[:5]:
                fields = _extract_raw_fields(expr, [f.field_id for f in self.catalog.fields])
                if fields:
                    yield RawCandidate(
                        expression=expr,
                        fields=fields[:4],
                        family="raw_experimental",
                        roles={"raw": ",".join(fields[:4])},
                        audit={
                            "source": "llm_raw_experimental",
                            "quality_score": 0.5,
                            "notes": ["Raw LLM expression accepted after field extraction."],
                        },
                    )

    def _family_candidates(
        self,
        theory: TheorySpec,
        family: str,
        role_pairs: list[tuple[str, str, list[FieldChoice], list[FieldChoice]]],
        windows: list[int],
    ) -> Iterable[RawCandidate]:
        requested = family in {f.lower() for f in (theory.template_families or [])}
        for w, (signal_role, condition_role, signals, conditions) in itertools.product(windows, role_pairs):
            for a, b in _field_pairs(signals, conditions, limit=10):
                expr = _render_family(family, a.field_id, b.field_id, w)
                if not expr:
                    continue
                roles = {signal_role: a.field_id, condition_role: b.field_id}
                fields = _dedupe([a.field_id, b.field_id])
                yield RawCandidate(
                    expression=expr,
                    fields=fields,
                    family=family,
                    roles=roles,
                    audit=_translation_audit(
                        theory=theory,
                        family=family,
                        requested_family=requested,
                        role_fields=roles,
                        selected_fields={a.field_id: a, b.field_id: b},
                        all_dataset_ids=self.catalog.dataset_ids,
                    ),
                )


def _first_role(roles: list[str], needles: tuple[str, ...]) -> str | None:
    for role in roles:
        low = role.lower()
        if any(n in low for n in needles):
            return role
    return None


def _is_core_role(role: str) -> bool:
    low = role.lower()
    return any(needle in low for needle in _CORE_ROLE_NEEDLES)


def _explicit_dataset_tokens(desc: Any, dataset_ids: set[str]) -> set[str]:
    if isinstance(desc, dict):
        text = " ".join(str(v) for v in desc.values())
    elif isinstance(desc, list):
        text = " ".join(str(v) for v in desc)
    else:
        text = str(desc)
    tokens = set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    return tokens & dataset_ids


def _expression_families(theory: TheorySpec) -> list[str]:
    requested = [f.lower() for f in (theory.template_families or []) if str(f).strip()]
    families = _dedupe(requested + _DEFAULT_EXPLORATION_FAMILIES)
    return [f for f in families if _family_supported(f)]


def _family_supported(family: str) -> bool:
    return bool(_render_family(family, "a", "b", 20))


def _role_pairs(
    role_fields: dict[str, list[FieldChoice]],
) -> list[tuple[str, str, list[FieldChoice], list[FieldChoice]]]:
    roles = [role for role, fields in role_fields.items() if fields]
    if not roles:
        return []
    signal_role = _first_role(roles, _SIGNAL_ROLE_NEEDLES) or roles[0]
    condition_roles = [
        role
        for role in roles
        if role != signal_role and any(needle in role.lower() for needle in _CONDITION_ROLE_NEEDLES)
    ]
    if not condition_roles:
        condition_roles = [role for role in roles if role != signal_role]
    if not condition_roles:
        condition_roles = [signal_role]

    pairs = []
    for condition_role in condition_roles[:4]:
        signals = role_fields.get(signal_role, [])[:5]
        conditions = role_fields.get(condition_role, [])[:5]
        if signals and conditions:
            pairs.append((signal_role, condition_role, signals, conditions))

    # If a theory exposes several non-primary roles, sample a few secondary interactions too.
    secondary_roles = [role for role in roles if role != signal_role]
    for left, right in itertools.combinations(secondary_roles[:4], 2):
        left_fields = role_fields.get(left, [])[:4]
        right_fields = role_fields.get(right, [])[:4]
        if left_fields and right_fields:
            pairs.append((left, right, left_fields, right_fields))
    return pairs


def _field_pairs(
    signals: list[FieldChoice],
    conditions: list[FieldChoice],
    *,
    limit: int,
) -> Iterable[tuple[FieldChoice, FieldChoice]]:
    seen: set[tuple[str, str]] = set()
    for rank_sum in range(len(signals) + len(conditions) - 1):
        for i in range(min(rank_sum, len(signals) - 1) + 1):
            j = rank_sum - i
            if j >= len(conditions):
                continue
            a, b = signals[i], conditions[j]
            key = (a.field_id, b.field_id)
            if a.field_id == b.field_id or key in seen:
                continue
            seen.add(key)
            yield a, b
            if len(seen) >= limit:
                return


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _translation_audit(
    *,
    theory: TheorySpec,
    family: str,
    requested_family: bool,
    role_fields: dict[str, str],
    selected_fields: dict[str, FieldChoice],
    all_dataset_ids: set[str],
) -> dict[str, Any]:
    missing_required: dict[str, list[str]] = {}
    matched_required: dict[str, list[str]] = {}
    selected_by_role = {}
    for role, field_id in role_fields.items():
        field = selected_fields.get(field_id)
        if not field:
            continue
        selected_by_role[role] = {
            "field_id": field.field_id,
            "dataset_id": field.dataset_id,
            "cluster_id": field.cluster_id,
        }
        required = _explicit_dataset_tokens((theory.data_roles or {}).get(role, ""), all_dataset_ids)
        if required:
            if field.dataset_id.lower() in required:
                matched_required[role] = sorted(required)
            else:
                missing_required[role] = sorted(required)

    quality = 1.0
    if missing_required:
        quality -= 0.35
    if not requested_family:
        quality -= 0.05
    return {
        "family": family,
        "family_source": "theory_requested" if requested_family else "exploration_added",
        "selected_roles": selected_by_role,
        "matched_required_datasets": matched_required,
        "missing_required_datasets": missing_required,
        "quality_score": max(0.0, round(quality, 3)),
    }


def _windows(horizon_days: list[int]) -> list[int]:
    vals = [int(v) for v in horizon_days if int(v) > 0]
    if not vals:
        vals = [20, 60]
    vals = sorted(set(vals))
    return vals[:3]


def _horizon_bucket(horizon_days: list[int]) -> str:
    med = sorted(horizon_days or [60])[len(horizon_days or [60]) // 2]
    if med <= 20:
        return "short"
    if med <= 120:
        return "medium"
    return "long"


def _render_family(family: str, a: str, b: str, w: int) -> str:
    fam = family.lower()
    slow = max(w, 20)
    fast = min(w, 20)
    if fam in {"confirm_product", "interaction", "confirmation"}:
        return (
            f"group_neutralize(multiply(rank(ts_zscore({a}, {slow})), "
            f"rank(ts_zscore({b}, {fast}))), subindustry)"
        )
    if fam in {"gated_trade_when", "conditional_gate", "gate"}:
        return (
            f"trade_when(group_neutralize(rank(ts_zscore({a}, {slow})), subindustry), "
            f"rank(ts_zscore({b}, {fast})), 0)"
        )
    if fam in {"bucket_condition", "bucketed_condition"}:
        return f"group_rank(ts_zscore({a}, {slow}), bucket(rank({b}), buckets=5))"
    if fam in {"residual_spread", "disagreement", "model_price_disagreement"}:
        return (
            f"group_neutralize(subtract(rank(ts_zscore({a}, {slow})), "
            f"rank(ts_zscore({b}, {fast}))), subindustry)"
        )
    if fam in {"additive_blend", "blend"}:
        return f"group_neutralize(add(rank(ts_zscore({a}, {slow})), rank(ts_zscore({b}, {fast}))), subindustry)"
    if fam in {"reversal_condition", "reversal"}:
        return (
            f"group_neutralize(multiply(rank(ts_zscore({a}, {fast})), "
            f"rank(reverse(ts_delta({b}, {fast})))), subindustry)"
        )
    if fam in {"vol_condition", "uncertainty_condition"}:
        return (
            f"group_neutralize(multiply(rank(ts_zscore({a}, {slow})), "
            f"rank(reverse(ts_rank({b}, {slow})))), subindustry)"
        )
    if fam in {"delta_confirmation", "change_confirmation"}:
        return (
            f"group_neutralize(multiply(rank(ts_delta({a}, {fast})), "
            f"rank(ts_zscore({b}, {slow}))), subindustry)"
        )
    if fam in {"delta_spread", "change_spread"}:
        return (
            f"group_neutralize(subtract(rank(ts_delta({a}, {fast})), "
            f"rank(ts_delta({b}, {fast}))), subindustry)"
        )
    if fam in {"ratio_level", "relative_ratio"}:
        return f"group_neutralize(rank(ts_zscore(divide({a}, {b}), {slow})), subindustry)"
    if fam in {"ratio_change", "relative_ratio_change"}:
        return f"group_neutralize(rank(ts_delta(divide({a}, {b}), {fast})), subindustry)"
    if fam in {"volatility_scaled", "vol_scaled"}:
        return (
            f"group_neutralize(multiply(rank(ts_zscore({a}, {slow})), "
            f"rank(ts_std_dev({b}, {slow}))), subindustry)"
        )
    if fam in {"uncertainty_reversal", "vol_reversal"}:
        return (
            f"group_neutralize(multiply(rank(reverse(ts_delta({a}, {fast}))), "
            f"rank(ts_std_dev({b}, {slow}))), subindustry)"
        )
    if fam in {"rank_blend", "additive_rank_blend"}:
        return f"group_neutralize(add(rank(ts_zscore({a}, {slow})), rank(ts_zscore({b}, {fast}))), subindustry)"
    if fam in {"correlation_alignment", "ts_correlation"}:
        return f"group_neutralize(rank(ts_corr({a}, {b}, {slow})), subindustry)"
    return ""


def _extract_raw_fields(expression: str, field_ids: list[str]) -> list[str]:
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression or ""))
    return [fid for fid in field_ids if fid in tokens]


def write_combined_plan(path: Path, candidates: list[CandidatePlanRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for candidate in candidates:
            f.write(__import__("json").dumps(candidate.to_dict(), sort_keys=True) + "\n")


def plan_name(prefix: str = "economic_plan") -> str:
    from datetime import datetime

    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
