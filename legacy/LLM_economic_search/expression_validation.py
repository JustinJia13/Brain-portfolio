"""Validation and diversity gates for LLM-generated FASTEXPR candidates."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .field_catalogs import parse_target_delay
from .schemas import TheorySpec, normalize_expression, semantic_normalize_expression


_FUNC_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?")
_BANNED_OPERATORS = {
    "group_mean": (
        "group_mean has a >90% UNKNOWN/flatline rate in local history and produced "
        "zero standalone-passable candidates; use group_neutralize, group_rank, "
        "group_zscore, rank, zscore, or residual forms instead."
    ),
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
    "timing",
)
_RESERVED = {
    "subindustry",
    "industry",
    "sector",
    "market",
    "country",
    "bucket",
    "buckets",
    "true",
    "false",
    "nan",
    "inf",
    "dense",
    "filter",
}


@dataclass(frozen=True)
class FieldMeta:
    field_id: str
    dataset_id: str
    category: str
    field_name: str
    coverage: float = 0.0
    alpha_count: int = 0
    delay: int | None = None
    universe: str = ""

    @property
    def cluster_id(self) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", (self.field_name or self.field_id).lower()).strip("_")
        clean = re.sub(r"_+", "_", clean)[:72] or self.field_id
        return f"{self.dataset_id}:{clean}"


@dataclass
class AcceptedExpression:
    source: dict[str, Any]
    expression: str
    fields: list[str]
    field_roles: dict[str, str]
    operator_skeleton: list[str]
    windowless_fingerprint: str
    role_binding_fingerprint: str
    datasets: list[str]
    categories: list[str]
    cluster_ids: list[str]
    novelty_score: float
    cross_delay_retest: bool
    validation_notes: list[str] = field(default_factory=list)


@dataclass
class BatchValidation:
    accepted: list[AcceptedExpression]
    rejected: list[dict[str, Any]]
    report: dict[str, Any]


class FieldUniverse:
    def __init__(
        self,
        field_csv: Path,
        *,
        target_delay: int | None = None,
        require_delay_metadata: bool = False,
    ) -> None:
        self.field_csv = Path(field_csv)
        self.target_delay = parse_target_delay(target_delay) if target_delay is not None else None
        self.require_delay_metadata = require_delay_metadata
        self.by_id: dict[str, FieldMeta] = {}
        self.dataset_ids: set[str] = set()
        self.load()

    def load(self) -> None:
        if not self.field_csv.exists():
            return
        with self.field_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_delay = "delay" in fieldnames
            if self.target_delay is not None and self.require_delay_metadata and not has_delay:
                raise ValueError(f"Field catalog lacks delay metadata: {self.field_csv}")
            for row in reader:
                field_id = str(row.get("field_id") or "").strip()
                if not field_id:
                    continue
                row_delay = _safe_int(row.get("delay")) if has_delay else None
                if self.target_delay is not None and has_delay and row_delay != self.target_delay:
                    continue
                meta = FieldMeta(
                    field_id=field_id,
                    dataset_id=str(row.get("dataset_id") or ""),
                    category=str(row.get("category") or "").lower().replace(" ", "_"),
                    field_name=str(row.get("field_name") or ""),
                    coverage=_safe_float(row.get("coverage")),
                    alpha_count=int(_safe_float(row.get("alpha_count"))),
                    delay=row_delay,
                    universe=str(row.get("universe") or ""),
                )
                self.by_id[field_id] = meta
        self.dataset_ids = {m.dataset_id.lower() for m in self.by_id.values() if m.dataset_id}

    def compact_fields_for_prompt(self, *, limit: int = 220) -> list[dict[str, Any]]:
        rows = sorted(self.by_id.values(), key=lambda f: (f.dataset_id, -f.alpha_count, f.field_id))[:limit]
        return [_field_prompt_row(f) for f in rows]

    def search(self, query: str, *, limit: int = 24) -> list[FieldMeta]:
        tokens = [
            t
            for t in re.findall(r"[a-zA-Z0-9_]+", str(query or "").lower())
            if len(t) >= 3 and t not in {"field", "fields", "signal", "condition", "with", "and", "the", "for"}
        ]
        explicit_datasets = {t for t in tokens if t in self.dataset_ids}
        fields = [
            f
            for f in self.by_id.values()
            if not explicit_datasets or f.dataset_id.lower() in explicit_datasets
        ]
        scored: list[tuple[float, FieldMeta]] = []
        for field in fields:
            hay = f"{field.field_id} {field.field_name} {field.dataset_id} {field.category}".lower()
            score = 0.0
            for token in tokens:
                if token == field.field_id.lower():
                    score += 16
                elif token in field.field_id.lower():
                    score += 8
                elif token in field.field_name.lower():
                    score += 3
                elif token == field.dataset_id.lower():
                    score += 2
                elif token in hay:
                    score += 0.25
            if score > 0:
                score += min(field.alpha_count, 100000) / 500000.0
                score += min(field.coverage, 1.0) * 0.2
                scored.append((score, field))
        if not scored and not tokens:
            scored = [(field.alpha_count + field.coverage, field) for field in fields]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:limit]]


class ExpressionValidator:
    def __init__(
        self,
        *,
        field_csv: Path,
        target_delay: int | None,
        history_index: Any | None = None,
        require_delay_metadata: bool = True,
        min_distinct_skeletons: int = 4,
        max_window_only_variants: int = 3,
    ) -> None:
        self.target_delay = parse_target_delay(target_delay) if target_delay is not None else None
        self.fields = FieldUniverse(
            field_csv,
            target_delay=self.target_delay,
            require_delay_metadata=require_delay_metadata,
        )
        self.history_index = history_index
        self.min_distinct_skeletons = max(1, int(min_distinct_skeletons))
        self.max_window_only_variants = max(1, int(max_window_only_variants))
        self.allowed_functions = allowed_function_operators()

    def role_field_context(self, theory: TheorySpec, *, per_role_limit: int = 28) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for role, desc in (theory.data_roles or {}).items():
            out[str(role)] = [_field_prompt_row(f) for f in self.fields.search(_role_query(desc), limit=per_role_limit)]
        if not out:
            query = " ".join(theory.field_cluster_targets or [])
            out["signal"] = [_field_prompt_row(f) for f in self.fields.search(query, limit=per_role_limit)]
        return out

    def validate_batch(
        self,
        *,
        theory: TheorySpec,
        candidates: Iterable[dict[str, Any]],
        requested_count: int,
        existing_expressions: set[str] | None = None,
        cross_delay_expressions: set[str] | None = None,
    ) -> BatchValidation:
        existing = {normalize_expression(e) for e in (existing_expressions or set())}
        cross_delay = {normalize_expression(e) for e in (cross_delay_expressions or set())}
        accepted: list[AcceptedExpression] = []
        rejected: list[dict[str, Any]] = []
        seen_exprs: set[str] = set()
        window_counts: dict[str, int] = {}
        skeleton_counts: dict[str, int] = {}
        max_per_skeleton = max(1, math.ceil(max(requested_count, 1) / self.min_distinct_skeletons))

        for raw in candidates:
            result = self._validate_one(
                theory=theory,
                raw=raw,
                existing=existing,
                cross_delay=cross_delay,
            )
            if isinstance(result, dict):
                rejected.append(result)
                continue
            if result.expression in seen_exprs:
                rejected.append(_rejection(raw, "duplicate_in_llm_batch", expression=result.expression))
                continue
            if window_counts.get(result.windowless_fingerprint, 0) >= self.max_window_only_variants:
                rejected.append(_rejection(raw, "too_many_window_or_constant_variants", expression=result.expression))
                continue
            skeleton_key = ">".join(result.operator_skeleton)
            if requested_count >= self.min_distinct_skeletons and skeleton_counts.get(skeleton_key, 0) >= max_per_skeleton:
                rejected.append(_rejection(raw, "too_many_same_operator_skeleton", expression=result.expression))
                continue
            accepted.append(result)
            seen_exprs.add(result.expression)
            window_counts[result.windowless_fingerprint] = window_counts.get(result.windowless_fingerprint, 0) + 1
            skeleton_counts[skeleton_key] = skeleton_counts.get(skeleton_key, 0) + 1
            if len(accepted) >= requested_count:
                break

        required = sorted(_required_core_datasets(theory, self.fields.dataset_ids))
        covered = sorted({d.lower() for row in accepted for d in row.datasets})
        missing_required = [d for d in required if d not in covered]
        distinct_skeletons = len({tuple(a.operator_skeleton) for a in accepted})
        distinct_role_bindings = len({a.role_binding_fingerprint for a in accepted})
        needs_repair = (
            len(accepted) < requested_count
            or (len(accepted) >= self.min_distinct_skeletons and distinct_skeletons < self.min_distinct_skeletons)
            or (len(accepted) >= 2 and distinct_role_bindings < 2)
            or bool(missing_required)
        )
        report = {
            "requested_count": requested_count,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "target_delay": self.target_delay,
            "distinct_operator_skeletons": distinct_skeletons,
            "distinct_role_bindings": distinct_role_bindings,
            "required_core_datasets": required,
            "covered_datasets": covered,
            "missing_required_core_datasets": missing_required,
            "needs_repair": needs_repair,
            "repair_reasons": _repair_reasons(
                accepted_count=len(accepted),
                requested_count=requested_count,
                distinct_skeletons=distinct_skeletons,
                min_distinct_skeletons=self.min_distinct_skeletons,
                distinct_role_bindings=distinct_role_bindings,
                missing_required=missing_required,
            ),
        }
        return BatchValidation(accepted=accepted, rejected=rejected, report=report)

    def _validate_one(
        self,
        *,
        theory: TheorySpec,
        raw: dict[str, Any],
        existing: set[str],
        cross_delay: set[str],
    ) -> AcceptedExpression | dict[str, Any]:
        if not isinstance(raw, dict):
            return _rejection(raw, "candidate_is_not_object")
        expr = normalize_expression(str(raw.get("expression") or ""))
        if not expr:
            return _rejection(raw, "missing_expression")
        if expr in existing:
            return _rejection(raw, "duplicate_in_theory_history", expression=expr)
        if not _balanced_parentheses(expr):
            return _rejection(raw, "unbalanced_parentheses", expression=expr)
        functions = _FUNC_RE.findall(expr)
        banned_functions = sorted({fn for fn in functions if fn in _BANNED_OPERATORS})
        if banned_functions:
            return _rejection(
                raw,
                "banned_operator",
                expression=expr,
                details={
                    "operators": banned_functions,
                    "reasons": {op: _BANNED_OPERATORS[op] for op in banned_functions},
                },
            )
        unknown_functions = sorted({fn for fn in functions if fn not in self.allowed_functions})
        if unknown_functions:
            return _rejection(raw, "unknown_operator", expression=expr, details={"operators": unknown_functions})

        tokens = set(_IDENT_RE.findall(expr))
        fields = sorted(t for t in tokens if t not in _RESERVED and t in self.fields.by_id)
        non_field_tokens = sorted(
            t
            for t in tokens
            if t not in self.fields.by_id and t not in self.allowed_functions and t not in _RESERVED
        )
        if non_field_tokens:
            return _rejection(raw, "unknown_identifier_or_field", expression=expr, details={"identifiers": non_field_tokens})
        if not fields:
            return _rejection(raw, "no_known_fields", expression=expr)

        datasets = sorted({self.fields.by_id[f].dataset_id for f in fields if self.fields.by_id[f].dataset_id})
        categories = sorted({self.fields.by_id[f].category for f in fields if self.fields.by_id[f].category})
        missing_role_datasets = _missing_required_role_datasets(theory, fields, self.fields)
        if missing_role_datasets:
            return _rejection(
                raw,
                "missing_required_theory_dataset",
                expression=expr,
                details={"missing_by_role": missing_role_datasets},
            )

        if self.history_index and self.target_delay is not None:
            if self.history_index.expression_exists_for_delay(expr, self.target_delay) or (
                hasattr(self.history_index, "semantic_expression_exists_for_delay")
                and self.history_index.semantic_expression_exists_for_delay(expr, self.target_delay)
            ):
                return _rejection(raw, "duplicate_expression_same_delay", expression=expr)
        elif self.history_index and self.history_index.expression_exists(expr):
            return _rejection(raw, "duplicate_expression", expression=expr)

        if self.history_index and _prior_neutral_variant_exists(
            self.history_index,
            expr,
            target_delay=self.target_delay,
        ):
            return _rejection(raw, "trivial_neutral_wrapper_of_prior_expression", expression=expr)

        sign_flip_retest = bool(
            self.history_index
            and _prior_sign_flip_variant_exists(
                self.history_index,
                expr,
                target_delay=self.target_delay,
            )
        )

        skeleton = functions
        if self.history_index and self.history_index.field_skeleton_exists(fields, skeleton, target_delay=self.target_delay):
            return _rejection(raw, "field_skeleton_seen_in_history", expression=expr)

        cross_delay_retest = expr in cross_delay
        if self.history_index and self.target_delay is not None:
            if self.history_index.expression_exists_other_delay(expr, self.target_delay) or (
                hasattr(self.history_index, "semantic_expression_exists_other_delay")
                and self.history_index.semantic_expression_exists_other_delay(expr, self.target_delay)
            ):
                cross_delay_retest = True

        novelty = 1.0
        if self.history_index:
            risk = self.history_index.jaccard_risk(fields, skeleton, target_delay=self.target_delay)
            novelty = max(0.0, 1.0 - float(risk.get("max_jaccard") or 0.0))
            if risk.get("high") and not sign_flip_retest:
                return _rejection(raw, "high_field_operator_jaccard_history", expression=expr, details={"risk": risk})
        if cross_delay_retest:
            novelty = min(novelty, 0.5)
        validation_notes: list[str] = []
        if sign_flip_retest:
            novelty = min(novelty, 0.25)
            validation_notes.append("sign_flip_of_prior_expression")

        field_roles = _clean_field_roles(raw.get("field_roles") or {}, set(fields))
        cluster_ids = [self.fields.by_id[f].cluster_id for f in fields]
        return AcceptedExpression(
            source=raw,
            expression=expr,
            fields=fields,
            field_roles=field_roles,
            operator_skeleton=skeleton,
            windowless_fingerprint=windowless_fingerprint(expr),
            role_binding_fingerprint=_role_binding_fingerprint(field_roles),
            datasets=datasets,
            categories=categories,
            cluster_ids=cluster_ids,
            novelty_score=novelty,
            cross_delay_retest=cross_delay_retest,
            validation_notes=validation_notes,
        )


def allowed_function_operators() -> set[str]:
    try:
        from alpha_grammar.operators import OPERATOR_REGISTRY

        return {name for name in OPERATOR_REGISTRY if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name)}
    except Exception:
        return {
            "rank",
            "zscore",
            "winsorize",
            "ts_rank",
            "ts_zscore",
            "ts_mean",
            "ts_delta",
            "ts_delay",
            "ts_std_dev",
            "ts_corr",
            "ts_sum",
            "group_rank",
            "group_neutralize",
            "group_zscore",
            "group_scale",
            "trade_when",
            "bucket",
            "add",
            "subtract",
            "multiply",
            "divide",
            "reverse",
            "if_else",
        }


def operator_skeleton(expression: str) -> list[str]:
    return _FUNC_RE.findall(expression or "")


def windowless_fingerprint(expression: str) -> str:
    expr = normalize_expression(expression).lower()
    expr = _NUMBER_RE.sub("#", expr)
    return expr


def _prior_neutral_variant_exists(history_index: Any, expression: str, *, target_delay: int | None) -> bool:
    return _prior_variant_exists(
        history_index,
        _neutral_inner_variants(expression),
        target_delay=target_delay,
    )


def _prior_sign_flip_variant_exists(history_index: Any, expression: str, *, target_delay: int | None) -> bool:
    return _prior_variant_exists(
        history_index,
        _sign_flip_inner_variants(expression),
        target_delay=target_delay,
    )


def _prior_variant_exists(history_index: Any, variants: set[str], *, target_delay: int | None) -> bool:
    if not variants:
        return False
    for variant in variants:
        if target_delay is not None and hasattr(history_index, "semantic_expression_exists_for_delay"):
            if history_index.semantic_expression_exists_for_delay(variant, int(target_delay)):
                return True
        elif hasattr(history_index, "semantic_expression_exists"):
            if history_index.semantic_expression_exists(variant):
                return True
    return False


def _neutral_inner_variants(expression: str) -> set[str]:
    """Return inner expressions hidden by neutral wrappers."""
    return _inner_variants(expression, include_sign_flips=False)


def _sign_flip_inner_variants(expression: str) -> set[str]:
    """Return prior expressions reached by an explicit sign flip."""
    return _inner_variants(expression, include_sign_flips=True, sign_flips_only=True)


def _inner_variants(
    expression: str,
    *,
    include_sign_flips: bool,
    sign_flips_only: bool = False,
) -> set[str]:
    pending = [semantic_normalize_expression(expression)]
    seen: set[str] = set()
    variants: set[str] = set()
    while pending:
        expr = pending.pop()
        if expr in seen:
            continue
        seen.add(expr)
        call = _parse_top_level_call(expr)
        if not call:
            continue
        name, args = call
        inners: list[str] = []
        if include_sign_flips and name == "reverse" and len(args) == 1:
            inners.append(args[0])
        elif not sign_flips_only and name == "add" and len(args) == 2:
            if _is_zero(args[0]):
                inners.append(args[1])
            if _is_zero(args[1]):
                inners.append(args[0])
        elif name == "subtract" and len(args) == 2:
            if not sign_flips_only and _is_zero(args[1]):
                inners.append(args[0])
            if include_sign_flips and _is_zero(args[0]):
                inners.append(args[1])
        elif name == "multiply" and len(args) == 2:
            if not sign_flips_only and _is_one(args[0]):
                inners.append(args[1])
            if not sign_flips_only and _is_one(args[1]):
                inners.append(args[0])
            if include_sign_flips and _is_minus_one(args[0]):
                inners.append(args[1])
            if include_sign_flips and _is_minus_one(args[1]):
                inners.append(args[0])
        elif not sign_flips_only and name == "divide" and len(args) == 2 and _is_one(args[1]):
            inners.append(args[0])

        for inner in inners:
            if inner and inner not in seen:
                variants.add(inner)
                pending.append(inner)
    return variants


def _parse_top_level_call(expression: str) -> tuple[str, list[str]] | None:
    expr = semantic_normalize_expression(expression)
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(", expr)
    if not match or not expr.endswith(")"):
        return None
    name = match.group(1)
    open_pos = len(name)
    depth = 0
    for i, ch in enumerate(expr[open_pos:], open_pos):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(expr) - 1:
                return None
            if depth < 0:
                return None
    if depth != 0:
        return None
    return name, _split_top_level_args(expr[open_pos + 1 : -1])


def _split_top_level_args(value: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(value):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(value[start:i])
            start = i + 1
    args.append(value[start:])
    return [arg for arg in args if arg != ""]


def _is_zero(value: str) -> bool:
    return value in {"0", "0.0", "+0", "+0.0", "-0", "-0.0"}


def _is_one(value: str) -> bool:
    return value in {"1", "1.0", "+1", "+1.0"}


def _is_minus_one(value: str) -> bool:
    return value in {"-1", "-1.0"}


def _balanced_parentheses(expression: str) -> bool:
    depth = 0
    for ch in expression:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


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


def _field_prompt_row(field: FieldMeta) -> dict[str, Any]:
    return {
        "field_id": field.field_id,
        "dataset_id": field.dataset_id,
        "category": field.category,
        "field_name": field.field_name,
        "coverage": field.coverage,
        "alpha_count": field.alpha_count,
    }


def _role_query(desc: Any) -> str:
    if isinstance(desc, dict):
        return " ".join(str(v) for v in desc.values())
    if isinstance(desc, list):
        return " ".join(str(v) for v in desc)
    return str(desc or "")


def _required_core_datasets(theory: TheorySpec, dataset_ids: set[str]) -> set[str]:
    required: set[str] = set()
    for role, desc in (theory.data_roles or {}).items():
        if any(needle in str(role).lower() for needle in _CORE_ROLE_NEEDLES):
            required.update(_dataset_tokens(desc, dataset_ids))
    for target in theory.field_cluster_targets or []:
        required.update(_dataset_tokens(target, dataset_ids))
    return required


def _missing_required_role_datasets(theory: TheorySpec, fields: list[str], universe: FieldUniverse) -> dict[str, list[str]]:
    used_datasets = {universe.by_id[f].dataset_id.lower() for f in fields if f in universe.by_id}
    missing: dict[str, list[str]] = {}
    for role, desc in (theory.data_roles or {}).items():
        role_name = str(role)
        if not any(needle in role_name.lower() for needle in _CORE_ROLE_NEEDLES):
            continue
        required = _dataset_tokens(desc, universe.dataset_ids)
        if required and not (required & used_datasets):
            missing[role_name] = sorted(required)
    return missing


def _dataset_tokens(desc: Any, dataset_ids: set[str]) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z0-9_]+", _role_query(desc).lower()))
    return tokens & dataset_ids


def _clean_field_roles(raw_roles: Any, valid_fields: set[str]) -> dict[str, str]:
    if not isinstance(raw_roles, dict):
        return {}
    out: dict[str, str] = {}
    for role, field in raw_roles.items():
        if isinstance(field, list):
            value = next((str(f) for f in field if str(f) in valid_fields), "")
        else:
            value = str(field)
        if value in valid_fields:
            out[str(role)] = value
    return out


def _role_binding_fingerprint(field_roles: dict[str, str]) -> str:
    if not field_roles:
        return ""
    return "|".join(f"{k}:{v}" for k, v in sorted(field_roles.items()))


def _rejection(raw: Any, reason: str, *, expression: str = "", details: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {"reason": reason, "candidate": raw}
    if expression:
        row["expression"] = expression
    if details:
        row["details"] = details
    return row


def _repair_reasons(
    *,
    accepted_count: int,
    requested_count: int,
    distinct_skeletons: int,
    min_distinct_skeletons: int,
    distinct_role_bindings: int,
    missing_required: list[str],
) -> list[str]:
    reasons = []
    if accepted_count < requested_count:
        reasons.append(f"Only {accepted_count}/{requested_count} valid candidates survived validation.")
    if accepted_count >= min_distinct_skeletons and distinct_skeletons < min_distinct_skeletons:
        reasons.append(f"Only {distinct_skeletons} distinct operator skeletons; need at least {min_distinct_skeletons}.")
    if accepted_count >= 2 and distinct_role_bindings < 2:
        reasons.append("Accepted candidates do not vary field-role bindings enough.")
    if missing_required:
        reasons.append(f"Accepted candidates miss required datasets: {', '.join(missing_required)}.")
    return reasons
