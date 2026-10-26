"""Robustness-card helpers for semantic workshop.

The card is diagnostic.  It intentionally excludes delay perturbations and
``sign(A)`` checks.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from core.brain.submit import DEFAULT_SETTINGS, _regime_key
from alpha_enhancement.workshop.core import is_passing, record_regime


EXPRESSION_CHECKS: tuple[tuple[str, str, float], ...] = (
    ("rank_remap", "rank(({a}))", 0.50),
    ("rank_power_2", "power(rank(({a})), 2)", 0.80),
    ("rank_winsor", "rank(winsorize(({a})))", 0.50),
)

WINDOW_NEIGHBORS: dict[int, tuple[int, ...]] = {
    5: (3, 10),
    10: (5, 20),
    20: (10, 40),
    40: (20, 60),
    60: (40, 120),
    120: (60, 200),
    200: (120, 252),
    252: (200, 360),
}

ROBUST_UNIVERSES = ("TOP3000", "TOP2000", "TOP1000")
ROBUST_NEUTRALIZATIONS = ("SUBINDUSTRY", "INDUSTRY", "SECTOR")
ROBUST_DECAYS = (0, 2, 4, 8)
ROBUST_TRUNCATIONS = (0.01, 0.02, 0.04)


def load_cluster_members(path: str | Path) -> dict[str, list[str]]:
    """Return cluster_id -> member field ids from alpha_discovery/grammar_method cluster JSON."""
    cluster_path = Path(path)
    if not cluster_path.exists():
        return {}
    try:
        rows = json.loads(cluster_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, list[str]] = {}
    for row in rows if isinstance(rows, list) else []:
        cid = str(row.get("cluster_id") or "")
        members = [
            str(member.get("field_id"))
            for member in row.get("members") or []
            if member.get("field_id")
        ]
        if cid and members:
            out[cid] = members
    return out


def build_robustness_candidates(
    source_records: list[dict[str, Any]],
    *,
    root_expression: str,
    cluster_members: dict[str, list[str]] | None = None,
    known_keys: set[tuple[str, str]] | None = None,
    max_expression_variants: int | None = None,
    max_window_variants: int = 4,
    max_field_variants: int = 3,
    max_settings_variants: int | None = None,
) -> list[dict[str, Any]]:
    """Build diagnostic robustness candidates for source records."""
    known = known_keys if known_keys is not None else set()
    cluster_members = cluster_members or {}
    candidates: list[dict[str, Any]] = []
    for source in source_records:
        expr = str(source.get("expression") or "").strip()
        if not expr:
            continue
        source_id = _source_id(source)
        source_settings = _effective_settings(source)
        for idx, (check_name, template, min_ratio) in enumerate(EXPRESSION_CHECKS, 1):
            if max_expression_variants is not None and idx > max_expression_variants:
                break
            candidate_expr = template.format(a=expr)
            candidates.extend(_candidate_if_new(
                source,
                candidate_expr,
                source_settings,
                known,
                root_expression=root_expression,
                source_id=source_id,
                robust_kind="expression",
                robust_check=check_name,
                min_ratio=min_ratio,
            ))

        for idx, candidate_expr in enumerate(_window_variants(expr), 1):
            if idx > max_window_variants:
                break
            candidates.extend(_candidate_if_new(
                source,
                candidate_expr,
                source_settings,
                known,
                root_expression=root_expression,
                source_id=source_id,
                robust_kind="window",
                robust_check=f"window_perturb_{idx}",
                min_ratio=0.50,
            ))

        for idx, candidate_expr in enumerate(_field_sibling_variants(source, cluster_members), 1):
            if idx > max_field_variants:
                break
            candidates.extend(_candidate_if_new(
                source,
                candidate_expr,
                source_settings,
                known,
                root_expression=root_expression,
                source_id=source_id,
                robust_kind="field_sibling",
                robust_check=f"field_sibling_{idx}",
                min_ratio=0.40,
            ))

        for idx, (settings, axes) in enumerate(_settings_variants(source_settings), 1):
            if max_settings_variants is not None and idx > max_settings_variants:
                break
            candidates.extend(_candidate_if_new(
                source,
                expr,
                settings,
                known,
                root_expression=root_expression,
                source_id=source_id,
                robust_kind="settings",
                robust_check="+".join(axes),
                min_ratio=None,
            ))
    return candidates


def build_robustness_card(
    *,
    root_expression: str,
    source_records: list[dict[str, Any]],
    robustness_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize source and robustness simulations into a diagnostic card."""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in robustness_records:
        params = record.get("params") if isinstance(record.get("params"), dict) else {}
        if params.get("root_expression") != root_expression:
            continue
        source_expr = str(params.get("source_expression") or "")
        if source_expr:
            by_source.setdefault(source_expr, []).append(record)

    sources: list[dict[str, Any]] = []
    for source in source_records:
        source_expr = str(source.get("expression") or "")
        rows = by_source.get(source_expr, [])
        sources.append(_source_card(source, rows))

    if not sources:
        overall = "missing"
    elif all(source.get("status") == "missing" for source in sources):
        overall = "missing"
    elif any(source.get("status") == "fail" for source in sources):
        overall = "fail"
    elif any(source.get("status") == "weak" for source in sources):
        overall = "weak"
    elif any(source.get("status") == "missing" for source in sources):
        overall = "weak"
    else:
        overall = "robust"

    return {
        "root_expression": root_expression,
        "overall_status": overall,
        "sources": sources,
    }


def format_card_text(card: dict[str, Any]) -> str:
    lines = [
        f"Robustness card: {card.get('overall_status', 'missing')}",
        f"root: {card.get('root_expression', '')[:160]}",
    ]
    for idx, source in enumerate(card.get("sources") or [], 1):
        lines.append(
            f"{idx}. status={source.get('status')} expr_ret={_fmt(source.get('expression_retention_rate'))} "
            f"settings_pass={_fmt(source.get('settings_pass_rate'))} "
            f"field_pass={_fmt(source.get('field_pass_rate'))} "
            f"top1000_ret={_fmt(source.get('top1000_retention'))} "
            f"id={source.get('source_alpha_id') or '?'}"
        )
        notes = source.get("notes") or []
        for note in notes[:4]:
            lines.append(f"   - {note}")
    return "\n".join(lines)


def record_score_addition(record: dict[str, Any]) -> tuple[float | None, float | None]:
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    brain = _float_or_none(
        record.get("admission_brain_score_delta"),
        record.get("brain_score_delta"),
        admission.get("brain_score_delta"),
    )
    fallback = _float_or_none(
        record.get("admission_delta_pool_sharpe"),
        admission.get("delta_pool_sharpe"),
    )
    return brain, fallback


def final_recommendation(record: dict[str, Any], robustness_status: str | None = None) -> str:
    brain, fallback = record_score_addition(record)
    score_ok = (brain is not None and brain > 0.0) or (brain is None and fallback is not None and fallback > 0.0)
    if _standalone_passes(record) and score_ok and robustness_status in {None, "robust", "weak"}:
        return "submit"
    if record.get("status") == "done" and _standalone_passes(record):
        return "keep-watchlist"
    return "reject"


def _source_card(source: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_expr = str(source.get("expression") or "")
    source_sharpe = _float_or_none(source.get("is_sharpe"))
    expression_rows = [row for row in rows if _robust_kind(row) in {"expression", "window"} and row.get("status") == "done"]
    settings_rows = [row for row in rows if _robust_kind(row) == "settings" and row.get("status") == "done"]
    field_rows = [row for row in rows if _robust_kind(row) == "field_sibling" and row.get("status") == "done"]

    if not rows:
        return {
            "source_alpha_id": source.get("alpha_id"),
            "source_candidate_id": source.get("candidate_id"),
            "source_expression": source_expr,
            "source_sharpe": source.get("is_sharpe"),
            "source_fitness": source.get("is_fitness"),
            "expression_retention_rate": None,
            "settings_pass_rate": None,
            "field_pass_rate": None,
            "top1000_retention": None,
            "perturbation_summary": {},
            "worst_perturbation": None,
            "best_perturbation": None,
            "status": "missing",
            "n_expression_done": 0,
            "n_settings_done": 0,
            "n_field_done": 0,
            "notes": ["no robustness simulations available for this source"],
        }

    expr_retained = [_retained(row, source_sharpe) for row in expression_rows]
    expr_pass = [value for value in expr_retained if value]
    expression_rate = len(expr_pass) / len(expr_retained) if expr_retained else None

    settings_pass_rate = _pass_rate(settings_rows)
    field_pass_rate = _pass_rate(field_rows)
    top1000_retention = _top1000_retention(settings_rows, source_sharpe)
    notes: list[str] = []
    if expression_rate is not None and expression_rate < 0.50:
        notes.append("expression remaps/window perturbations retain weak signal")
    if settings_pass_rate is not None and settings_pass_rate < 0.35:
        notes.append("settings one-axis checks are fragile")
    if top1000_retention is not None and top1000_retention < 0.50:
        notes.append("TOP1000 Sharpe retention below 50%")
    if field_pass_rate is not None and field_pass_rate < 0.25:
        notes.append("same-cluster sibling replacement is weak")

    weak_votes = sum(
        value is not None and value < threshold
        for value, threshold in (
            (expression_rate, 0.50),
            (settings_pass_rate, 0.35),
            (top1000_retention, 0.50),
            (field_pass_rate, 0.25),
        )
    )
    status = "robust"
    if weak_votes >= 2:
        status = "fail"
    elif weak_votes == 1:
        status = "weak"

    return {
        "source_alpha_id": source.get("alpha_id"),
        "source_candidate_id": source.get("candidate_id"),
        "source_expression": source_expr,
        "source_sharpe": source.get("is_sharpe"),
        "source_fitness": source.get("is_fitness"),
        "expression_retention_rate": expression_rate,
        "settings_pass_rate": settings_pass_rate,
        "field_pass_rate": field_pass_rate,
        "top1000_retention": top1000_retention,
        "perturbation_summary": _perturbation_summary(rows, source_sharpe),
        "worst_perturbation": _extreme_perturbation(rows, source_sharpe, worst=True),
        "best_perturbation": _extreme_perturbation(rows, source_sharpe, worst=False),
        "status": status,
        "n_expression_done": len(expression_rows),
        "n_settings_done": len(settings_rows),
        "n_field_done": len(field_rows),
        "notes": notes,
    }


def _candidate_if_new(
    source: dict[str, Any],
    expr: str,
    settings: dict[str, Any],
    known: set[tuple[str, str]],
    *,
    root_expression: str,
    source_id: str,
    robust_kind: str,
    robust_check: str,
    min_ratio: float | None,
) -> list[dict[str, Any]]:
    regime = _regime_key(settings)
    key = (expr, regime)
    if key in known:
        return []
    known.add(key)
    cid = _stable_id(f"{source_id}\n{robust_kind}\n{robust_check}\n{regime}\n{expr}")
    params = {
        "root_expression": root_expression,
        "source_expression": source.get("expression"),
        "source_alpha_id": source.get("alpha_id"),
        "source_candidate_id": source.get("candidate_id"),
        "source_sharpe": source.get("is_sharpe"),
        "source_fitness": source.get("is_fitness"),
        "robustness_card": True,
        "robust_kind": robust_kind,
        "robust_check": robust_check,
        "robust_min_ratio": min_ratio,
    }
    return [{
        "candidate_id": cid,
        "template_id": f"robustness:{robust_kind}:{robust_check}",
        "operator_skeleton": robust_check,
        "experiment_stage": "semantic_robustness",
        "template_family": "robustness_card",
        "dataset_id": source.get("dataset_id", ""),
        "category": source.get("category", ""),
        "search_lane": "semantic_robustness",
        "expression": expr,
        "fields": source.get("fields") or [],
        "cluster_ids": source.get("cluster_ids") or [],
        "group": source.get("group", ""),
        "params": params,
        "settings_override": dict(settings),
        "skip_qtable_update": True,
        "skip_workshop": True,
    }]


def _perturbation_summary(rows: list[dict[str, Any]], source_sharpe: float | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for kind in ("expression", "window", "settings", "field_sibling"):
        kind_rows = [row for row in rows if _robust_kind(row) == kind and row.get("status") == "done"]
        ratios = [_sharpe_ratio(row, source_sharpe) for row in kind_rows]
        ratios = [ratio for ratio in ratios if ratio is not None]
        out[kind] = {
            "n_done": len(kind_rows),
            "n_pass": sum(1 for row in kind_rows if _standalone_passes(row)),
            "avg_sharpe_retention": sum(ratios) / len(ratios) if ratios else None,
            "min_sharpe_retention": min(ratios) if ratios else None,
            "max_sharpe_retention": max(ratios) if ratios else None,
        }
    return out


def _extreme_perturbation(
    rows: list[dict[str, Any]],
    source_sharpe: float | None,
    *,
    worst: bool,
) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        if row.get("status") != "done":
            continue
        ratio = _sharpe_ratio(row, source_sharpe)
        if ratio is None:
            continue
        scored.append((ratio, row))
    if not scored:
        return None
    _, row = min(scored, key=lambda item: item[0]) if worst else max(scored, key=lambda item: item[0])
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    return {
        "alpha_id": row.get("alpha_id"),
        "robust_kind": params.get("robust_kind"),
        "robust_check": params.get("robust_check"),
        "sharpe": row.get("is_sharpe"),
        "fitness": row.get("is_fitness"),
        "turnover": row.get("is_turnover"),
        "sharpe_retention": _sharpe_ratio(row, source_sharpe),
        "passes_checks": _standalone_passes(row),
        "expression": str(row.get("expression") or "")[:240],
    }


def _sharpe_ratio(row: dict[str, Any], source_sharpe: float | None) -> float | None:
    if source_sharpe is None or source_sharpe <= 0:
        return None
    sharpe = _float_or_none(row.get("is_sharpe"))
    if sharpe is None:
        return None
    return sharpe / source_sharpe


def _settings_variants(base: dict[str, Any]) -> list[tuple[dict[str, Any], list[str]]]:
    out: list[tuple[dict[str, Any], list[str]]] = []
    grids = {
        "universe": ROBUST_UNIVERSES,
        "neutralization": ROBUST_NEUTRALIZATIONS,
        "decay": ROBUST_DECAYS,
        "truncation": ROBUST_TRUNCATIONS,
    }
    for axis, values in grids.items():
        for value in values:
            if _same(value, base.get(axis)):
                continue
            settings = dict(base)
            settings[axis] = value
            out.append((settings, [axis]))
    return out


def _window_variants(expr: str) -> list[str]:
    matches = list(_window_matches(expr))
    out: list[str] = []
    seen: set[str] = set()
    for start, end, window in matches:
        for replacement in WINDOW_NEIGHBORS.get(window, ()):
            candidate = expr[:start] + str(replacement) + expr[end:]
            if candidate != expr and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def _window_matches(expr: str) -> Iterable[tuple[int, int, int]]:
    # Known lookback windows only.  Avoid decimals and k= parameters.
    pattern = re.compile(r"(?<![\w.=])(?:5|10|20|40|60|120|200|252)(?![\w.])")
    for match in pattern.finditer(expr):
        try:
            window = int(match.group())
        except ValueError:
            continue
        prefix = expr[max(0, match.start() - 3):match.start()]
        if "k=" in prefix:
            continue
        yield match.start(), match.end(), window


def _field_sibling_variants(source: dict[str, Any], cluster_members: dict[str, list[str]]) -> list[str]:
    expr = str(source.get("expression") or "")
    fields = [str(field) for field in (source.get("fields") or []) if field]
    clusters = [str(cluster) for cluster in (source.get("cluster_ids") or []) if cluster]
    if not expr or not fields or not clusters:
        return []
    primary = fields[0]
    members = cluster_members.get(clusters[0], [])
    out: list[str] = []
    for member in members:
        if member == primary:
            continue
        candidate = re.sub(rf"\b{re.escape(primary)}\b", member, expr, count=1)
        if candidate != expr:
            out.append(candidate)
    return out


def _retained(record: dict[str, Any], source_sharpe: float | None) -> bool:
    if source_sharpe is None or source_sharpe <= 0:
        return False
    sharpe = _float_or_none(record.get("is_sharpe"))
    if sharpe is None or sharpe <= 0:
        return False
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    min_ratio = _float_or_none(params.get("robust_min_ratio"))
    if min_ratio is None:
        min_ratio = 0.50
    return sharpe / source_sharpe >= min_ratio


def _pass_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if _standalone_passes(row)) / len(rows)


def _top1000_retention(rows: list[dict[str, Any]], source_sharpe: float | None) -> float | None:
    if source_sharpe is None or source_sharpe <= 0:
        return None
    values = []
    for row in rows:
        settings = row.get("settings") if isinstance(row.get("settings"), dict) else row.get("settings_override")
        if not isinstance(settings, dict):
            continue
        if str(settings.get("universe") or "").upper() != "TOP1000":
            continue
        sharpe = _float_or_none(row.get("is_sharpe"))
        if sharpe is not None:
            values.append(sharpe / source_sharpe)
    return max(values) if values else None


def _standalone_passes(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "done"
        and (_float_or_none(record.get("is_sharpe")) or -999.0) >= 1.25
        and (_float_or_none(record.get("is_fitness")) or -999.0) >= 1.0
        and is_passing(record)
    )


def _robust_kind(record: dict[str, Any]) -> str:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    return str(params.get("robust_kind") or "")


def _effective_settings(record: dict[str, Any]) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    stored = record.get("settings")
    if isinstance(stored, dict):
        settings.update(stored)
    override = record.get("settings_override")
    if isinstance(override, dict):
        settings.update(override)
    return settings


def _same(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return str(left).upper() == str(right).upper()


def _float_or_none(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _source_id(record: dict[str, Any]) -> str:
    return str(record.get("alpha_id") or record.get("candidate_id") or _stable_id(str(record.get("expression") or "")))


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "build_robustness_candidates",
    "build_robustness_card",
    "final_recommendation",
    "format_card_text",
    "load_cluster_members",
    "record_score_addition",
]
