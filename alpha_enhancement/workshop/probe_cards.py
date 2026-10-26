"""Compact probe-card helpers for semantic workshop diagnostics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_FIELD_PROBE_KEYS: set[str] = set()


def load_probe_prior(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load cluster and field lookup rows from ``grammar_prior_from_probe.json``."""
    global _FIELD_PROBE_KEYS
    _FIELD_PROBE_KEYS = set()
    prior_path = Path(path)
    if not prior_path.exists():
        return {}
    payload = json.loads(prior_path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("cluster_priors") or []:
        cluster_id = str(row.get("cluster_id") or "")
        if cluster_id:
            out[cluster_id] = row
        field = str(row.get("field") or "")
        if field:
            _FIELD_PROBE_KEYS.add(field)
            key = f"field::{field}"
            existing = out.get(key)
            if existing is None or (_float(row.get("prior_score")) or -999.0) > (_float(existing.get("prior_score")) or -999.0):
                out[key] = row
    return out


def build_probe_card(seed: dict[str, Any], probe_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    clusters = [str(cluster) for cluster in (seed.get("cluster_ids") or []) if cluster]
    cluster_id = clusters[0] if clusters else ""
    row = probe_index.get(cluster_id)
    match_basis = "cluster"
    seed_fields = seed_fields_from_record(seed)
    if row is None:
        for field in seed_fields:
            row = probe_index.get(f"field::{field}")
            if row is not None:
                cluster_id = str(row.get("cluster_id") or cluster_id)
                match_basis = f"field:{field}"
                break
    if row is None:
        return {
            "cluster_id": cluster_id,
            "computed_trust": "missing",
            "seed_fields": seed_fields,
            "trust_reason": "no probe prior row matched seed primary cluster or fields",
        }
    card = {
        "cluster_id": cluster_id,
        "match_basis": match_basis,
        "seed_fields": seed_fields,
        "field": row.get("field"),
        "probe_family": row.get("probe_family"),
        "probe_template": row.get("probe_template"),
        "preferred_direction": row.get("preferred_direction"),
        "signal_bucket": row.get("signal_bucket"),
        "probe_quality": _float(row.get("probe_quality")),
        "prior_score": _float(row.get("prior_score")),
        "is_fitness": _float(row.get("is_fitness")),
        "is_sharpe": _float(row.get("is_sharpe")),
        "is_turnover": _float(row.get("is_turnover")),
        "max_concentration": _float(row.get("max_concentration")),
        "weak_plus_rate": _float(row.get("weak_plus_rate")),
        "best_probe_expression": row.get("expression"),
    }
    trust, reason = compute_probe_trust(seed, card)
    card["computed_trust"] = trust
    card["trust_reason"] = reason
    return card


def compute_probe_trust(seed: dict[str, Any], card: dict[str, Any]) -> tuple[str, str]:
    quality = _float(card.get("probe_quality")) or 0.0
    bucket = str(card.get("signal_bucket") or "")
    max_conc = _float(card.get("max_concentration"))
    seed_strength = max(abs(_float(seed.get("is_fitness")) or 0.0), abs(_float(seed.get("is_sharpe")) or 0.0))
    if max_conc is not None and max_conc >= 0.40:
        return "contradictory", "best probe had extreme concentration"
    if bucket == "background_noise" or (quality < 0.50 and seed_strength >= 1.0):
        return "contradictory", "probe evidence is weak relative to seed strength"

    family = str(card.get("probe_family") or "")
    seed_ops = expression_ops(str(seed.get("expression") or ""))
    family_ops = FAMILY_OPERATOR_HINTS.get(family, set())
    same_field = bool(card.get("field") and card.get("field") in set(seed_fields_from_record(seed)))
    if seed_ops & family_ops:
        if same_field:
            return "high", f"seed uses probe-family operator(s): {sorted(seed_ops & family_ops)}"
        return "medium", f"seed operator family aligns but uses a sibling field: {sorted(seed_ops & family_ops)}"
    if seed_ops & RELATED_OPERATOR_HINTS.get(family, set()):
        return "medium", "seed uses related transform family"
    return "low", "same primary cluster or field but seed expression does not match probe transform family"


def seed_fields_from_record(seed: dict[str, Any]) -> list[str]:
    fields = [str(field) for field in (seed.get("fields") or []) if field]
    if fields:
        return fields
    expr = str(seed.get("expression") or "")
    out: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr):
        if token in _FIELD_PROBE_KEYS and token not in seen:
            out.append(token)
            seen.add(token)
    return out


FAMILY_OPERATOR_HINTS: dict[str, set[str]] = {
    "level": {"rank", "zscore", "winsorize", "scale"},
    "relative_value": {"group_rank", "group_zscore", "group_neutralize", "group_scale"},
    "time_series_rank": {"ts_rank"},
    "time_series_zscore": {"ts_zscore"},
    "smooth_level": {"ts_mean", "ts_decay_linear", "ts_backfill", "group_backfill"},
    "delta_reversal": {"ts_delta", "last_diff_value", "ts_av_diff"},
    "self_residual": {"ts_regression", "ts_corr", "ts_covariance"},
}

RELATED_OPERATOR_HINTS: dict[str, set[str]] = {
    "level": {"group_rank", "group_zscore"},
    "relative_value": {"rank", "zscore", "ts_rank", "ts_zscore"},
    "time_series_rank": {"rank", "group_rank", "ts_zscore"},
    "time_series_zscore": {"zscore", "group_zscore", "ts_rank"},
    "smooth_level": {"ts_rank", "ts_zscore", "kth_element"},
    "delta_reversal": {"reverse", "ts_rank", "group_rank"},
    "self_residual": {"-", "subtract", "ts_av_diff"},
}


def expression_ops(expr: str) -> set[str]:
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
    infix = {op for op in ("+", "-", "*", "/") if op in expr}
    return tokens | infix


def compact_probe_card(card: dict[str, Any] | None) -> dict[str, Any] | None:
    if not card:
        return None
    keys = (
        "cluster_id",
        "match_basis",
        "seed_fields",
        "field",
        "probe_family",
        "probe_template",
        "preferred_direction",
        "signal_bucket",
        "probe_quality",
        "prior_score",
        "is_fitness",
        "is_sharpe",
        "is_turnover",
        "max_concentration",
        "weak_plus_rate",
        "best_probe_expression",
        "computed_trust",
        "trust_reason",
    )
    return {key: card.get(key) for key in keys if key in card}


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "build_probe_card",
    "compact_probe_card",
    "compute_probe_trust",
    "load_probe_prior",
    "seed_fields_from_record",
]
