"""Disabled legacy alpha-combination optimizer.

The active pipeline submits PROMOTE candidates individually after portfolio
admission review. Pair/triple expression combining, serial settings tuning,
weight tuning, and constant tuning are intentionally disabled because they are
highly exposed to in-sample selection bias.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


DEFAULT_DECAY_GRID: list[int] = []
DEFAULT_NEUTRALIZATION_GRID: list[str] = []
DEFAULT_UNIVERSE_GRID: list[str] = []
DEFAULT_TRUNCATION_GRID: list[float] = []
DEFAULT_GROUP_GRID: list[str] = []

EvaluateFn = Callable[[list[dict], str], list[dict]]


@dataclass(frozen=True)
class OptimizerConfig:
    """Compatibility config for the disabled optimizer."""

    max_seeds: int = 0
    pair_only: bool = True
    decay_grid: list[int] = field(default_factory=list)
    neutralization_grid: list[str] = field(default_factory=list)
    universe_grid: list[str] = field(default_factory=list)
    truncation_grid: list[float] = field(default_factory=list)
    group_grid: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ComboSpec:
    """Compatibility holder for legacy combo records.

    Rendering/submission is disabled; callers should review and submit the
    component alpha rows individually instead.
    """

    component_expressions: list[str]
    source_expressions: list[str]
    source_alpha_ids: list[str]
    source_deltas: list[float | None]
    source_fields: list[list[str]]
    source_datasets: list[str]
    source_categories: list[str]
    weights: list[float]
    target_delay: int
    settings_override: dict[str, Any] = field(default_factory=dict)
    parent_candidate_id: str | None = None
    optimizer_stage: str = "combo_disabled"
    tuned_axis: str = ""
    branch_root_alpha_id: str = ""
    branch_root_expression: str = ""

    def render_expression(self) -> str:
        raise RuntimeError("Legacy combo rendering is disabled; submit alphas individually.")

    def to_candidate(self, *, search_lane: str | None = None, tuned_axis: str | None = None) -> dict:
        raise RuntimeError("Legacy combo candidate generation is disabled; submit alphas individually.")

    @classmethod
    def from_record(cls, record: dict) -> "ComboSpec":
        params = record.get("params") or {}
        source_expressions = list(params.get("source_expressions") or params.get("component_expressions") or [])
        component_expressions = list(params.get("component_expressions") or source_expressions)
        return cls(
            component_expressions=component_expressions,
            source_expressions=source_expressions or component_expressions,
            source_alpha_ids=[str(value) for value in params.get("source_alpha_ids") or []],
            source_deltas=[_safe_float_or_none(value) for value in params.get("source_deltas") or []],
            source_fields=[list(value or []) for value in params.get("source_fields") or []],
            source_datasets=[str(value or "") for value in params.get("source_datasets") or []],
            source_categories=[str(value or "") for value in params.get("source_categories") or []],
            weights=[float(value) for value in params.get("weights") or []],
            target_delay=_record_delay(record),
            settings_override=dict(record.get("settings_override") or record.get("settings") or {}),
            parent_candidate_id=str(params.get("parent_candidate_id") or record.get("candidate_id") or ""),
            optimizer_stage=str(params.get("optimizer_stage") or record.get("search_lane") or "combo_disabled"),
            tuned_axis=str(params.get("tuned_axis") or ""),
            branch_root_alpha_id=str(params.get("branch_root_alpha_id") or ""),
            branch_root_expression=str(params.get("branch_root_expression") or ""),
        )


def load_seed_records(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def select_seed_buckets(records: list[dict], config: OptimizerConfig) -> dict[int, list[dict]]:
    return {}


def build_pair_candidates(seed_records: list[dict]) -> list[dict]:
    return []


def build_triple_candidates(pair_record: dict, seed_records: list[dict]) -> list[dict]:
    return []


def combo_advances(record: dict, reference_delta: float | None) -> bool:
    return False


def record_score_tuple(record: dict) -> tuple[float, float, float, float, float]:
    return (
        float(_pass_rank(record)),
        _score_value(_admission_delta(record)),
        _score_value(record.get("is_sharpe")),
        _score_value(record.get("is_fitness")),
        _score_value(record.get("brain_score_delta")),
    )


def choose_best_record(records: Iterable[dict]) -> dict | None:
    best: dict | None = None
    for record in records:
        if best is None or record_score_tuple(record) > record_score_tuple(best):
            best = record
    return best


def stage_counts(seed_buckets: dict[int, list[dict]]) -> list[tuple[int, int, int]]:
    return [(delay, len(seeds), 0) for delay, seeds in sorted(seed_buckets.items())]


def select_branch_winners(records: list[dict]) -> list[dict]:
    return []


def batched(items: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size <= 0:
        return [list(items)]
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def dedupe_candidates(candidates: list[dict]) -> list[dict]:
    return []


def match_candidate_records(candidates: list[dict], logged_records: list[dict]) -> list[dict]:
    return []


def evaluate_candidate_batches(
    candidates: list[dict],
    *,
    batch_size: int,
    submitter: Callable[[list[dict]], None],
    lookup_records: Callable[[list[dict]], list[dict]],
    annotater: Callable[[list[dict]], list[dict]],
) -> list[dict]:
    return []


def run_serial_settings_search(record: dict, evaluator: EvaluateFn, config: OptimizerConfig) -> dict:
    return record


def run_serial_weight_search(record: dict, evaluator: EvaluateFn) -> dict:
    return record


def run_serial_constant_search(record: dict, evaluator: EvaluateFn, config: OptimizerConfig) -> dict:
    return record


def optimize_delay_bucket(
    seed_records: list[dict],
    *,
    evaluator: EvaluateFn,
    config: OptimizerConfig,
) -> dict[str, list[dict]]:
    return {
        "pairs": [],
        "optimized_pairs": [],
        "branch_pairs": [],
        "triples": [],
        "promoted_triples": [],
    }


def parse_int_grid(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_float_grid(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_text_grid(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def config_from_sources(
    payload: dict | None = None,
    *,
    max_seeds: int | None = None,
    pair_only: bool | None = None,
    decay_grid: str = "",
    neutralization_grid: str = "",
    universe_grid: str = "",
    truncation_grid: str = "",
) -> OptimizerConfig:
    payload = payload or {}
    return OptimizerConfig(
        max_seeds=int(max_seeds if max_seeds is not None else payload.get("max_seeds", 0)),
        pair_only=bool(pair_only if pair_only is not None else payload.get("pair_only", True)),
        decay_grid=parse_int_grid(decay_grid) if decay_grid else [],
        neutralization_grid=parse_text_grid(neutralization_grid) if neutralization_grid else [],
        universe_grid=[value.upper() for value in parse_text_grid(universe_grid)] if universe_grid else [],
        truncation_grid=parse_float_grid(truncation_grid) if truncation_grid else [],
        group_grid=[],
    )


def _record_delay(record: dict) -> int:
    settings = record.get("settings") or record.get("settings_override") or {}
    try:
        return int(settings.get("delay", record.get("admission_delay", 1)))
    except (TypeError, ValueError):
        return 1


def _admission_delta(record: dict) -> float | None:
    value = record.get("admission_brain_score_delta")
    if value is None:
        value = record.get("admission_delta_pool_sharpe")
    return _safe_float_or_none(value)


def _pass_rank(record: dict) -> int:
    decision = str(record.get("admission_decision") or "").upper()
    if decision == "PROMOTE":
        return 2
    if decision == "WATCH" or _standalone_passing(record):
        return 1
    return 0


def _standalone_passing(record: dict) -> bool:
    checks = record.get("checks") or []
    return bool(checks) and not any(
        isinstance(check, dict)
        and check.get("result") == "FAIL"
        and check.get("name") != "SELF_CORRELATION"
        for check in checks
    )


def _score_value(value: Any) -> float:
    parsed = _safe_float_or_none(value)
    return parsed if parsed is not None else -999.0


def _safe_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ComboSpec",
    "OptimizerConfig",
    "build_pair_candidates",
    "build_triple_candidates",
    "combo_advances",
    "config_from_sources",
    "dedupe_candidates",
    "evaluate_candidate_batches",
    "load_seed_records",
    "match_candidate_records",
    "optimize_delay_bucket",
    "run_serial_constant_search",
    "run_serial_settings_search",
    "run_serial_weight_search",
    "select_seed_buckets",
    "stage_counts",
]
