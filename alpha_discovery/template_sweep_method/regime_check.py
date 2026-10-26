"""Predeclared regime checks for passing alpha expressions.

This lane evaluates the same expression under a fixed settings grid. It is a
robustness/regime diagnostic, not a greedy parameter optimizer.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from core.brain.submit import DEFAULT_SETTINGS, _regime_key
from alpha_enhancement.workshop.core import is_passing

REGIME_AXES = ("universe", "neutralization", "truncation", "decay")
CORE_SETTING_KEYS = ("delay", "universe", "neutralization", "truncation", "decay")

DEFAULT_DELAYS = (1,)
DEFAULT_UNIVERSES = ("TOP3000", "TOP2000", "TOP1000")
DEFAULT_NEUTRALIZATIONS = ("SUBINDUSTRY", "INDUSTRY", "SECTOR", "MARKET")
DEFAULT_TRUNCATIONS = (0.01, 0.02, 0.04)
DEFAULT_DECAYS = (0, 2, 4, 8)


def load_jsonl(paths: Iterable[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_text_grid(raw: str, default: Iterable[str]) -> list[str]:
    values = (
        [part.strip().upper() for part in raw.split(",") if part.strip()]
        if raw
        else [str(value).upper() for value in default]
    )
    return list(dict.fromkeys(values))


def parse_float_grid(raw: str, default: Iterable[float]) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()] if raw else list(default)
    return list(dict.fromkeys(values))


def parse_int_grid(raw: str, default: Iterable[int]) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()] if raw else list(default)
    return list(dict.fromkeys(values))


def known_expression_regimes(records: Iterable[dict]) -> set[tuple[str, str]]:
    known: set[tuple[str, str]] = set()
    for record in records:
        expr = str(record.get("expression") or "")
        if not expr:
            continue
        settings = _effective_settings(record)
        known.add((expr, _regime_key(settings)))
    return known


def build_regime_check_candidates(
    source_records: list[dict],
    *,
    universes: list[str] | None = None,
    delays: list[int] | None = None,
    neutralizations: list[str] | None = None,
    truncations: list[float] | None = None,
    decays: list[int] | None = None,
    mode: str = "one-axis",
    include_baseline: bool = False,
    known_pairs: set[tuple[str, str]] | None = None,
    max_alphas: int = 0,
    promote_only: bool = False,
) -> list[dict]:
    """Build a fixed regime-check plan.

    ``one-axis`` varies one settings axis at a time from the source regime.
    ``grid`` evaluates the full predeclared cross-product. Neither mode adapts
    based on interim results.
    """
    if mode not in {"one-axis", "grid"}:
        raise ValueError("mode must be 'one-axis' or 'grid'")

    known_pairs = known_pairs if known_pairs is not None else set()
    candidates: list[dict] = []
    usable_sources = _eligible_sources(source_records, promote_only=promote_only)
    if max_alphas > 0:
        usable_sources = usable_sources[:max_alphas]

    for source in usable_sources:
        expr = str(source.get("expression") or "").strip()
        if not expr:
            continue
        base_settings = _effective_settings(source)
        grids = {
            "delay": _ensure_base_value(list(delays or DEFAULT_DELAYS), int(base_settings["delay"])),
            "universe": _ensure_base_value(parse_text_grid("", universes or DEFAULT_UNIVERSES), base_settings["universe"]),
            "neutralization": _ensure_base_value(
                parse_text_grid("", neutralizations or DEFAULT_NEUTRALIZATIONS),
                base_settings["neutralization"],
            ),
            "truncation": _ensure_base_value(
                list(truncations or DEFAULT_TRUNCATIONS),
                float(base_settings["truncation"]),
            ),
            "decay": _ensure_base_value(list(decays or DEFAULT_DECAYS), int(base_settings["decay"])),
        }
        axes = _active_axes(base_settings, grids)

        variants = _one_axis_variants(base_settings, grids, axes)
        if mode == "grid":
            variants = _grid_variants(base_settings, grids, axes)
        if include_baseline:
            variants.insert(0, (dict(base_settings), []))

        for settings, changed_axes in variants:
            regime = _regime_key(settings)
            if (expr, regime) in known_pairs:
                continue
            known_pairs.add((expr, regime))
            source_id = str(source.get("alpha_id") or source.get("candidate_id") or _stable_id(expr))
            candidate = {
                "candidate_id": _stable_id(f"{expr}\n{regime}\n{source_id}"),
                "template_id": f"regime_check:{source_id}",
                "dataset_id": source.get("dataset_id", ""),
                "category": source.get("category", ""),
                "search_lane": "regime_check",
                "experiment_stage": "regime_check",
                "expression": expr,
                "fields": source.get("fields") or [],
                "group": source.get("group", ""),
                "params": {
                    "source_alpha_id": source.get("alpha_id"),
                    "source_candidate_id": source.get("candidate_id"),
                    "source_expression": expr,
                    "source_regime": _regime_key(base_settings),
                    "regime": regime,
                    "regime_check_mode": mode,
                    "changed_axes": changed_axes,
                    "predeclared_regime_check": True,
                },
                "settings_override": dict(settings),
                "skip_qtable_update": True,
                "skip_workshop": True,
            }
            candidates.append(candidate)
    return candidates


def summarize_regime_check(records: list[dict], *, min_pass_rate: float = 0.5) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        params = record.get("params") or {}
        if record.get("search_lane") != "regime_check" and not params.get("predeclared_regime_check"):
            continue
        source_expr = str(params.get("source_expression") or record.get("expression") or "")
        if source_expr:
            groups[source_expr].append(record)

    summaries: list[dict] = []
    for source_expr, rows in groups.items():
        done = [row for row in rows if row.get("status") == "done"]
        passing = [row for row in done if _standalone_passes(row)]
        best = max(done, key=_record_rank, default=None)
        pass_rate = len(passing) / len(done) if done else 0.0
        summaries.append({
            "source_expression": source_expr,
            "n_total": len(rows),
            "n_done": len(done),
            "n_pass": len(passing),
            "pass_rate": pass_rate,
            "robust": bool(done and pass_rate >= min_pass_rate),
            "best_alpha_id": best.get("alpha_id") if best else None,
            "best_regime": _record_regime(best) if best else "",
            "best_sharpe": best.get("is_sharpe") if best else None,
            "best_fitness": best.get("is_fitness") if best else None,
            "best_settings": _core_settings(_effective_settings(best)) if best else {},
        })
    summaries.sort(key=lambda row: (row["robust"], row["pass_rate"], row.get("best_fitness") or -999.0), reverse=True)
    return summaries


def write_summary_text(summaries: list[dict], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Regime check summary",
        "decision is diagnostic only; no greedy setting promotion is applied",
        "",
        f"{'#':>3} {'done':>4} {'pass':>4} {'rate':>6} {'robust':>6} {'best_fit':>8} {'best_sh':>8} {'best_regime':<45} expression",
        "-" * 130,
    ]
    for idx, row in enumerate(summaries, 1):
        best_fit = _fmt(row.get("best_fitness"))
        best_sh = _fmt(row.get("best_sharpe"))
        lines.append(
            f"{idx:>3} {row['n_done']:>4} {row['n_pass']:>4} {row['pass_rate']:>6.2f} "
            f"{str(row['robust']):>6} {best_fit:>8} {best_sh:>8} "
            f"{str(row['best_regime'])[:45]:<45} {row['source_expression'][:70]}"
        )
    out.write_text("\n".join(lines) + "\n")


def _eligible_sources(records: list[dict], *, promote_only: bool) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for record in records:
        expr = str(record.get("expression") or "").strip()
        if not expr or expr in seen:
            continue
        if promote_only and str(record.get("admission_decision") or "").upper() not in {"", "PROMOTE"}:
            continue
        seen.add(expr)
        out.append(record)
    return out


def _effective_settings(record: dict | None) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if not record:
        return settings
    stored = record.get("settings")
    if isinstance(stored, dict):
        settings.update(stored)
    override = record.get("settings_override")
    if isinstance(override, dict):
        settings.update(override)
    return settings


def _core_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {key: settings[key] for key in CORE_SETTING_KEYS if key in settings}


def _one_axis_variants(
    base_settings: dict[str, Any],
    grids: dict[str, list[Any]],
    axes: Iterable[str],
) -> list[tuple[dict[str, Any], list[str]]]:
    variants: list[tuple[dict[str, Any], list[str]]] = []
    for axis in axes:
        for value in grids[axis]:
            if _same(value, base_settings[axis]):
                continue
            settings = dict(base_settings)
            settings[axis] = value
            variants.append((settings, [axis]))
    return variants


def _grid_variants(
    base_settings: dict[str, Any],
    grids: dict[str, list[Any]],
    axes: Iterable[str],
) -> list[tuple[dict[str, Any], list[str]]]:
    axis_list = list(axes)
    variants: list[tuple[dict[str, Any], list[str]]] = []
    for values in product(*(grids[axis] for axis in axis_list)):
        settings = dict(base_settings)
        changed_axes: list[str] = []
        for axis, value in zip(axis_list, values):
            settings[axis] = value
            if not _same(value, base_settings[axis]):
                changed_axes.append(axis)
        if changed_axes:
            variants.append((settings, changed_axes))
    return variants


def _ensure_base_value(values: list[Any], base_value: Any) -> list[Any]:
    out = list(values)
    if not any(_same(value, base_value) for value in out):
        out.insert(0, base_value)
    return out


def _active_axes(base_settings: dict[str, Any], grids: dict[str, list[Any]]) -> tuple[str, ...]:
    axes = ["universe", "neutralization", "truncation", "decay"]
    if any(not _same(value, base_settings["delay"]) for value in grids["delay"]):
        axes.insert(0, "delay")
    return tuple(axes)


def _same(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return str(left).upper() == str(right).upper()


def _standalone_passes(record: dict) -> bool:
    return (
        record.get("status") == "done"
        and (record.get("is_sharpe") or 0.0) >= 1.25
        and (record.get("is_fitness") or 0.0) >= 1.0
        and is_passing(record)
    )


def _record_rank(record: dict) -> tuple[float, float, float]:
    return (
        1.0 if _standalone_passes(record) else 0.0,
        float(record.get("is_fitness") or -999.0),
        float(record.get("is_sharpe") or -999.0),
    )


def _record_regime(record: dict | None) -> str:
    if not record:
        return ""
    return _regime_key(_effective_settings(record))


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "DEFAULT_DELAYS",
    "DEFAULT_DECAYS",
    "DEFAULT_NEUTRALIZATIONS",
    "DEFAULT_TRUNCATIONS",
    "DEFAULT_UNIVERSES",
    "build_regime_check_candidates",
    "known_expression_regimes",
    "load_jsonl",
    "parse_float_grid",
    "parse_int_grid",
    "parse_text_grid",
    "summarize_regime_check",
    "write_jsonl",
    "write_summary_text",
]
