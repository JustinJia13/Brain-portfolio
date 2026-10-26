"""Analyze probe simulation logs into seed and prior summaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PASSING_IGNORED_CHECKS = {"SELF_CORRELATION", "UNITS"}
REPAIRABLE_CHECKS = {"CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize field probe simulation results.")
    p.add_argument("--sim-log", default="artifacts/probe/probe_sim.jsonl")
    p.add_argument("--out-summary", default="artifacts/probe/probe_summary.json")
    p.add_argument("--out-seeds", default="artifacts/probe/probe_seeds.jsonl")
    p.add_argument("--min-weak-abs-sharpe", type=float, default=0.70)
    p.add_argument("--min-weak-abs-fitness", type=float, default=0.50)
    p.add_argument("--min-near-fitness", type=float, default=0.80)
    p.add_argument("--min-neg-abs-fitness", type=float, default=1.00)
    return p.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _failing_checks(record: dict[str, Any]) -> list[str]:
    checks = record.get("checks") or []
    return [
        str(check.get("name"))
        for check in checks
        if check.get("result") != "PASS"
        and check.get("name") not in PASSING_IGNORED_CHECKS
    ]


def _checks_pass(record: dict[str, Any]) -> bool:
    checks = record.get("checks") or []
    if checks:
        return not _failing_checks(record)
    return str(record.get("grade") or "").upper() in {"PASS", "AVERAGE", "GOOD", "EXCELLENT"}


def _float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(record.get(key) or default)
    except (TypeError, ValueError):
        return default


def _max_concentration(record: dict[str, Any]) -> float | None:
    value = record.get("max_concentration")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    for check in record.get("checks") or []:
        if check.get("name") == "CONCENTRATED_WEIGHT" and check.get("value") is not None:
            try:
                return float(check["value"])
            except (TypeError, ValueError):
                return None
    return None


def _extreme_concentration(record: dict[str, Any]) -> bool:
    value = _max_concentration(record)
    return value is not None and value >= 0.40


def _classify(record: dict[str, Any], args: argparse.Namespace) -> str:
    if record.get("status") != "done":
        return "not_done"
    sharpe = _float(record, "is_sharpe")
    fitness = _float(record, "is_fitness")
    passing = _checks_pass(record)
    failing = set(_failing_checks(record))
    if passing and sharpe >= 1.25 and fitness >= 1.0:
        return "elite"
    if fitness <= -args.min_neg_abs_fitness and not _extreme_concentration(record):
        return "negate_seed"
    if (
        fitness >= args.min_near_fitness
        and failing
        and failing <= REPAIRABLE_CHECKS
        and not _extreme_concentration(record)
    ):
        return "near_repair"
    if (
        max(abs(sharpe), 0.0) >= args.min_weak_abs_sharpe
        or max(abs(fitness), 0.0) >= args.min_weak_abs_fitness
    ) and not _extreme_concentration(record):
        return "weak_signal"
    return "reject"


def _score(record: dict[str, Any], bucket: str) -> float:
    sharpe = _float(record, "is_sharpe")
    fitness = _float(record, "is_fitness")
    base = abs(fitness) + 0.35 * abs(sharpe)
    if bucket == "elite":
        base += 3.0
    elif bucket == "near_repair":
        base += 1.5
    elif bucket == "negate_seed":
        base += 1.0
    elif bucket == "weak_signal":
        base += 0.25
    if not bool(record.get("is_sub_universe_sharpe_ok", True)):
        base *= 0.75
    if _extreme_concentration(record):
        base = 0.0
    return round(base, 6)


def _seed_row(record: dict[str, Any], bucket: str) -> dict[str, Any]:
    params = record.get("params") or {}
    return {
        "bucket": bucket,
        "score": _score(record, bucket),
        "alpha_id": record.get("alpha_id"),
        "expression": record.get("expression"),
        "dataset_id": record.get("dataset_id"),
        "category": record.get("category"),
        "fields": record.get("fields") or [],
        "cluster_ids": record.get("cluster_ids") or [],
        "probe_template": params.get("probe_template") or record.get("operator_skeleton"),
        "probe_family": params.get("probe_family") or record.get("template_family"),
        "probe_horizon": params.get("probe_horizon"),
        "probe_direction": params.get("probe_direction"),
        "is_sharpe": record.get("is_sharpe"),
        "is_fitness": record.get("is_fitness"),
        "is_turnover": record.get("is_turnover"),
        "grade": record.get("grade"),
        "failing_checks": _failing_checks(record),
        "max_self_correlation": record.get("max_self_correlation"),
        "max_concentration": _max_concentration(record),
    }


def analyze(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bucket_counts: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()
    by_template: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seeds: list[dict[str, Any]] = []

    for record in rows:
        bucket = _classify(record, args)
        bucket_counts[bucket] += 1
        if bucket in {"elite", "near_repair", "negate_seed", "weak_signal"}:
            seed = _seed_row(record, bucket)
            seeds.append(seed)
            by_dataset[str(seed.get("dataset_id") or "")] += 1
            by_template[str(seed.get("probe_template") or "")] += 1
            by_family[str(seed.get("probe_family") or "")] += 1
            for field in seed.get("fields") or []:
                by_field[str(field)].append(seed)

    seeds.sort(key=lambda row: row["score"], reverse=True)
    field_summary = []
    for field, field_seeds in by_field.items():
        field_seeds.sort(key=lambda row: row["score"], reverse=True)
        field_summary.append(
            {
                "field": field,
                "n_seeds": len(field_seeds),
                "best_score": field_seeds[0]["score"],
                "best_bucket": field_seeds[0]["bucket"],
                "best_template": field_seeds[0]["probe_template"],
                "dataset_id": field_seeds[0]["dataset_id"],
            }
        )
    field_summary.sort(key=lambda row: (row["best_score"], row["n_seeds"]), reverse=True)

    summary = {
        "n_rows": len(rows),
        "bucket_counts": dict(bucket_counts),
        "seed_counts_by_dataset": by_dataset.most_common(),
        "seed_counts_by_template": by_template.most_common(),
        "seed_counts_by_family": by_family.most_common(),
        "top_fields": field_summary[:100],
        "top_seeds": seeds[:100],
    }
    return summary, seeds


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(Path(args.sim_log))
    summary, seeds = analyze(rows, args)
    out_summary = Path(args.out_summary)
    out_seeds = Path(args.out_seeds)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with out_seeds.open("w") as handle:
        for seed in seeds:
            handle.write(json.dumps(seed, sort_keys=True) + "\n")
    print(f"Rows: {len(rows)}")
    print(f"Seeds: {len(seeds)}")
    print(f"Buckets: {summary['bucket_counts']}")
    print(f"Wrote summary: {out_summary}")
    print(f"Wrote seeds: {out_seeds}")


if __name__ == "__main__":
    main()
