"""Build deterministic A-derived enhancement and robustness plans.

This module is intentionally narrow: it takes already-promising seed alphas and
emits expression-only variants around the same signal A. It is not a raw miner
and it should not feed Q-table/workshop statistics.

Examples
--------
Build structural A-derived variants from probe seeds:

    python -m alpha_discovery.probe_method.enhance_plan build \
        --input artifacts/probe/probe_seeds.jsonl \
        --out artifacts/enhance/self_enhance_plan.jsonl \
        --pack all \
        --max-alphas 50

Submit with:

    python -m core.brain.submit \
        --plan artifacts/enhance/self_enhance_plan.jsonl \
        --sim-log artifacts/enhance/self_enhance_sim.jsonl \
        --batch-size 3

Summarize completed variants:

    python -m alpha_discovery.probe_method.enhance_plan summary \
        --sim-log artifacts/enhance/self_enhance_sim.jsonl \
        --out-txt artifacts/enhance/self_enhance_summary.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from core.brain.submit import DEFAULT_SETTINGS
from alpha_enhancement.workshop.core import is_passing


DEFAULT_OUT = Path("artifacts/enhance/self_enhance_plan.jsonl")
DEFAULT_SIM_LOG = Path("artifacts/enhance/self_enhance_sim.jsonl")
DEFAULT_KNOWN_LOGS = (
    Path("artifacts/sim/master.jsonl"),
    Path("artifacts/workshop/workshop_sim.jsonl"),
    Path("artifacts/enhance/self_enhance_sim.jsonl"),
)

SETTING_KEYS = (
    "delay",
    "universe",
    "neutralization",
    "truncation",
    "decay",
    "pasteurization",
    "unitHandling",
    "nanHandling",
)


@dataclass(frozen=True)
class AlphaVariantTemplate:
    template_id: str
    family: str
    kind: str
    expression: str
    description: str
    min_ratio: float | None = None

    def render(self, a: str) -> str:
        return self.expression.format(a=a)


ROBUSTNESS_TEMPLATES: tuple[AlphaVariantTemplate, ...] = (
    AlphaVariantTemplate(
        "rank_remap",
        "distribution",
        "robustness_check",
        "rank(({a}))",
        "Monotone remap test: signal should survive full cross-sectional ranking.",
        min_ratio=0.50,
    ),
    AlphaVariantTemplate(
        "sign_remap",
        "distribution",
        "robustness_check",
        "sign(({a}))",
        "Direction-only remap test: checks whether only a few magnitudes are carrying the result.",
        min_ratio=0.40,
    ),
    AlphaVariantTemplate(
        "rank_power_2",
        "tail_shape",
        "robustness_check",
        "power(rank(({a})), 2)",
        "Right-tail concentration remap test.",
        min_ratio=0.80,
    ),
    AlphaVariantTemplate(
        "rank_power_half",
        "tail_shape",
        "robustness_check",
        "power(rank(({a})), 0.5)",
        "Flatter tail remap test.",
        min_ratio=0.50,
    ),
    AlphaVariantTemplate(
        "winsor_rank",
        "distribution",
        "robustness_check",
        "rank(winsorize(({a})))",
        "Clipping test for outlier dependence.",
        min_ratio=0.50,
    ),
)


ENHANCEMENT_TEMPLATES: tuple[AlphaVariantTemplate, ...] = (
    AlphaVariantTemplate(
        "industry_relative",
        "cross_section",
        "self_enhance",
        "group_rank(({a}), industry)",
        "Re-express A as an industry-relative signal.",
    ),
    AlphaVariantTemplate(
        "sector_zscore",
        "cross_section",
        "self_enhance",
        "group_zscore(({a}), sector)",
        "Sector-relative standardization.",
    ),
    AlphaVariantTemplate(
        "cap_bucket_neut",
        "exposure_repair",
        "self_enhance",
        "group_neutralize(({a}), bucket(rank(cap), range='0.1,1,0.1'))",
        "Remove broad size-bucket exposure while preserving A.",
    ),
    AlphaVariantTemplate(
        "cap_regression_neut",
        "exposure_repair",
        "self_enhance",
        "regression_neut(({a}), log(cap))",
        "Residualize A against size.",
    ),
    AlphaVariantTemplate(
        "ir_regression_neut",
        "exposure_repair",
        "self_enhance",
        "regression_neut(({a}), abs(ts_mean(returns, 252)/ts_std_dev(returns, 252)))",
        "Residualize A against stock-level return IR.",
    ),
    AlphaVariantTemplate(
        "smooth_5",
        "turnover_repair",
        "self_enhance",
        "rank(ts_mean(({a}), 5))",
        "Short smoothing for high-turnover A.",
    ),
    AlphaVariantTemplate(
        "smooth_10",
        "turnover_repair",
        "self_enhance",
        "rank(ts_mean(({a}), 10))",
        "Medium smoothing for high-turnover A.",
    ),
    AlphaVariantTemplate(
        "decay_5",
        "turnover_repair",
        "self_enhance",
        "rank(ts_decay_linear(({a}), 5))",
        "Linear decay of A.",
    ),
    AlphaVariantTemplate(
        "hump",
        "turnover_repair",
        "self_enhance",
        "hump(({a}))",
        "Limit day-to-day position changes.",
    ),
    AlphaVariantTemplate(
        "self_strength_weight",
        "self_gate",
        "self_enhance",
        "rank(({a}) * rank(abs(({a}))))",
        "Use A's own cross-sectional strength to emphasize high-conviction names.",
    ),
    AlphaVariantTemplate(
        "self_strength_gate_50",
        "self_gate",
        "self_enhance",
        "trade_when(rank(abs(({a}))) > 0.50, ({a}), -1)",
        "Trade A only when its own absolute strength is above median.",
    ),
    AlphaVariantTemplate(
        "self_strength_gate_70",
        "self_gate",
        "self_enhance",
        "trade_when(rank(abs(({a}))) > 0.70, ({a}), -1)",
        "Trade A only in stronger tails.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build A-derived enhancement/robustness candidates.")
    build.add_argument("--input", nargs="+", required=True, help="Source JSONL records with expressions.")
    build.add_argument("--out", default=str(DEFAULT_OUT))
    build.add_argument(
        "--pack",
        choices=("robustness", "enhance", "all"),
        default="all",
        help="robustness = remap tests; enhance = structural A-derived variants; all = both.",
    )
    build.add_argument("--known-log", nargs="*", default=[str(p) for p in DEFAULT_KNOWN_LOGS])
    build.add_argument("--no-skip-known", action="store_true")
    build.add_argument("--max-alphas", type=int, default=50)
    build.add_argument("--max-candidates", type=int, default=0)
    build.add_argument(
        "--source-bucket",
        nargs="*",
        default=["elite", "near_repair", "weak_signal"],
        help="When input is probe_seeds.jsonl, keep only these buckets.",
    )
    build.add_argument("--promote-only", action="store_true", help="Keep only rows with admission_decision PROMOTE.")
    build.add_argument("--passing-only", action="store_true", help="Keep only rows passing standalone checks.")
    build.add_argument(
        "--allow-multistatement",
        action="store_true",
        help="Allow wrapping expressions containing semicolons. Off by default because wrappers are often invalid.",
    )
    build.add_argument("--dry-run", action="store_true")

    summary = sub.add_parser("summary", help="Summarize completed A-derived simulations.")
    summary.add_argument("--sim-log", nargs="+", default=[str(DEFAULT_SIM_LOG)])
    summary.add_argument("--out-txt", default="artifacts/enhance/self_enhance_summary.txt")
    summary.add_argument("--out-jsonl", default="artifacts/enhance/self_enhance_summary.jsonl")
    summary.add_argument("--min-pass-rate", type=float, default=0.35)

    return parser.parse_args()


def _load_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _known_expressions(paths: Iterable[str | Path]) -> set[str]:
    return {
        str(row["expression"])
        for row in _load_jsonl(paths)
        if row.get("expression")
    }


def _templates(pack: str) -> tuple[AlphaVariantTemplate, ...]:
    if pack == "robustness":
        return ROBUSTNESS_TEMPLATES
    if pack == "enhance":
        return ENHANCEMENT_TEMPLATES
    if pack == "all":
        return ROBUSTNESS_TEMPLATES + ENHANCEMENT_TEMPLATES
    raise ValueError(f"unknown pack: {pack}")


def _source_id(record: dict[str, Any]) -> str:
    return str(record.get("alpha_id") or record.get("candidate_id") or _stable_id(str(record.get("expression") or "")))


def _source_score(record: dict[str, Any]) -> float:
    stored = record.get("score")
    if stored is not None:
        try:
            return float(stored)
        except (TypeError, ValueError):
            pass
    return abs(_float(record.get("is_fitness"))) + 0.35 * abs(_float(record.get("is_sharpe")))


def _eligible_sources(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    allowed_buckets = set(args.source_bucket or [])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        expr = str(record.get("expression") or "").strip()
        if not expr or expr in seen:
            continue
        if not args.allow_multistatement and ";" in expr:
            continue
        bucket = record.get("bucket")
        if bucket and allowed_buckets and bucket not in allowed_buckets:
            continue
        if args.promote_only and str(record.get("admission_decision") or "").upper() != "PROMOTE":
            continue
        if args.passing_only and not _standalone_passes(record):
            continue
        seen.add(expr)
        out.append(record)
    out.sort(key=_source_score, reverse=True)
    if args.max_alphas > 0:
        out = out[: args.max_alphas]
    return out


def _standalone_passes(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "done"
        and _float(record.get("is_sharpe")) >= 1.25
        and _float(record.get("is_fitness")) >= 1.0
        and is_passing(record)
    )


def _settings_override(record: dict[str, Any]) -> dict[str, Any]:
    settings = record.get("settings_override")
    if not isinstance(settings, dict):
        settings = record.get("settings")
    if not isinstance(settings, dict):
        return {}
    return {
        key: settings[key]
        for key in SETTING_KEYS
        if key in settings and settings[key] != DEFAULT_SETTINGS.get(key)
    }


def _candidate(source: dict[str, Any], template: AlphaVariantTemplate, expression: str) -> dict[str, Any]:
    source_expr = str(source.get("expression") or "")
    sid = _source_id(source)
    cid = _stable_id(f"{sid}\n{template.template_id}\n{expression}")
    return {
        "candidate_id": cid,
        "template_id": f"{template.kind}:{template.template_id}",
        "operator_skeleton": template.template_id,
        "experiment_stage": template.kind,
        "template_family": template.family,
        "dataset_id": source.get("dataset_id", ""),
        "category": source.get("category", ""),
        "search_lane": template.kind,
        "expression": expression,
        "fields": source.get("fields") or [],
        "cluster_ids": source.get("cluster_ids") or [],
        "settings_override": _settings_override(source),
        "params": {
            "source_alpha_id": source.get("alpha_id"),
            "source_candidate_id": source.get("candidate_id"),
            "source_expression": source_expr,
            "source_sharpe": source.get("is_sharpe"),
            "source_fitness": source.get("is_fitness"),
            "source_bucket": source.get("bucket"),
            "variant_template": template.template_id,
            "variant_family": template.family,
            "variant_kind": template.kind,
            "variant_description": template.description,
            "robustness_min_ratio": template.min_ratio,
            "a_derived_variant": True,
        },
        "priority_score": _source_score(source),
        "skip_qtable_update": True,
        "skip_workshop": True,
    }


def build_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    sources = _eligible_sources(_load_jsonl(args.input), args)
    templates = _templates(args.pack)
    known = set() if args.no_skip_known else _known_expressions(args.known_log)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        source_expr = str(source.get("expression") or "").strip()
        for template in templates:
            expr = template.render(source_expr)
            if expr in known or expr in seen or expr == source_expr:
                continue
            seen.add(expr)
            candidates.append(_candidate(source, template, expr))
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    return candidates


def _summarize_build(candidates: list[dict[str, Any]]) -> None:
    print(f"Candidates: {len(candidates)}")
    print("Kinds:", Counter(c["search_lane"] for c in candidates).most_common())
    print("Families:", Counter(c["template_family"] for c in candidates).most_common())
    for row in candidates[:8]:
        print(f"  {row['search_lane']:<18} {row['operator_skeleton']:<22} {row['expression'][:100]}")


def summarize_records(records: list[dict[str, Any]], *, min_pass_rate: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        params = record.get("params") or {}
        if not params.get("a_derived_variant"):
            continue
        source_expr = str(params.get("source_expression") or "")
        if source_expr:
            grouped[source_expr].append(record)

    summaries: list[dict[str, Any]] = []
    for source_expr, rows in grouped.items():
        done = [row for row in rows if row.get("status") == "done"]
        passing = [row for row in done if _standalone_passes(row)]
        best = max(done, key=_record_rank, default=None)
        robustness_rows = [row for row in done if _variant_kind(row) == "robustness_check"]
        enhance_rows = [row for row in done if _variant_kind(row) == "self_enhance"]
        source_sharpe = _source_metric(rows, "source_sharpe")
        robustness_passes = [
            row for row in robustness_rows
            if _retention_ratio(row, source_sharpe) >= _required_ratio(row)
        ]
        pass_rate = len(passing) / len(done) if done else 0.0
        robustness_rate = len(robustness_passes) / len(robustness_rows) if robustness_rows else 0.0
        summaries.append({
            "source_expression": source_expr,
            "source_sharpe": source_sharpe,
            "source_fitness": _source_metric(rows, "source_fitness"),
            "n_total": len(rows),
            "n_done": len(done),
            "n_pass": len(passing),
            "pass_rate": pass_rate,
            "n_robustness_done": len(robustness_rows),
            "n_robustness_retained": len(robustness_passes),
            "robustness_retention_rate": robustness_rate,
            "robust": bool(robustness_rows and robustness_rate >= min_pass_rate),
            "n_enhance_done": len(enhance_rows),
            "best_alpha_id": best.get("alpha_id") if best else None,
            "best_variant": _variant_template(best) if best else "",
            "best_kind": _variant_kind(best) if best else "",
            "best_sharpe": best.get("is_sharpe") if best else None,
            "best_fitness": best.get("is_fitness") if best else None,
        })
    summaries.sort(
        key=lambda row: (
            row["robust"],
            row["robustness_retention_rate"],
            row["pass_rate"],
            row.get("best_fitness") or -999.0,
        ),
        reverse=True,
    )
    return summaries


def write_summary_text(summaries: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "A-derived enhancement / expression robustness summary",
        "robustness uses per-template retention ratios, not standalone optimization",
        "",
        f"{'#':>3} {'done':>4} {'pass':>4} {'pass%':>6} {'rob%':>6} {'robust':>6} {'best_fit':>8} {'best_sh':>8} {'best_variant':<22} expression",
        "-" * 132,
    ]
    for idx, row in enumerate(summaries, 1):
        lines.append(
            f"{idx:>3} {row['n_done']:>4} {row['n_pass']:>4} {row['pass_rate']:>6.2f} "
            f"{row['robustness_retention_rate']:>6.2f} {str(row['robust']):>6} "
            f"{_fmt(row.get('best_fitness')):>8} {_fmt(row.get('best_sharpe')):>8} "
            f"{str(row.get('best_variant') or '')[:22]:<22} {row['source_expression'][:68]}"
        )
    out.write_text("\n".join(lines) + "\n")


def _variant_kind(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    return str((record.get("params") or {}).get("variant_kind") or record.get("search_lane") or "")


def _variant_template(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    return str((record.get("params") or {}).get("variant_template") or record.get("operator_skeleton") or "")


def _source_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in rows:
        value = (row.get("params") or {}).get(key)
        if value is not None:
            return _float(value)
    return None


def _required_ratio(record: dict[str, Any]) -> float:
    raw = (record.get("params") or {}).get("robustness_min_ratio")
    if raw is None:
        return 0.50
    return _float(raw, default=0.50)


def _retention_ratio(record: dict[str, Any], source_sharpe: float | None) -> float:
    if source_sharpe is None or source_sharpe <= 0.0:
        return 0.0
    sharpe = _float(record.get("is_sharpe"))
    if sharpe <= 0.0:
        return 0.0
    return sharpe / source_sharpe


def _record_rank(record: dict[str, Any]) -> tuple[float, float, float]:
    return (
        1.0 if _standalone_passes(record) else 0.0,
        _float(record.get("is_fitness"), default=-999.0),
        _float(record.get("is_sharpe"), default=-999.0),
    )


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _build(args: argparse.Namespace) -> None:
    candidates = build_candidates(args)
    _summarize_build(candidates)
    if args.dry_run:
        return
    _write_jsonl(candidates, args.out)
    manifest = {
        "args": vars(args),
        "n_candidates": len(candidates),
        "templates": [asdict(template) for template in _templates(args.pack)],
    }
    manifest_path = Path(args.out).with_suffix(Path(args.out).suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote plan: {args.out}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Submit with: python -m core.brain.submit --plan {args.out} --sim-log {DEFAULT_SIM_LOG}")


def _summary(args: argparse.Namespace) -> None:
    records = _load_jsonl(args.sim_log)
    summaries = summarize_records(records, min_pass_rate=args.min_pass_rate)
    write_summary_text(summaries, args.out_txt)
    _write_jsonl(summaries, args.out_jsonl)
    print(f"Wrote text summary: {args.out_txt}")
    print(f"Wrote JSONL summary: {args.out_jsonl}")
    if summaries:
        top = summaries[0]
        print(
            "Top result: "
            f"robust={top['robust']} robust_rate={top['robustness_retention_rate']:.2f} "
            f"best={top['best_variant']} fitness={top.get('best_fitness')} sharpe={top.get('best_sharpe')}"
        )


def main() -> None:
    args = parse_args()
    if args.command == "build":
        _build(args)
    elif args.command == "summary":
        _summary(args)


if __name__ == "__main__":
    main()
