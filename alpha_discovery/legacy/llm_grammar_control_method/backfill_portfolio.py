"""Backfill portfolio admission scores for historical sim records.

Fetches PnL for submittable alphas (no FAIL checks) from master.jsonl and any
other specified sim logs, fetches Brain contribution score with local fallback,
persists results, and optionally rebuilds all_qtable.json with portfolio-aware rewards.

Usage (from world_quant/ root):
    python -m alpha_discovery/legacy.llm_grammar_control_method.backfill_portfolio --credentials brain.txt
    python -m alpha_discovery/legacy.llm_grammar_control_method.backfill_portfolio --credentials brain.txt --max-fetch 100 --no-rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from core.qtable import (
    QTable,
    SkeletonCounter,
    bootstrap_qtable,
    extract_primary_op,
)
from core.selfcorr import load_portfolio_alpha_data
from core.brain.api import fetch_pnl_series_with_retry, sign_in
from core.logs import run_log_path
from core.portfolio.admission import fetch_brain_performance_comparison, score_record_admission

_LOG = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).parent
_ARTIFACTS_DIR = _MODULE_DIR / "artifacts"
_GRAMMAR_ARTIFACTS = _MODULE_DIR.parent / "alpha_discovery/grammar_method" / "artifacts"


def _contribution_delta(metrics: dict) -> float | None:
    value = metrics.get("brain_score_delta")
    if value is None:
        value = metrics.get("delta_pool_sharpe")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None

_PNL_CACHE_PATH = _ARTIFACTS_DIR / "pnl_cache.json"
_PERFORMANCE_CACHE_PATH = _ARTIFACTS_DIR / "performance_comparison_cache.json"
_PORTFOLIO_SCORE_HISTORY_PATH = _ARTIFACTS_DIR / "portfolio_score_history.json"
_EXPRESSION_PORTFOLIO_SCORES_PATH = _ARTIFACTS_DIR / "expression_portfolio_scores.json"
_DEFAULT_MASTER_LOG = _MODULE_DIR.parent / "artifacts" / "sim" / "master.jsonl"
_DEFAULT_LLM_LOG = _ARTIFACTS_DIR / "llm_sim.jsonl"
_DEFAULT_QTABLE_PATH = _GRAMMAR_ARTIFACTS / "all_qtable.json"


def _unique_existing(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _discover_all_sim_logs() -> list[Path]:
    """Return every known simulation JSONL log in the workspace."""
    root = _MODULE_DIR.parent
    patterns = [
        root / "artifacts" / "sim" / "master.jsonl",
        root / "artifacts" / "sim" / "sessions" / "*.jsonl",
        root / "artifacts" / "sim" / "archive" / "*.jsonl",
        root / "artifacts" / "*sim*.jsonl",
        root / "artifacts" / "workshop" / "*sim*.jsonl",
        root / "alpha_discovery/grammar_method" / "artifacts" / "*sim*.jsonl",
        root / "alpha_discovery/legacy.llm_grammar_control_method" / "artifacts" / "*sim*.jsonl",
        root / "alpha_search" / "artifacts" / "*sim*.jsonl",
        root / "alpha_discovery/legacy.economic_search_method" / "artifacts" / "*sim*.jsonl",
        root / "alpha_discovery/legacy.economic_search_method" / "artifacts" / "continuous" / "*sim*.jsonl",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(pattern.parent.glob(pattern.name)))
    paths = [path for path in paths if not path.name.endswith("_scored.jsonl")]
    return _unique_existing(paths)


def _is_submittable(record: dict) -> bool:
    """Return True if no check result is FAIL (PENDING is acceptable)."""
    checks = record.get("checks") or []
    return bool(checks) and not any(c.get("result") == "FAIL" for c in checks)


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def run_backfill(
    session,
    sim_paths: list[Path],
    portfolio_alpha_data: dict,
    *,
    max_fetch: int = 0,
    sleep_seconds: float = 1.0,
    rebuild_qtable: bool = True,
    qtable_sim_paths: list[Path] | None = None,
    competition_id: str = "IQC2026S1",
    refresh_performance: bool = False,
) -> dict[str, dict]:
    """Fetch PnL, compute contribution delta, persist artifacts, optionally rebuild Q-table.

    Returns expression_portfolio_scores dict (expression → admission metrics).
    """
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load PnL cache
    pnl_cache: dict[str, dict] = {}
    if _PNL_CACHE_PATH.exists():
        try:
            pnl_cache = json.loads(_PNL_CACHE_PATH.read_text())
        except Exception:
            pnl_cache = {}
    _LOG.info("PnL cache loaded: %d entries", len(pnl_cache))

    performance_cache: dict[str, dict | None] = {}
    if _PERFORMANCE_CACHE_PATH.exists() and not refresh_performance:
        try:
            performance_cache = json.loads(_PERFORMANCE_CACHE_PATH.read_text())
        except Exception:
            performance_cache = {}
    elif refresh_performance:
        _LOG.info("Ignoring cached Brain performance comparisons because --refresh-performance was set")
    _LOG.info("Brain performance cache loaded: %d entries", len(performance_cache))

    # Load existing expression scores
    expr_scores: dict[str, dict] = {}
    if _EXPRESSION_PORTFOLIO_SCORES_PATH.exists():
        try:
            expr_scores = json.loads(_EXPRESSION_PORTFOLIO_SCORES_PATH.read_text())
        except Exception:
            expr_scores = {}

    # Collect submittable records across all sim logs
    all_records: dict[str, dict] = {}  # alpha_id → record (dedup by alpha_id, keep first)
    for path in sim_paths:
        for rec in _load_jsonl(path):
            alpha_id = rec.get("alpha_id")
            if not alpha_id:
                continue
            if not _is_submittable(rec):
                continue
            if alpha_id not in all_records:
                all_records[alpha_id] = rec

    total_candidates = len(all_records)
    _LOG.info("Submittable records with alpha_id: %d", total_candidates)

    # Determine which need PnL fetching
    to_fetch = [
        (alpha_id, rec) for alpha_id, rec in all_records.items()
        if alpha_id not in pnl_cache
    ]
    _LOG.info("Need PnL fetch: %d (max_fetch=%d)", len(to_fetch), max_fetch or len(to_fetch))
    if max_fetch > 0:
        to_fetch = to_fetch[:max_fetch]

    # Fetch PnL
    fetched = 0
    for alpha_id, _rec in to_fetch:
        try:
            pnl = fetch_pnl_series_with_retry(
                session, alpha_id, max_attempts=3, sleep_seconds=sleep_seconds
            )
            pnl_cache[alpha_id] = pnl
            fetched += 1
            if fetched % 20 == 0:
                _LOG.info("Fetched PnL for %d/%d alphas ...", fetched, len(to_fetch))
                _PNL_CACHE_PATH.write_text(json.dumps(pnl_cache))
        except Exception as exc:
            _LOG.debug("PnL fetch failed for %s: %s", alpha_id, exc)
        time.sleep(sleep_seconds)

    # Save final PnL cache
    _PNL_CACHE_PATH.write_text(json.dumps(pnl_cache))
    _LOG.info("PnL cache saved: %d entries total", len(pnl_cache))

    # Compute admission scores
    portfolio_history: dict[str, dict] = {}
    if _PORTFOLIO_SCORE_HISTORY_PATH.exists():
        try:
            portfolio_history = json.loads(_PORTFOLIO_SCORE_HISTORY_PATH.read_text())
        except Exception:
            portfolio_history = {}

    scored = 0
    zeroed = 0
    for alpha_id, rec in all_records.items():
        pnl = pnl_cache.get(alpha_id)
        if not pnl:
            continue
        expression = rec.get("expression") or ""
        if not expression:
            continue
        try:
            if alpha_id in performance_cache:
                performance = performance_cache[alpha_id]
            else:
                performance = fetch_brain_performance_comparison(
                    session,
                    alpha_id,
                    competition_id,
                    max_attempts=5,
                    sleep_seconds=2.0,
                )
                performance_cache[alpha_id] = performance
                if len(performance_cache) % 20 == 0:
                    _PERFORMANCE_CACHE_PATH.write_text(json.dumps(performance_cache))
            metrics = score_record_admission(
                rec,
                pnl,
                portfolio_alpha_data,
                brain_performance=performance,
            )
            # Inject primary op
            field_ids = rec.get("fields") or []
            if field_ids:
                op = extract_primary_op(expression, str(field_ids[0]))
                if op:
                    metrics["_primary_op"] = op
                    entry = portfolio_history.setdefault(
                        op, {"n_pos": 0, "n_neg": 0, "n_none": 0, "sum_delta": 0.0}
                    )
                    delta = _contribution_delta(metrics)
                    if delta is None:
                        entry["n_none"] += 1
                    elif delta > 0:
                        entry["n_pos"] += 1
                        entry["sum_delta"] += delta
                    else:
                        entry["n_neg"] += 1
                        entry["sum_delta"] += delta

            expr_scores[expression] = {
                "brain_score_delta": metrics.get("brain_score_delta"),
                "contribution_delta": _contribution_delta(metrics),
                "delta_pool_sharpe": _contribution_delta(metrics),
                "fallback_delta_pool_sharpe": metrics.get("delta_pool_sharpe"),
                "signal": metrics.get("signal"),
                "decision": metrics.get("decision"),
            }
            scored += 1
            delta = _contribution_delta(metrics)
            if delta is not None and delta <= 0:
                zeroed += 1
        except Exception as exc:
            _LOG.debug("Admission scoring failed for %s: %s", alpha_id, exc)

    _LOG.info("Scored %d alphas; %d with contribution_delta <= 0", scored, zeroed)

    # Persist artifacts
    _EXPRESSION_PORTFOLIO_SCORES_PATH.write_text(json.dumps(expr_scores))
    _PERFORMANCE_CACHE_PATH.write_text(json.dumps(performance_cache))
    _PORTFOLIO_SCORE_HISTORY_PATH.write_text(json.dumps(portfolio_history, indent=2))
    _LOG.info("Saved expression_portfolio_scores (%d entries)", len(expr_scores))
    _LOG.info("Saved Brain performance cache (%d entries)", len(performance_cache))
    _LOG.info("Saved portfolio_score_history (%d operators)", len(portfolio_history))

    # Rebuild Q-table
    if rebuild_qtable:
        paths_to_rebuild = qtable_sim_paths or sim_paths
        _LOG.info("Rebuilding Q-table from %d sim logs with portfolio scores ...", len(paths_to_rebuild))
        qtable = QTable()
        skeleton_counter = SkeletonCounter()
        count = bootstrap_qtable(
            paths_to_rebuild,
            qtable,
            skeleton_counter=skeleton_counter,
            portfolio_scores=expr_scores,
            dedupe=True,
        )
        _LOG.info("Processed %d records", count)
        qtable.save(_DEFAULT_QTABLE_PATH)
        _LOG.info("Q-table saved to %s", _DEFAULT_QTABLE_PATH)

    return expr_scores


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(run_log_path("llm_backfill"))),
        ],
    )

    p = argparse.ArgumentParser(
        description="Backfill portfolio admission scores for historical sim records"
    )
    p.add_argument(
        "--credentials",
        default=str(Path(__file__).parent.parent / "brain.txt"),
        help="Path to brain.txt credentials (default: ../brain.txt)",
    )
    p.add_argument(
        "--sim-log",
        action="append",
        dest="sim_logs",
        default=None,
        help="Sim log path to backfill (can repeat; default: master.jsonl + llm_sim.jsonl; use --all-logs for workspace scan)",
    )
    p.add_argument(
        "--all-logs",
        action="store_true",
        help="Discover and use all simulation JSONL logs in the workspace for score backfill and Q-table rebuild.",
    )
    p.add_argument(
        "--max-fetch",
        type=int,
        default=0,
        help="Max PnL fetches per run (0 = all; useful to resume incrementally)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between PnL fetches (default 1.0)",
    )
    p.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip Q-table rebuild (only update score artifacts)",
    )
    p.add_argument(
        "--refresh-performance",
        action="store_true",
        help="Ignore cached Brain performance-comparison responses and fetch them again.",
    )
    p.add_argument(
        "--competition-id",
        default="IQC2026S1",
        help="Competition ID for portfolio loading",
    )
    args = p.parse_args()

    sim_paths: list[Path] = []
    if args.all_logs:
        sim_paths = _discover_all_sim_logs()
    elif args.sim_logs:
        sim_paths = [Path(p) for p in args.sim_logs]
    else:
        sim_paths = _unique_existing([_DEFAULT_MASTER_LOG, _DEFAULT_LLM_LOG])
    _LOG.info("Simulation logs selected: %d", len(sim_paths))
    for path in sim_paths[:20]:
        _LOG.info("  %s", path)
    if len(sim_paths) > 20:
        _LOG.info("  ... %d more", len(sim_paths) - 20)

    _LOG.info("Signing in to Brain ...")
    session = sign_in(args.credentials)

    _LOG.info("Loading portfolio PnL (competition=%s) ...", args.competition_id)
    portfolio_alpha_data = load_portfolio_alpha_data(
        session, args.competition_id, max_pnl_attempts=5
    )
    _LOG.info("Portfolio loaded: %d alphas", len(portfolio_alpha_data))

    # For default mode, include grammar/workshop sim logs as well. In --all-logs
    # mode they are already included in sim_paths.
    grammar_sim_paths = [] if args.all_logs else [p for p in [
        _GRAMMAR_ARTIFACTS / "all_sim.jsonl",
        _MODULE_DIR.parent / "artifacts" / "workshop" / "workshop_sim.jsonl",
    ] if p.exists()]
    qtable_sim_paths = _unique_existing(sim_paths + grammar_sim_paths)

    run_backfill(
        session,
        sim_paths=sim_paths,
        portfolio_alpha_data=portfolio_alpha_data,
        max_fetch=args.max_fetch,
        sleep_seconds=args.sleep,
        rebuild_qtable=not args.no_rebuild,
        qtable_sim_paths=qtable_sim_paths,
        competition_id=args.competition_id,
        refresh_performance=args.refresh_performance,
    )
    _LOG.info("Backfill complete.")


if __name__ == "__main__":
    main()
