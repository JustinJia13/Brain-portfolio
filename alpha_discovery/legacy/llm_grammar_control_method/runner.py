"""LLM exploration loop: propose → filter → submit → distill → repeat.

LLMExplorationLoop orchestrates:
  1. AlphaProposerAgent: LLM picks a GrammarConfig patch; grammar samples candidates
  2. SimilarityFilter: drops near-duplicates vs submitted history
  3. Brain submission (reuses run_pipeline.run_batch_submission)
  4. In-memory Q-table update from new results
  5. MemoryDistiller: LLM distills results into persistent memory facts

Usage (from world_quant/ root):
    python -m alpha_discovery.legacy.llm_grammar_control_method.runner [--rounds N] [--batch-size 40] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from core.fields import build_data_profiles, cluster_profiles, filter_profiles
from core.expr.models import FieldCluster
from core.expr.operators import BANNED_GENERATION_OPS, OPERATOR_REGISTRY
from alpha_discovery.legacy.economic_search_method.field_catalogs import default_fields_csv
from core.qtable import (
    QTable,
    SkeletonCounter,
    bootstrap_qtable,
    apply_pool_delta_reward_veto,
    compute_q_reward,
    extract_primary_op,
    is_degenerate_sim_record,
)
from core.feedback import SimResult
from core.selfcorr import load_portfolio_alpha_data
from core.clusters import load_clusters_from_csv
from alpha_discovery.grammar_method.run_pipeline import DEFAULT_SIMULATION_SETTINGS, run_batch_submission
from core.brain.api import get_datafields
from core.brain.api import fetch_pnl_series_with_retry
from core.logs import MASTER, merge_session_to_master, run_log_path
from core.portfolio.admission import fetch_brain_performance_comparison, score_record_admission

from .history_summarizer import (
    ExhaustionAnalysisAgent,
    LLMFailurePatternAgent,
    PoolHurtPatternAgent,
)
from .memory import ExplorationMemory, MemoryDistiller
from .proposer import AlphaProposerAgent, build_concentration_summary, format_portfolio_context
from .semantic_novelty import (
    AlphaSemanticCriticAgent,
    AlphaSemanticNoveltyGate,
    SemanticThresholds,
)
from .similarity import SimilarityFilter

_LOG = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).parent
_ARTIFACTS_DIR = _MODULE_DIR / "artifacts"

# Where cluster data is cached (reuse from alpha_discovery.grammar_method artifacts if available)
_GRAMMAR_ARTIFACTS = Path(__file__).parent.parent / "alpha_discovery/grammar_method" / "artifacts"

# Master sim log shared with the rest of the pipeline
_DEFAULT_SIM_LOG = _ARTIFACTS_DIR / "llm_sim.jsonl"

_PNL_CACHE_PATH = _ARTIFACTS_DIR / "pnl_cache.json"
_PORTFOLIO_SCORE_HISTORY_PATH = _ARTIFACTS_DIR / "portfolio_score_history.json"
_EXPRESSION_PORTFOLIO_SCORES_PATH = _ARTIFACTS_DIR / "expression_portfolio_scores.json"
_MASTER_SIM_LOG = _MODULE_DIR.parent / "artifacts" / "sim" / "master.jsonl"
_WORKSHOP_SIM_LOG = _MODULE_DIR.parent / "artifacts" / "workshop" / "workshop_sim.jsonl"
_PORTFOLIO_ADMISSION_LOG = _MODULE_DIR.parent / "artifacts" / "portfolio_admission" / "ranked.jsonl"
_SEMANTIC_NOVELTY_LOG = _ARTIFACTS_DIR / "semantic_novelty.jsonl"


def _contribution_delta(metrics: dict[str, Any]) -> float | None:
    value = metrics.get("brain_score_delta")
    if value is None:
        value = metrics.get("delta_pool_sharpe")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Q-table update helper (in-memory, no file I/O)
# ---------------------------------------------------------------------------

def _update_qtable_in_memory(
    records: list[dict],
    qtable: QTable,
    skeleton_counter: SkeletonCounter | None = None,
    portfolio_scores: dict[str, dict] | None = None,
) -> int:
    """Update qtable in place from sim records.  Returns count of records processed.

    portfolio_scores: optional dict mapping expression → admission metrics.
        Brain score delta is preferred; fallback delta_pool_sharpe is used only
        when Brain comparison was unavailable.
    """
    count = 0
    for rec in records:
        if rec.get("status") != "done":
            continue
        if is_degenerate_sim_record(rec):
            continue
        expression = rec.get("expression") or ""
        field_ids = rec.get("fields") or []
        cluster_ids = rec.get("cluster_ids") or []
        if not field_ids or not expression:
            continue

        primary_field = str(field_ids[0])
        primary_cluster = str(cluster_ids[0]) if cluster_ids else ""
        specific_op = extract_primary_op(expression, primary_field)

        result = SimResult(
            candidate_id=rec.get("candidate_id", ""),
            expression=expression,
            cluster_ids=tuple(str(c) for c in cluster_ids),
            is_sharpe=float(rec.get("is_sharpe") or 0.0),
            is_fitness=float(rec.get("is_fitness") or 0.0),
            passed=bool((rec.get("grade") or "").upper() == "PASS"),
            is_turnover=float(rec["is_turnover"]) if rec.get("is_turnover") is not None else None,
            concentrated_weight_ok=bool(rec.get("is_concentrated_weight_ok", True)),
            sub_universe_sharpe_ok=bool(rec.get("is_sub_universe_sharpe_ok", True)),
            max_self_correlation=float(rec["max_self_correlation"]) if rec.get("max_self_correlation") is not None else None,
            max_self_correlation_alpha_sharpe=float(rec["max_self_correlation_alpha_sharpe"]) if rec.get("max_self_correlation_alpha_sharpe") is not None else None,
            max_concentration=float(rec["max_concentration"]) if rec.get("max_concentration") is not None else None,
        )
        reward = compute_q_reward(result)
        ps = portfolio_scores.get(expression) if portfolio_scores else None
        reward = apply_pool_delta_reward_veto(rec, reward, ps)
        qtable.update(primary_cluster, primary_field, specific_op, reward)

        if skeleton_counter is not None:
            tid = rec.get("template_id", "")
            if tid:
                skeleton_counter.increment(tid)

        count += 1
    return count


# ---------------------------------------------------------------------------
# Cluster loading helpers
# ---------------------------------------------------------------------------

def _load_clusters_from_csv(
    csv_path: str,
    max_clusters: int = 400,
    exclude_datasets: list[str] | None = None,
) -> list[FieldCluster]:
    return load_clusters_from_csv(
        csv_path,
        max_clusters=max_clusters,
        exclude_datasets=exclude_datasets,
    )


def _load_clusters_from_api(
    session,
    max_clusters: int = 400,
    min_coverage: float = 0.10,
) -> list[FieldCluster]:
    search_scope = {
        "region": "USA", "delay": 1, "universe": "TOP3000", "instrumentType": "EQUITY",
    }
    raw_fields = get_datafields(session, search_scope, max_retries=10, default_sleep_seconds=60.0)
    profiles = build_data_profiles(raw_fields)
    profiles = filter_profiles(profiles, allowed_types=("MATRIX",), min_coverage=min_coverage)
    return cluster_profiles(profiles, max_clusters=max_clusters)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class LLMExplorationLoop:
    """Orchestrates LLM-guided alpha exploration in round-based batches.

    Each round:
      1. Proposer LLM call → GrammarConfig patch → grammar samples ~120 candidates
      2. SimilarityFilter drops near-duplicates → ~40 remain
      3. Submit to Brain API → collect sim records
      4. Q-table updated in memory from new records
      5. Distiller LLM call → updated memory facts

    Args:
        clusters:       FieldCluster list (loaded from CSV or Brain API)
        session:        authenticated requests.Session (Brain API)
        sim_log:        path to write simulation records (JSONL)
        api_key:        OpenAI API key (default: OPENAI_API_KEY env var)
        memory_path:    path to memory.json (default: alpha_discovery/legacy/llm_grammar_control_method/artifacts/memory.json)
        batch_size:     target number of alphas to submit per round (default 40)
        credentials:    path to brain.txt credentials file
        settings:       Brain simulation settings dict (default: standard EQUITY USA settings)
        qtable:         pre-loaded QTable (optional; empty if not provided)
        skeleton_counter: pre-loaded SkeletonCounter (optional)
        dry_run:        if True, generate and filter candidates but do not submit
        similarity_bootstrap_logs: extra history logs for novelty filtering
        semantic_novelty: if True, run the alpha_discovery.legacy.llm_grammar_control_method semantic novelty gate
    """

    def __init__(
        self,
        clusters: list[FieldCluster],
        session,
        sim_log: str | Path = _DEFAULT_SIM_LOG,
        api_key: str | None = None,
        memory_path: Path | None = None,
        batch_size: int = 40,
        credentials: str = "brain.txt",
        settings: dict | None = None,
        qtable: QTable | None = None,
        skeleton_counter: SkeletonCounter | None = None,
        dry_run: bool = False,
        competition_id: str = "IQC2026S1",
        similarity_bootstrap_logs: list[str | Path] | None = None,
        semantic_novelty: bool = True,
        semantic_thresholds: SemanticThresholds | dict[str, float] | None = None,
        semantic_agent: AlphaSemanticCriticAgent | None = None,
    ) -> None:
        self._clusters = clusters
        self._session = session
        self._sim_log = Path(sim_log)
        self._sim_log.parent.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size
        self._credentials = str(credentials)
        self._settings = settings or dict(DEFAULT_SIMULATION_SETTINGS)
        self._dry_run = dry_run
        self._competition_id = competition_id
        self._similarity_bootstrap_logs = [Path(p) for p in (similarity_bootstrap_logs or [])]

        self._qtable = qtable or QTable()
        self._skeleton_counter = skeleton_counter or SkeletonCounter()

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._api_key = resolved_key
        if isinstance(semantic_thresholds, dict):
            thresholds = SemanticThresholds(**semantic_thresholds)
        else:
            thresholds = semantic_thresholds or SemanticThresholds()

        self._proposer = AlphaProposerAgent(
            api_key=resolved_key,
            target_batch_size=batch_size,
            qtable=self._qtable,
            skeleton_counter=self._skeleton_counter,
        )
        self._similarity_filter = SimilarityFilter()
        self._semantic_gate = (
            AlphaSemanticNoveltyGate(
                thresholds=thresholds,
                agent=semantic_agent or AlphaSemanticCriticAgent(api_key=resolved_key),
                artifact_path=_SEMANTIC_NOVELTY_LOG,
            )
            if semantic_novelty
            else None
        )
        self._memory = ExplorationMemory(path=memory_path)
        self._distiller = MemoryDistiller(api_key=resolved_key)

        # Bootstrap similarity filter from active and external history logs.
        self._bootstrap_filter()

        # Fetch current portfolio once at startup:
        # - portfolio_alpha_data: {alpha_id: {pnl, is_sharpe, is_fitness}} for self-corr computation
        # - portfolio_context: formatted string for LLM prompt
        self._portfolio_alpha_data: dict = {}
        self._portfolio_context: str = self._fetch_portfolio_context()

        # Concentration kill-rate summary for LLM prompt (refreshed after each round)
        self._concentration_summary: str = self._build_concentration_summary()

        # All-time op exploration stats (superset of old portfolio_score_summary)
        # _alltime_op_counts is updated incrementally each round; _alltime_op_stats is the
        # formatted string injected into the proposer prompt.
        self._alltime_op_counts: dict[str, int] = {}
        self._alltime_op_stats: str = ""

        # Startup analysis agent outputs (JSON strings loaded from artifacts)
        self._exhaustion_analysis: str = ""
        self._llm_failure_analysis: str = ""
        self._pool_hurt_patterns: str = ""

        # HARVESTED vs FRONTIER operator table (built from full sim history at startup)
        self._sim_history_summary: str = ""

        # Run all startup analysis (scan logs, run 3 agents, bootstrap memory)
        self._run_startup_analysis()

    def _similarity_bootstrap_sources(self) -> list[tuple[Path, bool]]:
        sources: list[tuple[Path, bool]] = [(self._sim_log, True)]
        for path in [
            _MASTER_SIM_LOG,
            _WORKSHOP_SIM_LOG,
            _GRAMMAR_ARTIFACTS / "all_sim.jsonl",
            _PORTFOLIO_ADMISSION_LOG,
            *self._similarity_bootstrap_logs,
        ]:
            sources.append((Path(path), False))

        unique: list[tuple[Path, bool]] = []
        seen: set[str] = set()
        for path, count_skeletons in sources:
            key = str(path.expanduser().resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append((path, count_skeletons))
        return unique

    def _bootstrap_filter(self) -> None:
        total = 0
        for path, count_skeletons in self._similarity_bootstrap_sources():
            if not path.exists():
                continue
            records = _load_jsonl(path)
            if not records:
                continue
            self._similarity_filter.bootstrap(
                records,
                count_skeletons=count_skeletons,
                source_name=str(path),
            )
            total += len(records)
        if total:
            _LOG.info("Bootstrapped similarity filter from %d total history records", total)

    def _fetch_portfolio_context(self) -> str:
        """Fetch submitted competition alphas.

        Populates self._portfolio_alpha_data for self-correlation computation
        (same as grammar pipeline) and returns a formatted string for LLM context.
        """
        try:
            # load_portfolio_alpha_data fetches PnL + IS metrics for all competition alphas
            _LOG.info("Loading portfolio PnL for self-correlation (competition=%s) ...", self._competition_id)
            self._portfolio_alpha_data = load_portfolio_alpha_data(
                self._session,
                self._competition_id,
                max_pnl_attempts=10,
            )
            _LOG.info("Portfolio loaded: %d alphas with PnL", len(self._portfolio_alpha_data))
        except Exception as exc:
            _LOG.warning("Could not load portfolio PnL: %s — self-correlation will be skipped", exc)
            self._portfolio_alpha_data = {}

        # Also fetch alpha details (expression, settings, grade) for LLM prompt
        try:
            from core.brain.api import fetch_competition_alpha_ids
            alpha_ids = fetch_competition_alpha_ids(self._session, self._competition_id)
            alphas = []
            for aid in alpha_ids:
                try:
                    detail = self._session.get(
                        f"https://api.worldquantbrain.com/alphas/{aid}", timeout=30
                    ).json()
                    is_s = detail.get("is") or {}
                    settings = detail.get("settings") or {}
                    expr = detail.get("regular") or detail.get("expression") or ""
                    alphas.append({
                        "id": aid,
                        "expression": expr,
                        "sharpe": is_s.get("sharpe"),
                        "fitness": is_s.get("fitness"),
                        "turnover": is_s.get("turnover"),
                        "grade": detail.get("grade"),
                        "neutralization": settings.get("neutralization"),
                        "decay": settings.get("decay"),
                        "delay": settings.get("delay"),
                        "universe": settings.get("universe"),
                    })
                except Exception as exc:
                    _LOG.debug("Failed to fetch alpha detail %s: %s", aid, exc)
            return format_portfolio_context(alphas)
        except Exception as exc:
            _LOG.warning("Could not fetch portfolio details: %s", exc)
            return ""

    def _build_concentration_summary(self) -> str:
        """Build per-operator concentration kill-rate table from all available sim records."""
        records: list[dict] = []
        # LLM sim log
        if self._sim_log.exists():
            records.extend(_load_jsonl(self._sim_log))
        # Grammar sim log (bootstrap source, if present)
        grammar_log = _GRAMMAR_ARTIFACTS / "all_sim.jsonl"
        if grammar_log.exists():
            records.extend(_load_jsonl(grammar_log))
        if not records:
            return ""
        summary = build_concentration_summary(records)
        _LOG.info("[LLM] Concentration summary built from %d records", len(records))
        return summary

    # ------------------------------------------------------------------
    # Portfolio admission scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _is_submittable(record: dict) -> bool:
        """Return True if no check result is FAIL (PENDING is acceptable)."""
        checks = record.get("checks") or []
        return bool(checks) and not any(c.get("result") == "FAIL" for c in checks)

    def _score_batch_admission(self, raw_records: list[dict]) -> dict[str, dict]:
        """Fetch Brain contribution score for submittable alphas, with local fallback.

        Only fetches for records where all checks pass (no FAIL result) and
        alpha_id is present.  Uses a persistent PnL cache to avoid re-fetching.

        Returns dict mapping expression → admission metrics dict. Brain score delta
        is preferred; local delta_pool_sharpe is present only when Brain comparison
        stayed unavailable after retries.
        """
        if not self._portfolio_alpha_data:
            _LOG.debug("[LLM] No portfolio data — skipping admission scoring")
            return {}

        # Load PnL cache
        pnl_cache: dict[str, dict] = {}
        if _PNL_CACHE_PATH.exists():
            try:
                pnl_cache = json.loads(_PNL_CACHE_PATH.read_text())
            except Exception:
                pnl_cache = {}

        candidates = [
            r for r in raw_records
            if r.get("alpha_id") and self._is_submittable(r)
        ]
        _LOG.info("[LLM] Portfolio admission: %d submittable records to score", len(candidates))

        results: dict[str, dict] = {}
        cache_updated = False

        for rec in candidates:
            alpha_id = str(rec["alpha_id"])
            expression = rec.get("expression") or ""

            # Fetch PnL (use cache if available)
            if alpha_id not in pnl_cache:
                try:
                    pnl = fetch_pnl_series_with_retry(
                        self._session, alpha_id, max_attempts=3, sleep_seconds=1.0
                    )
                    pnl_cache[alpha_id] = pnl
                    cache_updated = True
                except Exception as exc:
                    _LOG.debug("[LLM] PnL fetch failed for %s: %s", alpha_id, exc)
                    continue
            else:
                pnl = pnl_cache[alpha_id]

            if not pnl:
                continue

            try:
                performance = fetch_brain_performance_comparison(
                    self._session,
                    alpha_id,
                    max_attempts=5,
                    sleep_seconds=2.0,
                )
                metrics = score_record_admission(
                    rec,
                    pnl,
                    self._portfolio_alpha_data,
                    brain_performance=performance,
                )
                # Inject primary operator so history tracker can attribute it
                field_ids = rec.get("fields") or []
                if field_ids:
                    metrics["_primary_op"] = extract_primary_op(expression, str(field_ids[0]))
                results[expression] = metrics
                delta = _contribution_delta(metrics)
                _LOG.debug(
                    "[LLM] Admission %s: contribution_delta=%s signal=%s decision=%s",
                    alpha_id, delta, metrics.get("signal"), metrics.get("decision"),
                )
            except Exception as exc:
                _LOG.debug("[LLM] Admission scoring failed for %s: %s", alpha_id, exc)

        # Persist updated PnL cache
        if cache_updated:
            try:
                _PNL_CACHE_PATH.write_text(json.dumps(pnl_cache))
            except Exception as exc:
                _LOG.warning("[LLM] Could not save PnL cache: %s", exc)

        zeroed = sum(
            1 for m in results.values()
            if _contribution_delta(m) is not None and (_contribution_delta(m) or 0.0) <= 0
        )
        _LOG.info(
            "[LLM] Portfolio admission: %d scored, %d with delta<=0 (reward=0)",
            len(results), zeroed,
        )
        return results

    def _update_portfolio_score_history(self, admission_results: dict[str, dict]) -> None:
        """Accumulate per-operator portfolio contribution stats from admission results.

        Loads portfolio_score_history.json, merges new data, and saves back.
        Schema: {operator: {n_pos: int, n_neg: int, n_none: int, sum_delta: float}}
        """
        if not admission_results:
            return

        # Load existing history
        history: dict[str, dict] = {}
        if _PORTFOLIO_SCORE_HISTORY_PATH.exists():
            try:
                history = json.loads(_PORTFOLIO_SCORE_HISTORY_PATH.read_text())
            except Exception:
                history = {}

        # Also update expression-level scores for backfill reuse
        expr_scores: dict[str, dict] = {}
        if _EXPRESSION_PORTFOLIO_SCORES_PATH.exists():
            try:
                expr_scores = json.loads(_EXPRESSION_PORTFOLIO_SCORES_PATH.read_text())
            except Exception:
                expr_scores = {}

        for expression, metrics in admission_results.items():
            delta = _contribution_delta(metrics)
            decision = metrics.get("decision")

            # Update expression-level store
            expr_scores[expression] = {
                "brain_score_delta": metrics.get("brain_score_delta"),
                "contribution_delta": delta,
                "delta_pool_sharpe": delta,
                "fallback_delta_pool_sharpe": metrics.get("delta_pool_sharpe"),
                "signal": metrics.get("signal"),
                "decision": decision,
            }

            # Attribute delta to primary operator (injected by _score_batch_admission)
            op = metrics.get("_primary_op")
            if op is None:
                continue

            entry = history.setdefault(op, {"n_pos": 0, "n_neg": 0, "n_none": 0, "sum_delta": 0.0})
            if delta is None:
                entry["n_none"] += 1
            elif delta > 0:
                entry["n_pos"] += 1
                entry["sum_delta"] += delta
            else:
                entry["n_neg"] += 1
                entry["sum_delta"] += delta

        try:
            _PORTFOLIO_SCORE_HISTORY_PATH.write_text(json.dumps(history, indent=2))
            _EXPRESSION_PORTFOLIO_SCORES_PATH.write_text(json.dumps(expr_scores))
        except Exception as exc:
            _LOG.warning("[LLM] Could not save portfolio score history: %s", exc)

    # ------------------------------------------------------------------
    # All-time op stats (startup scan + incremental update each round)
    # ------------------------------------------------------------------

    def _get_all_sim_log_paths(self) -> list[Path]:
        """Return all available sim log paths for all-time op counting."""
        paths = []
        for p in [
            self._sim_log,
            _GRAMMAR_ARTIFACTS / "all_sim.jsonl",
            _MASTER_SIM_LOG,
            _WORKSHOP_SIM_LOG,
        ]:
            if p.exists():
                paths.append(p)
        return paths

    def _scan_alltime_op_counts(self) -> dict[str, int]:
        """Scan all available sim logs and return {primary_op: n_submitted} counts."""
        counts: dict[str, int] = {}
        for log_path in self._get_all_sim_log_paths():
            for rec in _load_jsonl(log_path):
                if rec.get("status") != "done":
                    continue
                expr = rec.get("expression", "")
                fields = rec.get("fields") or []
                if not fields or not expr:
                    continue
                op = extract_primary_op(expr, str(fields[0]))
                if op:
                    counts[op] = counts.get(op, 0) + 1
        return counts

    def _build_alltime_op_stats_table(self, op_counts: dict[str, int]) -> str:
        """Format all-time exploration stats as a table for the LLM prompt.

        Merges submission counts with portfolio_score_history.json delta stats.
        """
        if not op_counts:
            return ""

        history: dict[str, dict] = {}
        if _PORTFOLIO_SCORE_HISTORY_PATH.exists():
            try:
                history = json.loads(_PORTFOLIO_SCORE_HISTORY_PATH.read_text())
            except Exception:
                pass

        lines = []
        for op, n_tried in sorted(op_counts.items(), key=lambda kv: -kv[1]):
            hist = history.get(op, {})
            n_pos = hist.get("n_pos", 0)
            n_neg = hist.get("n_neg", 0)
            n_scored = n_pos + n_neg
            if n_scored > 0:
                pos_pct = n_pos / n_scored
                mean_delta = hist.get("sum_delta", 0.0) / n_scored
                sign = "+" if mean_delta >= 0 else ""
                flag = ""
                if pos_pct <= 0.25:
                    flag = "  ** frequently hurts pool"
                elif pos_pct <= 0.50:
                    flag = "  * often hurts pool"
                lines.append(
                    f"  {op:<25} tried={n_tried:4}  pool_scored={n_scored:3}"
                    f"  pos_delta={pos_pct:.0%}  mean_delta={sign}{mean_delta:.3f}{flag}"
                )
            else:
                lines.append(
                    f"  {op:<25} tried={n_tried:4}  pool_scored=  0"
                    f"  (no pool scoring data)"
                )
        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # Startup analysis: pre-aggregate data and run 3 analysis agents
    # ------------------------------------------------------------------

    def _prepare_llm_failure_data(self, llm_records: list[dict]) -> dict[str, dict]:
        """Aggregate LLM sim records into failure pattern data for LLMFailurePatternAgent.

        Returns {primary_op: {fail_check_name: {n_fail, n_total, fail_rate, examples}}}
        """
        from collections import Counter

        op_totals: Counter = Counter()
        # (op, check_name) → list of example expressions
        failures: dict[tuple[str, str], list[str]] = {}

        for rec in llm_records:
            if rec.get("status") != "done":
                continue
            expr = rec.get("expression", "")
            fields = rec.get("fields") or []
            if not fields or not expr:
                continue
            op = extract_primary_op(expr, str(fields[0]))
            if not op:
                continue
            op_totals[op] += 1
            for check in (rec.get("checks") or []):
                if check.get("result") == "FAIL":
                    key = (op, check.get("name", "UNKNOWN"))
                    failures.setdefault(key, []).append(expr[:80])

        result: dict[str, dict] = {}
        for (op, check_name), exprs in failures.items():
            n_total = op_totals[op]
            n_fail = len(exprs)
            fail_rate = f"{n_fail / n_total:.0%}" if n_total > 0 else "?"
            result.setdefault(op, {})[check_name] = {
                "n_fail": n_fail,
                "n_total": n_total,
                "fail_rate": fail_rate,
                "examples": exprs[:3],
            }
        return result

    def _prepare_pool_hurt_examples(self) -> list[dict]:
        """Load top-30 submittable alphas with non-positive contribution delta.

        Joins expression_portfolio_scores.json with available sim logs for grade + fields.
        No Brain API calls — data comes from pre-computed backfill artifact.
        """
        if not _EXPRESSION_PORTFOLIO_SCORES_PATH.exists():
            return []
        try:
            expr_scores: dict[str, dict] = json.loads(
                _EXPRESSION_PORTFOLIO_SCORES_PATH.read_text()
            )
        except Exception:
            return []

        hurt_exprs = {
            expr: data
            for expr, data in expr_scores.items()
            if data.get("contribution_delta", data.get("delta_pool_sharpe")) is not None
            and data.get("contribution_delta", data.get("delta_pool_sharpe")) <= 0
        }
        if not hurt_exprs:
            return []

        examples: list[dict] = []
        seen: set[str] = set()

        for log_path in [_MASTER_SIM_LOG, self._sim_log]:
            if not log_path.exists():
                continue
            for rec in _load_jsonl(log_path):
                expr = rec.get("expression", "")
                if expr not in hurt_exprs or expr in seen:
                    continue
                seen.add(expr)
                fields = rec.get("fields") or []
                op = extract_primary_op(expr, str(fields[0])) if fields else None
                examples.append({
                    "grade": rec.get("grade", "?"),
                    "expression": expr,
                    "primary_op": op,
                    "fields": fields[:3],
                    "contribution_delta": hurt_exprs[expr].get("contribution_delta", hurt_exprs[expr].get("delta_pool_sharpe")),
                    "delta_pool_sharpe": hurt_exprs[expr].get("contribution_delta", hurt_exprs[expr].get("delta_pool_sharpe")),
                })

        examples.sort(key=lambda x: x["contribution_delta"] or 0)
        return examples[:30]

    # Normalization wrappers — always needed as outer layers.
    _NORMALIZATION_OPS: frozenset[str] = frozenset({
        "group_rank", "group_zscore", "group_neutralize", "group_scale",
        "group_backfill", "group_mean",
        "rank", "zscore", "winsorize", "scale",
    })

    def _build_sim_history_summary(self, all_records: list[dict]) -> str:
        """Build a simple viable-operator directive from sim history.

        Computes which signal operators still have room (never tried, or tried
        but never produced a passing alpha) and outputs a concise USE / AVOID
        table the LLM can act on directly.
        """
        from collections import Counter

        n_tried: Counter = Counter()
        n_passing: Counter = Counter()

        for rec in all_records:
            if rec.get("status") != "done":
                continue
            expr = rec.get("expression", "")
            fields = rec.get("fields") or []
            if not fields or not expr:
                continue
            op = extract_primary_op(expr, str(fields[0]))
            if not op or op in self._NORMALIZATION_OPS:
                continue
            n_tried[op] += 1
            sharpe = float(rec.get("is_sharpe") or 0.0)
            fitness = float(rec.get("is_fitness") or 0.0)
            checks_ok = not any(c.get("result") == "FAIL" for c in (rec.get("checks") or []))
            if sharpe >= 1.25 and fitness >= 1.0 and checks_ok:
                n_passing[op] += 1

        # Ops that are used inside expressions (as arithmetic/comparison) but are never
        # the primary signal source — exclude from the VIABLE signal list.
        _NON_SIGNAL_OPS: frozenset[str] = frozenset({
            "+", "-", "*", "/", "add", "subtract", "multiply", "divide",
            "<", "<=", "==", ">", ">=", "!=", "and", "or", "not",
            "abs", "sign", "log", "sqrt", "reverse", "inverse", "hump",
            "days_from_last_change", "ts_step", "vec_avg", "vec_sum",
            "densify", "bucket", "trade_when", "normalize",
        })

        # Viable = not yet saturated: either untried or tried with 0 passes
        viable = [
            op for op in OPERATOR_REGISTRY
            if op not in BANNED_GENERATION_OPS
            and op not in self._NORMALIZATION_OPS
            and op not in _NON_SIGNAL_OPS
            and n_passing.get(op, 0) == 0
        ]
        viable.sort(key=lambda o: n_tried.get(o, 0))  # least-tried first

        saturated = [
            op for op in n_passing
            if n_passing[op] > 0 and op not in self._NORMALIZATION_OPS
        ]
        saturated.sort(key=lambda o: -n_passing[o])

        lines = [
            "== VIABLE SIGNAL OPS (use ONLY these in allowed_*_ops) ==",
            "These have never produced a passing alpha — this is where unexplored signal lives.",
            "Sorted by least-tried first (top = highest priority):",
        ]
        for op in viable:
            n = n_tried.get(op, 0)
            lines.append(f"  {op:<28} tried={n:4}  passed=  0")

        lines += [
            "",
            "== SATURATED OPS (avoid — already have passing alphas, similarity filter blocks duplicates) ==",
        ]
        for op in saturated:
            lines.append(f"  {op:<28} tried={n_tried[op]:4}  passed={n_passing[op]:3}")

        lines += [
            "",
            "== NORMALIZATION OPS (always available — use freely as outer wrappers) ==",
            "  " + ", ".join(sorted(self._NORMALIZATION_OPS)),
        ]

        return "\n".join(lines)

    def _run_startup_analysis(self) -> None:
        """Scan all sim logs and run 3 startup analysis agents.

        Sets self._alltime_op_counts, self._alltime_op_stats,
        self._exhaustion_analysis, self._llm_failure_analysis, self._pool_hurt_patterns,
        and self._sim_history_summary.  Also bootstraps memory if empty.
        Failures are logged and ignored — the loop continues with empty strings.
        """
        _LOG.info("[LLM] Running startup analysis ...")

        # Scan all logs for op counts
        self._alltime_op_counts = self._scan_alltime_op_counts()
        self._alltime_op_stats = self._build_alltime_op_stats_table(self._alltime_op_counts)
        _LOG.info("[LLM] All-time op counts: %d operators", len(self._alltime_op_counts))

        # Build HARVESTED vs FRONTIER summary from all available sim history
        all_records: list[dict] = []
        for log_path in self._get_all_sim_log_paths():
            all_records.extend(_load_jsonl(log_path))
        self._sim_history_summary = self._build_sim_history_summary(all_records)
        _LOG.info("[LLM] Sim history summary built from %d records", len(all_records))

        # Bootstrap memory from history if currently empty
        if not self._memory.facts and all_records:
            _LOG.info("[LLM] Memory is empty — bootstrapping from %d historical records", len(all_records))
            try:
                self._distiller.bootstrap_from_history(all_records, self._memory)
            except Exception as exc:
                _LOG.warning("[LLM] Memory bootstrap failed: %s", exc)

        # Prepare per-agent inputs
        qtable_summary = self._qtable.exploration_summary(clusters=self._clusters)
        llm_records = _load_jsonl(self._sim_log) if self._sim_log.exists() else []
        failure_data = self._prepare_llm_failure_data(llm_records)
        pool_hurt_examples = self._prepare_pool_hurt_examples()

        _LOG.info(
            "[LLM] Startup inputs: %d failure patterns, %d pool-hurt examples",
            len(failure_data), len(pool_hurt_examples),
        )

        # Agent A: Exhaustion
        try:
            agent_a = ExhaustionAnalysisAgent(api_key=self._api_key)
            self._exhaustion_analysis = agent_a.analyze(self._alltime_op_counts, qtable_summary)
        except Exception as exc:
            _LOG.warning("[LLM] ExhaustionAnalysisAgent failed: %s", exc)
            self._exhaustion_analysis = ""

        # Agent B: LLM Failure Patterns
        try:
            agent_b = LLMFailurePatternAgent(api_key=self._api_key)
            self._llm_failure_analysis = agent_b.analyze(failure_data)
        except Exception as exc:
            _LOG.warning("[LLM] LLMFailurePatternAgent failed: %s", exc)
            self._llm_failure_analysis = ""

        # Agent C: Pool Hurt Patterns
        try:
            agent_c = PoolHurtPatternAgent(api_key=self._api_key)
            self._pool_hurt_patterns = agent_c.analyze(pool_hurt_examples)
        except Exception as exc:
            _LOG.warning("[LLM] PoolHurtPatternAgent failed: %s", exc)
            self._pool_hurt_patterns = ""

        _LOG.info(
            "[LLM] Startup agents done. exhaustion=%d chars, failure=%d chars, pool_hurt=%d chars",
            len(self._exhaustion_analysis), len(self._llm_failure_analysis),
            len(self._pool_hurt_patterns),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, n_rounds: int = 0) -> None:
        """Run the exploration loop.

        Args:
            n_rounds: number of rounds to run (0 = run until Ctrl+C).
        """
        batch_index = 0
        try:
            while n_rounds == 0 or batch_index < n_rounds:
                _LOG.info("[LLM] ===== Round %d =====", batch_index + 1)
                records = self.run_one_round(batch_index, seed=batch_index * 137 + 42)
                passed = sum(1 for r in records if (r.get("grade") or "").upper() == "PASS")
                _LOG.info(
                    "[LLM] Round %d complete: %d submitted, %d passed",
                    batch_index + 1, len(records), passed,
                )
                batch_index += 1
        except KeyboardInterrupt:
            _LOG.info("[LLM] Interrupted — stopping cleanly after round %d", batch_index)
        finally:
            if not self._dry_run and self._sim_log.exists():
                n_new, n_updated = merge_session_to_master(self._sim_log)
                _LOG.info("[LLM] Merged to master: %d new, %d updated (%s)", n_new, n_updated, MASTER)
                print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")

    def run_one_round(self, batch_index: int, seed: int) -> list[dict]:
        """Run one propose → filter → submit → distill round.

        Returns list of sim record dicts from this round (may be empty on dry-run).
        """
        # 1. Propose (oversampled)
        memory_facts = self._memory.context_for_proposer(top_n=20)
        qtable_summary = self._qtable.exploration_summary(clusters=self._clusters)

        candidates = self._proposer.propose_batch(
            self._clusters,
            memory_facts=memory_facts,
            qtable_summary=qtable_summary,
            batch_index=batch_index,
            seed=seed,
            portfolio_context=self._portfolio_context,
            concentration_summary=self._concentration_summary,
            alltime_op_stats=self._alltime_op_stats,
            exhaustion_analysis=self._exhaustion_analysis,
            llm_failure_analysis=self._llm_failure_analysis,
            pool_hurt_patterns=self._pool_hurt_patterns,
            sim_history_summary=self._sim_history_summary,
        )
        _LOG.info("[LLM] Proposed %d oversampled candidates", len(candidates))

        # 2. Deterministic similarity filter. Use a provisional clone so
        # within-batch duplicates are blocked, but the durable filter is only
        # registered after semantic novelty accepts the final candidates.
        filtered: list[CandidateSpec] = []
        n_skipped = 0
        provisional_filter = self._similarity_filter.clone()
        deterministic_pool_limit = len(candidates) if self._semantic_gate else self._batch_size
        for c in candidates:
            verdict = provisional_filter.is_too_similar(c)
            if verdict.skip:
                n_skipped += 1
                _LOG.debug("[Filter] Skip %s: %s", c.expression[:50], verdict.reason)
            else:
                provisional_filter.register(c)
                filtered.append(c)
                if len(filtered) >= deterministic_pool_limit:
                    break

        _LOG.info(
            "[LLM] After filtering: %d accepted, %d skipped (total proposed=%d)",
            len(filtered), n_skipped, len(candidates),
        )

        if not filtered:
            _LOG.warning("[LLM] No candidates passed similarity filter — skipping submission")
            return []

        # 2b. Semantic novelty gate, then register only final accepted candidates.
        if self._semantic_gate:
            semantic_result = self._semantic_gate.filter_candidates(
                filtered,
                similarity_filter=self._similarity_filter,
                requested_count=self._batch_size,
                batch_index=batch_index,
            )
            _LOG.info(
                "[LLM] Semantic novelty: %d accepted, %d quarantined, %d reviewed",
                len(semantic_result.accepted),
                len(semantic_result.quarantined),
                semantic_result.report.get("reviewed_count", 0),
            )
            filtered = semantic_result.accepted
        else:
            filtered = filtered[: self._batch_size]

        for c in filtered:
            self._similarity_filter.register(c)

        if not filtered:
            _LOG.warning("[LLM] No candidates passed semantic novelty gate — skipping submission")
            return []

        # 3. Dry run: print and return
        if self._dry_run:
            print(f"\n[DRY RUN] Round {batch_index + 1}: {len(filtered)} candidates:")
            for c in filtered:
                print(f"  {c.expression}")
            return []

        # 4. Submit
        candidate_dicts = [c.as_record() for c in filtered]
        _LOG.info("[LLM] Submitting %d candidates to Brain ...", len(candidate_dicts))

        try:
            _sim_results, raw_records = run_batch_submission(
                self._session,
                candidate_dicts,
                self._settings,
                str(self._sim_log),
                run_log="",
                batch_size=3,
                max_candidates=0,
                sleep_seconds=30,
                max_submit_retries=5,
                credentials_path=self._credentials,
                portfolio_alpha_data=self._portfolio_alpha_data or None,
                label="LLM",
            )
        except Exception as exc:
            _LOG.error("[LLM] Submission failed: %s", exc)
            return []

        _LOG.info("[LLM] Received %d sim records", len(raw_records))

        # 5a. Score portfolio admission for submittable alphas
        admission_results = self._score_batch_admission(raw_records)

        # 5b. Update Q-table (portfolio_scores zeros reward for pool-hurting alphas)
        updated = _update_qtable_in_memory(
            raw_records, self._qtable, self._skeleton_counter,
            portfolio_scores=admission_results,
        )
        _LOG.info("[LLM] Updated Q-table with %d records", updated)

        # 5c. Persist portfolio score history and refresh all-time op stats table
        self._update_portfolio_score_history(admission_results)
        for rec in raw_records:
            if rec.get("status") != "done":
                continue
            expr = rec.get("expression", "")
            fields = rec.get("fields") or []
            if fields and expr:
                op = extract_primary_op(expr, str(fields[0]))
                if op:
                    self._alltime_op_counts[op] = self._alltime_op_counts.get(op, 0) + 1
        self._alltime_op_stats = self._build_alltime_op_stats_table(self._alltime_op_counts)

        # 6. Distill memory (pool-aware: annotate top-10 with admission results)
        try:
            new_facts = self._distiller.distill(
                raw_records, self._memory.facts, batch_index,
                admission_results=admission_results,
            )
            self._memory.update(new_facts)
            _LOG.info("[LLM] Memory updated: %d facts", len(new_facts))
        except Exception as exc:
            _LOG.warning("[LLM] Memory distillation failed: %s", exc)

        # 7. Refresh concentration summary and sim history summary with new data
        self._concentration_summary = self._build_concentration_summary()
        all_records_fresh: list[dict] = []
        for log_path in self._get_all_sim_log_paths():
            all_records_fresh.extend(_load_jsonl(log_path))
        self._sim_history_summary = self._build_sim_history_summary(all_records_fresh)

        return raw_records


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-guided alpha exploration loop"
    )
    p.add_argument(
        "--credentials",
        default=str(Path(__file__).parent.parent / "brain.txt"),
        help="Path to brain.txt credentials (default: ../brain.txt)",
    )
    p.add_argument(
        "--fields-csv",
        default="",
        help=(
            "Path to fields CSV for field catalog. "
            "Defaults to all_fields.csv for brain.txt or all_fields_<tag>.csv "
            "for other credentials (e.g. all_fields_brain2.csv for brain2.txt)."
        ),
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Number of rounds to run (0 = run until Ctrl+C)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Target alphas per round (default 40)",
    )
    p.add_argument(
        "--sim-log",
        default=str(_DEFAULT_SIM_LOG),
        help="Simulation log path",
    )
    p.add_argument(
        "--min-coverage",
        type=float,
        default=0.10,
        help="Minimum field coverage threshold (default 0.10)",
    )
    p.add_argument(
        "--max-clusters",
        type=int,
        default=400,
        help="Maximum number of field clusters (default 400)",
    )
    p.add_argument(
        "--exclude-dataset",
        nargs="*",
        default=[],
        metavar="DATASET_ID",
        help="Dataset IDs to exclude (e.g. --exclude-dataset analyst4)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and filter candidates but do not submit to Brain",
    )
    p.add_argument(
        "--bootstrap-sim-log",
        default="",
        help="Additional sim log path to bootstrap Q-table from (e.g. master.jsonl)",
    )
    p.add_argument(
        "--similarity-bootstrap-log",
        action="append",
        default=[],
        help="Additional sim log path to bootstrap alpha_discovery.legacy.llm_grammar_control_method novelty filtering from; can repeat",
    )
    p.add_argument(
        "--no-semantic-novelty",
        action="store_true",
        help="Disable alpha_discovery.legacy.llm_grammar_control_method semantic novelty critic",
    )
    p.add_argument(
        "--semantic-review-threshold",
        type=float,
        default=55.0,
        help="Deterministic neighbor score that triggers semantic review (default 55)",
    )
    p.add_argument(
        "--semantic-simulate-threshold",
        type=float,
        default=65.0,
        help="LLM similarity score below which candidates pass directly (default 65)",
    )
    p.add_argument(
        "--semantic-quarantine-threshold",
        type=float,
        default=85.0,
        help="LLM similarity score at or above which candidates are quarantined (default 85)",
    )
    p.add_argument(
        "--competition-id",
        default="IQC2026S1",
        help="Competition ID for portfolio context (default: IQC2026S1)",
    )
    p.add_argument(
        "--delay",
        type=int,
        default=1,
        choices=[0, 1],
        help="Brain simulation delay setting (default: 1)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument("--log-file", default=None, help="Log file path (default: logs/YYYY-MM-DD/llm.log)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log_level = getattr(logging, args.log_level)
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_file = args.log_file or str(run_log_path("llm"))
    logging.basicConfig(
        level=log_level,
        format=log_fmt,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    # OpenAI API key — env var takes priority, then openaiapi.txt
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        key_file = Path(__file__).parent.parent / "openaiapi.txt"
        if key_file.exists():
            api_key = key_file.read_text().strip()
        if not api_key:
            sys.exit("No OpenAI API key found. Set OPENAI_API_KEY or place key in openaiapi.txt")

    # Auth
    from core.brain.api import sign_in
    _LOG.info("Authenticating with Brain ...")
    session = sign_in(args.credentials)
    _LOG.info("Authenticated.")

    # Load clusters — auto-resolve field CSV from credentials if not explicitly passed
    if args.fields_csv:
        csv_path = Path(args.fields_csv)
    else:
        csv_path = default_fields_csv(args.credentials)
    _LOG.info("Using field catalog: %s", csv_path)
    if csv_path.exists():
        _LOG.info("Loading clusters from %s ...", csv_path)
        clusters = _load_clusters_from_csv(
            str(csv_path),
            max_clusters=args.max_clusters,
            exclude_datasets=args.exclude_dataset or None,
        )
    else:
        _LOG.info("all_fields.csv not found — fetching from Brain API ...")
        clusters = _load_clusters_from_api(
            session,
            max_clusters=args.max_clusters,
            min_coverage=args.min_coverage,
        )
    _LOG.info("Loaded %d field clusters", len(clusters))

    # Load Q-table
    qtable = QTable()
    skeleton_counter = SkeletonCounter()

    bootstrap_logs = []
    llm_sim = Path(args.sim_log)
    if llm_sim.exists():
        bootstrap_logs.append(llm_sim)
    if args.bootstrap_sim_log:
        extra = Path(args.bootstrap_sim_log)
        if extra.exists():
            bootstrap_logs.append(extra)

    if bootstrap_logs:
        n = bootstrap_qtable(bootstrap_logs, qtable, skeleton_counter=skeleton_counter)
        _LOG.info("Bootstrapped Q-table from %d records", n)

    similarity_bootstrap_logs = [Path(p) for p in args.similarity_bootstrap_log]
    if args.bootstrap_sim_log:
        similarity_bootstrap_logs.append(Path(args.bootstrap_sim_log))
    semantic_thresholds = SemanticThresholds(
        review_threshold=args.semantic_review_threshold,
        simulate_threshold=args.semantic_simulate_threshold,
        quarantine_threshold=args.semantic_quarantine_threshold,
    )

    # Run
    settings = dict(DEFAULT_SIMULATION_SETTINGS)
    settings["delay"] = args.delay
    loop = LLMExplorationLoop(
        clusters=clusters,
        session=session,
        sim_log=args.sim_log,
        api_key=api_key,
        batch_size=args.batch_size,
        credentials=args.credentials,
        settings=settings,
        qtable=qtable,
        skeleton_counter=skeleton_counter,
        dry_run=args.dry_run,
        competition_id=args.competition_id,
        similarity_bootstrap_logs=similarity_bootstrap_logs,
        semantic_novelty=not args.no_semantic_novelty,
        semantic_thresholds=semantic_thresholds,
    )
    loop.run(n_rounds=args.rounds)


def compare(
    grammar_log: str = "alpha_discovery/grammar_method/artifacts/all_sim.jsonl",
    llm_log: str = "alpha_discovery/legacy/llm_grammar_control_method/artifacts/llm_sim.jsonl",
) -> None:
    """Print a side-by-side comparison of grammar vs LLM sim logs."""
    from collections import Counter

    def analyze(path: str) -> dict:
        records = _load_jsonl(Path(path))
        done = [r for r in records if r.get("status") == "done"]
        sharpes = [float(r["is_sharpe"]) for r in done if r.get("is_sharpe") is not None]
        fitnesses = [float(r["is_fitness"]) for r in done if r.get("is_fitness") is not None and r.get("is_fitness") != 0]
        grades = Counter(r.get("grade") or "?" for r in done)
        passing = [s for s in sharpes if s >= 1.25]
        passing_fit = [f for f in fitnesses if f >= 1.0]
        corrs = [float(r["max_self_correlation"]) for r in done if r.get("max_self_correlation") is not None]
        top = sorted(done, key=lambda r: float(r.get("is_sharpe") or 0), reverse=True)[:5]
        return {
            "total": len(done), "sharpes": sharpes, "fitnesses": fitnesses,
            "grades": grades, "passing": passing, "passing_fit": passing_fit,
            "corrs": corrs, "top": top,
        }

    def _mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def _median(xs): return sorted(xs)[len(xs) // 2] if xs else 0.0

    g = analyze(grammar_log)
    l = analyze(llm_log)

    print(f"\n{'':30}  {'GRAMMAR':>12}  {'LLM':>12}  {'DELTA':>10}")
    print(f"{'─'*68}")
    print(f"{'Done records':30}  {g['total']:>12}  {l['total']:>12}")
    gp = len(g['passing']); lp = len(l['passing'])
    gpr = gp/g['total']*100 if g['total'] else 0
    lpr = lp/l['total']*100 if l['total'] else 0
    print(f"{'Pass (sharpe>=1.25)':30}  {gp:>9} {gpr:>4.1f}%  {lp:>9} {lpr:>4.1f}%  {lpr-gpr:>+8.1f}pp")
    print(f"{'Mean sharpe':30}  {_mean(g['sharpes']):>12.3f}  {_mean(l['sharpes']):>12.3f}  {_mean(l['sharpes'])-_mean(g['sharpes']):>+10.3f}")
    print(f"{'Median sharpe':30}  {_median(g['sharpes']):>12.3f}  {_median(l['sharpes']):>12.3f}  {_median(l['sharpes'])-_median(g['sharpes']):>+10.3f}")
    print(f"{'Max sharpe':30}  {max(g['sharpes'],default=0):>12.3f}  {max(l['sharpes'],default=0):>12.3f}  {max(l['sharpes'],default=0)-max(g['sharpes'],default=0):>+10.3f}")
    print(f"{'Mean fitness (non-zero)':30}  {_mean(g['fitnesses']):>12.3f}  {_mean(l['fitnesses']):>12.3f}  {_mean(l['fitnesses'])-_mean(g['fitnesses']):>+10.3f}")
    print(f"{'Max fitness':30}  {max(g['fitnesses'],default=0):>12.3f}  {max(l['fitnesses'],default=0):>12.3f}  {max(l['fitnesses'],default=0)-max(g['fitnesses'],default=0):>+10.3f}")
    if g['corrs'] or l['corrs']:
        print(f"{'Mean self-corr':30}  {_mean(g['corrs']):>12.3f}  {_mean(l['corrs']):>12.3f}")
    print()
    print("Grade distribution:")
    all_grades = sorted(set(list(g['grades']) + list(l['grades'])), key=lambda x: (x is None, x))
    for grade in all_grades:
        gn = g['grades'].get(grade, 0); ln = l['grades'].get(grade, 0)
        print(f"  {str(grade):15}  grammar={gn:4}  llm={ln:4}")

    for label, stat in [("GRAMMAR", g), ("LLM", l)]:
        print(f"\nTop 5 {label}:")
        for r in stat["top"]:
            corr = r.get("max_self_correlation")
            corr_s = f" corr={corr:.2f}" if corr is not None else ""
            print(f"  sh={r['is_sharpe']:.3f} fit={r.get('is_fitness',0):.3f} {r.get('grade','?')}{corr_s}")
            print(f"    {r['expression'][:90]}")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "compare":
        g = _sys.argv[2] if len(_sys.argv) > 2 else "alpha_discovery/grammar_method/artifacts/all_sim.jsonl"
        l = _sys.argv[3] if len(_sys.argv) > 3 else "alpha_discovery/legacy/llm_grammar_control_method/artifacts/llm_sim.jsonl"
        compare(g, l)
    else:
        main()
