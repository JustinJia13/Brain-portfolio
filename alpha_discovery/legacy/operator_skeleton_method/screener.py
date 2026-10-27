"""Submit screening expressions to Brain and parse results.

Reuses submit_alpha / wait_for_completion from alpha_discovery.grammar_method.run_pipeline
so we don't duplicate the retry / re-auth logic.
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.brain.api import fetch_yearly_stats_with_retry
from core.selfcorr import get_alpha_detail
from alpha_discovery.grammar_method.run_pipeline import submit_alpha, wait_for_completion, append_jsonl


DEFAULT_SIM_SETTINGS: dict[str, Any] = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.01,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}


@dataclass
class ScreeningResult:
    expression: str
    template_id: str
    field_a: str
    field_b: str
    field_c: str | None
    label_a: str
    label_b: str
    label_c: str | None
    window: int | None
    # Brain results
    alpha_id: str | None
    status: str                        # "done" | "no_alpha_id" | "submit_failed"
    is_sharpe: float
    is_fitness: float
    is_turnover: float | None
    is_returns: float | None
    is_drawdown: float | None
    yearly_sharpes: dict[int, float]
    concentrated_weight_ok: bool
    sub_universe_sharpe_ok: bool
    grade: str
    logged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def year_consistency_score(self) -> float:
        """mean_sharpe / (1 + std_sharpe) — penalises single-year spikes."""
        vals = list(self.yearly_sharpes.values())
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean
        std = statistics.stdev(vals)
        return mean / (1.0 + std)

    @property
    def screening_score(self) -> float:
        """Primary ranking metric: IS Sharpe × year consistency."""
        return self.is_sharpe * self.year_consistency_score

    def passes_promotion(
        self,
        sharpe_threshold: float = 1.25,
    ) -> bool:
        return (
            self.status == "done"
            and self.is_sharpe >= sharpe_threshold
            and self.concentrated_weight_ok
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "logged_at": self.logged_at,
            "search_lane": "operator_search",
            "template_id": self.template_id,
            "expression": self.expression,
            "field_a": self.field_a,
            "field_b": self.field_b,
            "field_c": self.field_c,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "label_c": self.label_c,
            "window": self.window,
            "alpha_id": self.alpha_id,
            "status": self.status,
            "is_sharpe": self.is_sharpe,
            "is_fitness": self.is_fitness,
            "is_turnover": self.is_turnover,
            "is_returns": self.is_returns,
            "is_drawdown": self.is_drawdown,
            "yearly_sharpes": self.yearly_sharpes,
            "concentrated_weight_ok": self.concentrated_weight_ok,
            "sub_universe_sharpe_ok": self.sub_universe_sharpe_ok,
            "grade": self.grade,
            "year_consistency_score": self.year_consistency_score,
            "screening_score": self.screening_score,
        }


def _parse_is_checks(is_checks: list[dict]) -> dict[str, bool]:
    return {c["name"]: c.get("result", "FAIL") == "PASS" for c in (is_checks or [])}


def _build_screening_result(
    expression: str,
    template_id: str,
    field_a: str,
    field_b: str,
    field_c: str | None,
    label_a: str,
    label_b: str,
    label_c: str | None,
    window: int | None,
    alpha_id: str | None,
    status: str,
    alpha_detail: dict[str, Any],
    yearly_stats: dict[int, dict],
) -> ScreeningResult:
    is_stats = alpha_detail.get("is") or {}
    raw_checks = is_stats.get("checks") or []
    checks = _parse_is_checks(raw_checks)
    yearly_sharpes = {
        int(yr): float(stats.get("sharpe") or 0.0)
        for yr, stats in (yearly_stats or {}).items()
        if stats.get("sharpe") is not None
    }
    raw_turnover = is_stats.get("turnover")
    raw_returns = is_stats.get("returns")
    raw_drawdown = is_stats.get("drawdown")
    return ScreeningResult(
        expression=expression,
        template_id=template_id,
        field_a=field_a,
        field_b=field_b,
        field_c=field_c,
        label_a=label_a,
        label_b=label_b,
        label_c=label_c,
        window=window,
        alpha_id=alpha_id,
        status=status,
        is_sharpe=float(is_stats.get("sharpe") or 0.0),
        is_fitness=float(is_stats.get("fitness") or 0.0),
        is_turnover=float(raw_turnover) if raw_turnover is not None else None,
        is_returns=float(raw_returns) if raw_returns is not None else None,
        is_drawdown=float(raw_drawdown) if raw_drawdown is not None else None,
        yearly_sharpes=yearly_sharpes,
        concentrated_weight_ok=checks.get("CONCENTRATED_WEIGHT", True),
        sub_universe_sharpe_ok=checks.get("LOW_SUB_UNIVERSE_SHARPE", True),
        grade=alpha_detail.get("grade") or "",
    )


def submit_screening_batch(
    session,
    candidates: list[dict[str, Any]],
    log_path: str | Path,
    *,
    settings: dict[str, Any] | None = None,
    batch_size: int = 3,
    sleep_seconds: int = 30,
    max_submit_retries: int = 5,
    credentials_path: str = "",
) -> list[ScreeningResult]:
    """Submit candidates using a rolling window of batch_size concurrent simulations.

    Each candidate dict must have:
        expression, template_id, field_a, field_b, field_c (or None), window (or None)

    Results are appended to log_path as JSONL after each completion.
    """
    sim_settings = settings or DEFAULT_SIM_SETTINGS
    log_path = str(log_path)
    results: list[ScreeningResult] = []

    done_exprs: set[str] = set()
    if Path(log_path).exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("expression"):
                        done_exprs.add(rec["expression"])
                except json.JSONDecodeError:
                    pass

    remaining = [c for c in candidates if c["expression"] not in done_exprs]
    print(f"  Screening: {len(candidates)} candidates, {len(remaining)} new")

    if not remaining:
        return results

    total = len(remaining)

    def _simulate_one(cand: dict) -> ScreeningResult:
        expr = cand["expression"]
        print(f"[SCREEN]   submit: {expr[:80]}")
        url = submit_alpha(
            session, expr, sim_settings,
            max_retries=max_submit_retries,
            sleep_seconds=sleep_seconds,
            credentials_path=credentials_path,
        )
        if not url:
            print(f"[SCREEN]   FAILED to submit: {expr[:60]}")
            return _build_screening_result(
                expr, cand["template_id"],
                cand["field_a"], cand["field_b"],
                cand.get("field_c"),
                cand.get("label_a", ""), cand.get("label_b", ""),
                cand.get("label_c"),
                cand.get("window"),
                None, "submit_failed", {}, {},
            )

        print(f"[SCREEN]   wait:   {expr[:80]}")
        final = wait_for_completion(session, url, sleep_seconds=sleep_seconds,
                                    credentials_path=credentials_path)
        alpha_id = final.get("alpha")
        if alpha_id:
            detail = get_alpha_detail(session, alpha_id)
            yearly = fetch_yearly_stats_with_retry(session, alpha_id)
            status = "done"
        else:
            detail, yearly, status = {}, {}, "no_alpha_id"

        return _build_screening_result(
            expr, cand["template_id"],
            cand["field_a"], cand["field_b"],
            cand.get("field_c"),
            cand.get("label_a", ""), cand.get("label_b", ""),
            cand.get("label_c"),
            cand.get("window"),
            alpha_id, status, detail, yearly,
        )

    logged = 0
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_simulate_one, c): c for c in remaining}
        for future in as_completed(futures):
            sr = future.result()
            logged += 1
            sharpe_str = f"{sr.is_sharpe:.3f}"
            fit_str = f"{sr.is_fitness:.3f}"
            print(f"[SCREEN]   done [{logged}/{total}]: sharpe={sharpe_str} fitness={fit_str} grade={sr.grade}")
            append_jsonl(log_path, sr.as_record())
            results.append(sr)

    return results
