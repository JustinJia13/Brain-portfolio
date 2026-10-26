"""Deterministic novelty checks and result interpretation."""

from __future__ import annotations

import statistics
from typing import Any

from .history_index import HistoryIndex
from .schemas import TheorySpec


def theory_text(theory: TheorySpec) -> str:
    return " ".join(
        [
            theory.title,
            theory.mechanism,
            str(dict(theory.data_roles)),
            " ".join(theory.field_cluster_targets),
            " ".join(theory.template_families),
            theory.novelty_claim,
        ]
    )


class DeterministicNoveltyCritic:
    """Fast rule-based filter before optional LLM critique."""

    def __init__(
        self,
        history_index: HistoryIndex,
        portfolio_context: dict[str, Any] | None = None,
        target_delay: int | None = None,
    ) -> None:
        self.history_index = history_index
        self.portfolio_context = portfolio_context or {}
        self.target_delay = target_delay

    def critique(self, theory: TheorySpec) -> dict[str, Any]:
        report = self.history_index.theory_risk_report(theory_text(theory))
        text = theory_text(theory).lower()
        reasons = list(report.get("risks", []))
        categories = set(report.get("matches", {}).get("category", []))
        datasets = set(report.get("matches", {}).get("dataset", []))
        portfolio_overlap = self._portfolio_overlap(text)
        if portfolio_overlap:
            reasons.append(
                "overlaps submitted portfolio themes: " + ", ".join(portfolio_overlap[:5])
            )
        non_fundamental_terms = {"news", "option", "price_volume", "social_media", "model"}
        has_cross_family = bool(categories & non_fundamental_terms) or any(
            term in text
            for term in ("news", "option", "social", "sentiment", "volume", "price", "volatility", "model")
        )

        if portfolio_overlap and not has_cross_family:
            decision = "REJECT"
        elif len(portfolio_overlap) >= 3:
            decision = "REVISE"
        elif report.get("saturated_single_family"):
            decision = "REJECT"
        elif (
            len(datasets) <= 1
            and (datasets & {"analyst4", "fundamental6", "fundamental2"})
            and not has_cross_family
        ):
            decision = "REVISE"
            reasons.append("single crowded dataset needs an orthogonal conditioning dataset")
        elif not has_cross_family and categories <= {"analyst", "fundamental"}:
            decision = "REVISE"
            reasons.append("needs explicit non-fundamental interaction or conditioning")
        else:
            decision = "ACCEPT"

        return {
            "decision": decision,
            "reason": "; ".join(reasons) if reasons else "passes deterministic novelty gates",
            "risk_report": report,
            "portfolio_overlap": portfolio_overlap,
            "target_delay": self.target_delay,
        }

    def _portfolio_overlap(self, text: str) -> list[str]:
        matches: list[str] = []
        for item in self.portfolio_context.get("do_not_repeat") or []:
            low = str(item).lower()
            tokens = [t for t in low.replace("|", " ").replace(":", " ").split() if len(t) >= 5]
            if tokens and sum(1 for t in tokens if t in text) >= min(2, len(tokens)):
                matches.append(str(item))
        for exposure in self.portfolio_context.get("theme_exposures") or []:
            theme = str(exposure.get("theme") or "")
            low = theme.lower()
            tokens = [t for t in low.replace("|", " ").replace("+", " ").split() if len(t) >= 5]
            if tokens and sum(1 for t in tokens if t in text) >= min(2, len(tokens)):
                matches.append(theme)
        seen: set[str] = set()
        out: list[str] = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                out.append(match)
        return out


class DeterministicResultInterpreter:
    """Apply fixed stopping rules after simulation result ingestion."""

    def interpret(
        self,
        theory: TheorySpec,
        results: list[dict[str, Any]],
        *,
        target_delay: int | None = None,
    ) -> dict[str, Any]:
        if target_delay is not None:
            results = [r for r in results if _record_delay(r) == int(target_delay)]
        done = [r for r in results if r.get("status") == "done"]
        if not done:
            return {
                "decision": "ITERATE",
                "reason": "no completed simulations ingested yet",
                "metrics": {"target_delay": target_delay},
                "updated_status": theory.status,
                "target_delay": target_delay,
            }

        sharpes = [_float(r.get("is_sharpe")) for r in done if r.get("is_sharpe") is not None]
        fitness = [_float(r.get("is_fitness")) for r in done if r.get("is_fitness") is not None]
        best = max(sharpes) if sharpes else 0.0
        median = statistics.median(sharpes) if sharpes else 0.0
        standalone = [r for r in done if _standalone_passing(r)]
        def _contribution_delta(row: dict[str, Any]) -> float | None:
            value = row.get("admission_brain_score_delta")
            if value is None:
                value = row.get("admission_delta_pool_sharpe")
            return _float(value) if value is not None else None

        pos_delta = [r for r in done if (_contribution_delta(r) or 0.0) > 0]
        positive_residual = [r for r in done if _float(r.get("admission_residual_sharpe")) > 0.0]
        high_corr_bad = [
            r
            for r in standalone
            if _corr(r) is not None
            and _corr(r) >= 0.70
            and ((_contribution_delta(r) or 0.0) <= 0)
        ]

        if any((r.get("admission_decision") == "PROMOTE") for r in standalone) or (standalone and pos_delta):
            decision = "PROMOTE"
            reason = "standalone passer has positive marginal portfolio evidence"
            updated_status = "PROMOTED"
        elif best < 0.8:
            decision = "KILL"
            reason = "best Sharpe below 0.8 kill threshold"
            updated_status = "KILLED"
        elif standalone and len(high_corr_bad) == len(standalone):
            decision = "KILL"
            reason = "standalone passers are high-correlation with non-positive contribution delta"
            updated_status = "KILLED"
        elif len([r for r in done if _float(r.get("is_sharpe")) > 1.0]) >= 2 and (
            pos_delta or not any(_contribution_delta(r) is not None for r in done)
        ):
            decision = "EXPAND"
            reason = "multiple variants show signal and no conclusive negative pool evidence"
            updated_status = "EXPAND"
        else:
            decision = "ITERATE"
            reason = "signal is not dead, but evidence is below expansion threshold"
            updated_status = "ITERATE"

        return {
            "decision": decision,
            "reason": reason,
            "updated_status": updated_status,
            "target_delay": target_delay,
            "metrics": {
                "target_delay": target_delay,
                "n_done": len(done),
                "best_sharpe": best,
                "median_sharpe": median,
                "best_fitness": max(fitness) if fitness else None,
                "n_standalone_pass": len(standalone),
                "n_positive_delta": len(pos_delta),
                "n_positive_residual": len(positive_residual),
            },
        }

    def memory_markdown(self, theory: TheorySpec, decision: dict[str, Any]) -> str:
        metrics = decision.get("metrics") or {}
        lines = [
            f"# {theory.title}",
            "",
            f"Status: {decision.get('updated_status', theory.status)}",
            f"Decision: {decision.get('decision')} - {decision.get('reason')}",
            "",
            "## Metrics",
        ]
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines += [
            "",
            "## Next Rule",
            "Stop when kill/promote is reached. Otherwise run at most the configured max_iterations.",
        ]
        return "\n".join(lines)


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _standalone_passing(record: dict[str, Any]) -> bool:
    checks = record.get("checks") or []
    return bool(checks) and not any(c.get("result") == "FAIL" for c in checks)


def _record_delay(record: dict[str, Any]) -> int:
    for source in (
        {"delay": record.get("target_delay")},
        record.get("settings") or {},
        record.get("settings_override") or {},
        {"delay": record.get("admission_delay")},
    ):
        try:
            return int(source.get("delay", 1))
        except (AttributeError, TypeError, ValueError):
            continue
    return 1


def _corr(record: dict[str, Any]) -> float | None:
    value = record.get("admission_global_max_corr")
    if value is None:
        value = record.get("max_self_correlation")
    if value is None:
        return None
    return _float(value)
