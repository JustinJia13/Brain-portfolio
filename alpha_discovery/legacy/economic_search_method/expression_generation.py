"""LLM-first expression generation for accepted economic theories."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .expression_validation import AcceptedExpression, BatchValidation, ExpressionValidator
from .grammar_context import build_brain_grammar_context
from .history_index import HistoryIndex
from .llm_agents import CandidateGeneratorAgent, ExpansionAgent, IterationAgent
from .portfolio_context import compact_portfolio_context
from .schemas import CandidatePlanRecord, TheorySpec
from .semantic_gate import SemanticGateResult, SemanticNoveltyGate
from .theory_store import TheoryStore


class LLMExpressionGenerator:
    def __init__(
        self,
        *,
        field_csv: Path,
        history_index: HistoryIndex | None,
        target_delay: int,
        min_distinct_skeletons: int = 4,
        max_window_only_variants: int = 3,
        agent: CandidateGeneratorAgent | None = None,
        iteration_agent: IterationAgent | None = None,
        expansion_agent: ExpansionAgent | None = None,
        semantic_gate: SemanticNoveltyGate | None = None,
    ) -> None:
        self.field_csv = Path(field_csv)
        self.history_index = history_index
        self.target_delay = int(target_delay)
        self.validator = ExpressionValidator(
            field_csv=self.field_csv,
            target_delay=self.target_delay,
            history_index=history_index,
            require_delay_metadata=True,
            min_distinct_skeletons=min_distinct_skeletons,
            max_window_only_variants=max_window_only_variants,
        )
        self.agent = agent or CandidateGeneratorAgent()
        self.iteration_agent = iteration_agent or IterationAgent()
        self.expansion_agent = expansion_agent or ExpansionAgent()
        self.semantic_gate = semantic_gate or SemanticNoveltyGate(
            history_index=history_index,
            target_delay=self.target_delay,
        )

    def generate(
        self,
        *,
        theory: TheorySpec,
        store: TheoryStore,
        stage: str,
        max_candidates: int,
        portfolio_context: dict[str, Any],
        history_summary: dict[str, Any],
        field_catalog_summary: dict[str, Any],
        mock: bool = False,
        max_repairs: int = 1,
        generation_mode: str = "probe",
    ) -> tuple[list[CandidatePlanRecord], dict[str, Any]]:
        requested = max(1, int(max_candidates or theory.max_probe_candidates))
        requested = min(requested, 60)
        mode = _generation_mode(generation_mode or stage)
        existing = store.theory_candidate_expressions(theory.theory_id, target_delay=self.target_delay)
        cross_delay = store.theory_candidate_expressions(theory.theory_id, exclude_delay=self.target_delay)
        prior_results = _compact_results(store.all_results(theory.theory_id, target_delay=self.target_delay))
        branch_decision = store.read_decision(theory.theory_id, target_delay=self.target_delay)
        request_payload = self._request_payload(
            theory=theory,
            stage=stage,
            generation_mode=mode,
            requested=requested,
            portfolio_context=portfolio_context,
            history_summary=history_summary,
            field_catalog_summary=field_catalog_summary,
            prior_results=prior_results,
            existing_expressions=sorted(existing)[-80:],
            branch_decision=branch_decision,
        )
        active_agent = self._agent_for_mode(mode)

        responses: list[dict[str, Any]] = []
        if mock:
            raw_candidates = _mock_expression_candidates(theory, self.validator, requested)
            responses.append({"attempt": 0, "mock": True, "parsed": raw_candidates})
        else:
            data = active_agent.generate_expressions(payload=request_payload)
            raw_candidates = _extract_candidate_list(data)
            responses.append(_response_payload(active_agent, attempt=0, parsed=data))

        validation = self.validator.validate_batch(
            theory=theory,
            candidates=raw_candidates,
            requested_count=requested,
            existing_expressions=existing,
            cross_delay_expressions=cross_delay,
        )

        repairs = 0
        while (
            not mock
            and validation.report.get("needs_repair")
            and repairs < max(0, int(max_repairs))
        ):
            repairs += 1
            repair_payload = dict(request_payload)
            repair_payload["repair_request"] = {
                "attempt": repairs,
                "validation_report": validation.report,
                "rejected_candidates": validation.rejected[:40],
                "currently_valid_candidates": [
                    _accepted_source(row) for row in validation.accepted
                ],
                "instruction": (
                    "Generate additional concrete FASTEXPR candidates that fix the validation failures. "
                    "Do not repeat valid or rejected expressions. Increase operator-structure and "
                    "field-role diversity instead of only changing windows."
                ),
            }
            data = active_agent.generate_expressions(payload=repair_payload)
            repair_candidates = _extract_candidate_list(data)
            responses.append(_response_payload(active_agent, attempt=repairs, parsed=data))
            raw_candidates = [_accepted_source(row) for row in validation.accepted] + repair_candidates
            validation = self.validator.validate_batch(
                theory=theory,
                candidates=raw_candidates,
                requested_count=requested,
                existing_expressions=existing,
                cross_delay_expressions=cross_delay,
            )

        semantic_result = self.semantic_gate.apply(
            theory=theory,
            candidates=validation.accepted,
            requested_count=requested,
            portfolio_context=portfolio_context,
            history_summary=history_summary,
            mock=mock,
        )
        validation.report["deterministic_accepted_count"] = validation.report.get("accepted_count", len(validation.accepted))
        validation.report["accepted_count"] = len(semantic_result.accepted)
        validation.report["semantic_quarantined_count"] = len(semantic_result.quarantined)
        validation.accepted = semantic_result.accepted
        records = [
            _candidate_record(
                theory=theory,
                accepted=row,
                stage=stage,
                target_delay=self.target_delay,
            )
            for row in validation.accepted
        ]
        artifact_paths = self._persist_generation(
            store=store,
            theory=theory,
            stage=stage,
            request_payload=request_payload,
            responses=responses,
            validation=validation,
            semantic_result=semantic_result,
        )
        report = dict(validation.report)
        report["artifact_paths"] = artifact_paths
        report["generation_mode"] = mode
        report["semantic_gate"] = semantic_result.report
        return records, report

    def _agent_for_mode(self, mode: str) -> Any:
        if mode == "iterate":
            return self.iteration_agent
        if mode == "expand":
            return self.expansion_agent
        return self.agent

    def _request_payload(
        self,
        *,
        theory: TheorySpec,
        stage: str,
        generation_mode: str,
        requested: int,
        portfolio_context: dict[str, Any],
        history_summary: dict[str, Any],
        field_catalog_summary: dict[str, Any],
        prior_results: list[dict[str, Any]],
        existing_expressions: list[str],
        branch_decision: dict[str, Any],
    ) -> dict[str, Any]:
        result_summary = _result_feedback_summary(prior_results)
        return {
            "task": "generate_diverse_fast_expr_candidates",
            "theory": theory.to_dict(),
            "stage": stage,
            "generation_mode": generation_mode,
            "target_delay": self.target_delay,
            "delay_definition": (
                "delay 0 uses yesterday's available data to trade today; "
                "delay 1 uses data from two days ago to trade today."
            ),
            "requested_count": requested,
            "brain_grammar_context": build_brain_grammar_context(),
            "field_catalog_summary": field_catalog_summary,
            "role_candidate_fields": self.validator.role_field_context(theory),
            "field_catalog_sample": self.validator.fields.compact_fields_for_prompt(limit=220),
            "prior_theory_expressions": existing_expressions,
            "prior_theory_results": prior_results,
            "prior_result_summary": result_summary,
            "branch_decision": branch_decision,
            "history_summary": _compact_history_summary(history_summary),
            "current_submitted_portfolio": compact_portfolio_context(
                portfolio_context or {},
                target_delay=self.target_delay,
            ),
            "novelty_rules": [
                "Do not repeat submitted economic incentives, field/operator structures, or high-correlation families.",
                "Do not use window/constant changes as a substitute for novelty. Sign flips are only acceptable when prior evidence suggests the mechanism direction may be wrong; they will be labeled low-novelty repair candidates.",
                "Use required datasets named in the theory roles when they exist in the supplied field catalog.",
                "Prefer multiple independent realizations of the same theory: interaction, gating, residual, reversal, rank/bucket, correlation, and change-based forms are examples of different realizations, not templates to copy mechanically.",
            ],
            "portfolio_objective": {
                "primary_metric": "Brain before/after score delta is fetched from the competition endpoint.",
                "fallback_metric": "If Brain score comparison is unavailable after retries, fallback delta_pool_sharpe is computed against submitted alphas with the same target_delay.",
                "global_correlation_metric": "global self-correlation is checked against all submitted alphas across delay 0 and delay 1.",
                "objective": "Generate candidates likely to pass standalone checks and improve competition score contribution.",
                "priority_metrics": [
                    "admission_decision=PROMOTE",
                    "positive admission_brain_score_delta, or positive fallback admission_delta_pool_sharpe",
                    "acceptable admission_global_max_corr",
                    "standalone checks pass",
                ],
            },
            "output_contract": {
                "format": _output_contract_format(generation_mode),
                "candidate_keys": [
                    "expression",
                    "field_roles",
                    "mechanism_translation",
                    "expected_sign",
                    "horizon_days",
                    "novelty_argument",
                    "risk_notes",
                    "generation_source",
                ],
                "expression_policy": "Concrete Brain FASTEXPR only; no templates or placeholders.",
            },
        }

    def _persist_generation(
        self,
        *,
        store: TheoryStore,
        theory: TheorySpec,
        stage: str,
        request_payload: dict[str, Any],
        responses: list[dict[str, Any]],
        validation: BatchValidation,
        semantic_result: SemanticGateResult,
    ) -> dict[str, str]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = store.path_for(theory.theory_id) / f"generation_{stage}_delay{self.target_delay}_{stamp}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        request_path = prefix.with_name(prefix.name + "_request.json")
        response_path = prefix.with_name(prefix.name + "_response.json")
        validation_path = prefix.with_name(prefix.name + "_validation.json")
        rejected_path = prefix.with_name(prefix.name + "_rejected.jsonl")
        semantic_request_path = prefix.with_name(prefix.name + "_semantic_request.jsonl")
        semantic_response_path = prefix.with_name(prefix.name + "_semantic_response.jsonl")
        semantic_quarantine_path = prefix.with_name(prefix.name + "_semantic_quarantined.jsonl")
        request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True, default=str))
        response_path.write_text(json.dumps(responses, indent=2, sort_keys=True, default=str))
        validation_path.write_text(json.dumps(validation.report, indent=2, sort_keys=True, default=str))
        with rejected_path.open("w") as f:
            for row in validation.rejected:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        with semantic_request_path.open("w") as f:
            for row in semantic_result.requests:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        with semantic_response_path.open("w") as f:
            for row in semantic_result.responses:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        with semantic_quarantine_path.open("w") as f:
            for row in semantic_result.quarantined:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

        store.append_event(
            theory.theory_id,
            "llm_generated_expressions",
            {
                "stage": stage,
                "target_delay": self.target_delay,
                "accepted_count": len(validation.accepted),
                "rejected_count": len(validation.rejected),
                "semantic_quarantined_count": len(semantic_result.quarantined),
                "request_path": str(request_path),
                "response_path": str(response_path),
            },
        )
        store.append_event(
            theory.theory_id,
            "semantic_gate_applied",
            {
                "stage": stage,
                "target_delay": self.target_delay,
                "accepted_count": len(semantic_result.accepted),
                "quarantined_count": len(semantic_result.quarantined),
                "request_path": str(semantic_request_path),
                "response_path": str(semantic_response_path),
                "quarantined_path": str(semantic_quarantine_path),
                "decisions": semantic_result.report.get("decisions", {}),
            },
        )
        if validation.rejected:
            store.append_event(
                theory.theory_id,
                "validation_rejected",
                {
                    "stage": stage,
                    "target_delay": self.target_delay,
                    "n_rejected": len(validation.rejected),
                    "path": str(rejected_path),
                    "top_reasons": _top_rejection_reasons(validation.rejected),
                },
            )
        return {
            "request": str(request_path),
            "response": str(response_path),
            "validation": str(validation_path),
            "rejected": str(rejected_path),
            "semantic_request": str(semantic_request_path),
            "semantic_response": str(semantic_response_path),
            "semantic_quarantined": str(semantic_quarantine_path),
        }


def _extract_candidate_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        status = str(data.get("status") or "").upper()
        if status in {"NO_GENERATION", "NO_ITERATION", "NO_EXPANSION"}:
            return []
        for key in ("candidates", "expressions", "items", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _generation_mode(value: str) -> str:
    mode = str(value or "probe").lower()
    if mode in {"iterate", "iteration", "repair"}:
        return "iterate"
    if mode in {"expand", "expansion"}:
        return "expand"
    return "probe"


def _output_contract_format(mode: str) -> str:
    if mode == "iterate":
        return "JSON array of candidate objects, or {'status':'NO_ITERATION','reason':'...'}"
    if mode == "expand":
        return "JSON array of candidate objects, or {'status':'NO_EXPANSION','reason':'...'}"
    return "JSON array of candidate objects, or {'status':'NO_GENERATION','reason':'...'}"


def _response_payload(agent: CandidateGeneratorAgent, *, attempt: int, parsed: Any) -> dict[str, Any]:
    client = getattr(agent, "client", None)
    return {
        "attempt": attempt,
        "parsed": parsed,
        "raw_text": getattr(client, "last_raw_text", None),
        "stop_reason": getattr(client, "last_stop_reason", None),
    }


def _candidate_record(
    *,
    theory: TheorySpec,
    accepted: AcceptedExpression,
    stage: str,
    target_delay: int,
) -> CandidatePlanRecord:
    fields = accepted.fields
    first_field = fields[0] if fields else ""
    first_meta = None
    # Field metadata is already compacted into validation outputs. Reconstruct dataset/category
    # from accepted lists so candidate records stay submit-compatible without template expansion.
    dataset_id = accepted.datasets[0] if accepted.datasets else ""
    category = accepted.categories[0] if accepted.categories else ""
    skeleton = ">".join(accepted.operator_skeleton) or "llm_expression"
    source = accepted.source
    params = {
        "theory_title": theory.title,
        "expected_sign": source.get("expected_sign") or theory.expected_sign,
        "mechanism_translation": source.get("mechanism_translation", ""),
        "novelty_argument": source.get("novelty_argument", ""),
        "risk_notes": source.get("risk_notes", []),
        "generation_source": source.get("generation_source") or "llm_expression_generator",
        "branch_action": source.get("branch_action") or stage,
        "repaired_failure_modes": source.get("repaired_failure_modes", []),
        "expected_metric_effect": source.get("expected_metric_effect", ""),
        "expansion_axis": source.get("expansion_axis", ""),
        "pool_contribution_argument": source.get("pool_contribution_argument", ""),
        "operator_skeleton": accepted.operator_skeleton,
        "target_delay": target_delay,
        "cross_delay_retest": accepted.cross_delay_retest,
        "validation": {
            "windowless_fingerprint": accepted.windowless_fingerprint,
            "role_binding_fingerprint": accepted.role_binding_fingerprint,
            "datasets": accepted.datasets,
        },
        "semantic_similarity_score": source.get("semantic_similarity_score"),
        "semantic_similarity_dimensions": source.get("semantic_similarity_dimensions", {}),
        "semantic_similarity_confidence": source.get("semantic_similarity_confidence"),
        "semantic_tags": source.get("semantic_tags", []),
        "semantic_nearest_neighbors": source.get("semantic_nearest_neighbors", []),
        "semantic_gate_decision": source.get("semantic_gate_decision", ""),
        "semantic_gate_reason": source.get("semantic_gate_reason", ""),
        "semantic_gate_allowed_by_quota": source.get("semantic_gate_allowed_by_quota", False),
        "nearest_matching_mechanism": source.get("nearest_matching_mechanism", ""),
    }
    return CandidatePlanRecord.from_expression(
        theory_id=theory.theory_id,
        expression=accepted.expression,
        fields=fields,
        cluster_ids=accepted.cluster_ids,
        dataset_id=dataset_id,
        category=category,
        template_family="llm_generated",
        template_id=skeleton,
        field_roles=accepted.field_roles,
        experiment_stage=stage,
        horizon_bucket=_horizon_bucket(source.get("horizon_days") or theory.horizon_days),
        novelty_score=accepted.novelty_score,
        target_delay=target_delay,
        cross_delay_retest=accepted.cross_delay_retest,
        params=params,
    )


def _mock_expression_candidates(
    theory: TheorySpec,
    validator: ExpressionValidator,
    requested: int,
) -> list[dict[str, Any]]:
    role_context = validator.role_field_context(theory)
    role_names = list(role_context)
    signal_role = next((r for r in role_names if "signal" in r.lower() or "primary" in r.lower()), role_names[0] if role_names else "signal")
    condition_role = next(
        (
            r
            for r in role_names
            if r != signal_role and any(n in r.lower() for n in ("condition", "confirm", "gate", "context", "anchor", "timing"))
        ),
        next((r for r in role_names if r != signal_role), "condition"),
    )
    signal_fields = [row["field_id"] for row in role_context.get(signal_role, [])]
    condition_fields = [row["field_id"] for row in role_context.get(condition_role, [])]
    all_fields = list(validator.fields.by_id)
    if not signal_fields:
        signal_fields = all_fields[:4]
    if not condition_fields:
        condition_fields = [f for f in all_fields if f not in signal_fields[:1]][:4] or signal_fields
    a = signal_fields[0]
    b = condition_fields[0] if condition_fields else a
    c = signal_fields[1] if len(signal_fields) > 1 else a
    d = condition_fields[1] if len(condition_fields) > 1 else b
    templates = [
        f"group_neutralize(multiply(rank(ts_zscore({a}, 20)), rank(ts_zscore({b}, 10))), subindustry)",
        f"trade_when(rank(ts_zscore({b}, 20)), group_neutralize(rank(ts_delta({a}, 5)), subindustry), 0)",
        f"group_rank(ts_zscore({a}, 60), bucket(rank({b}), buckets=5))",
        f"group_neutralize(subtract(rank(ts_zscore({a}, 40)), rank(ts_zscore({b}, 20))), subindustry)",
        f"group_neutralize(rank(ts_corr({a}, {b}, 20)), subindustry)",
        f"group_neutralize(multiply(rank(reverse(ts_delta({a}, 10))), rank(ts_std_dev({b}, 20))), subindustry)",
        f"group_neutralize(multiply(rank(ts_delta({c}, 10)), rank(ts_zscore({d}, 30))), subindustry)",
        f"trade_when(rank(ts_delta({d}, 10)), group_neutralize(rank(ts_zscore({c}, 40)), subindustry), 0)",
        f"group_rank(ts_delta({c}, 20), bucket(rank({d}), buckets=5))",
        f"group_neutralize(rank(ts_covariance({c}, {d}, 20)), subindustry)",
        f"group_neutralize(add(rank(ts_zscore({a}, 20)), rank(reverse(ts_delta({d}, 10)))), subindustry)",
        f"group_neutralize(multiply(rank(ts_rank({c}, 20)), rank(ts_av_diff({b}, 20))), subindustry)",
    ]
    out = []
    for expr in templates[:requested]:
        out.append(
            {
                "expression": expr,
                "field_roles": {signal_role: a if a in expr else c, condition_role: b if b in expr else d},
                "mechanism_translation": "Mock diverse expression for testing the LLM-first generation path.",
                "expected_sign": theory.expected_sign,
                "horizon_days": (theory.horizon_days or [20])[0],
                "novelty_argument": "Mock candidate uses a different operator realization.",
                "risk_notes": [],
                "generation_source": "mock_llm_expression_generator",
            }
        )
    return out


def _result_feedback_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in results if r.get("status") == "done"]
    if not done:
        return {"n_done": 0}
    ranked = sorted(done, key=lambda r: _safe_float(r.get("is_sharpe")), reverse=True)
    failed_checks: dict[str, int] = {}
    admission: dict[str, int] = {}
    pool_hurt: list[dict[str, Any]] = []
    pool_positive: list[dict[str, Any]] = []
    for row in done:
        for check in row.get("checks") or []:
            if isinstance(check, dict) and check.get("result") == "FAIL":
                name = str(check.get("name") or "UNKNOWN")
                failed_checks[name] = failed_checks.get(name, 0) + 1
        decision = row.get("admission_decision")
        if decision:
            admission[str(decision)] = admission.get(str(decision), 0) + 1
        delta = row.get("admission_brain_score_delta")
        if delta is None:
            delta = row.get("admission_delta_pool_sharpe")
        if delta is not None:
            item = {
                "expression": str(row.get("expression") or "")[:180],
                "is_sharpe": row.get("is_sharpe"),
                "is_fitness": row.get("is_fitness"),
                "admission_decision": row.get("admission_decision"),
                "admission_reason": row.get("admission_reason"),
                "admission_brain_score_delta": row.get("admission_brain_score_delta"),
                "admission_delta_pool_sharpe": delta,
                "admission_signal": row.get("admission_signal"),
                "admission_global_max_corr": row.get("admission_global_max_corr"),
            }
            if _safe_float(delta) > 0:
                pool_positive.append(item)
            else:
                pool_hurt.append(item)
    pool_hurt.sort(key=lambda r: _safe_float(r.get("admission_delta_pool_sharpe")))
    pool_positive.sort(key=lambda r: _safe_float(r.get("admission_delta_pool_sharpe")), reverse=True)
    return {
        "n_done": len(done),
        "best": [_compact_result_row(r) for r in ranked[:8]],
        "worst": [_compact_result_row(r) for r in ranked[-5:]],
        "failed_checks": failed_checks,
        "admission_decisions": admission,
        "positive_pool_examples": pool_positive[:8],
        "pool_hurt_examples": pool_hurt[:8],
    }


def _compact_history_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in ("total_records", "standalone_pass_records", "dataset", "category", "field", "operator", "failed_checks", "delay", "pool_impact"):
        value = summary.get(key)
        if isinstance(value, list):
            value = value[:20]
        out[key] = value
    return out


def _compact_results(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in list(rows)[-40:]:
        out.append(_compact_result_row(row))
    return out


def _compact_result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "expression": row.get("expression"),
        "experiment_stage": row.get("experiment_stage"),
        "status": row.get("status"),
        "is_sharpe": row.get("is_sharpe"),
        "is_fitness": row.get("is_fitness"),
        "is_turnover": row.get("is_turnover"),
        "checks": row.get("checks", [])[:8] if isinstance(row.get("checks"), list) else [],
        "admission_decision": row.get("admission_decision"),
        "admission_reason": row.get("admission_reason"),
        "admission_brain_score_delta": row.get("admission_brain_score_delta"),
        "admission_delta_pool_sharpe": row.get("admission_delta_pool_sharpe"),
        "admission_signal": row.get("admission_signal"),
        "admission_global_max_corr": row.get("admission_global_max_corr"),
        "target_delay": row.get("target_delay"),
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _accepted_source(row: AcceptedExpression) -> dict[str, Any]:
    source = dict(row.source)
    source["expression"] = row.expression
    source["field_roles"] = row.field_roles
    return source


def _top_rejection_reasons(rejected: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rejected:
        reason = str(row.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10])


def _horizon_bucket(value: Any) -> str:
    if isinstance(value, int):
        vals = [value]
    elif isinstance(value, list):
        vals = []
        for item in value:
            try:
                vals.append(int(item))
            except (TypeError, ValueError):
                pass
    else:
        vals = []
    if not vals:
        vals = [60]
    med = sorted(vals)[len(vals) // 2]
    if med <= 20:
        return "short"
    if med <= 120:
        return "medium"
    return "long"
