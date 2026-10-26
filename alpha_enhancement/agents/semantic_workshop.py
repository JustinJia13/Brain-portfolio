"""Two-agent LLM helpers for post-grammar semantic workshop.

The diagnostic agent never emits FASTEXPR for submission.  It annotates a seed
or round result into a typed semantic template and failure hypothesis.

The generator agent emits candidate expressions plus internal rationale.  The
runner submits only the ``expression`` field.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from core import llm as anthropic
from core.json_utils import extract_json_text, parse_json_object


_FASTEXPR_RULES = """
FASTEXPR syntax rules:
- Group identifiers are barewords: sector, subindustry, industry, market.
- rank/zscore/normalize/scale/winsorize take exactly one argument.
- group_rank(expr, group), group_zscore(expr, group), group_neutralize(expr, group).
- ts_sum, ts_mean, ts_std_dev, ts_rank, ts_zscore take (expr, window).
- ts_delta(expr, days), ts_delay(expr, days), ts_decay_linear(expr, window).
- kth_element(expr, window, k=N).
- if_else(condition, true_expr, false_expr).
"""


_DIAGNOSTIC_SYSTEM = f"""You are a WorldQuant Brain alpha diagnostic agent.

You do NOT generate FASTEXPR expressions.  Your job is to turn a grammar or
workshop seed into a semantic template and a failure/repair hypothesis.

Use the provided simulation metrics, field list, prior workshop results, and
robustness card.  Be concrete and conservative.  Do not optimize exact
robustness rows; use robustness only to diagnose whether signal identity is
stable.

Evaluation objective:
- Primary goal is higher admission_brain_score_delta, or fallback
  admission_delta_pool_sharpe when Brain score is unavailable.
- A positive score delta is not enough by itself.  Treat weak standalone
  Sharpe/Fitness, fragile checks, excessive turnover, high self-correlation, or
  identity failure as real defects to diagnose.
- Prefer repairs that preserve or improve Fitness and Sharpe, reduce turnover
  when it is problematic, lower self-correlation when possible, and directly
  address named failing checks such as LOW_SUB_UNIVERSE_SHARPE, LOW_FITNESS,
  LOW_SHARPE, LOW_TURNOVER, HIGH_TURNOVER, or CONCENTRATED_WEIGHT.
- If score addition improves while standalone quality deteriorates, diagnose
  that as a watchlist/fragile outcome, not a solved alpha.
- Do not recommend parameter chasing.  Valid repairs must preserve or
  explicitly reframe the semantic signal identity.

If a probe card is provided, treat it as noisy prior evidence from simple
field/cluster probes.  It is not ground truth.  First decide whether the seed
alpha's simulated expression identity aligns with, weakly relates to,
contradicts, or should ignore the probe evidence.  Do not force a seed to match
the probe card when the seed expression and metrics imply a different signal.

Return ONLY one JSON object with this exact shape:
{{
  "signal_identity": "short label",
  "semantic_template": "typed template family",
  "field_roles": {{
    "main_signal": "role and field(s)",
    "secondary_fields": "compatible secondary field types",
    "gate_fields": "compatible trade_when/gating fields",
    "neutralizer_fields": "compatible neutralizer/control fields"
  }},
  "valid_mutation_axes": ["axis", "..."],
  "invalid_mutations": ["mutation to avoid", "..."],
  "failure_diagnosis": "diagnosis based on metrics and robustness",
  "overfit_risk": "low|medium|high",
  "recommendation": "generate|repair|abandon",
  "probe_relevance": "high|medium|low|contradictory|missing",
  "probe_use": "anchor|weak_hint|ignore",
  "seed_vs_probe_alignment": "short explanation"
}}
"""


_GENERATOR_SYSTEM = f"""You are a WorldQuant Brain FASTEXPR generator agent.

You receive a Diagnostic Agent JSON object and must generate structural
signal-enhancement variants.  The LLM is allowed to create genuinely new ideas,
but each idea must preserve or explicitly reframe the seed's semantic identity.

Optimization objective:
- Primary objective is to increase admission_brain_score_delta, or fallback
  admission_delta_pool_sharpe when Brain score is unavailable.
- Improve score through better signal structure, not by overfitting a single
  metric row.
- Preserve or improve standalone Sharpe and Fitness when possible.
- Keep turnover in a healthy range; reduce turnover when prior results show
  turnover stress, and do not create high-turnover artifacts.
- Avoid high self-correlation unless the expression has a clear Sharpe/Fitness
  advantage over the correlated alpha.
- Repair named failing checks directly.  For example, LOW_SUB_UNIVERSE_SHARPE
  calls for universe/size/liquidity-stability logic, not a blind window sweep;
  LOW_FITNESS/LOW_SHARPE calls for stronger signal identity; turnover failures
  call for smoothing, gating, or slower-moving structure.
- A candidate with positive score delta but weak Sharpe/Fitness/check quality
  is not a final success; generate variants that can become submit/watchlist
  candidates, not only score artifacts.

Hard rules:
- No parameter-only variants.
- Do not emit variants that only change window, group, constants, weights, or
  Brain simulation settings.
- New secondary fields must be semantically compatible with the diagnostic
  template.
- If robustness shows identity failure, reframe the signal or produce fewer
  final-round ideas; do not patch individual robustness rows.
- Rounds 1 and 2 require exactly six expressions.
- Round 3 requires three to six expressions.

{_FASTEXPR_RULES}

Think through each structural purpose internally, but do not output the
reasoning. Return ONLY complete FASTEXPR expressions, one per line. No JSON,
no markdown, no bullets, no numbering, no comments, no code fences.
"""


@dataclass
class DiagnosticResult:
    payload: dict[str, Any]
    messages: list[dict[str, str]] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class GeneratedVariant:
    expression: str
    structural_purpose: str = ""
    mutation_axes: list[str] = field(default_factory=list)
    preserved_identity: str = ""
    expected_fix: str = ""
    overfit_risk: str = "medium"


@dataclass
class GenerationResult:
    variants: list[GeneratedVariant]
    messages: list[dict[str, str]] = field(default_factory=list)
    raw_response: str = ""


class SemanticDiagnosticAgent:
    """LLM diagnostic agent for semantic template annotation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.5",
        max_tokens: int = 1400,
    ) -> None:
        resolved_key = anthropic.load_api_key(api_key)
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    def diagnose(
        self,
        *,
        seed_record: dict[str, Any],
        prior_diagnosis: dict[str, Any] | None = None,
        round_results: list[dict[str, Any]] | None = None,
        robustness_card: dict[str, Any] | None = None,
        context_records: list[dict[str, Any]] | None = None,
        probe_card: dict[str, Any] | None = None,
        context_packet: dict[str, Any] | None = None,
    ) -> DiagnosticResult:
        message = (
            _diagnostic_packet_message(context_packet)
            if context_packet
            else _diagnostic_message(
                seed_record,
                prior_diagnosis=prior_diagnosis,
                round_results=round_results or [],
                robustness_card=robustness_card,
                context_records=context_records or [],
                probe_card=probe_card,
            )
        )
        messages = [{"role": "user", "content": message}]
        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=_DIAGNOSTIC_SYSTEM,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": raw})
        payload_raw = parse_json_object(raw)
        if payload_raw is None:
            repair_message = _diagnostic_format_repair_message(raw)
            messages.append({"role": "user", "content": repair_message})
            repair_response = self.client.messages.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                system=_DIAGNOSTIC_SYSTEM,
                messages=messages,
            )
            repaired = repair_response.content[0].text.strip()
            messages.append({"role": "assistant", "content": repaired})
            raw = repaired
            payload_raw = parse_json_object(raw)
        payload = _normalize_diagnosis(payload_raw or {})
        return DiagnosticResult(payload=payload, messages=messages, raw_response=raw)


class SemanticGeneratorAgent:
    """LLM expression generator constrained by a diagnostic template."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.5",
        max_tokens: int = 2200,
    ) -> None:
        resolved_key = anthropic.load_api_key(api_key)
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        *,
        seed_record: dict[str, Any],
        diagnosis: dict[str, Any],
        round_index: int,
        previous_results: list[dict[str, Any]] | None = None,
        robustness_card: dict[str, Any] | None = None,
        known_expressions: list[str] | None = None,
        context_packet: dict[str, Any] | None = None,
    ) -> GenerationResult:
        target_count = "exactly 6" if round_index in {1, 2} else "3 to 6"
        message = (
            _generator_packet_message(context_packet)
            if context_packet
            else _generator_message(
                seed_record,
                diagnosis,
                round_index=round_index,
                target_count=target_count,
                previous_results=previous_results or [],
                robustness_card=robustness_card,
                known_expressions=known_expressions or [],
            )
        )
        messages = [{"role": "user", "content": message}]
        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=_GENERATOR_SYSTEM,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": raw})
        variants = _parse_variants(raw)
        if not _variant_count_valid(variants, round_index):
            repair_message = _generator_format_repair_message(raw, variants, round_index)
            messages.append({"role": "user", "content": repair_message})
            repair_response = self.client.messages.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                system=_GENERATOR_SYSTEM,
                messages=messages,
            )
            repaired = repair_response.content[0].text.strip()
            repaired_variants = _parse_variants(repaired)
            if _variant_count_valid(repaired_variants, round_index):
                raw = repaired
                variants = repaired_variants
            messages.append({"role": "assistant", "content": repaired})
        return GenerationResult(variants=variants, messages=messages, raw_response=raw)


def _diagnostic_message(
    seed: dict[str, Any],
    *,
    prior_diagnosis: dict[str, Any] | None,
    round_results: list[dict[str, Any]],
    robustness_card: dict[str, Any] | None,
    context_records: list[dict[str, Any]],
    probe_card: dict[str, Any] | None,
) -> str:
    lines = ["Seed alpha:", _record_line(seed)]
    lines.append(f"Expression: {seed.get('expression', '')}")
    fields = seed.get("fields") or []
    clusters = seed.get("cluster_ids") or []
    lines.append(f"Fields: {fields}")
    lines.append(f"Clusters: {clusters}")
    params = seed.get("params") if isinstance(seed.get("params"), dict) else {}
    if params:
        for key in ("probe_template", "probe_family", "template_family", "root_expression", "original_expression"):
            if params.get(key):
                lines.append(f"{key}: {params[key]}")

    if prior_diagnosis:
        lines.append("\nPrior diagnostic JSON:")
        lines.append(json.dumps(prior_diagnosis, sort_keys=True))

    if round_results:
        lines.append("\nRecent generated-result summary:")
        for record in sorted(round_results, key=_score_sort_value, reverse=True)[:12]:
            lines.append(_record_line(record, include_expr=True))

    if robustness_card:
        lines.append("\nRobustness card summary:")
        lines.append(json.dumps(_compact_robustness_card(robustness_card), sort_keys=True))

    if probe_card:
        lines.append("\nProbe card, noisy prior only:")
        lines.append(
            "Use this only to assess seed/probe alignment. It is not ground truth "
            "and must not override the seed's actual expression and simulation metrics."
        )
        lines.append(json.dumps(_compact_probe_card(probe_card), sort_keys=True))

    if context_records:
        lines.append("\nNearby passing or prior context:")
        for record in context_records[:8]:
            lines.append(_record_line(record, include_expr=True))

    lines.append(
        "\nDiagnose the signal identity and whether the next action should be "
        "generate, repair, or abandon."
    )
    return "\n".join(lines)


def _diagnostic_packet_message(packet: dict[str, Any]) -> str:
    return (
        "Diagnose the semantic alpha using this auditable context packet. "
        "The context_report explains why history records are included or excluded. "
        "Probe evidence is noisy prior context only. Return exactly one JSON object "
        "matching the system schema.\n\n"
        + json.dumps(packet, sort_keys=True)
    )


def _compact_probe_card(card: dict[str, Any]) -> dict[str, Any]:
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


def _generator_message(
    seed: dict[str, Any],
    diagnosis: dict[str, Any],
    *,
    round_index: int,
    target_count: str,
    previous_results: list[dict[str, Any]],
    robustness_card: dict[str, Any] | None,
    known_expressions: list[str],
) -> str:
    lines = [
        f"Round: {round_index}",
        f"Required count: {target_count} expressions.",
        "Objective: improve score addition through structural repair/enhancement while preserving or improving Sharpe/Fitness, keeping turnover healthy, avoiding high self-correlation, and repairing named failing checks. Do not chase score with weak standalone quality.",
        "Seed alpha:",
        _record_line(seed),
        f"Expression: {seed.get('expression', '')}",
        "",
        "Diagnostic Agent JSON:",
        json.dumps(diagnosis, sort_keys=True),
    ]
    if previous_results:
        lines.append("\nPrevious generated results, best first:")
        for record in sorted(previous_results, key=_score_sort_value, reverse=True)[:14]:
            lines.append(_record_line(record, include_expr=True))
    if robustness_card:
        lines.append("\nRobustness card summary:")
        lines.append(json.dumps(_compact_robustness_card(robustness_card), sort_keys=True))
    if known_expressions:
        lines.append("\nDo not repeat these expressions:")
        for expr in known_expressions[-40:]:
            lines.append(expr[:240])
    if round_index in {1, 2}:
        lines.append("\nReturn exactly 6 complete FASTEXPR expressions, one per line. No JSON or markdown.")
    else:
        lines.append("\nReturn 3 to 6 complete FASTEXPR expressions, one per line; focus only on the best surviving ideas. No JSON or markdown.")
    return "\n".join(lines)


def _generator_packet_message(packet: dict[str, Any]) -> str:
    return (
        "Generate FASTEXPR variants from this auditable semantic context packet. "
        "Optimize for score addition through quality-preserving structural changes: "
        "preserve/improve Sharpe and Fitness, keep turnover healthy, avoid high "
        "self-correlation, and repair named failing checks. Do not treat positive "
        "score delta with weak standalone quality as final success. "
        "Use relevant_history and lineage_history as prior attempts; do not repeat "
        "known expressions. Robustness is diagnostic only, not a row-by-row tuning target. "
        "Return expression lines only, with the exact count requested in the packet.\n\n"
        + json.dumps(packet, sort_keys=True)
    )


def _diagnostic_format_repair_message(raw_output: str) -> str:
    return (
        "Your previous response was not parseable as the required diagnostic JSON object. "
        "Do not redo the analysis. Convert the same content into ONLY this JSON shape:\n"
        "{"
        "\"signal_identity\":\"...\","
        "\"semantic_template\":\"...\","
        "\"field_roles\":{\"main_signal\":\"...\",\"secondary_fields\":\"...\",\"gate_fields\":\"...\",\"neutralizer_fields\":\"...\"},"
        "\"valid_mutation_axes\":[\"...\"],"
        "\"invalid_mutations\":[\"...\"],"
        "\"failure_diagnosis\":\"...\","
        "\"overfit_risk\":\"low|medium|high\","
        "\"recommendation\":\"generate|repair|abandon\","
        "\"probe_relevance\":\"high|medium|low|contradictory|missing\","
        "\"probe_use\":\"anchor|weak_hint|ignore\","
        "\"seed_vs_probe_alignment\":\"...\""
        "}\n\nPrevious raw output:\n"
        f"{raw_output[:6000]}"
    )


def _generator_format_repair_message(raw_output: str, parsed: list[GeneratedVariant], round_index: int) -> str:
    if round_index in {1, 2}:
        requirement = "exactly 6 complete FASTEXPR expressions"
    else:
        requirement = "3 to 6 complete FASTEXPR expressions"
    accepted = "\n".join(variant.expression for variant in parsed if variant.expression)
    return (
        "Your previous response did not satisfy the required expression count or format. "
        f"Return ONLY {requirement}, one per line. No JSON, markdown, bullets, numbering, or prose. "
        "Do not resend context or explanations.\n\n"
        f"Parsed expressions from previous output:\n{accepted[:3000]}\n\n"
        f"Previous raw output:\n{raw_output[:6000]}"
    )


def _variant_count_valid(variants: list[GeneratedVariant], round_index: int) -> bool:
    n = len([variant for variant in variants if variant.expression.strip()])
    if round_index in {1, 2}:
        return n == 6
    return 3 <= n <= 6


def _record_line(record: dict[str, Any], *, include_expr: bool = False) -> str:
    admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
    score_delta = _first_non_none(record.get("admission_brain_score_delta"), admission.get("brain_score_delta"))
    pool_delta = _first_non_none(record.get("admission_delta_pool_sharpe"), admission.get("delta_pool_sharpe"))
    corr = _first_non_none(record.get("admission_global_max_corr"), record.get("max_self_correlation"))
    parts = [
        f"id={record.get('alpha_id') or record.get('candidate_id') or '?'}",
        f"lane={record.get('search_lane') or '?'}",
        f"sharpe={_fmt(record.get('is_sharpe'))}",
        f"fitness={_fmt(record.get('is_fitness'))}",
        f"turnover={_fmt(record.get('is_turnover'))}",
        f"score_delta={_fmt(score_delta, digits=0)}",
        f"pool_delta={_fmt(pool_delta)}",
        f"corr={_fmt(corr)}",
        f"grade={record.get('grade') or '?'}",
    ]
    failing = [
        str(check.get("name"))
        for check in record.get("checks") or []
        if check.get("result") != "PASS" and check.get("name") != "SELF_CORRELATION"
    ]
    if failing:
        parts.append(f"failing={','.join(failing)}")
    if include_expr:
        parts.append(f"expr={record.get('expression', '')[:240]}")
    return " | ".join(parts)


def _compact_robustness_card(card: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "root_expression": card.get("root_expression"),
        "overall_status": card.get("overall_status"),
        "sources": [],
    }
    for source in card.get("sources") or []:
        compact["sources"].append({
            "source_alpha_id": source.get("source_alpha_id"),
            "source_expression": str(source.get("source_expression") or "")[:180],
            "expression_retention_rate": source.get("expression_retention_rate"),
            "settings_pass_rate": source.get("settings_pass_rate"),
            "field_pass_rate": source.get("field_pass_rate"),
            "top1000_retention": source.get("top1000_retention"),
            "status": source.get("status"),
            "notes": source.get("notes"),
        })
    return compact


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    payload = parse_json_object(raw_text)
    return _normalize_diagnosis(payload or {})


def _parse_variants(raw_text: str) -> list[GeneratedVariant]:
    text = extract_json_text(raw_text)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("variants", "expressions", "candidates", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    payload = value
                    break
        if isinstance(payload, list):
            out: list[GeneratedVariant] = []
            for item in payload:
                if isinstance(item, str):
                    expr = item.strip()
                    if expr:
                        out.append(GeneratedVariant(expression=expr))
                    continue
                if not isinstance(item, dict):
                    continue
                expr = str(item.get("expression") or "").strip()
                if not expr:
                    continue
                axes = item.get("mutation_axes")
                if not isinstance(axes, list):
                    axes = []
                out.append(GeneratedVariant(
                    expression=expr,
                    structural_purpose=str(item.get("structural_purpose") or item.get("rationale") or ""),
                    mutation_axes=[str(axis) for axis in axes],
                    preserved_identity=str(item.get("preserved_identity") or ""),
                    expected_fix=str(item.get("expected_fix") or ""),
                    overfit_risk=str(item.get("overfit_risk") or "medium"),
                ))
            return out
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return [
        GeneratedVariant(expression=expr)
        for expr in _parse_expression_lines(raw_text)
    ]


def _parse_expression_lines(raw_text: str) -> list[str]:
    """Autoworkshop-style fallback: extract FASTEXPR-looking lines."""
    text = raw_text
    if "```" in text:
        parts = text.split("```")
        fenced = [
            part.strip().removeprefix("fastexpr").removeprefix("FASTEXPR").removeprefix("text").strip()
            for part in parts
            if "(" in part or "/" in part
        ]
        if fenced:
            text = "\n".join(fenced)

    out: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().rstrip(",")
        if not cleaned:
            continue
        cleaned = re.sub(r'^(\d+[\.)]|[-*])\s+', '', cleaned)
        if cleaned.startswith(("FASTEXPR:", "Expression:", "expression:")):
            cleaned = cleaned.split(":", 1)[1].strip()
        if cleaned.startswith(('"', "'")) and cleaned.endswith(('"', "'")):
            cleaned = cleaned[1:-1].strip()
        if not cleaned or cleaned.startswith(("#", "//", "{", "[", "]", "}")):
            continue
        assignment_style = bool(
            re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=', cleaned)
            and ';' in cleaned
            and '(' in cleaned
        )
        if not assignment_style and not re.match(r'^-?(\(|[a-zA-Z_][a-zA-Z0-9_]*\s*[\(/])', cleaned):
            if ' ' in cleaned or not re.match(r'^-?[a-zA-Z_][a-zA-Z0-9_/.*+\-()[\] ,=<>?:]*$', cleaned):
                continue
        if ' ' in cleaned and '(' not in cleaned:
            continue
        out.append(cleaned)
    return out


def _normalize_diagnosis(payload: dict[str, Any]) -> dict[str, Any]:
    field_roles = payload.get("field_roles")
    if not isinstance(field_roles, dict):
        field_roles = {}
    recommendation = str(payload.get("recommendation") or "generate").lower()
    if recommendation not in {"generate", "repair", "abandon"}:
        recommendation = "generate"
    risk = str(payload.get("overfit_risk") or "medium").lower()
    if risk not in {"low", "medium", "high"}:
        risk = "medium"
    return {
        "signal_identity": str(payload.get("signal_identity") or "unknown"),
        "semantic_template": str(payload.get("semantic_template") or "unknown"),
        "field_roles": {
            "main_signal": str(field_roles.get("main_signal") or ""),
            "secondary_fields": str(field_roles.get("secondary_fields") or ""),
            "gate_fields": str(field_roles.get("gate_fields") or ""),
            "neutralizer_fields": str(field_roles.get("neutralizer_fields") or ""),
        },
        "valid_mutation_axes": _string_list(payload.get("valid_mutation_axes")),
        "invalid_mutations": _string_list(payload.get("invalid_mutations")),
        "failure_diagnosis": str(payload.get("failure_diagnosis") or ""),
        "overfit_risk": risk,
        "recommendation": recommendation,
        "probe_relevance": _enum_value(
            payload.get("probe_relevance"),
            {"high", "medium", "low", "contradictory", "missing"},
            "missing",
        ),
        "probe_use": _enum_value(
            payload.get("probe_use"),
            {"anchor", "weak_hint", "ignore"},
            "ignore",
        ),
        "seed_vs_probe_alignment": str(payload.get("seed_vs_probe_alignment") or ""),
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _enum_value(value: Any, allowed: set[str], default: str) -> str:
    parsed = str(value or default).lower()
    return parsed if parsed in allowed else default


def _score_sort_value(record: dict[str, Any]) -> tuple[float, float, float, float]:
    score = _first_float(
        record.get("admission_brain_score_delta"),
        (record.get("admission") or {}).get("brain_score_delta") if isinstance(record.get("admission"), dict) else None,
    )
    pool = _first_float(
        record.get("admission_delta_pool_sharpe"),
        (record.get("admission") or {}).get("delta_pool_sharpe") if isinstance(record.get("admission"), dict) else None,
    )
    effective = score if score is not None else pool
    return (
        effective if effective is not None else -1e9,
        _first_float(record.get("is_fitness")) or -1e9,
        _first_float(record.get("is_sharpe")) or -1e9,
        -(_first_float(record.get("is_turnover")) or 9.0),
    )


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "DiagnosticResult",
    "GeneratedVariant",
    "GenerationResult",
    "SemanticDiagnosticAgent",
    "SemanticGeneratorAgent",
]
