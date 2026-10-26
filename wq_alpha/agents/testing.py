"""Deterministic local agents for workshop harness tests."""

from __future__ import annotations

import json
from typing import Any

from wq_alpha.agents.semantic_workshop import DiagnosticResult, GeneratedVariant, GenerationResult


class StubDiagnosticAgent:
    """Deterministic diagnostic agent for no-network script tests."""

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
        del prior_diagnosis, round_results, robustness_card, context_records, context_packet
        probe_relevance = str((probe_card or {}).get("computed_trust") or "missing")
        if probe_relevance not in {"high", "medium", "low", "contradictory", "missing"}:
            probe_relevance = "low"
        payload = {
            "signal_identity": "stub semantic identity",
            "semantic_template": "stub_semantic_template",
            "field_roles": {
                "main_signal": ",".join(seed_record.get("fields") or []),
                "secondary_fields": "compatible liquid price-volume controls",
                "gate_fields": "volume/returns gates",
                "neutralizer_fields": "industry/subindustry",
            },
            "valid_mutation_axes": ["rank remap", "group relative normalization"],
            "invalid_mutations": ["parameter-only sweep"],
            "failure_diagnosis": "stub diagnostic",
            "overfit_risk": "medium",
            "recommendation": "generate",
            "probe_relevance": probe_relevance,
            "probe_use": "weak_hint" if probe_relevance in {"high", "medium", "low"} else "ignore",
            "seed_vs_probe_alignment": (probe_card or {}).get("trust_reason", "no probe card"),
        }
        return DiagnosticResult(payload=payload, raw_response=json.dumps(payload))


class StubGeneratorAgent:
    """Deterministic generator; outputs predictable structural wrappers."""

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
        del diagnosis, previous_results, robustness_card, known_expressions, context_packet
        expr = str(seed_record.get("expression") or "")
        wrappers = (
            "rank(({a}))",
            "zscore(({a}))",
            "winsorize(({a}))",
            "group_rank(({a}), industry)",
            "group_zscore(({a}), subindustry)",
            "ts_rank(({a}), 60)",
        )
        variants = [
            GeneratedVariant(
                expression=template.format(a=expr),
                structural_purpose=f"stub structural variant round {round_index}",
                mutation_axes=["stub"],
                preserved_identity="stub",
                expected_fix="stub",
                overfit_risk="medium",
            )
            for template in wrappers
        ]
        return GenerationResult(variants=variants, raw_response="\n".join(v.expression for v in variants))


__all__ = ["StubDiagnosticAgent", "StubGeneratorAgent"]
