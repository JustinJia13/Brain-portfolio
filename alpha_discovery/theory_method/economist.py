"""TheoryEconomistAgent: pre-registered hypothesis cards from the LLM.

Modeled on the TemplateArchitectAgent pattern (alpha_enhancement/agents/semantic_templates.py)
but with prompts loaded from core/prompts/ instead of inline constants.
The agent proposes cards only — expressions are rendered by theory.enumerate_,
never by the LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core import llm as anthropic
from core.json_utils import parse_json_array, parse_json_object
from core.prompts import load_prompt
from alpha_discovery.theory_method.enumerate_ import slotify_skeletons, validate_card


@dataclass
class ProposalResult:
    cards: list[dict]
    rejected: list[tuple[dict, list[str]]] = field(default_factory=list)
    raw_response: str = ""
    truncated: bool = False


class TheoryEconomistAgent:
    # gpt-5.5 is a reasoning model: an explicit max_completion_tokens covers
    # hidden reasoning AND visible output, so a cap can cut the JSON array off
    # mid-card and silently parse to nothing. Default to no cap (None) — the
    # model stops when the bounded card list is done.
    def __init__(self, api_key: str | None = None, model: str = "gpt-5.5", max_tokens: int | None = None) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def propose(
        self,
        *,
        cluster_catalog: str,
        cluster_index: str = "",
        operator_menu: str,
        wiki_excerpts: str,
        mechanism_index: str,
        falsified_index: str,
        n_theories: int,
        neglected_quota_pct: int = 60,
        known_clusters: set[str],
        clusters: dict[str, dict] | None = None,
        saturated_structures: str = "(none)",
    ) -> ProposalResult:
        prompt = load_prompt("theory_propose")
        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.render_user(
                cluster_catalog=cluster_catalog,
                cluster_index=cluster_index,
                operator_menu=operator_menu,
                wiki_excerpts=wiki_excerpts,
                mechanism_index=mechanism_index,
                falsified_index=falsified_index,
                n_theories=n_theories,
                neglected_quota=f"{neglected_quota_pct}%",
                saturated_structures=saturated_structures,
            )}],
        )
        raw = response.content[0].text.strip()
        truncated = getattr(response, "stop_reason", None) == "max_tokens"
        cards = [c for c in parse_json_array(raw, allow_object=True) if isinstance(c, dict)]

        valid: list[dict] = []
        rejected: list[tuple[dict, list[str]]] = []
        for card in cards:
            if clusters:
                card = slotify_skeletons(card, clusters)
            problems = validate_card(card, known_clusters=known_clusters)
            if problems:
                rejected.append((card, problems))
            else:
                valid.append(card)
        return ProposalResult(
            cards=valid, rejected=rejected, raw_response=raw, truncated=truncated)

    def expand_from_seed(
        self,
        *,
        seed_expression: str,
        seed_sharpe: Any,
        seed_fitness: Any,
        seed_robustness: Any,
        seed_dataset: Any,
        seed_fields: str,
        cluster_catalog: str,
        cluster_index: str,
        operator_menu: str,
        mechanism_index: str,
        falsified_index: str,
        n_expansions: int,
        neglected_quota_pct: int = 50,
        known_clusters: set[str],
        clusters: dict[str, dict] | None = None,
    ) -> tuple[dict, ProposalResult]:
        """Decompose one proven alpha and expand it into decorrelated cards.

        Returns (decomposition, ProposalResult) where the ProposalResult.cards are
        the validated expansion cards (consumable by enumerate_family) and the
        decomposition is the LLM's reverse-engineered economic idea.
        """
        prompt = load_prompt("theory_decompose")
        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.render_user(
                seed_expression=seed_expression,
                seed_sharpe=seed_sharpe,
                seed_fitness=seed_fitness,
                seed_robustness=seed_robustness,
                seed_dataset=seed_dataset,
                seed_fields=seed_fields,
                cluster_catalog=cluster_catalog,
                cluster_index=cluster_index,
                operator_menu=operator_menu,
                mechanism_index=mechanism_index,
                falsified_index=falsified_index,
                n_expansions=n_expansions,
                neglected_quota=f"{neglected_quota_pct}%",
            )}],
        )
        raw = response.content[0].text.strip()
        truncated = getattr(response, "stop_reason", None) == "max_tokens"
        obj = parse_json_object(raw) or {}
        decomposition = obj.get("decomposition") or {}
        cards = [c for c in (obj.get("expansions") or []) if isinstance(c, dict)]

        valid: list[dict] = []
        rejected: list[tuple[dict, list[str]]] = []
        for card in cards:
            if clusters:
                card = slotify_skeletons(card, clusters)
            problems = validate_card(card, known_clusters=known_clusters)
            if problems:
                rejected.append((card, problems))
            else:
                valid.append(card)
        return decomposition, ProposalResult(
            cards=valid, rejected=rejected, raw_response=raw, truncated=truncated)

    def refine(
        self,
        card: dict,
        family_results: list[dict],
        verdict: str,
        *,
        known_clusters: set[str],
    ) -> dict | None:
        """One refinement round; returns a revised valid card, or None (abandon/invalid)."""
        prompt = load_prompt("theory_refine")
        rows = [
            {
                "expression": str(r.get("expression") or "")[:120],
                "is_sharpe": r.get("is_sharpe"),
                "is_fitness": r.get("is_fitness"),
                "robustness_score": r.get("robustness_score"),
            }
            for r in family_results
        ]
        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.render_user(
                original_card=json.dumps(card),
                family_results=json.dumps(rows, indent=1),
                family_verdict=verdict,
                theory_id=card.get("theory_id") or "",
            )}],
        )
        parsed: Any = parse_json_object(response.content[0].text)
        if not parsed or parsed.get("abandon"):
            if parsed and parsed.get("lesson"):
                card["falsification_reason"] = str(parsed["lesson"])
            return None
        if validate_card(parsed, known_clusters=known_clusters):
            return None
        parsed["refined_from"] = card.get("theory_id")
        return parsed


__all__ = ["ProposalResult", "TheoryEconomistAgent"]
