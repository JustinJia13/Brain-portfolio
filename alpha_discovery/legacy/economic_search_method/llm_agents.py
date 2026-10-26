"""Optional LLM agents and deterministic mock outputs."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .field_catalogs import DELAY_DEFINITION
from .portfolio_context import compact_portfolio_context
from .schemas import TheorySpec


PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
ROOT = PACKAGE_DIR.parent
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"


def _json_payload_score(value: Any) -> int:
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return 1000 + len(value)
        if value:
            return 100 + len(value)
        return 1
    if isinstance(value, dict):
        for key in ("theories", "theory_specs", "hypotheses", "proposals", "items", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return 500 + _json_payload_score(nested)
        return 10
    return 0


def _balanced_json_candidates(text: str) -> list[Any]:
    candidates: list[Any] = []
    for start_ch, end_ch in (("[", "]"), ("{", "}")):
        starts = [m.start() for m in re.finditer(re.escape(start_ch), text)]
        for start in starts:
            depth = 0
            in_string = False
            escaped = False
            for i, ch in enumerate(text[start:], start):
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == start_ch:
                    depth += 1
                elif ch == end_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            candidates.append(json.loads(text[start : i + 1]))
                        except json.JSONDecodeError:
                            pass
                        break
    return candidates


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```"))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidates = _balanced_json_candidates(text)
    if not candidates:
        return json.loads(text)
    return max(candidates, key=_json_payload_score)


def _read_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text()


def _debug_dir() -> Path | None:
    raw = os.environ.get("LLM_ECONOMIC_SEARCH_DEBUG_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_debug_payload(kind: str, payload: dict[str, Any]) -> None:
    path = _debug_dir()
    if path is None:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = path / f"{stamp}_{kind}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _coerce_json_list(data: Any, keys: tuple[str, ...]) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError("LLM returned non-list JSON")
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    for value in data.values():
        if isinstance(value, list):
            return value
    raise ValueError(f"LLM returned JSON object without a list payload; expected one of {keys}")


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    key_file = ROOT / "openaiapi.txt"
    return key_file.read_text().strip() if key_file.exists() else ""


class AnthropicJSONClient:
    def __init__(self, model: str | None = None, max_tokens: int | None = None) -> None:
        key = _api_key()
        if not key:
            raise ValueError("No OpenAI key found. Set OPENAI_API_KEY or use --mock.")
        from core import llm as anthropic  # imported lazily so mock/test mode has no dependency

        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.max_tokens = max_tokens or int(os.environ.get("ANTHROPIC_MAX_TOKENS", "12000"))

    def json_call(self, *, system: str, user: str) -> Any:
        _write_debug_payload(
            "request",
            {"model": self.model, "max_tokens": self.max_tokens, "system": system, "user": user},
        )
        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw_text = response.content[0].text
        parsed = _extract_json(raw_text)
        stop_reason = getattr(response, "stop_reason", None)
        self.last_raw_text = raw_text
        self.last_parsed = parsed
        self.last_stop_reason = stop_reason
        _write_debug_payload(
            "response",
            {
                "model": self.model,
                "stop_reason": stop_reason,
                "raw_text": raw_text,
                "parsed_type": type(parsed).__name__,
                "parsed_score": _json_payload_score(parsed),
                "parsed": parsed,
            },
        )
        if stop_reason == "max_tokens" and _json_payload_score(parsed) < 1000:
            raise ValueError(
                f"LLM response hit completion token limit={self.max_tokens} before a complete JSON payload was produced"
            )
        return parsed


class TheoryProposerAgent:
    def __init__(self, client: AnthropicJSONClient | None = None) -> None:
        self.client = client

    def propose(
        self,
        *,
        history_summary: dict[str, Any],
        existing_theories: list[TheorySpec],
        portfolio_context: dict[str, Any] | None = None,
        target_delay: int | None = None,
        field_catalog_summary: dict[str, Any] | None = None,
        sprint_focus: str = "",
        n: int = 6,
        mock: bool = False,
    ) -> list[TheorySpec]:
        if mock:
            return mock_theories(sprint_focus=sprint_focus, n=n)
        if self.client is None:
            self.client = AnthropicJSONClient()
        payload = {
            "sprint_focus": sprint_focus,
            "requested_count": n,
            "target_delay": target_delay,
            "delay_definition": DELAY_DEFINITION,
            "field_catalog_summary": field_catalog_summary or {},
            "history_summary": _compact_summary(history_summary),
            "current_submitted_portfolio": compact_portfolio_context(
                portfolio_context or {},
                target_delay=target_delay,
            ),
            "existing_theories": [t.to_dict() for t in existing_theories[-20:]],
            "output_contract": {
                "format": "JSON array of TheorySpec objects only",
                "target_count": n,
                "empty_array_policy": (
                    "Do not return [] for crowding or uncertainty. Return the least-redundant plausible "
                    "hypotheses and let the critic reject weak ones."
                ),
                "common_control_field_policy": (
                    "Common controls such as returns, volume, volatility, cap, market, industry, and "
                    "subindustry are allowed as conditioning/context fields. They are not primary "
                    "economic-incentive duplicates by themselves."
                ),
            },
        }
        system = _read_prompt("theory_proposer_system.txt")
        last_error = ""
        for attempt in range(2):
            if attempt:
                payload["retry_instruction"] = (
                    f"The previous response produced zero valid TheorySpec objects. "
                    f"Return a JSON array with exactly {n} valid, distinct TheorySpec objects. "
                    "Do not return an empty array. If some areas are crowded, choose more orthogonal "
                    "cross-dataset mechanisms rather than refusing the request. Common controls like "
                    "returns, volume, cap, market, industry, and subindustry may be used as secondary "
                    "conditioning variables; avoid only duplicated primary economic incentives."
                )
            user = json.dumps(payload, indent=2, sort_keys=True)
            data = self.client.json_call(system=system, user=user)
            try:
                items = _coerce_json_list(
                    data,
                    ("theories", "theory_specs", "hypotheses", "proposals", "items", "results"),
                )
            except ValueError as exc:
                last_error = str(exc)
                continue
            theories = [TheorySpec.from_dict(item) for item in items[:n] if isinstance(item, dict)]
            if theories:
                return theories
            last_error = "Theory proposer returned zero valid TheorySpec objects"
        raise ValueError(last_error or "Theory proposer returned zero valid TheorySpec objects")


class NoveltyCriticAgent:
    def __init__(self, client: AnthropicJSONClient | None = None) -> None:
        self.client = client

    def critique(
        self,
        *,
        theories: list[TheorySpec],
        deterministic_reports: dict[str, dict[str, Any]],
        history_summary: dict[str, Any],
        portfolio_context: dict[str, Any] | None = None,
        target_delay: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        if self.client is None:
            self.client = AnthropicJSONClient()
        user = json.dumps(
            {
                "theories": [t.to_dict() for t in theories],
                "deterministic_reports": deterministic_reports,
                "target_delay": target_delay,
                "delay_definition": DELAY_DEFINITION,
                "history_summary": _compact_summary(history_summary),
                "current_submitted_portfolio": compact_portfolio_context(
                    portfolio_context or {},
                    target_delay=target_delay,
                ),
            },
            indent=2,
            sort_keys=True,
        )
        data = self.client.json_call(system=_read_prompt("novelty_critic_system.txt"), user=user)
        if not isinstance(data, dict):
            raise ValueError("Novelty critic returned non-object JSON")
        return data


class CandidateGeneratorAgent:
    def __init__(self, client: AnthropicJSONClient | None = None) -> None:
        self.client = client

    def template_plan(
        self,
        *,
        theory: TheorySpec,
        field_context: dict[str, Any],
        theory_memory: str,
    ) -> dict[str, Any]:
        if self.client is None:
            self.client = AnthropicJSONClient()
        user = json.dumps(
            {
                "theory": theory.to_dict(),
                "field_context": field_context,
                "theory_memory": theory_memory,
            },
            indent=2,
            sort_keys=True,
        )
        data = self.client.json_call(system=_read_prompt("candidate_generator_system.txt"), user=user)
        if not isinstance(data, dict):
            raise ValueError("Candidate generator returned non-object JSON")
        return data

    def generate_expressions(self, *, payload: dict[str, Any]) -> Any:
        if self.client is None:
            self.client = AnthropicJSONClient()
        user = json.dumps(payload, indent=2, sort_keys=True)
        return self.client.json_call(system=_read_prompt("candidate_generator_system.txt"), user=user)


class IterationAgent:
    def __init__(self, client: AnthropicJSONClient | None = None) -> None:
        self.client = client

    def generate_expressions(self, *, payload: dict[str, Any]) -> Any:
        if self.client is None:
            self.client = AnthropicJSONClient()
        user = json.dumps(payload, indent=2, sort_keys=True)
        return self.client.json_call(system=_read_prompt("iteration_agent_system.txt"), user=user)


class ExpansionAgent:
    def __init__(self, client: AnthropicJSONClient | None = None) -> None:
        self.client = client

    def generate_expressions(self, *, payload: dict[str, Any]) -> Any:
        if self.client is None:
            self.client = AnthropicJSONClient()
        user = json.dumps(payload, indent=2, sort_keys=True)
        return self.client.json_call(system=_read_prompt("expansion_agent_system.txt"), user=user)


class ExpressionSemanticCriticAgent:
    def __init__(self, client: AnthropicJSONClient | None = None) -> None:
        self.client = client

    def score_candidates(self, *, payload: dict[str, Any]) -> Any:
        if self.client is None:
            self.client = AnthropicJSONClient()
        user = json.dumps(payload, indent=2, sort_keys=True)
        return self.client.json_call(system=_read_prompt("expression_semantic_critic_system.txt"), user=user)


class ResultInterpreterAgent:
    def __init__(self, client: AnthropicJSONClient | None = None) -> None:
        self.client = client

    def interpret(
        self,
        *,
        theory: TheorySpec,
        metrics_summary: dict[str, Any],
        current_memory: str,
    ) -> dict[str, Any]:
        if self.client is None:
            self.client = AnthropicJSONClient()
        user = json.dumps(
            {
                "theory": theory.to_dict(),
                "metrics_summary": metrics_summary,
                "current_memory": current_memory,
            },
            indent=2,
            sort_keys=True,
        )
        data = self.client.json_call(system=_read_prompt("result_interpreter_system.txt"), user=user)
        if not isinstance(data, dict):
            raise ValueError("Result interpreter returned non-object JSON")
        return data


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "total_records",
        "standalone_pass_records",
        "dataset",
        "category",
        "field",
        "operator",
        "failed_checks",
        "admission",
        "delay",
        "pool_impact",
        "high_corr_by_dataset",
    }
    out = {k: summary.get(k) for k in keep if k in summary}
    for key in ("dataset", "category", "field", "operator", "failed_checks", "high_corr_by_dataset", "delay"):
        if isinstance(out.get(key), list):
            out[key] = out[key][:15]
    if isinstance(out.get("pool_impact"), dict):
        pi = dict(out["pool_impact"])
        pi["worst_examples"] = pi.get("worst_examples", [])[:8]
        out["pool_impact"] = pi
    return out


def mock_theories(*, sprint_focus: str = "", n: int = 6) -> list[TheorySpec]:
    focus = sprint_focus.strip()
    suffix = re.sub(r"[^a-zA-Z0-9]+", "_", focus.lower()).strip("_")[:20]
    if not suffix:
        suffix = "general"
    payloads = [
        {
            "theory_id": f"analyst_news_confirmation_{suffix}",
            "title": "Analyst revision confirmation by news attention",
            "mechanism": (
                "Analyst estimate revision signals should work better when public news attention "
                "or sentiment confirms the information rather than contradicting it."
            ),
            "data_roles": {
                "signal": "analyst4 EPS revision and estimate revision fields",
                "condition": "news12 or news18 sentiment, attention, event intensity fields",
            },
            "expected_sign": "positive analyst revision confirmed by positive news is long",
            "horizon_days": [20, 40, 60],
            "template_families": ["confirm_product", "gated_trade_when", "bucket_condition"],
            "field_cluster_targets": ["analyst4 eps revision", "news sentiment attention"],
            "novelty_claim": "Uses news as a trading condition, not another analyst standalone revision variant.",
            "kill_criteria": ["no variant above Sharpe 0.8", "all passers have high global corr and non-positive contribution delta"],
            "expand_criteria": ["two variants above Sharpe 1.0", "positive Brain score delta or fallback pool delta"],
            "max_probe_candidates": 18,
            "max_iterations": 2,
        },
        {
            "theory_id": f"fundamental_pv_confirmation_{suffix}",
            "title": "Fundamental quality only when price-volume does not reject it",
            "mechanism": (
                "Accounting quality/value signals should be less redundant when conditioned on recent "
                "price-volume confirmation or reversal rather than submitted as standalone balance-sheet factors."
            ),
            "data_roles": {
                "signal": "fundamental6 profitability leverage or quality fields",
                "condition": "pv1 returns volume volatility liquidity fields",
            },
            "expected_sign": "quality/value long when price-volume confirmation is not adverse",
            "horizon_days": [20, 60, 120],
            "template_families": ["gated_trade_when", "residual_spread", "confirm_product"],
            "field_cluster_targets": ["fundamental quality profitability leverage", "price volume returns liquidity"],
            "novelty_claim": "Changes when a crowded fundamental signal trades using price-volume information.",
            "kill_criteria": ["all variants map to existing debt/assets or operating expense clusters", "delta pool Sharpe <= 0"],
            "expand_criteria": ["consistent sign across horizons", "low or moderate corr to submitted pool"],
            "max_probe_candidates": 18,
            "max_iterations": 2,
        },
        {
            "theory_id": f"options_uncertainty_fundamental_{suffix}",
            "title": "Options-implied uncertainty conditions fundamental surprises",
            "mechanism": (
                "Fundamental or analyst surprises should have different payoff under high implied uncertainty; "
                "options fields can separate uncertainty premia from accounting-level signals."
            ),
            "data_roles": {
                "signal": "analyst4 forecast surprise or fundamental profitability fields",
                "condition": "option8 option9 implied volatility put call ratio open interest fields",
            },
            "expected_sign": "positive surprise is more valuable when uncertainty is falling or not elevated",
            "horizon_days": [20, 40, 60],
            "template_families": ["confirm_product", "bucket_condition", "residual_spread"],
            "field_cluster_targets": ["analyst surprise fundamental profitability", "option implied volatility put call ratio"],
            "novelty_claim": "Introduces option uncertainty as a conditioning data family.",
            "kill_criteria": ["option leg creates concentrated weight", "no residual Sharpe after conditioning"],
            "expand_criteria": ["at least one standalone passer with global corr below 0.7", "positive Brain score delta or fallback pool delta"],
            "max_probe_candidates": 18,
            "max_iterations": 2,
        },
        {
            "theory_id": f"social_attention_reversal_{suffix}",
            "title": "Social attention shock reversal versus price momentum",
            "mechanism": (
                "Retail/social attention shocks can create short-term overreaction; combining social intensity "
                "with price-volume reversal may create a PnL shape unlike accounting factors."
            ),
            "data_roles": {
                "signal": "socialmedia12 sentiment or buzz fields",
                "condition": "pv1 returns volume volatility fields",
            },
            "expected_sign": "high attention after adverse short-term return can mean-revert",
            "horizon_days": [5, 10, 20],
            "template_families": ["reversal_condition", "gated_trade_when", "confirm_product"],
            "field_cluster_targets": ["social sentiment buzz", "price volume returns"],
            "novelty_claim": "Underrepresented social plus price-volume mechanism.",
            "kill_criteria": ["high turnover or concentrated sparse social fields", "no variant above Sharpe 0.8"],
            "expand_criteria": ["short-horizon variants pass turnover", "positive Brain score delta or fallback pool delta"],
            "max_probe_candidates": 16,
            "max_iterations": 2,
        },
        {
            "theory_id": f"model_price_disagreement_{suffix}",
            "title": "Model signal disagreement with recent price action",
            "mechanism": (
                "Model-derived scores may add independent value when recent price action disagrees, indicating "
                "underreaction or delayed incorporation rather than plain momentum."
            ),
            "data_roles": {
                "signal": "model16 or model51 factor score derivative fields",
                "condition": "pv1 returns volatility liquidity fields",
            },
            "expected_sign": "model strength with negative short-term price pressure is long",
            "horizon_days": [20, 60],
            "template_families": ["residual_spread", "bucket_condition", "gated_trade_when"],
            "field_cluster_targets": ["model factor score derivative", "price volume returns volatility"],
            "novelty_claim": "Explores underrepresented model datasets with a price-disagreement condition.",
            "kill_criteria": ["model fields duplicate known model16 split/score behavior", "no positive Brain score delta or fallback pool delta"],
            "expand_criteria": ["multiple windows produce positive sign", "low global corr"],
            "max_probe_candidates": 16,
            "max_iterations": 2,
        },
        {
            "theory_id": f"news_options_uncertainty_{suffix}",
            "title": "News event attention filtered by options uncertainty",
            "mechanism": (
                "News event signals may only be tradable when the options market is not pricing extreme "
                "uncertainty; implied volatility can filter event noise and reduce redundant news exposure."
            ),
            "data_roles": {
                "signal": "news12 news18 event sentiment attention fields",
                "condition": "option8 option9 implied volatility and put call ratio fields",
            },
            "expected_sign": "positive event sentiment under non-extreme uncertainty is long",
            "horizon_days": [10, 20, 40],
            "template_families": ["gated_trade_when", "bucket_condition", "confirm_product"],
            "field_cluster_targets": ["news sentiment attention event", "option implied volatility put call ratio"],
            "novelty_claim": "Combines two underrepresented non-fundamental families.",
            "kill_criteria": ["sparse event fields fail concentration", "all variants fail sub-universe Sharpe"],
            "expand_criteria": ["at least two variants above Sharpe 1.0", "positive Brain score delta or fallback pool delta"],
            "max_probe_candidates": 16,
            "max_iterations": 2,
        },
        {
            "theory_id": f"footnote_news_dislocation_{suffix}",
            "title": "Footnote disclosure dislocation confirmed by news tone",
            "mechanism": (
                "Detailed filing-footnote changes can expose accounting or disclosure risk that is not "
                "captured by headline fundamentals; news tone can separate material disclosure shocks from noise."
            ),
            "data_roles": {
                "signal": "fundamental2 footnote disclosure change and accounting footnote fields",
                "condition": "news12 news18 sentiment attention event intensity fields",
            },
            "expected_sign": "negative disclosure risk with weak news tone is short; positive clarification with supportive news is long",
            "horizon_days": [20, 60, 120],
            "template_families": ["confirm_product", "gated_trade_when", "bucket_condition"],
            "field_cluster_targets": ["fundamental2 footnote disclosure", "news sentiment event attention"],
            "novelty_claim": "Uses footnote-level disclosure information with a news materiality filter rather than standalone accounting factors.",
            "kill_criteria": ["footnote leg maps to saturated fundamental standalone clusters", "no positive Brain score delta or fallback pool delta"],
            "expand_criteria": ["consistent sign across footnote/news pairings", "positive Brain score delta or fallback pool delta in same-delay branch"],
            "max_probe_candidates": 16,
            "max_iterations": 2,
        },
        {
            "theory_id": f"option_pv_stress_relief_{suffix}",
            "title": "Options stress relief versus price-volume pressure",
            "mechanism": (
                "When price-volume pressure is adverse but options-implied stress is easing, forced selling "
                "or hedging pressure may be abating, creating a reversal that is distinct from pure momentum."
            ),
            "data_roles": {
                "signal": "option8 implied volatility skew put call ratio open interest fields",
                "condition": "pv1 pv13 returns volume volatility relationship fields",
            },
            "expected_sign": "falling option stress after adverse price pressure is long",
            "horizon_days": [5, 10, 20],
            "template_families": ["reversal_condition", "gated_trade_when", "residual_spread"],
            "field_cluster_targets": ["option implied volatility skew", "price volume pressure reversal"],
            "novelty_claim": "Targets a derivatives-plus-price pressure mechanism rather than accounting or analyst revisions.",
            "kill_criteria": ["option fields are too sparse or fail concentration", "turnover is structurally too high"],
            "expand_criteria": ["short-horizon sign is stable", "global correlation remains moderate"],
            "max_probe_candidates": 16,
            "max_iterations": 2,
        },
        {
            "theory_id": f"relationship_model_confirm_{suffix}",
            "title": "Inter-stock relationship break confirmed by model score",
            "mechanism": (
                "Relationship or peer-move fields can identify temporary dislocations; a model score condition "
                "can distinguish justified relative moves from mean-reverting relationship breaks."
            ),
            "data_roles": {
                "signal": "pv13 relationship peer return relative movement fields",
                "condition": "model16 model51 factor score derivative fields",
            },
            "expected_sign": "positive model score with adverse relative relationship pressure is long",
            "horizon_days": [10, 20, 60],
            "template_families": ["residual_spread", "bucket_condition", "gated_trade_when"],
            "field_cluster_targets": ["price volume relationship dislocation", "model factor score confirmation"],
            "novelty_claim": "Combines relationship structure with model confirmation rather than using either as a standalone signal.",
            "kill_criteria": ["relationship fields reproduce broad market beta", "no low-correlation standalone passers"],
            "expand_criteria": ["multiple peer/relationship fields keep the same sign", "positive Brain score delta or fallback pool delta"],
            "max_probe_candidates": 16,
            "max_iterations": 2,
        },
        {
            "theory_id": f"social_option_crowding_{suffix}",
            "title": "Social crowding filtered by options demand",
            "mechanism": (
                "Social buzz can identify crowded attention, but options demand can separate informed pressure "
                "from noisy retail attention and reduce dependence on simple social sentiment."
            ),
            "data_roles": {
                "signal": "socialmedia12 socialmedia8 sentiment buzz attention fields",
                "condition": "option8 put call ratio implied volatility open interest fields",
            },
            "expected_sign": "extreme social optimism without confirming options demand is short; social pessimism with easing option stress can rebound",
            "horizon_days": [5, 10, 20],
            "template_families": ["bucket_condition", "reversal_condition", "gated_trade_when"],
            "field_cluster_targets": ["social sentiment buzz", "option demand volatility"],
            "novelty_claim": "Uses two underrepresented non-accounting datasets to avoid submitted fundamental and analyst incentives.",
            "kill_criteria": ["social fields are too sparse", "candidate family produces unstable signs"],
            "expand_criteria": ["at least two variants survive concentration and turnover", "Brain score delta is positive, or fallback same-delay pool delta is positive"],
            "max_probe_candidates": 14,
            "max_iterations": 2,
        },
    ]
    return [TheorySpec.from_dict(p) for p in payloads[:n]]
