"""LLM-based alpha enhancement agents using the OpenAI-backed LLM wrapper.

Primary agent:

  AlphaEnhanceAgent — given the best passing alphas, suggests structural enhancements

The legacy OverseerAgent API is retained as a disabled compatibility shim. It
does not call an LLM and never emits parametric sweeps.

Usage:
    from wq_alpha.agents.enhance import AlphaEnhanceAgent, OverseerAgent

    enhance_agent = AlphaEnhanceAgent()
    result = enhance_agent.enhance(passing_records, history=[])
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import product
from typing import Any

from wq_alpha import llm as anthropic


# ---------------------------------------------------------------------------
# FASTEXPR syntax rules injected into both prompts
# ---------------------------------------------------------------------------

_FASTEXPR_RULES = """
FASTEXPR syntax rules (strictly enforced by Brain):
- Group identifiers are BAREWORDS: sector, subindustry, industry, market (never quoted)
- ts_quantile(x, fraction, lookback) — fraction arg comes BEFORE lookback
- subtract(a, b) — exactly 2 args, not variadic
- rank/zscore/normalize/scale/winsorize take exactly 1 arg
- group_rank(expr, group), group_zscore(expr, group), group_neutralize(expr, group)
- ts_arg_max(field, lookback) — returns days-since-last-event (integer 0..lookback-1)
- ts_sum, ts_mean, ts_std, ts_rank all take (expr, window)
- ts_delta(expr, delta_window), ts_delay(expr, days)
- if_else(condition, true_expr, false_expr) — exactly 3 args
"""

# ---------------------------------------------------------------------------
# AlphaEnhanceAgent
# ---------------------------------------------------------------------------

_ENHANCE_SYSTEM = f"""You are a WorldQuant Brain FASTEXPR alpha enhancement expert operating \
in a feedback loop.

Each iteration you receive the currently best passing alphas and (from the second iteration \
onwards) the results of your previous suggestions.

Your goal: suggest 5-8 new FASTEXPR expressions that improve both:
1. Fitness: higher IS Fitness, usually by improving return per turnover rather than just raw Sharpe.
2. Score contribution: positive Brain score delta when available, otherwise positive
   marginal contribution to the submitted equal-weight pool.

Do not optimize standalone Sharpe alone. A variant with higher Sharpe but worse Fitness,
higher turnover, high self-correlation, or negative pool contribution is not an improvement.
Work only within the same dataset(s) as the provided alphas — do not introduce fields from
other datasets.

Strategies:
- Change normalization structure, not just the group name
- Change signal framing, e.g. level vs delta, absolute vs relative, event vs smooth level
- Add a secondary signal interaction using fields already present in the passing alphas
- Change the operator path around the field, e.g. winsorize+scale instead of rank/zscore
- Add conditioning logic that changes when the signal trades
- Reduce turnover when Fitness is weak, but avoid flattening the signal into zero trades.
- If portfolio/admission metrics are shown, prefer variants likely to increase Brain score
  delta when available; otherwise improve fallback delta_pool_sharpe while
  keeping global correlation low.
- Do not produce variants that only tune a window, group, constant, weight, or Brain
  simulation setting. Those are not enhancements for this lane.

{_FASTEXPR_RULES}

Output ONLY a JSON array — no markdown, no preamble:
[
  {{"expression": "normalize(...)", "rationale": "one sentence"}},
  ...
]
"""


@dataclass
class EnhanceResult:
    """Outcome of one enhancement call."""
    expressions: list[str]
    rationale: dict[str, str]      # expression → rationale string
    updated_history: list[dict]    # full conversation history after this call
    raw_response: str = ""


class AlphaEnhanceAgent:
    """Suggests FASTEXPR enhancements for passing alphas via multi-turn LLM conversation.

    Args:
        api_key:    OpenAI API key. Defaults to OPENAI_API_KEY env var.
        model:      Model label for the LLM wrapper (default: gpt-5.5).
        max_tokens: Max tokens per response (default 2048).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.5",
        max_tokens: int = 2048,
    ) -> None:
        resolved_key = anthropic.load_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key found.\n"
                "Set ANTHROPIC_API_KEY env var or place the key in antropicapi.txt."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    def enhance(
        self,
        passing_records: list[dict],
        history: list[dict],
        last_batch_results: list[dict] | None = None,
    ) -> EnhanceResult:
        """Suggest enhancements for the best passing alphas.

        Args:
            passing_records:    Best deduplicated passing sim records.
            history:            Multi-turn conversation history (empty on first call).
            last_batch_results: Sim records from the previous enhance batch (None on first call).

        Returns:
            EnhanceResult with expressions, rationale, and updated history.
        """
        user_msg = _build_enhance_user_message(passing_records, last_batch_results)
        new_history = list(history) + [{"role": "user", "content": user_msg}]

        response = self.client.messages.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            system=_ENHANCE_SYSTEM,
            messages=new_history,
        )
        raw_text = response.content[0].text.strip()
        new_history.append({"role": "assistant", "content": raw_text})

        expressions, rationale = _parse_enhance_response(raw_text)

        return EnhanceResult(
            expressions=expressions,
            rationale=rationale,
            updated_history=new_history,
            raw_response=raw_text,
        )


def _build_enhance_user_message(
    passing_records: list[dict],
    last_batch_results: list[dict] | None,
) -> str:
    lines: list[str] = []

    if last_batch_results:
        lines.append(f"Results from last batch ({len(last_batch_results)} suggestions):")
        for rec in last_batch_results:
            expr = rec.get("expression", "")[:80]
            sharpe = rec.get("is_sharpe")
            fitness = rec.get("is_fitness")
            turnover = rec.get("is_turnover")
            grade = rec.get("grade", "")
            status = rec.get("status", "")
            checks = rec.get("checks") or []
            failing = [c["name"] for c in checks if c.get("result") == "FAIL"]
            if sharpe is not None:
                fit_str = f" Fitness={fitness:.2f}" if fitness is not None else " Fitness=N/A"
                turn_str = f" Turnover={turnover * 100:.1f}%" if turnover is not None else ""
                score_str = _score_feedback_line(rec)
                fail_str = f" ({', '.join(failing)})" if failing else ""
                lines.append(f'- "{expr}" → Sharpe={sharpe:.2f}{fit_str}{turn_str} {grade}{score_str}{fail_str}')
            else:
                lines.append(f'- "{expr}" → N/A ({status})')
        lines.append("")

    lines.append(f"Current best passing alphas ({len(passing_records)} families):")
    for i, rec in enumerate(passing_records, 1):
        expr = rec.get("expression", "")
        sharpe = rec.get("is_sharpe", 0.0)
        fitness = rec.get("is_fitness", 0.0)
        turnover = (rec.get("is_turnover") or 0.0) * 100
        grade = rec.get("grade", "")
        lines.append(
            f'{i}. expression="{expr}" '
            f"Sharpe={sharpe:.2f} Fitness={fitness:.2f} "
            f"Turnover={turnover:.1f}% grade={grade}{_score_feedback_line(rec)}"
        )

    lines.append("")
    lines.append(
        "Suggest structural enhancements that target both higher Fitness and better score contribution. "
        "Do not trade lower Fitness or negative pool contribution for standalone Sharpe. "
        "Do not suggest parameter-only variants that merely change windows, groups, constants, weights, "
        "or Brain simulation settings."
    )
    return "\n".join(lines)


def _score_feedback_line(record: dict) -> str:
    """Compact score/admission feedback for enhancement prompts."""
    parts: list[str] = []
    decision = record.get("admission_decision") or (record.get("admission") or {}).get("decision")
    reason = record.get("admission_reason") or (record.get("admission") or {}).get("reason")
    score_delta = record.get("admission_brain_score_delta")
    if score_delta is None:
        score_delta = (record.get("admission") or {}).get("brain_score_delta")
    delta = record.get("admission_delta_pool_sharpe")
    if delta is None:
        delta = (record.get("admission") or {}).get("delta_pool_sharpe")
    corr = record.get("admission_global_max_corr")
    if corr is None:
        corr = record.get("max_self_correlation")

    if decision:
        parts.append(f"admission={decision}")
    if reason:
        parts.append(f"reason={reason}")
    if score_delta is not None:
        parts.append(f"dScore={_fmt_metric(score_delta, 0)}")
    if delta is not None:
        parts.append(f"dPool={_fmt_metric(delta)}")
    if corr is not None:
        parts.append(f"corr={_fmt_metric(corr)}")
    return f" score[{', '.join(parts)}]" if parts else ""


def _fmt_metric(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _parse_enhance_response(raw_text: str) -> tuple[list[str], dict[str, str]]:
    """Parse JSON array from LLM response. Falls back to line-by-line on parse error."""
    # Strip markdown code fences if present
    text = raw_text
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip().lstrip("json").strip()
            if stripped.startswith("["):
                text = stripped
                break

    try:
        items = json.loads(text)
        if isinstance(items, list):
            expressions: list[str] = []
            rationale: dict[str, str] = {}
            for item in items:
                if isinstance(item, dict):
                    expr = item.get("expression", "").strip()
                    rat = item.get("rationale", "").strip()
                    if expr:
                        expressions.append(expr)
                        rationale[expr] = rat
            return expressions, rationale
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: one expression per line
    expressions = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip() and not line.strip().startswith(("#", "//", "{", "[", "]", "}"))
    ]
    return expressions, {}


# ---------------------------------------------------------------------------
# Disabled legacy OverseerAgent compatibility API
# ---------------------------------------------------------------------------


@dataclass
class SweepSpec:
    """A parametric sweep detected by the Overseer."""
    base_template: str
    param_grid: dict[str, list]
    rationale: str = ""

    def expand(self) -> list[str]:
        """Expand param_grid into concrete expressions via Cartesian product."""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        expressions: list[str] = []
        for combo in product(*values):
            substitutions = dict(zip(keys, combo))
            try:
                expr = self.base_template.format(**substitutions)
                expressions.append(expr)
            except (KeyError, ValueError):
                pass
        return expressions


@dataclass
class OverseerResult:
    """Outcome of one overseer call."""
    sweep_specs: list[SweepSpec]
    expanded_expressions: list[str]
    raw_response: str = ""


class OverseerAgent:
    """Disabled legacy parametric sweep detector.

    Args:
        api_key:    OpenAI API key. Defaults to OPENAI_API_KEY env var.
        model:      Ignored legacy model label.
        max_tokens: Max tokens per response (default 1024).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.5",
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def extract_sweeps(self, expressions: list[str]) -> OverseerResult:
        """Return no sweep specs.

        Parametric sweep expansion is intentionally disabled because it tends to
        produce window/group/weight tuning rather than structural alpha changes.
        """
        return OverseerResult(
            sweep_specs=[],
            expanded_expressions=[],
            raw_response="[]",
        )


# ---------------------------------------------------------------------------
# Selection helpers (reused by enhance_loop.py)
# ---------------------------------------------------------------------------

PASSING_GRADES = {"GOOD", "EXCELLENT", "SPECTACULAR", "AVERAGE"}


def select_passing_reps(
    sim_records: list[dict],
    *,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
    top_n: int = 8,
) -> list[dict]:
    """Return one representative record per template family, sorted by Sharpe.

    Filters to fully passing records (all checks PASS, no FAIL), then keeps
    the highest-Sharpe record per template_id.
    """
    passing: list[dict] = []
    for rec in sim_records:
        if rec.get("status") != "done":
            continue
        sharpe = float(rec.get("is_sharpe") or 0.0)
        fitness = float(rec.get("is_fitness") or 0.0)
        if sharpe < min_sharpe or fitness < min_fitness:
            continue
        checks = rec.get("checks") or []
        has_fail = any(c.get("result") == "FAIL" for c in checks)
        # Also check boolean convenience flags (populated even when checks list is empty)
        if not rec.get("is_concentrated_weight_ok", True):
            has_fail = True
        if not rec.get("is_sub_universe_sharpe_ok", True):
            has_fail = True
        if has_fail:
            continue
        passing.append(rec)

    # Deduplicate: keep best Sharpe per template_id
    best: dict[str, dict] = {}
    for rec in passing:
        tid = rec.get("template_id") or rec.get("candidate_id") or rec.get("expression", "")[:40]
        rec_sharpe = float(rec.get("is_sharpe") or 0.0)
        existing = best.get(tid)
        if existing is None or rec_sharpe > float(existing.get("is_sharpe") or 0.0):
            best[tid] = rec

    reps = sorted(best.values(), key=lambda r: float(r.get("is_sharpe") or 0.0), reverse=True)
    return reps[:top_n]


__all__ = [
    "AlphaEnhanceAgent",
    "EnhanceResult",
    "OverseerAgent",
    "OverseerResult",
    "SweepSpec",
    "select_passing_reps",
    "PASSING_GRADES",
]
