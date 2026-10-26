"""Startup analysis agents for the LLM exploration loop.

Three single-purpose agents run once at startup to produce artifacts that guide
the proposer throughout the session.  Each has one well-defined job:

  ExhaustionAnalysisAgent  — which op/cluster regions are mined out; what frontier remains
  LLMFailurePatternAgent   — which patterns the LLM keeps proposing that fail Brain checks
  PoolHurtPatternAgent     — which submittable patterns consistently hurt the pool score

All agents write a JSON artifact to alpha_discovery/legacy.llm_grammar_control_method/artifacts/ and return the JSON string.
On any LLM call or parse failure: log a warning, return "", leave the artifact empty.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core import llm as anthropic

_LOG = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).parent
_ARTIFACTS_DIR = _MODULE_DIR / "artifacts"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _call_claude(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int = 2048,
) -> str | None:
    """Make one Claude API call. Returns response text or None on failure."""
    try:
        response = client.messages.create(
            model=model,
            max_completion_tokens=max_completion_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        _LOG.warning("Claude API call failed: %s", exc)
        return None


def _extract_json(text: str) -> Any:
    """Extract JSON object or array from text, stripping markdown fences if present."""
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    # Try each container type
    for start_ch, end_ch in [('{', '}'), ('[', ']')]:
        start = text.find(start_ch)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Agent A: ExhaustionAnalysisAgent
# ---------------------------------------------------------------------------

_EXHAUSTION_SYSTEM = """\
You are an exploration coverage analyst for alpha discovery on the WorldQuant Brain platform.

Given total submission counts per operator and the Q-table exploration state, identify:
1. Which operator/cluster regions are exhausted (too many submissions, diminishing returns)
2. Which frontiers have been barely explored (few or zero submissions)

Output ONLY valid JSON with this exact structure — no prose, no markdown:
{
  "exhausted": [
    {"operator": "...", "n_submitted": 0, "note": "..."}
  ],
  "frontier": [
    {"cluster_category": "...", "operator_family": "...", "n_submitted": 0, "note": "..."}
  ]
}
"""


class ExhaustionAnalysisAgent:
    """Identifies exhausted search regions and frontier opportunities.

    Input:  alltime_op_counts dict + Q-table exploration summary string
    Output: alpha_discovery/legacy.llm_grammar_control_method/artifacts/exhaustion_analysis.json
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-6",
        artifacts_dir: Path | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model
        self._artifact_path = (artifacts_dir or _ARTIFACTS_DIR) / "exhaustion_analysis.json"

    def analyze(self, alltime_op_counts: dict[str, int], qtable_summary: str) -> str:
        """Run analysis and save artifact. Returns JSON string or '' on failure."""
        op_lines = "\n".join(
            f"  {op:<30} n_submitted={n}"
            for op, n in sorted(alltime_op_counts.items(), key=lambda kv: -kv[1])
        )
        user_msg = (
            "OPERATOR SUBMISSION COUNTS (all history):\n"
            + (op_lines or "  (no data yet)")
            + "\n\nQ-TABLE EXPLORATION STATE:\n"
            + (qtable_summary[:2000] if qtable_summary else "  (no data yet)")
            + "\n\nIdentify exhausted regions and frontier opportunities. Output JSON only."
        )

        raw = _call_claude(self._client, self._model, _EXHAUSTION_SYSTEM, user_msg)
        if not raw:
            _LOG.warning("[ExhaustionAgent] LLM call failed — artifact not written")
            return ""

        parsed = _extract_json(raw)
        if not parsed:
            _LOG.warning("[ExhaustionAgent] Could not parse JSON from response")
            return ""

        self._save(parsed)
        return json.dumps(parsed)

    def _save(self, data: Any) -> None:
        try:
            self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self._artifact_path.write_text(json.dumps(data, indent=2))
            _LOG.info("[ExhaustionAgent] Artifact saved: %s", self._artifact_path)
        except Exception as exc:
            _LOG.warning("[ExhaustionAgent] Could not save artifact: %s", exc)


# ---------------------------------------------------------------------------
# Agent B: LLMFailurePatternAgent
# ---------------------------------------------------------------------------

_LLM_FAILURE_SYSTEM = """\
You are an LLM proposal failure analyst for alpha discovery on the WorldQuant Brain platform.

Given aggregated failure patterns from LLM-proposed alphas that fail Brain's simulation checks,
identify the STRUCTURAL REASON each pattern fails:
- Which operator/field/dataset combinations cause CONCENTRATED_WEIGHT, LOW_RETURNS, etc.?
- Why does the combination structurally guarantee failure (e.g., sparse NaN fields, lookback too high)?

Output ONLY valid JSON — no prose, no markdown:
{
  "patterns": [
    {
      "operator": "...",
      "dataset": "...",
      "fail_check": "...",
      "n_fail": 0,
      "fail_rate": "0%",
      "reason": "..."
    }
  ]
}
"""


class LLMFailurePatternAgent:
    """Identifies structural patterns in LLM proposals that fail Brain checks.

    Input:  pre-aggregated failure data from llm_sim.jsonl
            {primary_op: {fail_check: {n_fail, n_total, fail_rate, examples}}}
    Output: alpha_discovery/legacy.llm_grammar_control_method/artifacts/llm_failure_analysis.json
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-6",
        artifacts_dir: Path | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model
        self._artifact_path = (artifacts_dir or _ARTIFACTS_DIR) / "llm_failure_analysis.json"

    def analyze(self, failure_data: dict[str, dict]) -> str:
        """Run analysis. Returns JSON string or '' on failure.

        failure_data: {primary_op: {fail_check_name: {n_fail, n_total, fail_rate, examples}}}
        Caps input to top-15 op/check combos by n_fail to keep response within token limit.
        """
        if not failure_data:
            _LOG.info("[LLMFailureAgent] No failure data — skipping")
            return ""

        # Flatten all (op, check, stats) entries and sort by n_fail descending
        all_entries = []
        for op, checks in failure_data.items():
            for check_name, stats in checks.items():
                all_entries.append((op, check_name, stats))
        all_entries.sort(key=lambda t: -t[2]["n_fail"])

        lines = []
        for op, check_name, stats in all_entries[:15]:
            examples = stats.get("examples", [])[:2]
            lines.append(
                f"  op={op}  fail_check={check_name}"
                f"  n_fail={stats['n_fail']}/{stats['n_total']}"
                f"  rate={stats['fail_rate']}"
                f"  examples={examples}"
            )

        user_msg = (
            "LLM PROPOSAL FAILURE PATTERNS (top 15 by fail count, from llm_sim.jsonl):\n"
            + "\n".join(lines)
            + "\n\nIdentify the structural reason each pattern fails Brain checks. Output JSON only."
        )

        raw = _call_claude(self._client, self._model, _LLM_FAILURE_SYSTEM, user_msg, max_completion_tokens=4096)
        if not raw:
            _LOG.warning("[LLMFailureAgent] LLM call failed — artifact not written")
            return ""

        parsed = _extract_json(raw)
        if not parsed:
            _LOG.warning("[LLMFailureAgent] Could not parse JSON from response")
            return ""

        self._save(parsed)
        return json.dumps(parsed)

    def _save(self, data: Any) -> None:
        try:
            self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self._artifact_path.write_text(json.dumps(data, indent=2))
            _LOG.info("[LLMFailureAgent] Artifact saved: %s", self._artifact_path)
        except Exception as exc:
            _LOG.warning("[LLMFailureAgent] Could not save artifact: %s", exc)


# ---------------------------------------------------------------------------
# Agent C: PoolHurtPatternAgent
# ---------------------------------------------------------------------------

_POOL_HURT_SYSTEM = """\
You are a pool-impact pattern analyst for alpha discovery on the WorldQuant Brain platform.

These alphas passed all Brain simulation checks but hurt or failed to improve the
competition score (Brain score delta <= 0, or fallback delta_pool_sharpe <= 0 when
Brain comparison was unavailable). Identify STRUCTURAL PATTERNS:
- Which operator/field/dataset combinations consistently produce pool-hurting alphas?
- Why do they pass individual Brain checks but still hurt the pool?
  (e.g., redundancy with existing pool alphas, correlated signals, weight concentration)

Output ONLY valid JSON — no prose, no markdown:
{
  "traps": [
    {
      "pattern": "...",
      "grade_range": "...",
      "failure": "...",
      "avoid_operators": ["..."],
      "avoid_datasets": ["..."]
    }
  ]
}
"""


class PoolHurtPatternAgent:
    """Identifies structural patterns from submittable alphas that hurt pool score.

    Input:  top-30 pool-hurting examples (any grade) from expression_portfolio_scores.json
            joined with sim records to get grade/fields/primary_op.
            Sorted by contribution delta ascending (worst first).
    Output: alpha_discovery/legacy.llm_grammar_control_method/artifacts/pool_hurt_analysis.json
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-6",
        artifacts_dir: Path | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model
        self._artifact_path = (artifacts_dir or _ARTIFACTS_DIR) / "pool_hurt_analysis.json"

    def analyze(self, pool_hurt_examples: list[dict]) -> str:
        """Run analysis. Returns JSON string or '' on failure.

        pool_hurt_examples: list of {grade, expression, primary_op, fields, contribution_delta}
        """
        if not pool_hurt_examples:
            _LOG.info("[PoolHurtAgent] No pool-hurt examples — skipping")
            return ""

        lines = []
        for ex in pool_hurt_examples[:30]:
            expr = (ex.get("expression") or "")[:80]
            lines.append(
                f"  grade={ex.get('grade', '?')}"
                f"  op={ex.get('primary_op', '?')}"
                f"  delta={ex.get('contribution_delta', ex.get('delta_pool_sharpe', 'N/A'))}"
                f"  fields={ex.get('fields', [])[:2]}"
                f"  expr={expr}"
            )

        user_msg = (
            "SUBMITTABLE ALPHAS WITH NEGATIVE SCORE CONTRIBUTION:\n"
            + "\n".join(lines)
            + "\n\nIdentify structural patterns that consistently hurt the pool score."
            + " Output JSON only."
        )

        raw = _call_claude(self._client, self._model, _POOL_HURT_SYSTEM, user_msg)
        if not raw:
            _LOG.warning("[PoolHurtAgent] LLM call failed — artifact not written")
            return ""

        parsed = _extract_json(raw)
        if not parsed:
            _LOG.warning("[PoolHurtAgent] Could not parse JSON from response")
            return ""

        self._save(parsed)
        return json.dumps(parsed)

    def _save(self, data: Any) -> None:
        try:
            self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self._artifact_path.write_text(json.dumps(data, indent=2))
            _LOG.info("[PoolHurtAgent] Artifact saved: %s", self._artifact_path)
        except Exception as exc:
            _LOG.warning("[PoolHurtAgent] Could not save artifact: %s", exc)


__all__ = [
    "ExhaustionAnalysisAgent",
    "LLMFailurePatternAgent",
    "PoolHurtPatternAgent",
]
