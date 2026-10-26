"""LLM-driven alpha candidate proposer.

AlphaProposerAgent makes one Claude API call per round to produce a
GrammarConfig patch (JSON dict), then uses GrammarPlanner to sample
grammatically valid FASTEXPR candidates from the patched config.

The LLM never generates raw FASTEXPR strings — it only decides *where*
to search (operator categories, budget, groups).  Validity is guaranteed
by the existing grammar sampler.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any

from wq_alpha import llm as anthropic

from alpha_grammar.grammar import GrammarConfig
from alpha_grammar.models import CandidateSpec, FieldCluster
from alpha_grammar.operators import BANNED_GENERATION_OPS, OPERATOR_REGISTRY
from alpha_grammar.planner import GrammarPlanner
from alpha_grammar.qtable import QTable, SkeletonCounter, extract_primary_op

_LOG = logging.getLogger(__name__)

def _extract_json_object(text: str) -> dict | None:
    """Extract JSON config from text.

    Checks for <config>...</config> tags first, then falls back to
    finding the first {...} object in the text.
    """
    import re
    # Prefer <config> tagged content
    m = re.search(r"<config>\s*(\{.*?\})\s*</config>", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    # Fallback: find first balanced { ... }
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


# GrammarConfig fields that hold operator name tuples.
_OP_TUPLE_FIELDS: frozenset[str] = frozenset({
    "allowed_ts_rolling_ops",
    "allowed_ts_delta_ops",
    "allowed_ts_paired_ops",
    "allowed_group_ops",
    "allowed_group_window_ops",
    "allowed_group_weighted_ops",
    "allowed_cs_ops",
    "allowed_binary_ops",
    "allowed_unary_ops",
})

# GrammarConfig fields that hold non-operator tuples.
_OTHER_TUPLE_FIELDS: frozenset[str] = frozenset({
    "windows",
    "delta_windows",
    "groups",
    "ts_regression_rettypes",
    "ts_quantile_fractions",
    "kth_element_k_values",  # empty [] disables kth_element
})

_KNOWN_OPS: frozenset[str] = frozenset(OPERATOR_REGISTRY.keys())
_VALID_GROUPS: frozenset[str] = frozenset({"market", "sector", "industry", "subindustry", "none"})

# ---------------------------------------------------------------------------
# Operator reference block embedded in the system prompt
# ---------------------------------------------------------------------------

_OP_REFERENCE = """\
Available operators by GrammarConfig field (use these exact names):

allowed_ts_rolling_ops:
  ts_rank, ts_zscore, ts_mean, ts_std_dev, ts_decay_linear, ts_av_diff,
  ts_scale, ts_quantile, ts_sum, ts_arg_max, ts_arg_min, ts_backfill,
  ts_product

allowed_ts_delta_ops:
  ts_delta, ts_delay, last_diff_value

allowed_ts_paired_ops:
  ts_corr, ts_covariance, ts_regression

allowed_group_ops:
  group_rank, group_zscore, group_neutralize, group_scale

allowed_group_window_ops:
  group_backfill

allowed_group_weighted_ops:
  disabled. Do not use group_mean.

allowed_cs_ops:
  rank, zscore, winsorize, scale

allowed_binary_ops:
  +, -, *, /, add, subtract, multiply, divide, power, signed_power, max, min

allowed_unary_ops:
  abs, sign, log, sqrt, reverse, inverse, hump, days_from_last_change

Budget schedule keys: "batch_1" (cheap, K=3-7), "batch_2" (mid K=3-9), "batch_3+" (deep K=3-12)
Groups: market, sector, industry, subindustry, none
Windows (days): 5, 10, 20, 40, 60, 120, 200, 252
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
You are an alpha expression search strategist for the WorldQuant Brain platform.

Your job: configure the grammar sampler for one batch of ~40 alpha candidates.
The grammar guarantees syntactically and semantically valid FASTEXPR expressions —
you never generate raw expressions, only decide WHICH part of the search space to explore.

You will receive:
  - Sim history summary: HARVESTED operators (already produced passing alphas — avoid)
    and FRONTIER operators (never or rarely tried — prioritize these)
  - Memory facts: structural patterns learned from past batches (use to pick WHICH frontier to enter)
  - Q-table summary: full map of which (cluster, op) pairs have been tried and their reward
  - Concentration kill rates, failure patterns, pool-hurt patterns

You output a JSON "config patch" — a partial override of GrammarConfig fields.

== EXPLORATION MANDATE (read this first) ==

You will receive a VIABLE SIGNAL OPS list — operators that have never produced a passing
alpha yet. Your config patch must use ONLY operators from that list in the allowed_*_ops
fields (plus normalization ops which are always available).

Do NOT include SATURATED ops (those with passing alphas) in allowed_*_ops — the similarity
filter blocks near-duplicate expressions anyway, so exploring them again wastes budget.

Use memory facts and the Q-table to pick WHICH viable ops to target and which field clusters
to point at. Use failure patterns and concentration kill-rates to remove viable ops that
consistently fail Brain checks.

{_OP_REFERENCE}

Patchable GrammarConfig fields (all optional — omit to use defaults):
  budget: int (3-12) — max expression complexity budget K
  budget_schedule: "batch_1" | "batch_2" | "batch_3+"
  allowed_ts_rolling_ops: list[str]  — empty list [] disables this entire category
  allowed_ts_delta_ops: list[str]    — empty list [] disables (ts_delta, ts_delay, last_diff_value)
  allowed_ts_paired_ops: list[str]   — empty list [] disables ts_corr, ts_covariance, ts_regression
  allowed_group_ops: list[str]
  allowed_group_window_ops: list[str] — empty list [] disables group_backfill
  allowed_group_weighted_ops: disabled; omit this field
  allowed_cs_ops: list[str]
  allowed_binary_ops: list[str]
  allowed_unary_ops: list[str]
  windows: list[int] — lookback days
  groups: list[str] — normalization groups (market/sector/industry/subindustry/none)
  alpha_binary_prob: float (0.0-1.0) — probability of composite (A op B) pattern
  trade_when_prob: float (0.0-1.0) — probability of conditional alpha pattern
  min_coverage: float (0.0-1.0) — minimum field coverage threshold
  kth_element_k_values: list[int] — k-values for kth_element; set to [] to DISABLE kth_element entirely
  bool_logical_prob: float (0.0-1.0) — probability of if_else expressions; set to 0.0 to DISABLE if_else

== CONCENTRATION WEIGHT RULES (hard constraints, not guidelines) ==

CONCENTRATED_WEIGHT >= 40% means the alpha puts nearly all portfolio weight into a tiny
number of stocks. Brain treats this as a fake/degenerate signal — it CANNOT be repaired
or submitted regardless of Sharpe. The Q-table gives these a hard reward of ZERO.

Structural causes of extreme concentration (avoid these patterns):
- ts_delta / ts_delay / last_diff_value on sparse fields (many NaN → only a few non-NaN
  stocks trade on that day, concentrating all weight there)
- group_backfill on sparse / event-driven fields (similar NaN pooling effect)
- days_from_last_change on infrequent fields — generates binary/event signals
  with very few "active" stocks
- kth_element on sparse time-series — same sparsity problem
- Any field where coverage << 1.0 combined with an op that doesn't fill/smooth NaNs
- if_else where the condition is sparse → most weight lands in one branch
- ts_regression on two sparse fields — degenerate when few stocks have both fields

Preferred patterns that avoid concentration:
- ts_rolling ops (ts_mean, ts_rank, ts_zscore, ts_std_dev) on dense MATRIX fields
- group_rank / group_zscore / group_neutralize — cross-sectional normalisation distributes
  weight broadly across all stocks in each group
- Use min_coverage >= 0.5 when targeting dense signal; use fields with high alpha_count
- Wrapping in rank() or zscore() before group ops helps distribute weight evenly

== OPERATOR HARD BAN ==

Do not use group_mean, is_nan, or ts_count_nans and do not include them in
any allowed_* operator list. Local history shows group_mean has a >90%
UNKNOWN/flatline rate, and the NaN-presence operators are already saturated in
the submitted pool with repetitive sparse alphas.

You will receive a "concentration kill rate by operator" table in the user message.
Operators with kill rate > 25% should be REMOVED from the config patch or avoided.
Operators with kill rate > 40% must be removed entirely.

== Strategy guidelines ==

1. Set allowed_*_ops to ONLY include operators from the VIABLE SIGNAL OPS list.
   Map each viable op to its correct category:
   - ts_rolling category:  ts_av_diff, ts_scale, ts_quantile, ts_rank, ts_product, ts_sum, ts_arg_max, ts_arg_min
   - ts_delta category:    ts_delta, ts_delay, last_diff_value
   - ts_paired category:   ts_covariance, ts_corr  (include ts_regression only if it's in VIABLE list)
   - binary category:      subtract, multiply, divide, add, power, signed_power, max, min
   - unary category:       abs, log, sqrt, inverse, hump, reverse, sign
   - cs category:          rank, zscore, winsorize, scale  (these are normalization — always ok)
   - group ops:            group_rank, group_zscore, group_neutralize, group_scale  (always ok)

2. Check Q-table for clusters with low n_obs within your chosen viable ops — target those.

3. Use memory facts to decide which viable op is most likely to work based on structural patterns.

4. Remove any viable op with concentration kill_rate > 25% from failure patterns.

5. Set budget:
   - ts_rolling / group ops: budget 5-7
   - ts_paired ops: budget 7+ (cost Δk=3)
   - binary composite: budget 8+, alpha_binary_prob=0.6
   - Set min_coverage >= 0.5

6. To disable kth_element: set kth_element_k_values: []
   To disable if_else: set bool_logical_prob: 0.0
   To disable ts_delta family: set allowed_ts_delta_ops: []

Output the config patch FIRST inside <config> tags, then optionally add brief reasoning after:
<config>{{"budget": 8, "allowed_ts_paired_ops": ["ts_corr", "ts_covariance"], "groups": ["sector", "subindustry"], "min_coverage": 0.5}}</config>
Brief reasoning: focusing on ts_paired ops with subindustry grouping based on SPECTACULAR pattern in portfolio.
"""


class AlphaProposerAgent:
    """Proposes alpha candidates by having the LLM configure the grammar sampler.

    One Claude API call per round to select a GrammarConfig patch.
    GrammarPlanner then samples grammatically valid FASTEXPR candidates.
    Returns an oversampled list (target_batch_size × 3) for downstream
    similarity filtering.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-6",
        max_tokens: int = 1024,
        target_batch_size: int = 40,
        qtable: QTable | None = None,
        skeleton_counter: SkeletonCounter | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key.\n"
                "Set OPENAI_API_KEY env var or pass api_key= to AlphaProposerAgent."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens
        self._target_batch_size = target_batch_size
        self._qtable = qtable or QTable()
        self._skeleton_counter = skeleton_counter

    def propose_batch(
        self,
        clusters: list[FieldCluster],
        memory_facts: list[dict],
        qtable_summary: str,
        batch_index: int,
        seed: int = 42,
        portfolio_context: str = "",
        concentration_summary: str = "",
        alltime_op_stats: str = "",
        exhaustion_analysis: str = "",
        llm_failure_analysis: str = "",
        pool_hurt_patterns: str = "",
        sim_history_summary: str = "",
    ) -> list[CandidateSpec]:
        """Propose ~target_batch_size × 3 candidates (oversampled for filtering).

        LLM picks a GrammarConfig patch → grammar samples → returns CandidateSpecs
        with search_lane="llm_grammar".

        portfolio_context: pre-formatted string describing already-submitted alphas.
        concentration_summary: per-operator concentration kill-rate table.
        alltime_op_stats: all-time tried/pool_scored/pos_delta table per operator.
        exhaustion_analysis: JSON from ExhaustionAnalysisAgent (startup artifact).
        llm_failure_analysis: JSON from LLMFailurePatternAgent (startup artifact).
        pool_hurt_patterns: JSON from PoolHurtPatternAgent (startup artifact).
        sim_history_summary: HARVESTED vs FRONTIER operator table from full sim history.
        """
        config = self._llm_pick_config(
            memory_facts, qtable_summary, batch_index, portfolio_context,
            concentration_summary, alltime_op_stats, exhaustion_analysis,
            llm_failure_analysis, pool_hurt_patterns, sim_history_summary,
        )
        candidates = self._sample_candidates(config, clusters, seed)
        _LOG.info(
            "[Proposer] batch=%d  config_budget=%d  schedule=%s  min_cov=%.2f  sampled=%d",
            batch_index, config.budget, config.budget_schedule, config.min_coverage, len(candidates),
        )
        return candidates

    # ------------------------------------------------------------------
    # LLM config selection
    # ------------------------------------------------------------------

    def _llm_pick_config(
        self,
        memory_facts: list[dict],
        qtable_summary: str,
        batch_index: int,
        portfolio_context: str = "",
        concentration_summary: str = "",
        alltime_op_stats: str = "",
        exhaustion_analysis: str = "",
        llm_failure_analysis: str = "",
        pool_hurt_patterns: str = "",
        sim_history_summary: str = "",
    ) -> GrammarConfig:
        patch = self._llm_pick_config_patch(
            memory_facts, qtable_summary, batch_index, portfolio_context,
            concentration_summary, alltime_op_stats, exhaustion_analysis,
            llm_failure_analysis, pool_hurt_patterns, sim_history_summary,
        )
        return self._apply_patch(patch)

    def _llm_pick_config_patch(
        self,
        memory_facts: list[dict],
        qtable_summary: str,
        batch_index: int,
        portfolio_context: str = "",
        concentration_summary: str = "",
        alltime_op_stats: str = "",
        exhaustion_analysis: str = "",
        llm_failure_analysis: str = "",
        pool_hurt_patterns: str = "",
        sim_history_summary: str = "",
    ) -> dict:
        user_msg = self._build_user_message(
            memory_facts, qtable_summary, batch_index, portfolio_context,
            concentration_summary, alltime_op_stats, exhaustion_analysis,
            llm_failure_analysis, pool_hurt_patterns, sim_history_summary,
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            _LOG.debug("[Proposer] LLM patch raw response: %s", raw[:400])
            patch = _extract_json_object(raw)
            if patch is None:
                _LOG.warning("[Proposer] Could not extract JSON from response: %r — using defaults", raw[:200])
                return {}
            return patch
        except Exception as exc:
            _LOG.warning("[Proposer] LLM config patch failed: %s — using defaults", exc)
            return {}

    def _apply_patch(self, patch: dict) -> GrammarConfig:
        """Merge patch onto default GrammarConfig.  Returns default on any error."""
        validated: dict[str, Any] = {}
        for key, value in patch.items():
            if key not in GrammarConfig.__dataclass_fields__:
                _LOG.debug("[Proposer] Unknown GrammarConfig field '%s' — skipping", key)
                continue

            if key in _OP_TUPLE_FIELDS:
                if not isinstance(value, list):
                    continue
                # Validate each operator name and enforce generation hard bans.
                clean = [
                    v for v in value
                    if isinstance(v, str) and v in _KNOWN_OPS and v not in BANNED_GENERATION_OPS
                ]
                # Empty list is intentional: LLM wants to disable this category.
                # Use an empty tuple — the grammar will skip this category when the tuple is empty.
                value = tuple(clean)
                if not value:
                    _LOG.debug("[Proposer] %s set to empty — disabling this op category", key)

            elif key in _OTHER_TUPLE_FIELDS:
                if not isinstance(value, list):
                    continue
                if key == "groups":
                    clean = [v for v in value if isinstance(v, str) and v in _VALID_GROUPS]
                    if not clean:
                        continue
                    value = tuple(clean)
                else:
                    value = tuple(value)

            validated[key] = value

        # Force oversample target
        validated["n_samples"] = self._target_batch_size * 3

        try:
            config = dataclasses.replace(GrammarConfig(), **validated)
            _LOG.debug("[Proposer] Applied patch keys: %s", list(validated.keys()))
            return config
        except Exception as exc:
            _LOG.warning("[Proposer] Failed to apply patch: %s — using defaults", exc)
            return dataclasses.replace(GrammarConfig(), n_samples=self._target_batch_size * 3)

    # ------------------------------------------------------------------
    # Grammar sampling
    # ------------------------------------------------------------------

    def _sample_candidates(
        self,
        config: GrammarConfig,
        clusters: list[FieldCluster],
        seed: int,
    ) -> list[CandidateSpec]:
        planner = GrammarPlanner(
            config,
            qtable=self._qtable,
            skeleton_counter=self._skeleton_counter,
        )
        raw_candidates = planner.build_plan(clusters, seed=seed)

        # Override search_lane for downstream tracking
        result: list[CandidateSpec] = []
        for c in raw_candidates:
            result.append(
                dataclasses.replace(
                    c,
                    search_lane="llm_grammar",
                    intention_id="llm_guided_grammar",
                    intention_label="LLM-guided grammar sampling",
                )
            )
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        memory_facts: list[dict],
        qtable_summary: str,
        batch_index: int,
        portfolio_context: str = "",
        concentration_summary: str = "",
        alltime_op_stats: str = "",
        exhaustion_analysis: str = "",
        llm_failure_analysis: str = "",
        pool_hurt_patterns: str = "",
        sim_history_summary: str = "",
    ) -> str:
        facts_block = (
            json.dumps(memory_facts, indent=2) if memory_facts
            else "  (no memory facts yet — rely on sim history summary and Q-table below)"
        )

        qtable_block = qtable_summary

        parts = [f"Batch #{batch_index}", ""]

        if sim_history_summary:
            parts += [
                sim_history_summary,
                "",
            ]

        if portfolio_context:
            parts += [
                "== SUBMITTED PORTFOLIO (already in competition) ==",
                portfolio_context,
                "",
            ]

        if concentration_summary:
            parts += [
                "== CONCENTRATION KILL RATES BY OPERATOR (from all sim history) ==",
                "Operators marked *** must be removed. Operators marked ** should be avoided.",
                "hard_kill = max_concentration >= 40% (Brain hard-kills this — reward = 0, not submittable).",
                concentration_summary,
                "",
            ]

        if alltime_op_stats:
            parts += [
                "== ALL-TIME EXPLORATION STATS (tried / pool_scored / pos_delta per operator) ==",
                "Operators marked ** frequently hurt the pool — avoid them.",
                "Operators with pool_scored=0 have never produced a submittable alpha.",
                alltime_op_stats,
                "",
            ]

        if exhaustion_analysis:
            parts += [
                "== EXHAUSTION ANALYSIS (regions mined out vs frontier) ==",
                exhaustion_analysis,
                "",
            ]

        if llm_failure_analysis:
            parts += [
                "== LLM FAILURE PATTERNS (your proposals that consistently fail Brain checks) ==",
                "These operator/dataset combos structurally guarantee Brain check failures — avoid them.",
                llm_failure_analysis,
                "",
            ]

        if pool_hurt_patterns:
            parts += [
                "== POOL HURT PATTERNS (submittable but hurt competition pool score) ==",
                "These passed Brain checks but hurt Brain score delta, or fallback delta_pool_sharpe when Brain comparison was unavailable. Do not repeat them.",
                pool_hurt_patterns,
                "",
            ]

        parts += [
            f"== MEMORY FACTS (top {len(memory_facts)} by confidence) ==",
            facts_block,
            "",
            "== Q-TABLE (n_obs=0 clusters are highest priority within viable ops) ==",
            qtable_block,
            "",
            "Output the config patch JSON (allowed_*_ops must only contain VIABLE ops):",
        ]
        return "\n".join(parts)


def build_concentration_summary(sim_records: list[dict], min_obs: int = 3) -> str:
    """Build a concentration kill-rate table by primary operator from sim records.

    Used to warn the LLM which operators consistently produce fake concentrated signals.
    Only includes operators with >= min_obs observations.
    """
    from collections import defaultdict, Counter

    op_total: Counter = Counter()
    op_hard_kills: Counter = Counter()   # max_concentration >= 40%
    op_any_conc: Counter = Counter()     # any concentration fail

    for rec in sim_records:
        if rec.get("status") != "done":
            continue
        expr = rec.get("expression", "")
        fields = rec.get("fields") or []
        if not fields or not expr:
            continue
        op = extract_primary_op(expr, fields[0])
        if op is None:
            continue
        op_total[op] += 1
        max_c = rec.get("max_concentration")
        if max_c is not None and max_c >= 0.40:
            op_hard_kills[op] += 1
        if not rec.get("is_concentrated_weight_ok", True):
            op_any_conc[op] += 1

    if not op_total:
        return "  (no concentration data yet)"

    lines = []
    # Sort by hard-kill rate descending, then by total observations
    for op in sorted(op_total, key=lambda o: (-op_hard_kills[o] / op_total[o], -op_total[o])):
        n = op_total[op]
        if n < min_obs:
            continue
        hard = op_hard_kills[op]
        any_c = op_any_conc[op]
        hard_rate = hard / n
        flag = ""
        if hard_rate >= 0.40:
            flag = "  *** REMOVE — extreme kill rate"
        elif hard_rate >= 0.25:
            flag = "  ** HIGH — avoid"
        elif any_c / n >= 0.30:
            flag = "  * elevated concentration"
        lines.append(
            f"  {op:<25} hard_kill={hard:2}/{n:2} ({hard_rate:.0%})  any_fail={any_c:2}/{n:2}{flag}"
        )

    if not lines:
        return "  (insufficient data — fewer than min_obs per operator)"
    return "\n".join(lines)


def format_portfolio_context(portfolio_alphas: list[dict]) -> str:
    """Format a list of submitted alpha dicts into a compact prompt block.

    Each dict should have keys: id, expression (str or dict), sharpe, fitness,
    turnover, grade, neutralization, decay, delay, universe.
    Sorted by grade tier then Sharpe descending.
    """
    if not portfolio_alphas:
        return "(no submitted alphas found)"

    grade_order = {"SPECTACULAR": 0, "EXCELLENT": 1, "GOOD": 2, "AVERAGE": 3}
    sorted_alphas = sorted(
        portfolio_alphas,
        key=lambda a: (
            grade_order.get((a.get("grade") or "").upper(), 9),
            -float(a.get("sharpe") or 0.0),
        ),
    )

    lines = []
    for a in sorted_alphas:
        expr = a.get("expression", "")
        if isinstance(expr, dict):
            expr = expr.get("code", "")
        # Collapse multiline expressions to one line for the prompt
        expr_oneline = " ".join(expr.split())
        if len(expr_oneline) > 100:
            expr_oneline = expr_oneline[:97] + "..."

        sharpe = a.get("sharpe")
        fitness = a.get("fitness")
        turnover = a.get("turnover")
        grade = (a.get("grade") or "?").upper()
        neutralization = a.get("neutralization", "?")
        decay = a.get("decay", "?")
        delay = a.get("delay", "?")
        universe = a.get("universe", "?")

        settings_str = (
            f"neutralization={neutralization}, decay={decay}, delay={delay}, universe={universe}"
        )
        perf_str = (
            f"sharpe={sharpe:.2f}, fitness={fitness:.2f}, turnover={turnover:.2%}"
            if sharpe is not None and fitness is not None and turnover is not None
            else "metrics unavailable"
        )
        lines.append(f"  [{grade}] {perf_str} | {settings_str}")
        lines.append(f"    {expr_oneline}")

    return "\n".join(lines)


__all__ = ["AlphaProposerAgent", "format_portfolio_context"]
