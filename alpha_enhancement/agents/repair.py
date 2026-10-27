"""LLM-based alpha repair agent using the OpenAI-backed LLM wrapper.

Given a near-miss alpha (good Sharpe/Fitness but failing CONCENTRATED_WEIGHT
or LOW_SUB_UNIVERSE_SHARPE), the agent suggests FASTEXPR repair expressions
by wrapping the original expression with appropriate operators.

Usage:
    from alpha_enhancement.agents.repair import AlphaRepairAgent

    agent = AlphaRepairAgent(api_key="sk-...")
    result = agent.repair(sim_record)

    for expr in result.repair_expressions:
        print(expr)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core import llm as anthropic


def _parse_expressions(raw_text: str) -> list[str]:
    """Extract only valid FASTEXPR lines from LLM output, discarding prose."""
    out = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip common markdown list prefixes: "1. expr", "- expr", "* expr"
        cleaned = re.sub(r'^(\d+\.|[-*])\s+', '', line)
        assignment_style = bool(
            re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=', cleaned)
            and ';' in cleaned
            and '(' in cleaned
        )
        # Must start with an identifier (optionally negated) followed by ( or /
        if not assignment_style and not re.match(r'^-?[a-zA-Z_][a-zA-Z0-9_]*\s*[\(/]', cleaned):
            # Also allow plain field names with no spaces
            if ' ' in cleaned or not re.match(r'^-?[a-zA-Z_][a-zA-Z0-9_/.*+\-()[\] ,=]*$', cleaned):
                continue
        # Reject if it reads like a sentence (no parens and has multiple words)
        if ' ' in cleaned and '(' not in cleaned:
            continue
        out.append(cleaned)
    return out


def _fmt_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _portfolio_feedback_line(record: dict, *, compact: bool = False) -> str:
    """Return prompt text describing marginal portfolio impact when available."""
    decision = record.get("admission_decision") or (record.get("admission") or {}).get("decision")
    reason = record.get("admission_reason") or (record.get("admission") or {}).get("reason")
    score_delta = record.get("admission_brain_score_delta")
    if score_delta is None:
        score_delta = (record.get("admission") or {}).get("brain_score_delta")
    delta = record.get("admission_delta_pool_sharpe")
    corr = record.get("admission_global_max_corr")
    if corr is None:
        corr = record.get("max_self_correlation")
    pool_size = record.get("admission_same_delay_pool_size")

    if decision is None and score_delta is None and delta is None and corr is None:
        return ""

    direction = ""
    if delta is not None:
        try:
            delta_f = float(delta)
            if delta_f >= 0.01:
                direction = "adds to same-delay pool Sharpe"
            elif delta_f > 0:
                direction = "barely adds to same-delay pool Sharpe"
            else:
                direction = "decreases same-delay pool Sharpe"
        except (TypeError, ValueError):
            direction = ""

    if compact:
        parts = []
        if decision:
            parts.append(str(decision))
        if reason:
            parts.append(str(reason))
        if score_delta is not None:
            parts.append(f"dScore={_fmt_float(score_delta, 0)}")
        if delta is not None:
            parts.append(f"dPool={_fmt_float(delta)}")
        if corr is not None:
            parts.append(f"corr={_fmt_float(corr)}")
        if direction:
            parts.append(direction)
        return "portfolio: " + " ".join(parts)

    lines = ["  Portfolio admission feedback:"]
    if decision or reason:
        lines.append(f"    Decision: {decision or 'UNSCORED'}  Reason: {reason or 'N/A'}")
    if score_delta is not None:
        before = record.get("admission_brain_score_before")
        after = record.get("admission_brain_score_after")
        lines.append(f"    Brain official score delta: {_fmt_float(score_delta, 0)}")
        if before is not None and after is not None:
            lines.append(f"    Brain score: {_fmt_float(before, 0)} → {_fmt_float(after, 0)}")
    if delta is not None:
        lines.append(
            f"    Fallback same-delay equal-weight delta_pool_sharpe: {_fmt_float(delta)}"
            f" ({direction or 'direction unknown'})"
        )
        before = record.get("admission_pool_sharpe_before")
        after = record.get("admission_pool_sharpe_after")
        if before is not None and after is not None:
            lines.append(
                f"    Pool Sharpe: {_fmt_float(before)} → {_fmt_float(after)}"
                + (f"  (pool size: {pool_size})" if pool_size else "")
            )
    if corr is not None:
        lines.append(f"    Global self-correlation vs all submitted alphas: {_fmt_float(corr)}")
    if pool_size is not None:
        lines.append(f"    Same-delay submitted pool size: {pool_size}")
    lines.append(
        "    Optimization target: prefer variants that improve Brain score delta when shown; "
        "otherwise use positive fallback delta_pool_sharpe and lower global correlation."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# System prompt: factual FASTEXPR reference only, no prescriptive rules.
# Repair guidance comes from few-shot examples in the user message (feedback loop).
# ---------------------------------------------------------------------------

_FASTEXPR_REFERENCE = """
WorldQuant Brain FASTEXPR — syntax rules:

  - Group identifiers are BAREWORD keywords, never quoted:
      group_rank(expr, sector)       ✓
      group_rank(expr, 'sector')     ✗  (Brain rejects quoted group names)
  - Valid group identifiers: market, sector, industry, subindustry
  - Named parameters: ts_decay_exp(expr, window, factor=0.5)
  - Nested calls: group_rank(ts_zscore(field, 60), subindustry)

Brain submission check definitions (what each check measures — no fix implied):
  CONCENTRATED_WEIGHT      — too much portfolio weight concentrated in a few positions
                             Common fix: winsorize(x, stf=2) to clip extreme weights before ranking
  LOW_SUB_UNIVERSE_SHARPE  — Sharpe poor when evaluated within sub-universes (e.g. per industry)
  HIGH_TURNOVER            — daily portfolio changes exceed the turnover limit (~70%)
  LOW_FITNESS              — return/turnover ratio (Fitness) below 1.0
  LOW_SHARPE               — in-sample Sharpe below 1.25
  SELF_CORRELATION         — alpha is too correlated with another of your submitted alphas
"""

_SYSTEM_PROMPT = f"""You are a WorldQuant Brain FASTEXPR alpha construction expert.

Your task: given a failing alpha and its full trial history, reason from the observed \
gradient to suggest 3-6 repair FASTEXPR expressions that are likely to fix the failing checks.

IMPORTANT: Your output is fed directly into the WorldQuant Brain simulation platform as-is. \
Each line is submitted verbatim as a FASTEXPR expression. Any line that is not a valid \
FASTEXPR expression will cause a simulation failure and waste a submission slot.

{_FASTEXPR_REFERENCE}

Reasoning approach — treat this like gradient descent on check values, OOS fitness, and portfolio contribution:
1. Read the trial history carefully. Each attempt shows the exact check value vs limit.
   The gap (value − limit) tells you how far off each check is and whether you are converging.
2. Also track 2023 Sharpe and 2023 Fitness per trial — these are out-of-sample signals.
   A repair that lifts IS metrics but degrades 2023 Fitness is not desirable.
   Prefer repairs that improve both the failing checks AND maintain or improve 2023 Fitness.
   CRITICAL — portfolio contribution: Brain score delta should be > 0 when shown.
   If Brain score delta is unavailable after retries, use delta_pool_sharpe > 0 as the
   fallback proxy. An alpha with negative portfolio contribution is rejected even if its
   standalone Sharpe is excellent. Treat these metrics as primary targets alongside fixing checks.
   Look for structural changes that diversify away from the portfolio (different
   neutralization scope, lookback window, or field transformation) to push
   the portfolio contribution from negative to positive.
   CRITICAL — structural repair vs. parameter tuning: do not propose variants
   that only change window numbers, group names, constants, or Brain simulation
   settings. A repair must alter the operator path, signal framing, data
   transformation, conditioning logic, or field interaction. Window/group changes
   are acceptable only when they accompany a real operator-structure change.
3. Identify which structural changes moved each check metric in the right direction and by how much.
   A change that halved the CW gap but dropped Sharpe is directionally useful — find a variant
   that keeps that structural idea while recovering Sharpe and 2023 Fitness.
4. Identify which directions are exhausted: if smoothing (ts_mean, ts_decay_linear) consistently
   kills Sharpe without fixing CW, do not propose more smoothing variants.
5. Use the passing examples to understand what final structural form the signal needs to reach —
   they are the target state, not just references. Pay attention to their 2023 Fitness values
   to understand what level of OOS quality is achievable on this dataset.
6. Propose expressions that explore genuinely new directions given the gradient evidence:
   different neutralization scope, cross-sectional ranking order, signal clipping approach,
   or field transformation that changes the weight distribution.
7. Do NOT repeat transformations that have already been tried and failed.
8. Do NOT use Brain settings as the repair mechanism. The output is expression-only;
   do not rely on changing delay, decay, universe, truncation, or simulation neutralization.
9. Do NOT apply generic rules from memory — derive every fix from the trial evidence provided.

Output format — CRITICAL:
- Output ONLY complete valid FASTEXPR expressions, one per line
- Every single line in your response will be submitted to Brain for simulation
- Any explanation, comment, markdown, or non-expression text will be simulated and fail
- No numbered lists, no dashes, no asterisks, no backticks — just the raw expression
- Each expression must be structurally distinct from every prior attempt in the trial history
- Parameter-only variants are invalid, even if they might improve the backtest
"""

_ENHANCEMENT_SYSTEM_PROMPT = f"""You are a WorldQuant Brain FASTEXPR alpha construction expert.

Your task: given a PASSING alpha, suggest 3-6 structural variants that improve both:
1. Fitness: higher IS Fitness, usually by improving return per turnover rather than just raw Sharpe.
2. Score contribution: positive Brain score delta when available, otherwise positive
   fallback marginal contribution to the submitted equal-weight pool.

Better standalone Sharpe is useful only if Fitness, turnover, self-correlation, and pool
contribution remain acceptable. A higher-Sharpe variant with worse Fitness or negative
portfolio contribution is not an improvement.

IMPORTANT: Your output is fed directly into the WorldQuant Brain simulation platform as-is. \
Each line is submitted verbatim as a FASTEXPR expression. Any line that is not a valid \
FASTEXPR expression will cause a simulation failure and waste a submission slot.

{_FASTEXPR_REFERENCE}

Submission criteria (all must hold for an alpha to be accepted):
- IS Sharpe >= 1.25
- IS Fitness >= 1.0
- Prefer Fitness improvement over raw Sharpe improvement when the two conflict.
- Turnover within allowed range
- SELF_CORRELATION < 0.70 with any already-submitted portfolio alpha
  EXCEPTION: if self-correlation >= 0.70 but IS Sharpe >= 1.10 x (correlated alpha's Sharpe),
  the alpha is still accepted. A sufficiently higher-Sharpe variant CAN be submitted even when
  highly correlated with an existing alpha.
  When the seed alpha already has high self-correlation, prefer variants that either:
  (a) reduce correlation structurally (different neutralization, grouping, or lookback), OR
  (b) substantially improve Sharpe so the 110%% threshold is met.

CRITICAL — structural change vs. parameter tuning:
Parameter tuning is NOT a structural change and produces near-identical alphas that waste
submission slots. The following are all parameter tuning, NOT structural changes — do NOT
produce more than one variant of any of these types:
  BAD (tuning only): changing a window number (5→10, 20→40)
  BAD (tuning only): swapping a group name (industry→subindustry, sector→market)
  BAD (tuning only): combining both of the above across multiple variants
  BAD (tuning only): adding/removing a single operator layer while keeping the rest identical

A structural change alters HOW the signal is computed, not just the parameters of the same
computation. Each variant must differ from the seed and from every other variant in operator
structure — not just in numbers or group names.

Examples of genuine structural changes:
  - Replace group_backfill with ts_backfill or ts_delay (time-series instead of cross-sectional fill)
  - Wrap the raw field in ts_zscore or ts_rank before the outer normalization
  - Add group_neutralize at a different point in the chain (before vs. after ranking)
  - Replace rank/zscore with winsorize + scale to change the weight distribution
  - Introduce ts_decay_linear or ts_decay_exp to down-weight stale values
  - Add a secondary signal interaction: divide, subtract, or multiply with a related field
  - Change from relative-value (group_zscore) to absolute-value (zscore) framing
  - Replace a lookback mean (ts_mean) with a momentum signal (ts_delta, ts_regression slope)

Reasoning approach:
1. Study the seed alpha's operator chain from inner to outer. Name each operator's role.
2. For each operator, ask: is there a structurally different operator that serves the same
   role but computes it differently? That is one genuine variant direction.
3. Check its correlation status (shown in the user message). If corr >= 0.70, every variant
   must use a structurally different normalization chain or signal framing.
4. Study similar passing alphas with higher Fitness — what operator-level differences do they
   have vs. the seed? Those are the structural directions worth exploring.
5. Preserve the core economic idea (the underlying field and its meaning) while changing
   the computational structure around it.
6. CRITICAL — portfolio contribution: the goal is positive Brain score delta when shown,
   otherwise delta_pool_sharpe > 0 as a fallback proxy after retries. An enhancement that hurts the
   submitted portfolio should not be used regardless of standalone Sharpe. If the seed's
   portfolio delta is negative or
   corr >= 0.70, focus entirely on structurally diverging variants: different normalization
   chain, different signal framing, or orthogonal operator sequence.
   CRITICAL — source alpha pool status: the source alpha's Brain score delta, fallback delta_pool_sharpe, and
   admission decision are shown in the Portfolio admission feedback section below.
   If the source alpha has negative portfolio delta or was rejected for a non-positive
   delta reason, it is a net drag on the submitted pool. In that case, every variant you propose
   MUST structurally diverge from the source with a different normalization chain,
   signal framing, data transformation, conditioning rule, or field interaction.
   Changing only neutralization group, grouping granularity, or lookback window is
   parameter tuning, not divergence. Near-variants of a pool-dragging alpha will
   also drag — do not propose them.
7. CRITICAL — Fitness: do not simply amplify the signal if it raises turnover or lowers
   return per turnover. Prefer cleaner normalization, slightly slower horizons, and
   transformations that maintain breadth while improving Fitness.
8. Do NOT apply generic rules from memory — derive variants from the examples provided.

Output format — CRITICAL:
- Output ONLY complete valid FASTEXPR expressions, one per line
- Every single line in your response will be submitted to Brain for simulation
- Any explanation, comment, markdown, or non-expression text will be simulated and fail
- No numbered lists, no dashes, no asterisks, no backticks — just the raw expression
- Each expression must differ from the original and from every other variant in OPERATOR
  STRUCTURE, not just in window sizes or group names
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """Outcome of one repair request."""
    original_expression: str
    failure_modes: list[str]
    repair_expressions: list[str]
    is_sharpe: float
    is_fitness: float
    raw_response: str = ""
    messages: list[dict] = field(default_factory=list)  # full conversation history


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def _build_round_feedback(sim_results: list[dict], n_variants: int) -> str:
    """Build the follow-up user message showing simulation results for the prior round."""
    lines = ["Simulation results for the expressions you suggested:"]
    for r in sim_results:
        expr = r.get("expression", "")
        sharpe = r.get("is_sharpe")
        fitness = r.get("is_fitness")
        checks = (
            r.get("checks")
            or (r.get("raw_alpha_detail") or {}).get("is", {}).get("checks")
            or []
        )
        failing = [
            c["name"] for c in checks
            if c.get("result") != "PASS" and c.get("name") != "SELF_CORRELATION"
        ]
        status = f"FAIL({', '.join(failing)})" if failing else "PASS"
        oos23f = r.get("oos_fitness_2023")
        oos_tag = f"  2023-fit={oos23f:.3f}" if oos23f is not None else ""
        sharpe_s = f"{sharpe:.3f}" if sharpe is not None else "N/A"
        fitness_s = f"{fitness:.3f}" if fitness is not None else "N/A"
        portfolio_tag = _portfolio_feedback_line(r, compact=True)
        portfolio_part = f", {portfolio_tag}" if portfolio_tag else ""
        lines.append(f"  [{sharpe_s} sharpe, {fitness_s} fitness{oos_tag}{portfolio_part}, {status}]  {expr}")
        for c in checks:
            name = c.get("name", "")
            if name == "SELF_CORRELATION":
                continue
            result_str = c.get("result", "")
            val = c.get("value")
            limit = c.get("limit")
            if val is not None and limit is not None:
                lines.append(f"    {name}: {result_str} (value={val:.4f}, limit={limit:.4f})")
    lines.append(
        f"\nBased on these results, reason from the gradient and suggest "
        f"{n_variants} new structurally distinct FASTEXPR expressions. Do not "
        "respond with variants that only change windows, groups, constants, or "
        "simulation settings. Favor variants that improve "
        "Brain score delta when available; otherwise improve fallback delta_pool_sharpe, "
        "not just standalone Sharpe:"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AlphaRepairAgent
# ---------------------------------------------------------------------------

class AlphaRepairAgent:
    """Uses an LLM to suggest FASTEXPR repair expressions.

    Args:
        api_key:   OpenAI API key. Defaults to OPENAI_API_KEY env var.
        model:     Model label for the LLM wrapper (default: gpt-5.5).
        max_tokens: Max tokens in the repair response (default 512).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.5",
        max_tokens: int = 4096,
    ) -> None:
        resolved_key = anthropic.load_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key found.\n"
                "The repair agent uses the Anthropic-backed LLM wrapper, separate from Brain.\n"
                "Set ANTHROPIC_API_KEY env var or place the key in antropicapi.txt."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    def _completion_text(self, *, system: str, messages: list[dict]) -> str:
        """Request visible text, retrying once when GPT spends the budget invisibly."""
        budgets = [self.max_tokens]
        retry_budget = max(self.max_tokens * 2, 4096)
        if retry_budget not in budgets:
            budgets.append(retry_budget)

        last_text = ""
        for budget in budgets:
            response = self.client.messages.create(
                model=self.model,
                max_completion_tokens=budget,
                system=system,
                messages=messages,
            )
            last_text = response.content[0].text.strip()
            if last_text:
                return last_text
        return last_text

    def repair(
        self,
        record: dict,
        passing_examples: list[dict] | None = None,
        trial_history: list[dict] | None = None,
        context_records: list[dict] | None = None,
        origin_note: str = "",
    ) -> RepairResult:
        """Generate repair expressions for one near-miss simulation record.

        Args:
            record:
                A sim.jsonl record with keys: expression, is_sharpe, is_fitness,
                checks, is_concentrated_weight_ok, is_sub_universe_sharpe_ok.
            passing_examples:
                Optional list of sim records of passing alphas from the same
                dataset/operator context. Injected as few-shot positive examples
                so the LLM reasons from evidence, not from prescribed rules.
            trial_history:
                Optional list of sim records for previous repair attempts on
                this specific alpha (same original_expression). Provides the
                LLM with the full trial trajectory — what was tried, exact
                check values/limits, and which direction moves which metric.

        Returns:
            RepairResult with a list of repaired FASTEXPR strings.
        """
        expression = record.get("expression", "")
        sharpe = float(record.get("is_sharpe") or 0.0)
        fitness = float(record.get("is_fitness") or 0.0)

        # Determine failure modes; fall back to raw_alpha_detail.is.checks for older logs
        checks_list = record.get("checks") or []
        if not checks_list:
            raw = record.get("raw_alpha_detail") or {}
            checks_list = (raw.get("is") or {}).get("checks") or []
        checks_map = {c["name"]: c.get("result", "FAIL") for c in checks_list}
        failure_modes = [name for name, result in checks_map.items() if result == "FAIL"]

        if not record.get("is_concentrated_weight_ok", True):
            if "CONCENTRATED_WEIGHT" not in failure_modes:
                failure_modes.append("CONCENTRATED_WEIGHT")
        if not record.get("is_sub_universe_sharpe_ok", True):
            if "LOW_SUB_UNIVERSE_SHARPE" not in failure_modes:
                failure_modes.append("LOW_SUB_UNIVERSE_SHARPE")

        if not failure_modes:
            failure_modes = ["UNKNOWN_FAIL"]

        # Build check detail string (with measured values for context)
        check_details = []
        for c in checks_list:
            name = c.get("name", "")
            result = c.get("result", "")
            val = c.get("value")
            limit = c.get("limit")
            if val is not None and limit is not None:
                check_details.append(f"  {name}: {result}  (value={val:.4f}, limit={limit:.4f})")
            else:
                check_details.append(f"  {name}: {result}")
        check_block = "\n".join(check_details) if check_details else "  (no check detail available)"

        # Build few-shot passing examples block
        examples_block = ""
        if passing_examples:
            lines = []
            for ex in passing_examples[:6]:
                ex_sharpe = ex.get("is_sharpe", 0)
                ex_fitness = ex.get("is_fitness", 0)
                ex_oos23f = ex.get("oos_fitness_2023")
                ex_expr = ex.get("expression", "")
                oos23_tag = f", {ex_oos23f:.2f} 2023-fit" if ex_oos23f is not None else ""
                portfolio_tag = _portfolio_feedback_line(ex, compact=True)
                portfolio_part = f", {portfolio_tag}" if portfolio_tag else ""
                lines.append(
                    f"  [{ex_sharpe:.2f} sharpe, {ex_fitness:.2f} fitness{oos23_tag}{portfolio_part}] {ex_expr}"
                )
            examples_block = "Passing alphas on this dataset (for reference):\n" + "\n".join(lines)

        # Build per-alpha trial history block (all prior attempts on THIS expression)
        trial_block = ""
        if trial_history:
            lines = [
                "Previous repair attempts on this specific alpha"
                " (sorted best→worst Sharpe — study the gradient):"
            ]
            for t in trial_history[:10]:
                t_expr = t.get("expression", "")
                t_sharpe = t.get("is_sharpe") or 0.0
                t_fitness = t.get("is_fitness") or 0.0
                t_oos23s = t.get("oos_sharpe_2023")
                t_oos23f = t.get("oos_fitness_2023")
                lines.append(f"  Expression: {t_expr}")
                t_turnover = t.get("is_turnover")
                t_returns = t.get("is_returns")
                t_drawdown = t.get("is_drawdown")
                stats_parts = [f"Sharpe={t_sharpe:.3f}", f"Fitness={t_fitness:.3f}"]
                if t_turnover is not None:
                    stats_parts += [
                        f"Turnover={t_turnover:.3f}",
                        f"Returns={t_returns:.3f}",
                        f"Drawdown={t_drawdown:.3f}",
                    ]
                oos23_str = ""
                if t_oos23s is not None or t_oos23f is not None:
                    s = f"{t_oos23s:.3f}" if t_oos23s is not None else "N/A"
                    f_ = f"{t_oos23f:.3f}" if t_oos23f is not None else "N/A"
                    oos23_str = f"  2023-Sharpe={s}  2023-Fitness={f_}"
                portfolio_tag = _portfolio_feedback_line(t, compact=True)
                portfolio_str = f"  {portfolio_tag}" if portfolio_tag else ""
                lines.append(f"  {'  '.join(stats_parts)}{oos23_str}{portfolio_str}")
                t_checks = (
                    t.get("checks")
                    or (t.get("raw_alpha_detail") or {}).get("is", {}).get("checks")
                    or []
                )
                for c in t_checks:
                    name = c.get("name", "")
                    result = c.get("result", "")
                    val = c.get("value")
                    limit = c.get("limit")
                    if val is not None and limit is not None:
                        lines.append(
                            f"    {name}: {result} (value={val:.4f}, limit={limit:.4f})"
                        )
                    else:
                        lines.append(f"    {name}: {result}")
                lines.append("")
            trial_block = "\n".join(lines)

        oos23_sharpe = record.get("oos_sharpe_2023")
        oos23_fitness = record.get("oos_fitness_2023")
        oos23_line = ""
        if oos23_sharpe is not None or oos23_fitness is not None:
            s = f"{oos23_sharpe:.3f}" if oos23_sharpe is not None else "N/A"
            f_ = f"{oos23_fitness:.3f}" if oos23_fitness is not None else "N/A"
            oos23_line = f"  2023 OOS Sharpe: {s}  2023 OOS Fitness: {f_}\n"

        # Build field context block (other structural variants of the same field)
        context_block = ""
        if context_records:
            lines = ["Other structural variants tried for this field (do not repeat these):"]
            for cx in context_records[:8]:
                cx_expr = cx.get("expression", "")
                cx_sharpe = cx.get("is_sharpe") or 0.0
                cx_fitness = cx.get("is_fitness") or 0.0
                cx_turnover = cx.get("is_turnover")
                cx_returns = cx.get("is_returns")
                cx_drawdown = cx.get("is_drawdown")
                cx_checks = (
                    cx.get("checks")
                    or (cx.get("raw_alpha_detail") or {}).get("is", {}).get("checks")
                    or []
                )
                failing = [c["name"] for c in cx_checks if c.get("result") == "FAIL"]
                status_str = f"FAIL({', '.join(failing)})" if failing else "PASS"
                cx_oos23f = cx.get("oos_fitness_2023")
                oos23_tag = f", 2023-fit={cx_oos23f:.2f}" if cx_oos23f is not None else ""
                extra = (
                    f", to={cx_turnover:.3f}, ret={cx_returns:.3f}, dd={cx_drawdown:.3f}"
                    if cx_turnover is not None else ""
                )
                cx_corr = cx.get("max_self_correlation")
                cx_bench_sh = cx.get("max_self_correlation_alpha_sharpe")
                if cx_corr is None:
                    corr_tag = ""
                elif cx_corr < 0.70:
                    corr_tag = f", corr={cx_corr:.2f}"
                else:
                    cx_thr = (cx_bench_sh or 0.0) * 1.10
                    corr_ok = cx_sharpe >= cx_thr if cx_bench_sh else False
                    corr_tag = (
                        f", corr={cx_corr:.2f}(HIGH,{'OK' if corr_ok else 'FAIL'}"
                        f"@need>={cx_thr:.2f})"
                    )
                portfolio_tag = _portfolio_feedback_line(cx, compact=True)
                portfolio_part = f", {portfolio_tag}" if portfolio_tag else ""
                lines.append(
                    f"  [{cx_sharpe:.2f} sharpe, {cx_fitness:.2f} fitness{oos23_tag}{extra}{corr_tag}{portfolio_part}, {status_str}] {cx_expr}"
                )
            context_block = "\n".join(lines)

        turnover = record.get("is_turnover")
        returns = record.get("is_returns")
        drawdown = record.get("is_drawdown")
        is_stats_line = (
            f"  IS Turnover: {turnover:.3f}  IS Returns: {returns:.3f}  IS Drawdown: {drawdown:.3f}\n"
            if turnover is not None else ""
        )
        portfolio_line = _portfolio_feedback_line(record)

        user_message = (
            (f"Note: {origin_note}\n\n") if origin_note else ""
        ) + (
            f"Failing alpha:\n"
            f"  Expression: {expression}\n"
            f"  IS Sharpe: {sharpe:.3f}  IS Fitness: {fitness:.3f}\n"
            + is_stats_line
            + oos23_line +
            f"  Dataset: {record.get('dataset_id', 'unknown')}\n"
            + portfolio_line +
            f"  Check results:\n{check_block}\n"
            f"\nFailing checks: {', '.join(failure_modes)}\n"
        )
        if context_block:
            user_message += f"\n{context_block}\n"
        if trial_block:
            user_message += f"\n{trial_block}\n"
        if examples_block:
            user_message += f"\n{examples_block}\n"
        user_message += (
            "\nSuggest 3-6 structurally distinct FASTEXPR repair expressions. "
            "Do not use parameter-only changes such as only changing windows, groups, constants, "
            "or Brain simulation settings:"
        )

        messages = [{"role": "user", "content": user_message}]
        raw_text = self._completion_text(system=_SYSTEM_PROMPT, messages=messages)
        messages.append({"role": "assistant", "content": raw_text})
        repair_exprs = _parse_expressions(raw_text)

        return RepairResult(
            original_expression=expression,
            failure_modes=failure_modes,
            repair_expressions=repair_exprs,
            is_sharpe=sharpe,
            is_fitness=fitness,
            raw_response=raw_text,
            messages=messages,
        )

    def repair_batch(self, records: list[dict]) -> list[RepairResult]:
        """Repair multiple near-miss records."""
        return [self.repair(rec) for rec in records]

    def repair_round(
        self,
        messages: list[dict],
        sim_results: list[dict],
        n_variants: int = 6,
    ) -> RepairResult:
        """Continue a repair conversation with simulation feedback.

        Args:
            messages:    Conversation history from a prior repair() or repair_round() call.
            sim_results: Simulation records for candidates generated in the previous round.
            n_variants:  Number of new variants to request.

        Returns:
            RepairResult with updated messages (prior history + new user+assistant turns).
        """
        follow_up = _build_round_feedback(sim_results, n_variants)
        new_messages = messages + [{"role": "user", "content": follow_up}]
        raw_text = self._completion_text(system=_SYSTEM_PROMPT, messages=new_messages)
        new_messages = new_messages + [{"role": "assistant", "content": raw_text}]
        return RepairResult(
            original_expression="",
            failure_modes=[],
            repair_expressions=_parse_expressions(raw_text),
            is_sharpe=0.0,
            is_fitness=0.0,
            raw_response=raw_text,
            messages=new_messages,
        )

    def enhance(
        self,
        record: dict,
        passing_examples: list[dict] | None = None,
        context_records: list[dict] | None = None,
        origin_note: str = "",
    ) -> RepairResult:
        """Generate enhancement variants for a passing alpha.

        Args:
            record:
                A sim.jsonl record for a passing alpha with keys:
                expression, is_sharpe, is_fitness, dataset_id.
            passing_examples:
                Optional list of passing sim records from the same dataset,
                ideally with higher Sharpe than the target. Used as evidence
                for structural patterns that work well.

        Returns:
            RepairResult with repair_expressions populated with enhancement
            variants. failure_modes will be empty (alpha already passes).
        """
        expression = record.get("expression", "")
        sharpe = float(record.get("is_sharpe") or 0.0)
        fitness = float(record.get("is_fitness") or 0.0)

        # Build few-shot passing examples block — prefer higher-Sharpe examples
        examples = sorted(
            passing_examples or [],
            key=lambda r: float(r.get("is_sharpe") or 0.0),
            reverse=True,
        )
        examples_block = ""
        if examples:
            lines = []
            for ex in examples[:8]:
                ex_sharpe = ex.get("is_sharpe", 0)
                ex_fitness = ex.get("is_fitness", 0)
                ex_oos23f = ex.get("oos_fitness_2023")
                ex_expr = ex.get("expression", "")
                if ex_expr != expression:
                    oos23_tag = f", {ex_oos23f:.2f} 2023-fit" if ex_oos23f is not None else ""
                    portfolio_tag = _portfolio_feedback_line(ex, compact=True)
                    portfolio_part = f", {portfolio_tag}" if portfolio_tag else ""
                    lines.append(f"  [{ex_sharpe:.2f} sharpe, {ex_fitness:.2f} fitness{oos23_tag}{portfolio_part}] {ex_expr}")
            if lines:
                examples_block = "Passing alphas on this dataset (highest Sharpe first):\n" + "\n".join(lines)

        # Build field context block (other structural variants of the same field)
        context_block = ""
        if context_records:
            lines = ["Other structural variants tried for this field (do not repeat these):"]
            for cx in context_records[:8]:
                cx_expr = cx.get("expression", "")
                cx_sharpe = cx.get("is_sharpe") or 0.0
                cx_fitness = cx.get("is_fitness") or 0.0
                cx_turnover = cx.get("is_turnover")
                cx_returns = cx.get("is_returns")
                cx_drawdown = cx.get("is_drawdown")
                cx_checks = (
                    cx.get("checks")
                    or (cx.get("raw_alpha_detail") or {}).get("is", {}).get("checks")
                    or []
                )
                failing = [c["name"] for c in cx_checks if c.get("result") == "FAIL"]
                status_str = f"FAIL({', '.join(failing)})" if failing else "PASS"
                cx_oos23f = cx.get("oos_fitness_2023")
                oos23_tag = f", 2023-fit={cx_oos23f:.2f}" if cx_oos23f is not None else ""
                extra = (
                    f", to={cx_turnover:.3f}, ret={cx_returns:.3f}, dd={cx_drawdown:.3f}"
                    if cx_turnover is not None else ""
                )
                cx_corr = cx.get("max_self_correlation")
                cx_bench_sh = cx.get("max_self_correlation_alpha_sharpe")
                if cx_corr is None:
                    corr_tag = ""
                elif cx_corr < 0.70:
                    corr_tag = f", corr={cx_corr:.2f}"
                else:
                    cx_thr = (cx_bench_sh or 0.0) * 1.10
                    corr_ok = cx_sharpe >= cx_thr if cx_bench_sh else False
                    corr_tag = (
                        f", corr={cx_corr:.2f}(HIGH,{'OK' if corr_ok else 'FAIL'}"
                        f"@need>={cx_thr:.2f})"
                    )
                portfolio_tag = _portfolio_feedback_line(cx, compact=True)
                portfolio_part = f", {portfolio_tag}" if portfolio_tag else ""
                lines.append(
                    f"  [{cx_sharpe:.2f} sharpe, {cx_fitness:.2f} fitness{oos23_tag}{extra}{corr_tag}{portfolio_part}, {status_str}] {cx_expr}"
                )
            context_block = "\n".join(lines)

        turnover = record.get("is_turnover")
        returns = record.get("is_returns")
        drawdown = record.get("is_drawdown")
        is_stats_line = (
            f"  IS Turnover: {turnover:.3f}  IS Returns: {returns:.3f}  IS Drawdown: {drawdown:.3f}\n"
            if turnover is not None else ""
        )

        corr = record.get("max_self_correlation")
        corr_benchmark_sharpe = record.get("max_self_correlation_alpha_sharpe")
        if corr is None:
            corr_line = "  Self-correlation: not yet computed\n"
        elif corr < 0.70:
            corr_line = f"  Self-correlation: {corr:.3f} (below 0.70 threshold — no issue)\n"
        else:
            threshold = (corr_benchmark_sharpe or 0.0) * 1.10
            passes = sharpe >= threshold if corr_benchmark_sharpe else False
            status = f"PASSES (sharpe {sharpe:.3f} >= 110%% of {corr_benchmark_sharpe:.3f})" if passes else \
                     f"FAILS (need sharpe >= {threshold:.3f} to pass 110%% rule)"
            corr_line = (
                f"  Self-correlation: {corr:.3f} >= 0.70 — HIGH CORRELATION\n"
                f"  Correlated-alpha Sharpe: {corr_benchmark_sharpe:.3f}  "
                f"110%% threshold: {threshold:.3f}  {status}\n"
            )
        portfolio_line = _portfolio_feedback_line(record)

        user_message = (
            (f"Note: {origin_note}\n\n") if origin_note else ""
        ) + (
            f"Passing alpha to enhance:\n"
            f"  Expression: {expression}\n"
            f"  IS Sharpe: {sharpe:.3f}  IS Fitness: {fitness:.3f}\n"
            + is_stats_line
            + corr_line
            + portfolio_line
            + f"  Dataset: {record.get('dataset_id', 'unknown')}\n"
            f"  All submission checks PASS\n"
        )
        if context_block:
            user_message += f"\n{context_block}\n"
        if examples_block:
            user_message += f"\n{examples_block}\n"
        user_message += "\nSuggest 3-6 structural enhancement variants:"
        user_message += (
            "\nOptimize for both higher Fitness and better score contribution. "
            "Do not trade lower Fitness or negative pool contribution for standalone Sharpe. "
            "Do not use parameter-only changes such as only changing windows, groups, constants, "
            "or Brain simulation settings."
        )

        messages = [{"role": "user", "content": user_message}]
        raw_text = self._completion_text(system=_ENHANCEMENT_SYSTEM_PROMPT, messages=messages)
        messages.append({"role": "assistant", "content": raw_text})
        variants = _parse_expressions(raw_text)

        return RepairResult(
            original_expression=expression,
            failure_modes=[],
            repair_expressions=variants,
            is_sharpe=sharpe,
            is_fitness=fitness,
            raw_response=raw_text,
            messages=messages,
        )

    def enhance_round(
        self,
        messages: list[dict],
        sim_results: list[dict],
        n_variants: int = 6,
    ) -> RepairResult:
        """Continue an enhance conversation with simulation feedback.

        Args:
            messages:    Conversation history from a prior enhance() or enhance_round() call.
            sim_results: Simulation records for candidates generated in the previous round.
            n_variants:  Number of new variants to request.

        Returns:
            RepairResult with updated messages (prior history + new user+assistant turns).
        """
        follow_up = _build_round_feedback(sim_results, n_variants)
        new_messages = messages + [{"role": "user", "content": follow_up}]
        raw_text = self._completion_text(system=_ENHANCEMENT_SYSTEM_PROMPT, messages=new_messages)
        new_messages = new_messages + [{"role": "assistant", "content": raw_text}]
        return RepairResult(
            original_expression="",
            failure_modes=[],
            repair_expressions=_parse_expressions(raw_text),
            is_sharpe=0.0,
            is_fitness=0.0,
            raw_response=raw_text,
            messages=new_messages,
        )


# ---------------------------------------------------------------------------
# Near-miss selection helpers
# ---------------------------------------------------------------------------

def select_near_misses(
    sim_records: list[dict],
    *,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
    max_results: int = 20,
) -> list[dict]:
    """Return records with good signal but failing submission checks.

    Sorts by IS Sharpe descending so the most promising alphas are repaired first.
    """
    near_misses = []
    for rec in sim_records:
        if rec.get("status") != "done":
            continue
        sharpe = float(rec.get("is_sharpe") or 0.0)
        fitness = float(rec.get("is_fitness") or 0.0)
        if sharpe < min_sharpe or fitness < min_fitness:
            continue

        # Check if it needs repair
        grade = (rec.get("grade") or "").upper()
        concentrated_ok = rec.get("is_concentrated_weight_ok", True)
        sub_sharpe_ok = rec.get("is_sub_universe_sharpe_ok", True)

        # Parse checks; fall back to raw_alpha_detail.is.checks for older logs
        checks_list = rec.get("checks") or []
        if not checks_list:
            raw = rec.get("raw_alpha_detail") or {}
            checks_list = (raw.get("is") or {}).get("checks") or []
        checks_map = {c["name"]: c.get("result", "FAIL") for c in checks_list}
        if not concentrated_ok:
            checks_map["CONCENTRATED_WEIGHT"] = "FAIL"
        if not sub_sharpe_ok:
            checks_map["LOW_SUB_UNIVERSE_SHARPE"] = "FAIL"

        has_failure = (
            grade != "PASS"
            or checks_map.get("CONCENTRATED_WEIGHT") == "FAIL"
            or checks_map.get("LOW_SUB_UNIVERSE_SHARPE") == "FAIL"
        )
        if has_failure:
            near_misses.append(rec)

    near_misses.sort(key=lambda r: float(r.get("is_sharpe") or 0.0), reverse=True)
    return near_misses[:max_results]


def load_sim_log(sim_log_path: str) -> list[dict]:
    """Load all records from a sim.jsonl file."""
    import json
    from pathlib import Path

    records = []
    path = Path(sim_log_path)
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


__all__ = [
    "AlphaRepairAgent",
    "RepairResult",
    "select_near_misses",
    "load_sim_log",
]
