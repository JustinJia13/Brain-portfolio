# `core/` — the shared platform

Everything both pillars need, and nothing either pillar owns. `core` is the bottom layer: it
imports nothing from [`alpha_discovery/`](../alpha_discovery/README.md) or
[`alpha_enhancement/`](../alpha_enhancement/README.md).

That constraint is the point. Before the restructure, several of these pieces lived inside the
grammar lane's 1,489-line `run_pipeline.py`, which forced the enhancement engine and the
portfolio gate to import from a *discovery method* just to compute a self-correlation. Lifting
them out here removed that cycle.

| Module | What it owns |
|---|---|
| `brain/` | All WorldQuant BRAIN I/O: authentication, simulation submit and poll, paged metric and PnL fetches. The only place that talks to the platform. |
| `expr/operators.py` | The FASTEXPR operator registry — 70 operators with their arity, types and constraints. Single source of truth for the expression vocabulary. |
| `expr/models.py` | `ExpressionNode`, `DataProfile`, `FieldCluster` — the data model every lane passes around. |
| `fields.py` | Builds, filters and clusters field profiles from the catalog. |
| `clusters.py` | Constructs and caches the field-cluster catalog, and loads it back. Four separate modules read this same catalog, so it lives here. |
| `qtable.py` | The two-level Q-table. Shared because several lanes write to it, not just the grammar lane. |
| `selfcorr.py` | Self-correlation, portfolio PnL loading, and simulation-record construction. |
| `quality.py` | The **submittable predicate** (`is_passing`, `resolve_checks`, `failing_checks`) and the quality ordering (`quality_rank_key`). Encodes the rule: submittable > score > fitness, never fitness-led. |
| `robustness.py` | Year-consistency robustness scoring — the repository-wide selection criterion. |
| `feedback.py` | `SimResult` and the quality weighting every lane produces. |
| `logs.py` | Session and master simulation-log handling, and the dedup key. |
| `llm.py` | The Anthropic client shim and API-key resolution for every agent. |
| `json_utils.py` | Tolerant JSON parsing for LLM responses. |
| `prompts/` | Every LLM prompt template, kept as data rather than embedded in code. |
| `portfolio/` | The score-add admission gate — [README](portfolio/README.md). |
| `wiki/` | The per-cluster posterior that is exported back as a sampling prior — [README](wiki/README.md). |

## Why the submittable predicate lives here

It looks like an enhancement-engine concern, since that is where alphas get routed by their
failing checks. But *every* lane needs to know whether a simulated alpha passed, and the
definition is not negotiable per lane: an alpha is submittable only when **all** platform checks
pass — not just Sharpe, fitness and turnover, but also concentrated weight, sub-universe Sharpe
and units. Keeping one definition in the shared layer is what stops two lanes from quietly
disagreeing about what "good" means.

## Layering

```
core/                     <- imports nothing above it
  ^              ^
alpha_discovery/   alpha_enhancement/        <- may import core
  ^              ^
tools/                                        <- may import anything
```

Verified mechanically: `alpha_enhancement → alpha_discovery` edges are zero, and `core` has one
deliberate, annotated exception — `core/brain/submit.py`'s opt-in `--auto-workshop` CLI path
lazily imports the workshop inside `main()`, so the submit *library* still has no upward
dependency.

## Prompts are data

`core/prompts/*.md` holds the prompt templates. The loader uses `string.Template.substitute()`
rather than `safe_substitute()` on purpose: a renamed placeholder then fails loudly instead of
silently leaking a literal `$name` into an LLM's context.
