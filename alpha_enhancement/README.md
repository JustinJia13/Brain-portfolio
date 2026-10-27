# The AI Alpha Enhancement Engine

## The problem

This is the second of the project's two pillars, and it exists because of a negative result.

Three separate attempts to make an LLM *generate* alphas all failed badly — see
[`../alpha_discovery/legacy/`](../alpha_discovery/legacy/README.md). LLM-authored expressions
were fluent, novel by every similarity measure, and empirically dead. Plain random sampling from
a grammar beat all of them.

But the diagnosis pointed at something specific. The LLM was being asked to do the one thing it
had no evidence for: predict, from the text of an expression alone, whether it would make money.
It had no feedback. Change that one condition — hand it an alpha that **already has a measured
backtest and a specific named failing check** — and the task becomes something an LLM is
genuinely good at: a constrained, diagnosable repair with a stated objective.

That is the whole thesis of this engine. **It never invents from nothing.** Its input is always
a simulated record with known metrics and known failures, and its job is to fix or extend it.

## How it enhances alpha

### Selection into the engine is hand-picked

This is deliberate and worth being explicit about, because it is the step a reader would
otherwise assume is automated. There is **no automatic promotion** from discovery into
enhancement. The researcher inspects the simulation ledger and the workshop audit, decides
which measured alphas are worth spending LLM and simulation budget on, and feeds that curated
list in — either as `--expr-file` (one expression per line) or `--from-audit <txt>`.

Only *after* that human selection does `route_record()` in `workshop/core.py` dispatch each
record automatically to one of three lanes.

### The three lanes

**negate** — the alpha has real structure but the wrong sign. The cheapest possible fix: flip
it and re-simulate. No LLM involved.

**repair** — the alpha fails a *named* check, and the repair is targeted at that check. This
splits in two. **Settings repair** changes neutralization, decay, truncation or universe while
leaving the expression untouched — often the right move for a concentration or turnover
failure, and it needs no LLM at all. **Semantic repair** asks an LLM to rewrite the expression,
but always with the failing check and the measured metrics in the prompt, and under a hard cap
on rounds so a failing repair cannot loop indefinitely.

**enhance** — the alpha already passes. The question is whether it can be made stronger without
becoming fragile, which is why robustness scoring rather than raw fitness is the acceptance
criterion.

A fourth component, `semantic_templates/`, abstracts families that were *already promoted* into
typed-slot templates and binds new fields into those slots — exploiting what worked rather than
repairing what failed.

```mermaid
flowchart TD
    MASTER[("artifacts/sim/master.jsonl plus workshop_sim.jsonl — 91969 already-simulated records")]
    DONE[("workshopped_sources.txt — 9478 regime-aware source keys already worked")]
    SEL["select_candidates — best done record per expression plus regime, excluded-field filter, tier negate then repair then enhance, cap 200 per run"]
    ROUTE{"route_record in workshop/core.py"}
    NEG["negate lane — deterministic _sign_flip, no LLM call"]
    SET["settings_repair — ONE LLM call names the single binding gate, at most 4 settings_override fixes from decay 4 8 16 32 or neutralization or truncation"]
    REP["repair plus enhance lanes — AlphaRepairAgent and AlphaEnhanceAgent, 3 iterations of generate then simulate then feed the sim results back, parameter-only edits banned"]
    SEM["semantic workshop — diagnostic agent emits typed template plus failure hypothesis, generator agent emits 6 FASTEXPR, rounds 1 2 3 ONLY"]
    SIM["submit_candidate_batch — Brain simulate, batch size 3"]
    PASS{"is_submittable — abs sharpe at or above 1.25 AND abs fitness at or above 1.0 AND turnover under 0.70 AND every check PASS incl CONCENTRATED_WEIGHT and LOW_SUB_UNIVERSE_SHARPE"}
    ROB["robustness card and finalist_gate — year-consistency score, window-plateau retention at or above 0.5 of source sharpe, neutralization sign held"]
    ADM{"portfolio admission _classify — brain score delta, else same-delay pool sharpe delta, corr limit 0.70"}
    OUT[("workshop_submittable.jsonl and promote_to_brain.jsonl")]
    HIST[("WorkshopHistory — multi-hop lineage root to variant across every lane")]
    DROP[("discarded — row still kept in workshop_sim.jsonl as negative evidence")]

    MASTER -->|"status done records plus checks"| SEL
    DONE -->|"skip keys, already-worked sources never re-enter"| SEL
    SEL -->|"seed record, metrics, failing-check names"| ROUTE
    ROUTE -->|"fitness at or below -1.2 and concentration under 0.40"| NEG
    ROUTE -->|"failing checks ALL repairable and fitness at or above 0.9"| REP
    ROUTE -->|"all checks pass, fitness at or above 1.0, robustness at or above 0.30"| REP
    ROUTE -->|"grammar seed routed repair or enhance, max 5 seeds"| SEM
    ROUTE -->|"sharpe 1.25 to 4.0, not submittable, turnover under 0.999"| SET
    ROUTE -->|"skipped — robustness under 0.30 or concentration at or above 0.40 or score delta at or below 0 or self-corr at or above 0.70"| DROP
    NEG -->|"negated FASTEXPR"| SIM
    REP -->|"structural variants"| SIM
    SEM -->|"6 variants per round"| SIM
    SET -->|"SAME expression, new decay or neutralization or truncation"| SIM
    SIM -->|"sim record, checks, yearly stats"| PASS
    SIM -->|"every result appended, lineage registered"| HIST
    SIM -->|"source key marked done only AFTER submit"| DONE
    HIST -->|"full prior-attempt gradient prepended to the next LLM call"| REP
    HIST -->|"previous_results plus latest robustness card"| SEM
    SEM -->|"round 1 produced no positive-delta standalone passer, or diagnostic returned abandon, or zero NEW valid expressions"| DROP
    PASS -->|"fails any one check — 587 sims gave 211 submittable"| DROP
    PASS -->|"submittable"| ROB
    ROB -->|"IS parameter spike, no year consistency"| DROP
    ROB -->|"plateau not peak"| ADM
    ADM -->|"HIGH_GLOBAL_CORR or non-positive delta — 22 of 22 in the 2026-06-22 run"| DROP
    ADM -->|"positive brain score delta — 206 PROMOTE in the 2026-06-17 to 06-19 slice"| OUT
```

## The code

| File | LOC | What it does |
|---|---|---|
| `alpha_enhancement/workshop/batch.py` **(CLI)** | 2494 | Batch auto-workshop over entire simulation history |
| `alpha_enhancement/semantic_templates/core.py` | 1623 | Deterministic helpers for semantic templates and template routing |
| `alpha_enhancement/workshop/semantic.py` **(CLI)** | 1249 | Post-grammar semantic LLM workshop runner |
| `alpha_enhancement/agents/repair.py` | 955 | LLM-based alpha repair agent using the OpenAI-backed LLM wrapper |
| `alpha_enhancement/agents/semantic_workshop.py` | 707 | Two-agent LLM helpers for post-grammar semantic workshop |
| `alpha_enhancement/workshop/alpha.py` **(CLI)** | 684 | Unified alpha fix/enhancement workshop |
| `alpha_enhancement/workshop/targeted.py` **(CLI)** | 595 | Targeted workshop runner for specific alphas |
| `alpha_enhancement/workshop/robustness_card.py` | 585 | Robustness-card helpers for semantic workshop |
| `alpha_enhancement/workshop/core.py` | 555 | Auto-workshop: route sim records to negate/repair/enhance and generate candidates |
| `alpha_enhancement/workshop/seed_resim.py` | 434 | Seed resimulation helpers for score-missing semantic/workshop sources |
| `alpha_enhancement/workshop/repair.py` **(CLI)** | 428 | Near-miss repair: scan simulation logs for high-Sharpe failing alphas and |
| `alpha_enhancement/semantic_templates/pipeline.py` **(CLI)** | 421 | Semantic template architect/router CLI |
| `alpha_enhancement/agents/enhance.py` | 408 | LLM-based alpha enhancement agents using the OpenAI-backed LLM wrapper |
| `alpha_enhancement/semantic_templates/handoff.py` | 406 | Semantic-workshop handoff into the semantic-template pipeline |
| `alpha_enhancement/workshop/semantic_context.py` | 371 | Auditable context packets for semantic workshop LLM prompts |
| `alpha_enhancement/agents/semantic_templates.py` | 360 | LLM agents for semantic template creation, field selection, and routing |
| `alpha_enhancement/workshop/scoring.py` | 306 | Shared workshop seed scoring and admission-context helpers |
| `alpha_enhancement/workshop/simulation.py` | 290 | Shared workshop candidate simulation helpers |
| `alpha_enhancement/workshop/history.py` | 197 | WorkshopHistory — shared trial history across all workshop lanes and iterations |
| `alpha_enhancement/workshop/probe_cards.py` | 186 | Compact probe-card helpers for semantic workshop diagnostics |

The submittable predicate (`is_passing`, `resolve_checks`, `failing_checks`) and the JSON parsing helpers live in the shared layer at `core/quality.py` and `core/json_utils.py`, so that discovery lanes can use them without importing the enhancement engine. Prompts are in `core/prompts/`.

## Running it

```bash
# dry run: show what would be routed where, no sims, no LLM calls
python tools/batch_workshop.py --dry-run

# hand-picked input, explicit lanes, capped LLM rounds
python -m alpha_enhancement.workshop.batch \
    --from-audit artifacts/workshop/workshop_audit.txt \
    --lane negate repair enhance \
    --workshop-iterations 3 \
    --credentials brain2.txt

# a single curated expression list
python -m alpha_enhancement.workshop.alpha \
    --expr-file my_picks.txt --credentials brain2.txt
```

## Results

### What is and is not evidenced

The workshop is the only lane in the project whose LLM calls clearly pay for themselves — its
submittable rate on hand-picked input is an order of magnitude above every generation lane.
That asymmetry is the project's central empirical finding.

`semantic_templates/` is the honest exception. It is wired, default-on inside the semantic
workshop runner, and has produced **no surviving output** — `artifacts/semantic_templates/`
does not exist. It is built, not yet evidenced, and is described that way rather than counted
as a working component.

