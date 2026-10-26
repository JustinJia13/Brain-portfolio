# Workshop: Negate, Repair, Enhance a Measured Alpha

*The LLM never invents from scratch here. It repairs an alpha that already has a measured backtest and a named failing check.*

[&larr; back to the pipeline overview](../../README.md)

Discovery lanes produce alphas with a *diagnosis attached* — a Sharpe, a turnover, and a specific failing submission check. The bet of this lane is that an LLM handed that evidence is a good repair engine even though it is a bad generator: 587 workshop sims produced 211 submittable (36%), while the same model proposing mechanisms from scratch produced 0/100. So the lane is shaped as a router, not a generator — `core.py:route_record` reads the metrics and sends each already-simulated record to the cheapest remedy that fits its defect (deterministic sign flip, settings-only fix, or structural LLM rewrite), and every proposal must re-clear the full submittable gate by re-simulation, never by argument.

## Pipeline

```mermaid
flowchart TD
    MASTER[("artifacts/sim/master.jsonl plus workshop_sim.jsonl — 96.6k already-simulated records")]
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

## Stages

| Stage | Implementation | What it does |
|---|---|---|
| 1. Select sources | `alpha_enhancement/workshop/batch.py:select_candidates` | Keep highest-|Sharpe| done record per expression+regime, drop 9478 already-worked keys and excluded fields, cap 200. |
| 2. Route by defect | `alpha_enhancement/workshop/core.py:route_record` | fitness<=-1.2 to negate; repairable failing checks + fitness>=0.9 to repair; all-pass + fitness>=1.0 + robustness>=0.30 to enhance. |
| 3a. Negate | `alpha_enhancement/workshop/repair.py:_sign_flip` | Deterministic sign flip of a strongly-negative alpha. No LLM, no cost. Blocked when concentration >= 0.40. |
| 3b. Settings repair | `tools/settings_repair.py + core/prompts/settings_repair.md` | One LLM call diagnoses the binding gate, returns <=4 settings_override fixes. Expression untouched, so execution-neutral. |
| 3c. Structural repair/enhance | `alpha_enhancement/agents/repair.py, alpha_enhancement/agents/enhance.py` | 3 rounds of generate then simulate then feed the sim verdict back. Parameter-only and settings-only variants rejected. |
| 3d. Semantic workshop | `alpha_enhancement/workshop/semantic.py + alpha_enhancement/agents/semantic_workshop.py` | Diagnostic agent types the signal and hypothesises the failure; generator agent emits 6 FASTEXPR. Hard 3-round cap. |
| 4. Submittable gate | `core/quality.py:is_submittable + workshop/core.py:is_passing` | Sharpe 1.25, fitness 1.0, turnover 0.70 AND every submission check PASS. Only SELF_CORRELATION and UNITS are warnings. |
| 5. Robustness then admission | `alpha_enhancement/workshop/robustness_card.py, core/portfolio/admission.py:_classify` | Perturbation retention kills IS spikes; then PROMOTE/WATCH/REJECT on brain score delta with a 0.70 global-corr veto. |

## Running it

```bash
python tools/batch_workshop.py --from-audit artifacts/workshop/workshop_audit.txt --workshop-iterations 3 --lane negate repair enhance --credentials brain.txt
```

## Outcome

This is the lane that empirically works, and the only one whose LLM calls pay for themselves. `artifacts/workshop/workshop_summary_20260620.txt` records the 2026-06-17 to 06-19 slice: 587 sims (enhance 327, repair 230, negate 30) -> 211 submittable (36%) -> 206 PROMOTE / 219 REJECT, 309 with positive score delta; the standouts were repair-lane fnd6_itci composites at Sharpe 2.1-2.3, fitness 2+. Contrast with the memory note `llm-generator-underperforms.md`: the same model proposing mechanisms from scratch went 0/100 submittable. Settings-repair is the cheapest sub-lane: 446 sims in `settings_repair_sim.jsonl` yielded 112 submittable rows in `settings_repair_submittable.jsonl` (~25%) by changing decay/neutralization alone. The honest failure mode is also on record - `artifacts/workshop/workshop_summary.txt` (2026-06-22) reads "56 simulated -> 22 standalone-passing -> 0 admission-promoted, REJECT=22", 18 of them HIGH_GLOBAL_CORR: once the workshop has mined a signal family hard, its own variants become near-duplicates of each other and the portfolio corr gate, not the alpha gate, becomes binding.

