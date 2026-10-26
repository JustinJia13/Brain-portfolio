# Alpha Research Pipeline

This repo is organized around one current objective: find structurally distinct
FASTEXPR alphas, repair or enhance them structurally, then submit individual
alphas that improve the Brain portfolio pool. Combo construction and adaptive
parameter tuning are disabled.

## Current Status (2026-06-12) — Read This First

`artifacts/sim/master.jsonl` has been RESTORED (81.5k records: .bak + all
sessions/archives + both accounts' `records/` shards merged). The `records/`
tree remains the durable source of truth after the June 2026 disk failure.

The pipeline was upgraded on 2026-06-12 to a theory-first architecture:

- **Quality ordering (everywhere):** an alpha is judged 1st by whether it is
  SUBMITTABLE (sharpe >= 1.25, fitness >= 1.0, turnover < 0.70, all
  submission-relevant checks pass), 2nd by its robustness SCORE
  (year-consistency, `core/robustness.py`), 3rd by fitness. Fitness alone
  never ranks an alpha as "good" — see `core/wiki.generate.quality_rank_key`.
- **Robustness-first selection:** yearly-consistency scores feed the Q-table
  reward, workshop routing (enhance blocked below 0.30 when data exists), and
  all ranking keys. Finalist gate (`tools/finalist_gate.py`) adds
  window-plateau + neutralization checks before submission.
- **Theory lane (primary generator):** `tools/theory_lane.py` — LLM proposes
  pre-registered hypothesis cards, the grammar enumerates designed expression
  families, verdicts are family-level, and the median-robustness members (not
  the max-sharpe spike) get promoted. Grammar random discovery continues as a
  background evidence collector.
- **Wiki (posterior store):** `wiki/` regenerates after every workshop/theory
  run; falsified mechanisms are first-class. `tools/build_wiki.py
  --export-priors` also feeds the grammar lane's probe-prior input.
- **All LLM prompts live in `core/prompts/`** (one file per prompt).

Where to find what:

| You need | Look at |
|----------|---------|
| Historical sim results (recovered, per account) | `records/brain/alphas_part*.jsonl`, `records/brain2/alphas_part*.jsonl` — master.jsonl schema, `raw_alpha_detail` inline |
| Surviving pre-crash master history (~17.9k rows, through 2026-04) | `artifacts/sim/master.jsonl.bak` |
| OS data for submitted alphas (yearly-stats, daily PnL) | `records/<account>/os_part*.jsonl` |
| Competition portfolio snapshot | `records/<account>/competition_IQC2026S1.json` |
| Backfill resume cursor / last-run metadata | `records/<account>/state.json` |
| Backfill/running-log implementation | `tools/fetch_brain_records.py` (module docstring covers pagination, shard rotation, resume) |
| Record schema and master merge/dedup logic | `core/logs.py` (`sim_record_key`, `merge_session_to_master`) |
| Brain auth, rate-limit retries, paged fetches | `core/brain/api.py` |

Operational rules for this state:

- To refresh records (also the ongoing running log; resumes + dedupes):
  `python tools/fetch_brain_records.py --all`. Account `brain2` works
  non-interactively; account `brain` triggers an interactive Persona face scan,
  so it must run in a terminal with the user present — never in background.
- Auth attempts are budgeted (25/day, ledger `artifacts/auth/auth_attempts.json`);
  do not sign in casually.
- Shards rotate before 95 MB; `records/` is intentionally committable (not
  gitignored) for GitHub history. Never write files ≥100 MB there.
- Master is already rebuilt; to refresh it after new fetches run
  `python tools/fetch_brain_records.py --all --merge-master`.
- Use the `worldquant` conda env
  (`${PYTHON:-python3}`) — base lacks
  numpy/openai.
- Active lanes are described below (theory lane + grammar discovery → workshop
  → finalist gate → portfolio admission, plus probe, semantic-template, wiki,
  and archaeology lanes).

## Code Space Map

| Path | Role | Current status |
|------|------|----------------|
| `alpha_discovery/theory_method/` | Theory lane: economist agent, mechanism dedup, family enumeration/verdicts, runner | **Primary discovery lane** |
| `core/prompts/` | All LLM prompts, one `.md` per prompt (`## system` / `## user`, `$placeholders`) | Shared LLM assets |
| `core/wiki/` | Wiki generator: structured cluster stats, narrative pages, grammar prior export | Knowledge compounding |
| `core/robustness.py` | Year-consistency robustness score + reward multiplier | Shared selection criterion |
| `alpha_discovery/grammar_method/` | Grammar discovery, field clustering, Q-table pruning, tested-pair registry | Background discovery lane |
| `core/brain/` | Brain API/session helpers, simulation submission, record normalization | Shared infrastructure |
| `core/portfolio/` | Pool/admission scoring and same-delay contribution checks | Main submission filter |
| `alpha_enhancement/workshop/` | Batch selection, routing, public simulation helpers, admission, history handling | Main repair/enhance orchestration |
| `alpha_enhancement/agents/` | LLM repair/enhance prompts, semantic agents, template agents, deterministic test stubs | Structural LLM lanes |
| `alpha_enhancement/semantic_templates/` | Promoted-alpha semantic template architect, router, plan builder, feedback | Active template lane |
| `alpha_discovery/template_sweep_method/` | Older template-search compatibility, submit pipeline, fixed regime check | Compatibility plus regime checks |
| `alpha_discovery/probe_method/` | Field-level medoid probes and signal summaries | Active prior-building lane |
| `tools/` | Maintained CLIs and harnesses; reusable logic should move into active packages | Active operational wrappers |
| `alpha_search/`, `alpha_multi/` | Compatibility facades pointing to `core` implementations | Keep for old scripts/tests |
| `legacy/` | Archived historical experiment lanes | No active imports |

Generated artifacts live mostly under `artifacts/`, `alpha_discovery/grammar_method/artifacts/`,
`alpha_search/artifacts/`, `alpha_multi/artifacts/`, and
`artifacts/semantic_templates/`. Treat those as run state, not core source.

## Current Flow

```text
Stage 1a: Theory lane (primary)
  tools/theory_lane.py
  -> LLM proposes pre-registered hypothesis cards (prompts/theory_propose.md)
  -> embedding dedup vs mechanism registry (artifacts/theory/mechanisms.jsonl)
  -> grammar-constrained family enumeration (5-20 designed variants)
  -> simulate (master-cache hits are free; early-stop after first 5 if 0 support)
  -> family verdict: VALIDATED / FALSIFIED / INCONCLUSIVE
  -> promote MEDIAN-robustness members into the workshop
  -> artifacts/sim/sessions/theory_<timestamp>.jsonl -> master.jsonl -> wiki

Stage 1b: Grammar discovery (background)
  alpha_discovery/grammar_method.run_pipeline  (now reads artifacts/probe/grammar_prior_from_wiki.json)
  -> structural FASTEXPR candidates
  -> artifacts/sim/sessions/grammar_<timestamp>.jsonl -> master.jsonl

Stage 2: Workshop repair/enhance
  tools/batch_workshop.py
  -> negate weak negative alphas deterministically
  -> repair near misses with structural LLM changes
  -> enhance passers with structural LLM changes (blocked when robustness < 0.30)
  -> artifacts/workshop/workshop_submittable.jsonl
  -> wiki refresh (deterministic pages)

Stage 2.4: Settings-repair (strong-but-not-submittable -> submittable)
  tools/settings_repair.py
  -> selects alphas with Sharpe>=1.25 that fail a submission gate (usually
     HIGH_TURNOVER, which also tanks fitness)
  -> one LLM call per alpha DIAGNOSES the binding constraint and proposes
     targeted sim-setting fixes (decay for turnover, neutralization/truncation
     for concentration) — principled single remedies, not a sweep
  -> re-sims the fixes; admits only those that become submittable AND robust
  -> artifacts/workshop/settings_repair_submittable.jsonl
  -> standalone; does NOT touch the legacy autoworkshop repair agent

Stage 2.5: Finalist robustness gate
  tools/finalist_gate.py
  -> window-plateau + neutralization-consistency perturbations (~6 sims each)
  -> artifacts/robustness/finalist_cards.jsonl (robust/weak/fail)
  -> "fail" = IS peak that will decay OS; do not submit

Stage 2.7: Promote brain2 winners to the brain pool
  tools/promote_to_brain.py
  -> reads artifacts/theory/promote_to_brain.jsonl (validated, submittable,
     robust winners discovered on brain2)
  -> re-simulates them on the brain account WITH the portfolio gates
     (self-correlation + Brain score-addition) that brain2 skips
  -> artifacts/theory/promoted_brain_sim.jsonl

Stage 3: Portfolio-aware review
  Review PROMOTE rows ranked submittable > robustness score > fitness.
  Submit alphas individually when they improve Brain score or same-delay pool Sharpe.

Knowledge loop: Wiki + archaeology
  tools/build_wiki.py [--llm] [--export-priors]
  -> wiki/structured/cluster_stats.jsonl (authoritative, machine-written)
  -> wiki/clusters/*.md with first-class "What failed" sections
  -> grammar prior export (revives the probe-prior lane)
  tools/archaeology_bootstrap.py
  -> retrospective theory cards from 78k historical sims (PRIOR, not posterior:
     retro cards need a live confirmation family before VALIDATED)

Optional: Fixed regime check
  tools.regime_check
  -> predeclared universe, neutralization, truncation, decay checks
  -> excluded from Q-table and workshop feedback

Optional: Semantic template expansion
  tools.semantic_templates
  -> architect semantic templates from promoted roots
  -> route medoid fields through probe-gated compatibility checks
  -> render bounded template plans
```

Preferred implementation ownership:

- Brain/session code: `core/brain`.
- Shared workshop candidate submission: `alpha_enhancement/workshop.simulation.submit_candidate_batch`.
- Shared JSONL record loading: `alpha_enhancement/workshop.records.load_records`.
- LLM agents and deterministic harness stubs: `alpha_enhancement/agents`.
- CLI-only code: `tools`.

## Stage 1a: Theory Lane (Primary)

Entry points:

```bash
# iterate prompts safely: propose + dedup + enumerate, no simulations
python tools/theory_lane.py --dry-run --n-theories 5

# pilot run: simulate the 2 best-ranked theories only
python tools/theory_lane.py --n-theories 5 --max-simulated 2

# full run with one refine round per inconclusive theory
python tools/theory_lane.py --n-theories 8 --refine
```

Rules:

- Hypothesis cards are PRE-REGISTERED: the LLM never sees results before the
  family is fixed, and the family is rendered by the grammar enumerator — the
  LLM declares clusters/operators/horizons/skeletons, never final expressions.
- The verdict is family-level. A lone max-sharpe spike inside a failed family
  is a multiple-testing artifact and is discarded with the family.
- Promotion takes the MEDIAN-robustness supporters, never the best member.
- Falsified mechanisms go to the graveyard (wiki + registry); the dedup
  adjudicator blocks re-proposals unless a card names an untested dimension.
- Retro (archaeology) theories are PRIOR, not posterior: they need a live
  confirmation family (~5 sims, often free via master-cache) before VALIDATED.

Key artifacts:

| File | Purpose |
|------|---------|
| `artifacts/theory/mechanisms.jsonl` | Mechanism registry (cards, Beta posteriors, verdicts, adjudications) |
| `artifacts/theory/mechanism_embeddings.npz` | Embedding sidecar for dedup |
| `artifacts/theory/theories.jsonl` | Per-run verdict log |
| `artifacts/sim/sessions/theory_*.jsonl` | Theory-lane simulation sessions |

## Alpha Quality Ordering (applies everywhere)

Good vs bad is decided lexicographically — fitness alone never ranks an alpha:

1. **Submittable**: sharpe >= 1.25, fitness >= 1.0, turnover < 0.70, and all
   submission-relevant checks pass (`core.quality.is_submittable`).
2. **Score**: year-consistency robustness score (`core/robustness.py`).
3. **Fitness**: tiebreaker only.

The canonical key is `core.quality.quality_rank_key`. It governs wiki
top-expression lists, archaeology exemplars, the grammar prior export quality
signal, theory-family promotion, and review ranking.

## Two Pools: brain2 (discovery) and brain (competition)

The two accounts are deliberately different pools:

- **brain2 = discovery pool.** The theory lane runs here (headless auth, cheap).
  It has NO self-correlation gate and NO competition score-addition gate, because
  those are properties of the *other* account's portfolio. brain2 alphas are
  ruled only by **submittable, robustness score, and fitness** — the IS backtest
  of an expression is account-independent (same region/universe/delay), so brain2
  IS metrics predict brain IS metrics exactly.
- **brain = competition pool.** Self-correlation and Brain score-addition apply
  here against the brain portfolio. An alpha must be simulated under the brain
  account to be submitted to the competition.

Promotion path: when a brain2 theory family is VALIDATED and yields submittable,
robust members, the runner queues them in `artifacts/theory/promote_to_brain.jsonl`.
`tools/promote_to_brain.py` re-simulates them on brain with the portfolio gates
active. Only then do correlation/score-addition decide actual submission. VECTOR
fields are auto-wrapped with `vec_avg(...)` during enumeration (raw VECTOR fields
error on Brain), mirroring `alpha_discovery/grammar_method._field_node`.

## Stage 1b: Grammar Discovery (Background)

Entry point:

```bash
python -m alpha_discovery/grammar_method.run_pipeline \
    --fields-csv all_fields.csv \
    --delay 1 \
    --exclude-dataset analyst4 \
    --n-samples 300 \
    --credentials brain2.txt
```

Purpose:

- Generate structurally new FASTEXPR expressions from the grammar.
- Use field clusters and operator grammar to explore new economic forms.
- Use Q-table pruning to avoid dead `(cluster, operator)` and `(field, operator)`
  combinations.
- Append deduped results into `artifacts/sim/master.jsonl`.

Anti-overfit boundary:

- Acceptable: trying new fields, operators, signal constructions, conditioning,
  neutralization inside the expression, and economic transformations.
- Risky: repeatedly nudging windows/constants/settings on the same expression
  until the in-sample score improves.

Key artifacts:

| File | Purpose |
|------|---------|
| `artifacts/sim/master.jsonl` | Permanent deduped simulation history |
| `artifacts/sim/sessions/grammar_<timestamp>.jsonl` | Per-run grammar session |
| `alpha_discovery/grammar_method/artifacts/all_qtable.json` | Learned pruning state |
| `alpha_discovery/grammar_method/artifacts/all_tested.json` | Tested field/operator registry |
| `alpha_discovery/grammar_method/artifacts/all_clusters.json` | Field cluster cache |

Keep delay 0 and delay 1 as separate mining regimes. Use explicit `--delay 0`
or `--delay 1` and keep their logs separated when comparing results.

## Field Probing

Probe medoid/champion fields with simple expression families before spending
larger grammar or template budgets. Probe output should become priors for field
cluster selection, direction, and route eligibility; it should not become an
adaptive tuning loop by itself.

Entry point:

```bash
python -m alpha_discovery/probe_method.build_plan \
    --clusters alpha_discovery/grammar_method/artifacts/all_clusters.json \
    --out artifacts/probe/probe_plan.jsonl \
    --pack core \
    --representatives champion

python -m core/brain.submit \
    --plan artifacts/probe/probe_plan.jsonl \
    --sim-log artifacts/probe/probe_sim.jsonl \
    --batch-size 3
```

## Stage 2: Workshop Repair And Enhance

Entry point:

```bash
python tools/batch_workshop.py
```

Dry-run audit:

```bash
python tools/batch_workshop.py --dry-run
python tools/batch_workshop.py --from-audit artifacts/workshop/workshop_audit.txt
```

Workshop routing:

| Lane | Condition | Action |
|------|-----------|--------|
| `negate` | Strong negative Sharpe and Fitness | Flip sign and simulate |
| `repair` | Good signal with failing checks | Ask LLM for structural repairs |
| `enhance` | Passing signal | Ask LLM for structural improvements |

The repair/enhance prompts explicitly reject parameter-only edits. The LLM is
asked for structural changes such as different operator paths, signal framing,
conditioning logic, turnover reduction, or interactions with fields already in
the source family.

Workshop artifacts:

| File | Purpose |
|------|---------|
| `artifacts/workshop/workshop_sim.jsonl` | Workshop simulation results |
| `artifacts/workshop/workshopped_sources.txt` | Processed source expressions |
| `artifacts/workshop/workshop_audit.txt` | Human-editable audit list |
| `artifacts/workshop/workshop_admission.jsonl` | PROMOTE/WATCH/REJECT admission rows |
| `artifacts/workshop/workshop_submittable.jsonl` | PROMOTE rows for review |
| `artifacts/workshop/new_templates.jsonl` | Compatibility copy of promoted rows |

## Stage 2.5: Finalist Robustness Gate

Before submitting, gate the top finalists with designed perturbations:

```bash
python tools/finalist_gate.py --dry-run          # show planned sims
python tools/finalist_gate.py --top 5            # ~6 sims per finalist
```

Verdict per finalist (`artifacts/robustness/finalist_cards.jsonl`): window
variants must retain >= 0.5 of source sharpe (plateau, not peak) and
INDUSTRY/SECTOR neutralization variants must keep the same sign with
reasonable retention. `fail` means the IS performance is a parameter spike —
expect OS decay; do not submit.

## Wiki And Archaeology (Knowledge Loop)

```bash
python tools/build_wiki.py                    # structured layer + deterministic pages (free)
python tools/build_wiki.py --llm              # LLM narrative for changed pages only
python tools/build_wiki.py --export-priors    # grammar prior -> artifacts/probe/grammar_prior_from_wiki.json
python tools/archaeology_bootstrap.py --limit 3   # pilot; drop --limit for full sweep
```

Two layers, separation is load-bearing: `wiki/structured/` is machine-written
from sim records only (authoritative; drives allocation and prior export);
narrative pages are LLM-distilled from the structured layer and may add
interpretation, never new facts. Falsified mechanisms get equal prominence to
validated ones ("What failed" sections) — they are what stop budget from being
re-burned. The wiki refreshes automatically after batch-workshop and
theory-lane runs.

## Stage 3: Portfolio-Aware Review

Primary review file:

```text
artifacts/workshop/workshop_submittable.jsonl
```

Review command:

```bash
python tools/show_results.py \
    --sim-log artifacts/workshop/workshop_submittable.jsonl \
    --min-sharpe 1.25 --passing-only
```

Admission uses Brain score delta when available. Otherwise it falls back to the
same-delay equal-weight pool Sharpe contribution, residual Sharpe, and global
self-correlation checks.

Submit individual alphas from `workshop_submittable.jsonl`. Do not combine them
into one expression for submission. Brain's final score is based on the submitted
pool, so two distinct low-correlation submissions can contribute differently than
one blended expression, and the blended expression can also obscure which leg is
actually contributing.

## Optional: Fixed Regime Check

Use this only as a robustness check, not as a greedy optimizer.

```bash
python -m tools.regime_check build \
    --input artifacts/workshop/workshop_submittable.jsonl \
    --output artifacts/regime_check/plan.jsonl

python -m alpha_search.pipeline submit \
    --plan artifacts/regime_check/plan.jsonl \
    --sim-log artifacts/regime_check/sim.jsonl

python -m tools.regime_check summary \
    --sim-log artifacts/regime_check/sim.jsonl
```

Default `regime_check` mode varies one axis at a time:

- Universe: `TOP3000`, `TOP2000`, `TOP1000`
- Neutralization: `SUBINDUSTRY`, `INDUSTRY`, `SECTOR`, `MARKET`
- Truncation: `0.01`, `0.02`, `0.04`
- Decay: `0`, `2`, `4`, `8`

Rules:

- The grid must be declared before simulation.
- Do not expand the grid after seeing a good result.
- Do not feed regime-check rows into Q-table learning.
- Do not feed regime-check rows back into workshop candidate selection.
- Treat a setting-only improvement as evidence about robustness, not as a new
  alpha idea by itself.

This repo does not tune arbitrary historical test periods. Brain settings used
here are universe, neutralization, truncation, decay, delay, region, language,
instrument type, pasteurization, unit handling, and NaN handling.

## Optional: Semantic Template Expansion

The active template lane starts from promoted/robust alphas, not raw syntactic
extraction. The architect agent converts a promoted root into a semantic
template with typed slots; the router agent maps compatible probe-gated clusters
into those slots; the plan builder renders a small medoid-first plan.

```bash
python -m tools.semantic_templates architect --max-roots 5

python -m tools.semantic_templates route \
    --candidate-limit 12 \
    --chunk-size 6

python -m tools.semantic_templates build-plan \
    --max-windows 1 \
    --max-groups 1 \
    --max-candidates-per-route 1
```

Mechanical template extraction into `shared/discovered_templates.json` has been
removed. Use semantic templates for promoted-root reuse.

## Optional: Legacy Template Search

The older template-search facade is still available for curated experiments:

```bash
python -m alpha_search.pipeline run \
    --sim-log alpha_search/artifacts/all_sim.jsonl \
    --credentials brain.txt
```

Use this for deliberate compatibility experiments only. The main pipeline should
prefer grammar/probe discovery, workshop repair/enhance, and semantic templates.
Do not use template variants as an adaptive parameter search loop.

## Disabled Or Legacy Paths

These paths are intentionally retained only for compatibility:

| Path/command | Status |
|--------------|--------|
| `python -m alpha_search.pipeline refine` | Disabled no-op; combo optimizer submits nothing |
| `alpha_enhancement/agents.enhance.OverseerAgent` | Disabled compatibility API; no parametric sweeps |

Do not reintroduce combo search, window sweeps, weight sweeps, or adaptive
setting sweeps into the main pipeline without a separate out-of-sample validation
design.

## Operating Checklist

1. Keep the wiki fresh: `tools/build_wiki.py --export-priors` after big merges;
   run archaeology when new history accumulates.
2. Run the theory lane (dry-run first) — the primary source of new candidates.
3. Run grammar discovery in the background with the wiki-exported prior.
4. Run workshop dry-run and inspect the audit file when you want human control.
5. Run workshop live for negate, repair, and enhance.
6. Gate finalists with `tools/finalist_gate.py` before considering submission.
7. Review `workshop_submittable.jsonl` (ranked submittable > score > fitness)
   and submit individual PROMOTE alphas.
8. Build semantic templates from promoted roots when you want bounded reuse.
9. Run `regime_check` only when you want predeclared robustness evidence.
10. Keep regime-check and manual sweep outputs out of Q-table/workshop learning.

## Verification Commands

```bash
python -m py_compile \
    alpha_discovery/template_sweep_method/pipeline.py \
    alpha_discovery/template_sweep_method/sweep.py \
    alpha_discovery/template_sweep_method/regime_check.py \
    alpha_enhancement/semantic_templates/pipeline.py \
    tools/regime_check.py \
    tools/semantic_templates.py \
    tools/alpha_multi_enhance_loop.py

python -m unittest tools.test_qtable_pool_delta -v
python -m tools.test_self_corr_verification --unit-only
python -m tools.semantic_templates --help
python -m alpha_search.pipeline refine \
    --seed-file artifacts/workshop/workshop_submittable.jsonl \
    --sim-log /tmp/refine_disabled.jsonl
python -m tools.regime_check build \
    --expression 'rank(close)' \
    --output /tmp/regime_plan.jsonl \
    --sim-log /tmp/regime_sim.jsonl
git diff --check
```
