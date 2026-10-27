# Operator Skeleton Method (retired) — fix the shape, search the fields

## The problem

Invert grammar search.

Grammar search derives expression *shapes* and binds fields into whatever it derives. This
method does the opposite: fix the operator **skeleton** first, drawn from a registry of
hypotheses with role-constrained and dataset-constrained slots, then search which fields belong
in those slots.

The appeal is control. If you believe the useful shapes are a small known set, then constraining
slots by *role* — this slot wants a size proxy, that one wants a sentiment series — should
concentrate the search on economically coherent combinations instead of arbitrary ones.

## How it discovers alpha

`templates.py` holds the skeleton registry with declared slot roles. `field_sampler.py` selects
candidate fields, `screener.py` filters, `promoter.py` advances survivors, and `pipeline.py`
drives the loop.

The design is clear. What it actually did at runtime was not the same thing — see below.

```mermaid
flowchart TD
  CL[("all_clusters.json cache - 5451 clusters")]
  TR[("templates.py registry - 11 screening plus 5 promotion plus 6 role-constrained skeletons")]
  QT[("all_qtable.json - level-2 pruned field-op pairs")]
  UCB["TOMBSTONE ucb_legacy.py - UCB arm keyed on cluster_id alone"]
  FL["Keep MATRIX champion fields from 14 numeric datasets"]
  SMP["Uniform random unordered cluster-pair sampling - 200 pairs per template"]
  ROLE{"slot role constraints enforced"}
  DEAD[("DEAD CODE - ClusterRecord carries no role so the 6 role-constrained skeletons never sample")]
  REND["Render FASTEXPR - 4 windowless plus 7 windowed over windows 20 60 120"]
  SIM["Brain screening sims - batch 3 - TOP3000 delay 1 SUBINDUSTRY truncation 0.01"]
  GATE{"IS Sharpe at or above 1.25 and concentrated-weight check PASS"}
  DROP[("screening_results.jsonl - rejected rows retained for audit")]
  SCORE["Rank by IS Sharpe times mean yearly Sharpe over 1 plus std yearly Sharpe"]
  PROMO["Top-K 20 pairs times 3 arity-2 promotion skeletons"]
  OPT["TOMBSTONE optimizer.py and regen_proposals.py - combine and tune DISABLED"]
  WS["tools.batch_workshop - repair and portfolio admission"]

  CL -->|"cluster_id plus label plus champion_field_id"| FL
  FL -->|"3188 eligible clusters"| SMP
  TR -->|"operator skeleton with slot a slot b and window"| SMP
  QT -->|"read-only pruned set for divide ts_corr ts_covariance ts_regression"| SMP
  UCB -.->|"superseded - one-way pruning replaces regret minimisation"| QT
  SMP -->|"cluster pair plus chosen skeleton"| ROLE
  ROLE -->|"never - slot role fields are declared but read by nothing"| DEAD
  ROLE -->|"dataset constraints only"| REND
  REND -->|"5000 screening expressions at default n-pairs 200"| SIM
  SIM -->|"IS stats plus yearly Sharpes plus IS checks"| GATE
  GATE -->|"fail - the large majority of the 5000"| DROP
  GATE -->|"pass"| SCORE
  SCORE -->|"20 surviving field pairs"| PROMO
  PROMO -->|"100 promotion sims as workshop-format candidate dicts"| WS
  PROMO -.->|"pair and triple combining plus weight and constant tuning - every stage function returns empty"| OPT
```

## The code

| File | LOC | What it does |
|---|---|---|
| `alpha_discovery/legacy/operator_skeleton_method/screener.py` | 271 | Submit screening expressions to Brain and parse results |
| `alpha_discovery/legacy/operator_skeleton_method/templates.py` | 271 | Relationship template registry for operator-first alpha discovery |
| `alpha_discovery/legacy/operator_skeleton_method/field_sampler.py` | 236 | Field/cluster loading and pair sampling for operator-first alpha discovery |
| `alpha_discovery/legacy/operator_skeleton_method/pipeline.py` **(CLI)** | 221 | Operator-first alpha discovery pipeline |
| `alpha_discovery/legacy/operator_skeleton_method/promoter.py` | 172 | Promote top-scoring cluster pairs from screening to the full template suite |
| `alpha_discovery/legacy/operator_skeleton_method/roles.py` | 27 | Semantic role constants for field slot classification |
| `alpha_discovery/legacy/operator_skeleton_method/__init__.py` | 1 | Operator-first alpha discovery: templates → field pairs → Brain screening |

## Running it

```bash
cd alpha_discovery/legacy
python -m operator_skeleton_method.pipeline --dry-run
```

Archived: not imported by active code.

### Two deliberately disabled successors

Also archived, in [`../archived_shims/`](../archived_shims/), are two components that were
switched **off on purpose** and whose own docstrings explain why:

- `../archived_shims/wq_alpha/search/optimizer.py` — an alpha-combination optimizer, disabled
  because selecting combination weights on in-sample results is heavily exposed to selection
  bias. It would have produced beautiful backtests and no out-of-sample score.
- `../archived_shims/tools/regen_proposals.py` — disabled on the same grounds.

Both are left in place as working stubs that refuse to run, with the reasoning in the module
docstring, so the decision is documented where someone would look for the behaviour.

Also here: `../archived_shims/alpha_grammar/ucb_legacy.py`, a UCB bandit superseded by the
Q-table. A bandit was the wrong frame for this problem — the arms are not stationary, and the
reward cannot be observed without spending the very resource being allocated, so the usual
regret guarantees do not apply.

## Results

### Why it was retired

No alpha from this lane ever reached the ledger as a promoted candidate.

The more interesting failure is that **its distinguishing mechanism was never actually live**.
The slot *roles* are declared on every skeleton, but nothing in the execution path reads them —
the cluster records carry no role field to match against. So what ran was uniform random pairing
of clusters into a handful of fixed skeletons: all the constraint machinery that made the idea
attractive was inert, and the lane was quietly a worse version of template sweep.

It is kept partly as a reminder that a declared constraint which no code reads is not a
constraint, and that the way to find out is to check the execution path rather than the design
document.

