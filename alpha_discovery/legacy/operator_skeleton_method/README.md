# Operator Skeleton Method (retired) — fix the shape, search the fields

## The problem

Turn the grammar method around. The grammar derives formula *shapes* and fills in whatever fields
it lands on. This method fixes the shape first — a small registry of operator "skeletons", each
encoding one economic hypothesis — and then searches for the fields to put in them.

The appeal is control. Each blank in a skeleton is labelled with the *role* it needs — a
fundamental quantity here, a size measure there — so the search should only ever try economically
coherent combinations instead of arbitrary ones.

## How it discovers alpha

1. **Start from a registry of skeletons** — fixed formula shapes, each blank labelled with the
   role of field it should take (`templates.py`).
2. **Fill the blanks with pairs of field clusters** (`field_sampler.py`).
3. **Screen cheaply first:** run a few skeletons over many cluster pairs and rank the pairs by how
   well they did (`screener.py`).
4. **Promote the best pairs to the full suite** of skeletons (`promoter.py`).

What actually happened at step 2 is the interesting part. The role labels were declared on every
skeleton, but **no code ever read them** — the sampler drew cluster pairs uniformly at random. So
the feature that made the idea attractive was never switched on.

```mermaid
flowchart TB
    SPACE[("search space<br/>7642 fields times 70 operators")]

    subgraph OPS["OPERATOR SKELETON METHOD, retired — fix the shape, then search for fields"]
        direction TB
        SKEL[("registry of skeletons<br/>fixed shapes with role-labelled blanks")]
        FILL["fill the blanks with pairs of field clusters<br/>role matching intended, never wired up"]
    end

    SIMBUS["BRAIN simulation<br/>the metered shared resource"]
    LEDGER[("simulation ledger<br/>every alpha with its metrics and its failed checks")]
    RANK{"a top pair<br/>in screening?"}
    FULL["run every skeleton<br/>on the promoted pairs"]
    DROP[("discarded")]

    SPACE -->|"field clusters"| FILL
    SKEL -->|"formula shapes"| FILL
    FILL -->|"screening formulas"| SIMBUS
    SIMBUS -->|"metrics and named failing checks"| LEDGER
    LEDGER -->|"screening results"| RANK
    RANK -->|"yes"| FULL
    RANK -->|"no"| DROP
    FULL -->|"full suite of formulas"| SIMBUS
```

**How to read this graph:** the labelled box is what this retired method put inside the *alpha discovery module* of the [overall pipeline](../../../README.md#alpha-discovery-vs-alpha-enhancement). The search space, the simulation and the ledger are shared by every method and drawn the same way in each graph.

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
attractive was inert, and the lane was quietly just random search over a handful of fixed shapes.

It is kept partly as a reminder that a declared constraint which no code reads is not a
constraint, and that the way to find out is to check the execution path rather than the design
document.

