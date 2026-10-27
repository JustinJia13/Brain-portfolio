# `alpha_discovery/` — alpha generation

The generation half of the pipeline: build candidate expressions from the field catalog and
decide which ones are worth simulating. The other half,
[`../alpha_enhancement/`](../alpha_enhancement/README.md), never generates — it only improves
alphas that already have a backtest. See the [root README](../README.md) for the competition
format and the overall problem.

## The baseline everything is measured against

`grammar_method/` is the reference implementation, built on the *AlphaCFG* idea of Yang, Hao,
Wang, Shi and Li (*Alpha Discovery via Grammar-Guided Learning and Search*): treat an alpha as
a derivation tree from a context-free grammar that can only produce valid expressions, and
bound the space with a per-rule `Δk` cost against a maximum length `K`.

Sampled **uniformly at random**, that bounded grammar is the baseline. It is a strong one —
valid-by-construction sampling over a size-controlled space beats unstructured search outright
— and every other method here has to earn its keep against it. Mostly they have not.

The paper's own next step, a learned MCTS policy over the grammar, is deliberately **not**
implemented: a rollout here is a metered simulation, so the project substitutes cheap persistent
pruning for learned search. Details and the archived bandit attempt are in
[`grammar_method/README.md`](grammar_method/README.md).

## The designs

| Method | The assumption it makes |
|---|---|
| [`grammar_method/`](grammar_method/README.md) | The useful expression *shapes* are unknown — derive them from the grammar and spend the effort on refusing bad draws. **The baseline, and the productive generator.** |
| [`probe_method/`](probe_method/README.md) | Signal is cluster-structured. Spend one cheap simulation per field cluster on a fixed operator pack and use the result as a prior over where to sample. |
| [`theory_method/`](theory_method/README.md) | A pre-registered economic mechanism can point at the right subspace — provided a deterministic enumerator, not the LLM, renders the expressions. |
| [`template_sweep_method/`](template_sweep_method/README.md) | The cheapest place to find a new alpha is next to one that already works: vary a proven alpha's windows and simulation settings, never its fields. Demoted for overfitting; its regime check survives. |
| [`sim_slot_gate/`](sim_slot_gate/README.md) | A pre-screen. Predict from structure whether a candidate is redundant or likely weak, and decline to simulate it. |
| [`legacy/`](legacy/README.md) | Three retired designs, archived with the reasoning that killed each. |

## How they compose

```mermaid
flowchart TB
    SPACE[("search space<br/>7642 fields times 70 operators")]

    subgraph GEN["ALPHA DISCOVERY — the methods in this folder"]
        direction TB
        PROBE["probe method<br/>test one field per cluster"]
        GRAM["grammar method<br/>the baseline: sample a bounded grammar"]
        THEORY["theory method<br/>test an economic hypothesis"]
        SWEEP["template sweep method<br/>vary the settings of a proven alpha"]
        SCREEN{"cheap pre-screen<br/>refuse before paying for a simulation"}
    end

    SIMBUS["BRAIN simulation<br/>the metered shared resource"]
    LEDGER[("simulation ledger<br/>every alpha with its metrics and its failed checks")]

    subgraph SIDE["REINFORCEMENT — results fed back into discovery"]
        direction TB
        POST[("posterior over where signal is and is not")]
    end

    DROP[("discarded")]

    SPACE --> PROBE
    SPACE --> GRAM
    SPACE --> THEORY
    PROBE -->|"one test per cluster"| SCREEN
    GRAM -->|"candidates"| SCREEN
    THEORY -->|"designed families"| SCREEN
    SWEEP -->|"variants of proven alphas"| SCREEN
    SCREEN -->|"predicted not worth it"| DROP
    SCREEN -->|"survivor"| SIMBUS
    SIMBUS -->|"metrics and named failing checks"| LEDGER
    LEDGER -->|"alphas that already work"| SWEEP
    LEDGER -->|"every result, win or failure"| POST
    POST -.->|"sampling prior"| GRAM
```

This is the *alpha discovery module* box of the
[overall pipeline](../README.md#alpha-discovery-vs-alpha-enhancement), opened up. The search
space, the simulation, the ledger and the reinforcement side are the same boxes in the same words,
and each method's own README opens up its box one level further.

The structural claim is **several generators behind one screen**. Adding a generator is cheap,
and individually none of them matters much; what matters is that the simulation budget flows
through a refusal step before it is spent, and that results come back as a prior rather than
being thrown away. Each method has its own cheap refusal — the grammar method's is the most
developed, and is where [`sim_slot_gate/`](sim_slot_gate/README.md) plugs in.

## Shared machinery

These methods deliberately do not own what they have in common. The FASTEXPR operator registry,
the expression and field models, the Q-table, self-correlation, the cluster catalog and the
submittable predicate all live in [`../core/`](../core/README.md), so no method imports from
another. That layering is verified: `alpha_enhancement → alpha_discovery` import edges are zero,
and `core` imports nothing from either pillar.
