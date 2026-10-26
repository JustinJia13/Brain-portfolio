# `alpha_discovery/legacy/` — retired methods, kept for their reasoning

Three methods that were built out properly, measured against live simulations, and switched off.
They are archived rather than deleted for two reasons: so the approaches are not re-attempted,
and so the simulation budget already spent learning from them is not re-spent.

**Policy:** nothing here is imported by active code, and no new work belongs here.

| Method | The idea | Why it was retired |
|---|---|---|
| [`economic_search_method/`](economic_search_method/README.md) | An LLM proposes a falsifiable economic mechanism *and* writes the FASTEXPR itself; a SQLite novelty index rations simulation slots. | Expressions were valid and certifiably novel but empirically dead. **Novelty and signal turned out to be close to independent** — a novelty index cannot substitute for a quality predictor. |
| [`llm_grammar_control_method/`](llm_grammar_control_method/README.md) | Don't let the LLM write expressions — have it emit a `GrammarConfig` patch, so the grammar guarantees validity and the LLM only steers the search distribution. | Lost to the *same grammar* sampled uniformly. LLM steering concentrated sampling in economically plausible, already-crowded regions. |
| [`operator_skeleton_method/`](operator_skeleton_method/README.md) | Fix the operator skeleton first from a role-constrained registry, then search which fields fill the slots. | No promoted alpha ever reached the ledger — and the role constraints that made the idea attractive were **never read by any code**, so what ran was random cluster pairing. |

## The arc these three describe

Read in order, they are three successively more careful attempts to put an LLM in the generator
seat, and they fail in a way that gets more informative each time:

1. **Let the model write alphas.** Fails — fluent and dead.
2. **Let the model steer a generator that cannot write invalid alphas.** Still fails, which rules
   out "the expressions were malformed" as the explanation.
3. **Constrain the shapes by economic role instead.** Fails, and the post-mortem finds the
   constraint was inert anyway.

The conclusion the active repository is built on: an LLM given no feedback cannot tell which
expressions will make money, and no amount of structural scaffolding around the generation step
fixes that. What *does* work is giving the model feedback — a measured backtest and a named
failing check — which is exactly what
[`../../alpha_enhancement/`](../../alpha_enhancement/README.md) does, and putting the remaining
intelligence into [`../sim_slot_gate/`](../sim_slot_gate/README.md), which predicts quality
instead of asserting it.

## `archived_shims/`

Compatibility shims, one-off scripts, and two components that were **deliberately disabled**:
an alpha-combination optimizer and a proposal regenerator, both switched off on in-sample
selection-bias grounds with the reasoning left in their docstrings. Also an archived UCB bandit
that the Q-table superseded. Details in
[`operator_skeleton_method/README.md`](operator_skeleton_method/README.md).

Two archived scripts here performed work at import time; they are now guarded inert so that
importing them has no side effects.
