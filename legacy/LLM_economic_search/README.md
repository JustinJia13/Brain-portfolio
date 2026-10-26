# LLM Economic Search

Plan-first economic hypothesis search for WorldQuant Brain alphas.

This workspace proposes economic theories, critiques them against prior history,
uses an LLM expression generator to produce diverse Brain FASTEXPR candidates,
validates those expressions deterministically, and can submit the resulting
candidate JSONL plans through the economic submitter.

## Quick Start

```bash
python -m LLM_economic_search.cli index
python -m LLM_economic_search.cli portfolio --credentials brain.txt
python -m LLM_economic_search.cli fields --delay 0 --credentials brain.txt
python -m LLM_economic_search.cli fields --delay 1 --credentials brain.txt
python -m LLM_economic_search.cli propose --delay 1 --mock --n 6
python -m LLM_economic_search.cli critique --delay 1 --no-llm-critic
python -m LLM_economic_search.cli generate --delay 1 --stage probe --max-candidates 18 --mock
python -m LLM_economic_search.cli summarize
```

Omit `--mock` and `--no-llm-critic` for the normal LLM-first path.

For a single delay-specific sprint command, use `sprint`. This wraps index,
portfolio context, field catalog refresh, propose, critique, generate, and
optionally simulation plus ingest:

```bash
python -m LLM_economic_search.cli sprint \
  --delay 0 \
  --n 10 \
  --max-candidates-per-theory 12 \
  --simulate \
  --credentials brain.txt \
  --alpha-ids "9qAqWGWV vRJRnlaz XgYg51pa 78X2MvgL 6XqGLgdL omzNmjnn QPl3EKlQ 9q6VEeA9 Gr8ebEYP O0mo6NZq E5bnLEAL pwEbxpN3" \
  --sprint-focus "Delay-0 probe: propose economically distinct, preferably cross-dataset mechanisms that are orthogonal to the submitted portfolio."
```

Omit `--simulate` to stop after writing the combined candidate plan.
If the LLM proposer fails and the sprint uses deterministic fallback theories,
simulation is skipped by default. Add `--allow-fallback-simulation` only when
you intentionally want to spend Brain simulations on the fallback seed set.
The novelty critic is LLM-backed by default; add `--no-llm-critic` only for a
deterministic dry run. The expression generator is LLM-backed by default; add
`--mock` only for local dry runs and tests.
The Anthropic JSON calls default to model `claude-opus-4-7` and
`max_tokens=12000`; override with `--anthropic-model` or
`--anthropic-max-tokens` if needed.

If you already know the exact submitted portfolio IDs, pin the context to those
IDs instead of relying on Brain status filters:

```bash
python -m LLM_economic_search.cli portfolio --credentials brain.txt \
  --alpha-ids "9qAqWGWV vRJRnlaz XgYg51pa 78X2MvgL 6XqGLgdL omzNmjnn QPl3EKlQ 9q6VEeA9 Gr8ebEYP O0mo6NZq E5bnLEAL pwEbxpN3"
```

The generated combined candidate plan is written under:

```text
LLM_economic_search/artifacts/candidate_plans/
```

Each theory is stored under:

```text
LLM_economic_search/artifacts/theories/<theory_id>/
```

The submitted-account portfolio context is stored at:

```text
LLM_economic_search/artifacts/current_portfolio_context.json
```

`propose` and `critique` load that file automatically when it exists, so the LLM
and deterministic critic can avoid economic incentives already represented in
submitted alphas.

Delay is explicit for research commands. `delay 0` uses yesterday's available
data to trade today; `delay 1` uses data from two days ago to trade today.
Generated candidates include `settings_override.delay`, LLM generation request
and validation artifacts are saved in each theory directory, and portfolio
admission continues to score marginal pool contribution against the same-delay
portfolio while checking self-correlation against all submitted alphas.

## Submit Later

Candidate plans should be submitted through the economic submitter, which has no
legacy template abandonment logic:

```bash
python -m LLM_economic_search.cli submit \
  --plan LLM_economic_search/artifacts/candidate_plans/<plan>.jsonl \
  --sim-log LLM_economic_search/artifacts/economic_sim.jsonl \
  --credentials brain.txt \
  --ingest
```

If you submit without `--ingest`, attach results later:

```bash
python -m LLM_economic_search.cli ingest \
  --results-jsonl LLM_economic_search/artifacts/economic_sim.jsonl
```

The economic submitter preserves theory metadata when possible, and ingest also
matches results back to theories by normalized expression and delay.
