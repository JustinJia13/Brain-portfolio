# Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Credentials

Every credential is read from a plaintext file at the repo root. None are committed;
copy the matching `.example` file and fill in your own values.

| File | Contents | Read by |
|---|---|---|
| `brain.txt` | `["username", "password"]` for WorldQuant BRAIN | `core/brain/api.py` |
| `brain2.txt` | second BRAIN account (same format) | `core/brain/api.py` |
| `antropicapi.txt` | Anthropic API key — or set `ANTHROPIC_API_KEY` | `core/llm.py` |
| `openaiapi.txt` | OpenAI API key (optional) | `core/llm.py` |

```bash
cp brain.txt.example brain.txt        # then edit
cp antropicapi.txt.example antropicapi.txt
```

## Offline unit tests

These need no credentials and no network. They are standalone modules, not
`unittest`-discoverable, so run each directly:

```bash
for t in parse channels corr_model surrogate; do python -m alpha_discovery/sim_slot_gate.tests.test_$t; done
```

Expected: `13/13 parse tests passed`, `9/9 channel tests passed`,
`11/11 corr_model tests passed`, `surrogate acceptance checks done`.

They also carry `def test_*` functions, so `pytest alpha_discovery/sim_slot_gate/tests/` works once
pytest is installed from `requirements.txt`.

## What will not run without an account

Every discovery lane submits to a metered external API, so an end-to-end run needs live
credentials and a simulation budget. What is inspectable offline:

```bash
# the sim-slot gate's unit tests - no account, no network
python -m alpha_discovery.sim_slot_gate.tests.test_parse
python -m alpha_discovery.sim_slot_gate.tests.test_channels
python -m alpha_discovery.sim_slot_gate.tests.test_corr_model
python -m alpha_discovery.sim_slot_gate.tests.test_surrogate
```

- `alpha_discovery/grammar_method/` - grammar derivation and the prune datastores are pure
  functions over `all_fields.csv`.
- `alpha_discovery/sim_slot_gate/` - the design doc (`metric.md`) and the calibrated parameters
  (`metric_params.json`). The fitted surrogate itself is a large binary and is not committed.
- `core/portfolio/local_admission.py` - a PnL-only scoring path that needs no account.
- `samples/` - truncated real output from each lane, as evidence of what a run produces.

## A note on generated paths

`artifacts/`, `records/`, `wiki/`, `logs/` and `test_results/` are generated and gitignored, so
paths into them that appear in the per-method READMEs will not exist in a fresh clone. The one
exception committed on purpose is
`alpha_discovery/grammar_method/artifacts/all_clusters.json`, the field-cluster catalog, because
four modules read it and the grammar lane cannot be exercised offline without it.
