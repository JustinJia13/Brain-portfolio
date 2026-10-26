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

The discovery lanes submit to the BRAIN REST API and consume a metered simulation
budget, so an end-to-end run needs a live account. The parts that are inspectable
without one:

- `alpha_discovery/grammar_method/` — grammar derivation and the Q-table pruner are pure functions over
  the field catalog in `all_fields.csv`.
- `alpha_discovery/sim_slot_gate/` — the sim-slot gate's design and calibrated parameters
  (`metric_params.json`). The fitted surrogate itself is a 12MB binary and is not committed.
- `core/portfolio/admission.py` — has a PnL-only local scoring path that needs no account.
- `samples/` — truncated real output from each lane, as evidence of what a run produces.
