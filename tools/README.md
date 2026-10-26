# Tools

This directory contains maintained command wrappers and research harnesses.
Keep command names stable; move reusable implementation into `wq_alpha/`,
`alpha_discovery/grammar_method/`, or `alpha_discovery/probe_method/`.

## Active Pipeline

- `batch_workshop.py` — autoworkshop repair/enhance lane.
- `semantic_workshop.py` — semantic workshop wrapper.
- `semantic_templates.py` — semantic template architect/router/plan CLI.
- `alpha_workshop.py` — one-off alpha workshop.
- `portfolio_admission.py` — portfolio contribution scoring.
- `regime_check.py` — fixed robustness/regime checks.

## Discovery And Probe Diagnostics

- `compare_probe_prior_grammar.py` — raw grammar vs probe-prior grammar comparison.
- `probe_unsubmitted_selfcorr.py` — self-correlation checks for probe outputs.
- `show_results.py` — inspect simulation/workshop logs.
- `full_history_scan.py` — audit historical Brain records.
- `consolidate_logs.py` — merge and dedupe simulation logs.
- `fetch_brain_records.py` — recover/maintain account alpha history from the
  Brain API into GitHub-safe rotated shards under `records/<account>/`
  (`--all` for brain+brain2, `--merge-master` to absorb into master.jsonl).

## Harnesses And Tests

- `test_prompt_hygiene.py` — semantic prompt context harness.
- `test_probe_semantic_workshop.py` — probe-card semantic workshop A/B runner.
- `test_semantic_template_loop.py` — template loop dry-run/live test.
- `test_qtable_pool_delta.py` — Q-table reward regression.
- `test_self_corr_verification.py` — self-correlation/admission regression.
- `test_grammar_novelty.py` — grammar novelty checks.

## Compatibility And Manual Utilities

- `alpha_multi_*.py` — compatibility CLIs for older multi-agent workflows.
- `workshop_by_id.py` — explicit Brain alpha ID workshop.
- `try_negations.py` — manual negation experiments.
- `browser_auth.py` — Brain auth helper.
- `refresh_fields_csv.py` — field catalog refresh.

If a script grows reusable logic, move that logic into an active package and
leave the script as a thin CLI wrapper.
