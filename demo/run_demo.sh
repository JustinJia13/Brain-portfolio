#!/usr/bin/env bash
# Offline walkthrough of the discovery and enhancement engines.
# Needs only `pip install -r requirements.txt`: no account, no API key, no network.
#
#   bash demo/run_demo.sh                          # from the repo root, venv active
#   PYTHON=/path/to/python bash demo/run_demo.sh   # or point it at an interpreter
#
# Outputs go to demo_out/ (gitignored), which is cleared at the start of each run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
OUT=demo_out
SEEDS=demo/enhancement_seeds.synthetic.jsonl
rm -rf "$OUT" && mkdir -p "$OUT"

step() {  # step <title> <description line>...
  printf '\n==== %s\n' "$1"; shift
  for line in "$@"; do printf '     %s\n' "$line"; done
  echo
}

step "1/8  Grammar method: sample candidate alphas" \
  "Random derivation from the bounded grammar, the baseline every other lane is measured against."
"$PY" demo/sample_grammar.py --budget 4 6 --n 4

step "2/8  Probe method: plan cheap probes" \
  "Simple templates x field clusters, to learn which fields carry signal before spending on complex alphas."
"$PY" -m alpha_discovery.probe_method.build_plan \
  --clusters alpha_discovery/grammar_method/artifacts/all_clusters.json --fields-csv all_fields.csv \
  --pack core --cluster-limit 40 --out "$OUT/probe_plan.jsonl"

step "3/8  Probe method: read measured results" \
  "Buckets 300 real simulation results from samples/ and keeps the seeds worth following up."
"$PY" -m alpha_discovery.probe_method.analyze_results --sim-log samples/sim_master.sample.jsonl \
  --out-summary "$OUT/probe_summary.json" --out-seeds "$OUT/probe_seeds.jsonl"

step "4/8  Template sweep: expand one template" \
  "One hand-written template, expanded over its parameter grid."
"$PY" -m alpha_discovery.template_sweep_method.pipeline build --output "$OUT/sweep_plan.jsonl"

step "5/8  Robustness check: one alpha across settings" \
  "Re-tests an alpha with one simulation setting changed at a time."
"$PY" -m tools.regime_check build --expression "group_rank(ts_rank(cashflow_fin, 200), market)" \
  --mode one-axis --output "$OUT/regime_plan.jsonl" --sim-log "$OUT/regime_sim.jsonl"

step "6/8  Sim-slot gate: unit tests" \
  "The pre-simulation filter that skips candidates predicted to be redundant or weak."
for t in parse channels corr_model surrogate; do
  "$PY" -m "alpha_discovery.sim_slot_gate.tests.test_$t"
done

step "7/8  Enhancement engine: route hand-picked alphas (dry run)" \
  "Each alpha goes to negate (flip the sign), repair (fix one failing check) or enhance (improve a passing alpha)." \
  "Input: $SEEDS, six synthetic records (not real results) built so that every route fires."
"$PY" -m alpha_enhancement.workshop.alpha --sim-log-input "$SEEDS" --dry-run --skip-portfolio-pnl \
  --sim-log "$OUT/workshop_sim.jsonl" --out-dir "$OUT/workshop"

step "8/8  Semantic workshop: one full loop with stand-in agents" \
  "Seed -> semantic template -> new candidates -> simulation -> A/B summary, on the same synthetic seeds." \
  "The agents and simulation results are deterministic stand-ins, so this shows the plumbing, not alpha quality."
"$PY" tools/harnesses/probe_semantic_workshop_harness.py --stub-agents --sim-log "$SEEDS" --source all \
  --no-score-seeds --out-root "$OUT/semantic" > "$OUT/semantic.log" 2>&1
RUN=$(ls -d "$OUT"/semantic/*/ | tail -1)
tail -n 1 "$OUT/semantic.log"
echo "Fields of the semantic template card that the LLM fills in (stand-in values here):"
"$PY" - "${RUN}probe_diagnostic/diagnostics.jsonl" <<'PYCODE'
import json, sys
card = json.loads(open(sys.argv[1]).readline())["diagnosis"]
for key in ("signal_identity", "field_roles", "valid_mutation_axes", "invalid_mutations", "recommendation"):
    print(f"  {key}: {card[key]}")
PYCODE
echo "Full A/B summary: ${RUN}summary.txt"

printf '\nDone. Every output is in %s/\n' "$OUT"
