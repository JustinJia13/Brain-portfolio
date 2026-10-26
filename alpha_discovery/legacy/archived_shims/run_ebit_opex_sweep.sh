#!/usr/bin/env bash
# Sweep runner for ebit_opex_cov_size_capx_v1
# Usage:
#   ./run_ebit_opex_sweep.sh              # batch-size 3
#   ./run_ebit_opex_sweep.sh 5            # batch-size 5
#   ./run_ebit_opex_sweep.sh 3 50         # batch-size 3, cap at 50 candidates
set -euo pipefail
cd "$(dirname "$0")"

PLAN=alpha_search/artifacts/ebit_opex_sweep.jsonl
SIM_LOG=alpha_search/artifacts/ebit_opex_sim.jsonl
CREDS=brain.txt
BATCH_SIZE=${1:-3}
MAX_CANDIDATES=${2:-0}

echo "=== [1/2] Building candidate plan ==="
python -m alpha_search.sweep --output "$PLAN"

echo ""
echo "=== [2/2] Submitting to Brain ==="
echo "    plan:           $PLAN"
echo "    sim-log:        $SIM_LOG"
echo "    batch-size:     $BATCH_SIZE"
echo "    max-candidates: ${MAX_CANDIDATES} (0 = no cap)"
echo ""

python -m alpha_search.submit \
    --plan "$PLAN" \
    --sim-log "$SIM_LOG" \
    --credentials "$CREDS" \
    --batch-size "$BATCH_SIZE" \
    --max-candidates "$MAX_CANDIDATES"

echo ""
echo "Done. Results in $SIM_LOG"
