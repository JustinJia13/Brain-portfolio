#!/usr/bin/env bash
# Settings-repair over ALL strong-but-not-submittable alphas (no cap).
#
# Per alpha: diagnose the binding constraint (LLM), apply targeted sim-setting
# fixes (decay for turnover, neutralization/truncation for concentration),
# re-sim, keep submittable+robust, then a LOCAL score-addition gate (self-corr +
# delta_pool_sharpe vs the brain portfolio on disk — no brain account needed)
# queues only likely-additive survivors. promote_to_brain then gets the official
# Brain score-addition on those survivors.
#
# Account is yours to choose (PnL is account-independent, so the local gate works
# regardless of which account ran the sims). brain2 is headless/perpetual and
# best for the bulk re-sim; brain triggers the face scan and is session-limited.
#
# On 'brain.txt' the OFFICIAL Brain score-addition is fetched per fix; on any
# other account the local score-add proxy (corr + delta_pool_sharpe) is used and
# survivors are queued for promote_to_brain.
#
# Usage:
#   ./run_repair.sh                          # brain2.txt, ALL alphas, 1 round
#   ./run_repair.sh brain.txt                # run sims on brain1, fetch official score
#   ./run_repair.sh brain2.txt theory        # only recent theory-lane discoveries
#   ./run_repair.sh brain2.txt all 200 5     # cap 200/round, 5 rounds
#   ./run_repair.sh brain2.txt theory 0 0    # no cap, perpetual rounds
set -u

PY="$HOME"/miniconda3/envs/worldquant/bin/python
CRED=${1:-brain2.txt}
SCOPE=${2:-all}
MAX=${3:-0}                       # 0 = no cap (every strong-not-submittable alpha) per round
ROUNDS=${4:-1}                    # repair passes; <=0 = until a round finds no targets
LOG=artifacts/workshop/repair_run.log

export PYTHONUNBUFFERED=1         # tee block-buffers Python stdout otherwise
cd "$(dirname "$0")"
mkdir -p artifacts/workshop

SIMLOG=()
if [ "$SCOPE" = "theory" ]; then
  SIMLOG=(--sim-log artifacts/sim/sessions/theory_*.jsonl)
fi

echo "=== settings-repair (account=$CRED scope=$SCOPE max=$MAX) — $(date) ===" | tee -a "$LOG"
"$PY" tools/settings_repair.py \
  --max-alphas "$MAX" \
  --rounds "$ROUNDS" \
  --credentials "$CRED" \
  "${SIMLOG[@]}" 2>&1 | tee -a "$LOG"
