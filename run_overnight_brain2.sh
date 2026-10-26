#!/usr/bin/env bash
# Perpetual brain2 theory-lane discovery run.
#
# brain2 is the discovery pool: headless auth (no face scan), so this loops
# forever and is freely restartable. Everything is persisted per round
# (registry, master merge, wiki, promotion queue), so killing it loses at most
# the in-progress round. Validated submittable winners queue to
# artifacts/theory/promote_to_brain.jsonl for later re-sim on brain.
#
# Usage:
#   ./run_overnight_brain2.sh            # run in this terminal (Ctrl-C to stop)
#   nohup ./run_overnight_brain2.sh &    # detach; survives logout
#
# Tunables (env overrides): N_THEORIES, MAX_SIM, BATCH_SIZE
set -u

PY="$HOME"/miniconda3/envs/worldquant/bin/python
CRED=brain2.txt

# Stream output live: through a pipe (tee) Python block-buffers stdout, hiding
# progress. Force unbuffered.
export PYTHONUNBUFFERED=1

# brain2 and brain discovery may run SIMULTANEOUSLY: the theory lane serializes
# only its brief per-round state commit (registry/master/wiki) via an internal
# fcntl lock, never the simulation. So no whole-run lock here.
N_THEORIES=${N_THEORIES:-10}    # cards proposed per round
MAX_SIM=${MAX_SIM:-3}           # theories simulated per round (Thompson-ranked)
BATCH_SIZE=${BATCH_SIZE:-3}     # platform caps concurrent sims at 3/account;
                                # scale throughput by running 2 accounts, not by raising this
LOG=artifacts/theory/overnight_brain2.log

cd "$(dirname "$0")"
echo "=== brain2 perpetual theory lane — started $(date) ===" | tee -a "$LOG"

# --rounds 0 loops forever in one process (one sign-in, headless re-auth on
# expiry). The outer while-loop is belt-and-suspenders: if the process dies
# (crash, dropped connection), it restarts after a short pause. brain2 re-auth
# is headless, so restarts need no human.
while true; do
  "$PY" tools/theory_lane.py \
    --rounds 0 \
    --n-theories "$N_THEORIES" \
    --max-simulated "$MAX_SIM" \
    --batch-size "$BATCH_SIZE" \
    --credentials "$CRED" 2>&1 | tee -a "$LOG"
  echo "=== process exited ($(date)); restarting in 30s — Ctrl-C now to stop ===" | tee -a "$LOG"
  sleep 30
done
