#!/usr/bin/env bash
# brain (competition account) runner — INTERACTIVE, BOUNDED.
#
# brain.txt requires the Persona face scan at sign-in, and a session-expiry
# re-auth re-triggers it. So this must run in a real terminal (NOT nohup /
# detached), and stay bounded enough to finish inside one session (~a few
# hours). No restart loop — a restart would mean another face scan.
#
# Two modes:
#   ./run_brain.sh promote      # re-sim brain2 discovery winners on brain
#                               # (applies self-correlation + score-addition gates)
#   ./run_brain.sh discover [N] # run the theory lane directly on brain for N
#                               # rounds (default 3). Usually unnecessary — prefer
#                               # promote, since brain2 already found the signal.
set -u

PY="$HOME"/miniconda3/envs/worldquant/bin/python
CRED=brain.txt
MODE=${1:-promote}
ROUNDS=${2:-3}                  # discover mode only; keep small (one session)
LOG=artifacts/theory/brain_run.log

cd "$(dirname "$0")"

# Stream output live (and the face-scan URL) — tee would otherwise let Python
# block-buffer stdout and you'd never see the biometric link.
export PYTHONUNBUFFERED=1

if [ ! -t 0 ]; then
  echo "ERROR: brain needs an interactive terminal for the face scan." >&2
  echo "Run this directly (no nohup, no pipe into it)." >&2
  exit 1
fi

# May run concurrently with the brain2 perpetual run — the theory lane's
# internal fcntl lock serializes only the brief per-round state commit
# (shared registry/master/wiki), not the simulation.

case "$MODE" in
  promote)
    echo "=== brain promote (re-sim brain2 winners) — $(date) ===" | tee -a "$LOG"
    "$PY" tools/promote_to_brain.py --credentials "$CRED" 2>&1 | tee -a "$LOG"
    ;;
  discover)
    echo "=== brain discover — $ROUNDS rounds — $(date) ===" | tee -a "$LOG"
    "$PY" tools/theory_lane.py \
      --rounds "$ROUNDS" \
      --n-theories 10 \
      --max-simulated 3 \
      --batch-size 3 \
      --credentials "$CRED" 2>&1 | tee -a "$LOG"
    ;;
  *)
    echo "usage: $0 {promote|discover [rounds]}" >&2
    exit 2
    ;;
esac
