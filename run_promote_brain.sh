#!/usr/bin/env bash
# Re-simulate ALL valid brain2 discovery winners on the brain (competition) account
# to fetch the OFFICIAL Brain SCORE-ADDITION (self-correlation + score delta vs your
# competition portfolio). Queue: artifacts/theory/promote_to_brain.jsonl — built from
# fully-submittable (sharpe>=1.25, fitness>=1.0, turnover<0.7, AND concentrated-weight
# + sub-universe checks PASS) + self-corr<0.7, deduped to distinct motifs.
#
#   ./run_promote_brain.sh            # DRY-RUN: print the queue, no sims, no auth
#   ./run_promote_brain.sh go         # re-sim on brain.txt (face-scan), fetch score
#
# Re-sims write to artifacts/theory/promoted_brain_sim.jsonl (+ merged to master);
# review the admission rows there, then submit the score-adding survivors.
# brain.txt is session-limited — re-run 'go' across sessions until the queue drains
# (already-simulated exprs are skipped automatically).
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1          # else tee block-buffers stdout and the log looks "stuck"
PY="$HOME"/miniconda3/envs/worldquant/bin/python
DRY="--dry-run"; [ "${1:-}" = "go" ] && DRY=""
"$PY" tools/promote_to_brain.py --credentials brain.txt --competition-id IQC2026S1 \
      --batch-size 3 $DRY 2>&1 | tee -a artifacts/theory/promote_run.log
