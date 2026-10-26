#!/usr/bin/env bash
# Re-simulate + robustness-gate the diverse brain2 candidates (records/brain2_resim.txt)
# on the headless brain2 account. finalist_gate simulates a designed perturbation set
# (window neighbours + neutralization variants) per expression and cards each
# pass/weak/fail — IS peaks that collapse under perturbation (would decay OS) get "fail".
# Already-logged variants are skipped (free historical evidence). No submission here.
#   ./run_resim_brain2.sh              # dry-run (plan only, no sims)
#   ./run_resim_brain2.sh go           # actually simulate on brain2
set -u
cd "$(dirname "$0")"
PY="$HOME"/miniconda3/envs/worldquant/bin/python
EXPR=records/brain2_resim.txt
DRY="--dry-run"; [ "${1:-}" = "go" ] && DRY=""
"$PY" tools/finalist_gate.py --expressions "$EXPR" --credentials brain2.txt \
      --max-sims-per-finalist 6 --batch-size 3 $DRY 2>&1 | tee -a artifacts/workshop/resim_brain2.log
