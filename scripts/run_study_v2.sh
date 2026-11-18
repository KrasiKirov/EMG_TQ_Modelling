#!/usr/bin/env bash
# Study v2 runs (EXPERIMENTS/protocol/study_v2.json). Finished runs are skipped, so every step can be stopped and
# started again. Results go to EXPERIMENTS/results_v2/ (the TS2_/FE_ ids).
#
#   scripts/run_study_v2.sh round R1     # R1 of every active cell (9 configs, seed 0, 60 epochs)
#   scripts/run_study_v2.sh screen 2s    # D23 gate: R1 of the 2 s window at full overlap, then the gate decision
#   scripts/run_study_v2.sh round R2     # the 3 configs R1 kept per cell, 400 epochs
#   scripts/run_study_v2.sh round R3     # the 2 configs R2 kept per cell, seed 1 added
#   scripts/run_study_v2.sh final        # final runs from results_v2/choice.json (EXPERIMENTS/tuning/finalize.py)
#
# Order (plan § Compute): R1 of every always-run cell, the 2 s screen, R2, R3; 5 s and 10 s screens only if the
# shorter length passed. Study runs refuse uncommitted code. WORKERS (default 6) runs train at once, 1 thread each.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-.venv/bin/python}
WORKERS=${WORKERS:-6}
SUBJECTS=$($PY -c "from EXPERIMENTS.protocol.protocol import load; print(' '.join(load()['people']))")
queue() { caffeinate -dimsu "$PY" -m EXPERIMENTS.pipeline.run_queue --subjects $SUBJECTS --workers "$WORKERS" "$@"; }
# Study runs refuse untracked or edited specs, so the specs a step writes are committed (only those files) first.
commit_specs() {
  local paths=(); for i in "$@"; do paths+=("EXPERIMENTS/configs/$i.json"); done
  [ ${#paths[@]} -eq 0 ] && return
  git add -- "${paths[@]}"
  git diff --cached --quiet -- "${paths[@]}" || git commit -q -m "Study v2 specs: $STEP" -- "${paths[@]}"
}
seeds() { $PY -c "from EXPERIMENTS.protocol.protocol import load; print(' '.join(map(str, load()['rounds']['$1']['seeds'])))"; }

case "${1:-}" in
  round)
    R=${2:?round R1, R2 or R3}
    ids=$($PY -m EXPERIMENTS.tuning.make_specs --round "$R")
    STEP="round $R"; commit_specs $ids
    queue --exps $ids --seeds $(seeds "$R") ;;
  screen)
    case "${2:-}" in 2s) C=w200_full ;; 5s) C=w500_full ;; 10s) C=w1000_full ;; *) echo "screen 2s|5s|10s" >&2; exit 2 ;; esac
    ids=$($PY -m EXPERIMENTS.tuning.make_specs --round R1 --cell "$C")
    STEP="screen $C"; commit_specs $ids
    queue --exps $ids --seeds $(seeds R1)
    $PY -m EXPERIMENTS.tuning.select --gate "$C" --json "EXPERIMENTS/results_v2/gate_$C.json" ;;
  final)
    $PY -m EXPERIMENTS.tuning.finalize --jobs EXPERIMENTS/results_v2/final_jobs.json
    STEP="final"; commit_specs $($PY -c "import json; print(' '.join(sorted({j[0] for j in json.load(open('EXPERIMENTS/results_v2/final_jobs.json'))})))")
    queue --jobs-file EXPERIMENTS/results_v2/final_jobs.json ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
