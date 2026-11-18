#!/usr/bin/env bash
# Study v2 runs (EXPERIMENTS/protocol/study_v2.json). Finished runs are skipped, so every step can be stopped and
# started again. Results go to EXPERIMENTS/results_v2/ (the TS2_/FE_ ids).
#
#   scripts/run_study_v2.sh round R1     # coarse: 6 shared random draws + the anchor per active cell, seed 0
#   scripts/run_study_v2.sh budget       # after 24 h of R1 and when R1 ends: run-time projection, fallback steps
#                                        # (results_v2/budget.json, committed); exits 1 if still over 6 days
#   scripts/run_study_v2.sh screen 2s    # D23 gate: R1 of the 2 s window at full overlap and the gate decision
#                                        # (6-day budget: only 2 s is screened, a pass is not tuned)
#   scripts/run_study_v2.sh round R2     # fine: 4 draws (2 after fine_2) around each cell's best R1 config, seed 0
#   scripts/run_study_v2.sh round R3     # seed check: the best 2 of R1 + R2 per cell get seed 1, then the winner
#   Every round trains to early stopping under the 400-epoch cap.
#   scripts/run_study_v2.sh final        # final runs from results_v2/choice.json (EXPERIMENTS/tuning/finalize.py)
#
# Order (plan § Compute): R1 of every always-run cell (75 % overlap last), budget check after 24 h, budget check
# when R1 ends, the 2 s screen, R2, R3, final. Study runs refuse uncommitted code. WORKERS (default 6) runs train at once, 1 thread each.
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
    budget_ok=$($PY -c "from EXPERIMENTS.protocol.protocol import load; from EXPERIMENTS.tuning import budget as B; b = load()['budget']; print('$C' in b['screens'] and 'skip_2s_check' not in B.applied())")
    [ "$budget_ok" = True ] || { echo "screen $C is not run under the budget (protocol budget.screens, results_v2/budget.json)" >&2; exit 2; }
    ids=$($PY -m EXPERIMENTS.tuning.make_specs --round R1 --cell "$C")
    STEP="screen $C"; commit_specs $ids
    queue --exps $ids --seeds $(seeds R1)
    $PY -m EXPERIMENTS.tuning.select --gate "$C" --json "EXPERIMENTS/results_v2/gate_$C.json"
    # a passed length gets its other 3 overlap cells, R1 first (then R2 and R3 with every other cell)
    tune=$($PY -c "from EXPERIMENTS.protocol.protocol import load; print(load()['budget']['tune_gated'])")
    if [ "$tune" = True ] && [ "$($PY -c "import json; print(json.load(open('EXPERIMENTS/results_v2/gate_$C.json'))['continues'])")" = True ]; then
      W=${C%_full}
      ids=$($PY -m EXPERIMENTS.tuning.make_specs --round R1 --cell "${W}_o75" "${W}_o50" "${W}_o0")
      STEP="R1 $W overlaps"; commit_specs $ids
      queue --exps $ids --seeds $(seeds R1)
    fi ;;
  budget)
    rc=0; $PY -m EXPERIMENTS.tuning.budget --check || rc=$?
    STEP="budget check"; git add -- EXPERIMENTS/results_v2/budget.json
    git diff --cached --quiet -- EXPERIMENTS/results_v2/budget.json || git commit -q -m "Study v2 budget check" -- EXPERIMENTS/results_v2/budget.json
    exit $rc ;;
  final)
    $PY -m EXPERIMENTS.tuning.finalize --jobs EXPERIMENTS/results_v2/final_jobs.json
    STEP="final"; commit_specs $($PY -c "import json; print(' '.join(sorted({j[0] for j in json.load(open('EXPERIMENTS/results_v2/final_jobs.json'))})))")
    queue --jobs-file EXPERIMENTS/results_v2/final_jobs.json ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
