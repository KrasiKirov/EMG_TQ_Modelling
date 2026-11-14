#!/usr/bin/env bash
# Run the window study (EXPERIMENTS/configs/W*.json) one setting at a time, in priority order:
#   1. the reference (both directions) and the total-torque reference, 10 seeds;
#   2. every other setting with seeds 0-4, cheapest first within each block, each followed by its
#      total-torque version (<id>_tot);
#   3. carried state at five chunk lengths, seeds 0-4 (the slowest runs, so last).
# Finished runs are skipped, so the script can be stopped and started again at any time.
#
#   scripts/run_window_study.sh                          # the whole remaining study
#   scripts/run_window_study.sh WO4_00:0-4               # one setting, one seed block (session 1 -> 2)
#   scripts/run_window_study.sh W00_reference:0-9:S2toS1 # the reverse direction
#
# WORKERS (default 6) sets how many runs train at once (one core each). The Mac is kept awake while it runs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-.venv/bin/python}
WORKERS=${WORKERS:-6}
SUBJECTS="YES JM EG IES CH HM"
grid() { for o in 00 50 75 90; do printf 'WG_%s_o%s ' "$1" "$o"; done; }
PHASE1="W00_reference:0-9 W00_reference:0-9:S2toS1 WT1_total:0-9"
FIVE="WO4_00 WO3_50 WO2_75 WO1_90 WL1_0500ms $(grid 0500ms) WL2_1s $(grid 1s) WS1_shuffled WL3_2s $(grid 2s) WL4_5s $(grid 5s) WL2c_1s_10ms WL5_10s $(grid 10s)"
CARRY="WC1_carry WC2_carry_500ms WC3_carry_1s WC4_carry_2s WC5_carry_5s"   # slowest per run: last, 5 seeds

steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
  for s in $PHASE1; do steps+=("$s"); done
  for e in $FIVE; do steps+=("$e:0-4" "${e}_tot:0-4"); done
  for e in $CARRY; do steps+=("$e:0-4"); done
fi

for step in "${steps[@]}"; do
  IFS=: read -r exp range direction <<< "$step"
  direction=${direction:-S1toS2}
  seeds=$(seq "${range%-*}" "${range#*-}" | tr '\n' ' ')
  echo "START $exp seeds $range $direction $(date '+%F %T')"
  caffeinate -i env TF_CPP_MIN_LOG_LEVEL=3 "$PY" -m EXPERIMENTS.pipeline.run_queue --exps "$exp" \
    --subjects $SUBJECTS --directions "$direction" --seeds $seeds --workers "$WORKERS"
  echo "SETTING_DONE $exp seeds $range $direction $(date '+%F %T')"
done
