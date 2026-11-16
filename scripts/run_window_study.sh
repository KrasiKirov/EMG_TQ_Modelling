#!/usr/bin/env bash
# Run the window study (EXPERIMENTS/configs/W*.json) one setting at a time, in priority order, so the study can be
# cut at any point and the most important results are already in:
#   1. the reference (both directions) and the total-torque reference, 10 seeds;
#   2. overlaps at 200 ms, then the full overlap x length grid, 5 seeds, each followed by its total-torque version;
#   3. 500 ms, 1 s and the shuffled comparison (with total torque), 2 s (active only), 5 seeds;
#   4. carried state at two chunk lengths (500 ms, 1 s), then the long windows at full overlap (5 s, 1 s at 10 ms, 10 s):
#      seeds 0-2 first, active torque only, because one run costs 5-16 reference runs;
#   5. the total-torque reference with a 400-epoch limit, 2 seeds (epoch-limit check for total torque);
#   6. seeds 3-4 of the carried-state and long-window settings, so every setting has 5 seeds.
# Finished runs are skipped, so the script can be stopped and started again, or given more seeds later.
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
both() { for e in "$@"; do printf '%s:0-4 %s_tot:0-4 ' "$e" "$e"; done; }
grid() { for L in 0500ms 1s 2s 5s 10s; do for o in 00 50 75 90; do both "WG_${L}_o${o}"; done; done; }
DEFAULT="W00_reference:0-9 W00_reference:0-9:S2toS1 WT1_total:0-9
 $(both WO4_00 WO3_50 WO2_75 WO1_90)
 $(grid)
 $(both WL1_0500ms WL2_1s WS1_shuffled) WL3_2s:0-4
 WC2_carry_500ms:0-2 WC3_carry_1s:0-2
 WL4_5s:0-2 WL2c_1s_10ms:0-2 WL5_10s:0-2
 WT1_total_400:0-1
 WC2_carry_500ms:3-4 WC3_carry_1s:3-4 WL4_5s:3-4 WL2c_1s_10ms:3-4 WL5_10s:3-4"

steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
  for s in $DEFAULT; do steps+=("$s"); done
fi
if [ "${LIST_ONLY:-}" = 1 ]; then printf '%s\n' "${steps[@]}"; exit 0; fi

for step in "${steps[@]}"; do
  IFS=: read -r exp range direction <<< "$step"
  direction=${direction:-S1toS2}
  seeds=$(seq "${range%-*}" "${range#*-}" | tr '\n' ' ')
  echo "START $exp seeds $range $direction $(date '+%F %T')"
  caffeinate -i env TF_CPP_MIN_LOG_LEVEL=3 "$PY" -m EXPERIMENTS.pipeline.run_queue --exps "$exp" \
    --subjects $SUBJECTS --directions "$direction" --seeds $seeds --workers "$WORKERS"
  echo "SETTING_DONE $exp seeds $range $direction $(date '+%F %T')"
done
