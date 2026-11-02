#!/bin/bash
# 12 learning-curve runs (X00f and D01, 6 subjects, seed 0, learn S1 -> test S2), 3 at a time
cd "C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137"
source ML/experiments/env.sh
for e in X00f_baseline_framework D01_cleanpassive+autogain+equalweight; do for s in YES EG JM IES CH HM; do echo "$e $s"; done; done | \
  xargs -P 3 -L 1 bash -c 'f=EXPERIMENTS/results/learning_curves/$0/$1/S1toS2/seed0.json; [ -f "$f" ] || $PY EXPERIMENTS/code/learning_curves.py --exp $0 --subject $1 > EXPERIMENTS/logs/lc__$0__$1.log 2>&1; echo "done $0 $1"'
echo LC_DONE
