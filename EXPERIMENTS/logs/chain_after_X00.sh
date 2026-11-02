#!/bin/bash
# waits for the X00 seeds 3-9 queue to finish, then runs the presentation job list with 14 workers
cd "C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137"
source ML/experiments/env.sh
until grep -q QUEUE_DONE EXPERIMENTS/logs/queue_X00_seeds3-9.txt 2>/dev/null; do sleep 60; done
$PY EXPERIMENTS/code/run_queue.py --subjects YES --jobs-file EXPERIMENTS/logs/jobs_presentation_2026-10-24.json --workers 13
