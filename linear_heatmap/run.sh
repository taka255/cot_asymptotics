#!/usr/bin/env bash
set -euo pipefail

OUTDIR="/home/takanami/cot_asymptotics/linear_heatmap/results/latest_linear_heatmap2"
LOG="/home/takanami/cot_asymptotics/linear_heatmap/run.log2"

nohup python "/home/takanami/cot_asymptotics/linear_heatmap/linear_heatmap.py" \
  --outdir "$OUTDIR" > "$LOG" 2>&1 &
echo "PID=$!"
echo "OUTDIR=$OUTDIR"
echo "LOG=$LOG"
