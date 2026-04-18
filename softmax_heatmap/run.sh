#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$BASE_DIR/results/softmax_heatmap_${TS}"
LOGFILE="$BASE_DIR/log_${TS}.txt"
mkdir -p "$BASE_DIR/results"

nohup bash -lc "cd \"$BASE_DIR\" && python softmax_heatmap.py --outdir \"$OUTDIR\" && OUTDIR=\"$OUTDIR\" jupyter nbconvert --to notebook --execute \"plot.ipynb\" --output \"plot_executed_${TS}.ipynb\" --ExecutePreprocessor.timeout=-1" > "$LOGFILE" 2>&1 &

echo "PID=$!"
echo "OUTDIR=$OUTDIR"
echo "LOG=$LOGFILE"
echo "tail -f \"$LOGFILE\""

