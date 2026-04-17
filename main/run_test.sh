#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$BASE_DIR/log_${TS}.txt"

# CPUを使い切る設定（必要なら手動で値を下げてください）
CPU_THREADS="$(nproc)"

nohup env \
  OMP_NUM_THREADS="$CPU_THREADS" \
  MKL_NUM_THREADS="$CPU_THREADS" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS" \
  PYTHONUNBUFFERED=1 \
  python "$BASE_DIR/exp_fixed_D_vary_NM_dynamics.py" > "$LOGFILE" 2>&1 &

echo "PID=$!"
echo "LOG=$LOGFILE"
echo "THREADS=$CPU_THREADS"
echo "tail -f \"$LOGFILE\""
