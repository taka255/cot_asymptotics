#!/usr/bin/env bash
set -euo pipefail

nohup python "/home/takanami/cot_asymptotics/linear_dynamics/linear_dynamics.py" \
  > "/home/takanami/cot_asymptotics/linear_dynamics/run.log" 2>&1 &
echo "PID=$!"
echo "LOG=/home/takanami/cot_asymptotics/linear_dynamics/run.log"
