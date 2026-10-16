#!/usr/bin/env bash
# Full experiment campaign (all GPUs share one job queue; see
# experiments/run_campaign.py).  Runs in the background: closing the terminal
# or Ctrl+C does not stop it.
#
#   bash run_experiments.sh          # stop old runs, run tests, start (or resume)
#   bash run_experiments.sh status   # progress + ETA
#   bash run_experiments.sh stop     # stop everything (finished chunks are kept)
#
# Results: experiments/results_v2  (override: MPR_RESULTS=/path bash run_experiments.sh)
# Restarting resumes: finished chunks are skipped, interrupted ones rerun.
cd "$(dirname "$0")"
export MPR_RESULTS="${MPR_RESULTS:-$PWD/experiments/results_v2}"
JOBS='python[0-9.]* (-u )?e(1b|1|2|3|4|5|6|7|8|9|10|11|12)_[a-z]+\.py'

stop_all() {
  pkill -f 'run_campaign\.py' 2>/dev/null
  pkill -f -- "$JOBS" 2>/dev/null
  sleep 3
  pkill -9 -f -- "$JOBS" 2>/dev/null
}

case "${1:-start}" in
  stop)
    stop_all; echo "stopped."; exit 0 ;;
  status)
    cat "$MPR_RESULTS/status.txt" 2>/dev/null || echo "no status yet"
    echo; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    exit 0 ;;
  start) ;;
  *) echo "usage: bash run_experiments.sh [start|status|stop]"; exit 1 ;;
esac

if pgrep -f 'run_campaign\.py' >/dev/null; then
  echo "already running -> bash run_experiments.sh status"; exit 1
fi
if pgrep -f -- "$JOBS" >/dev/null; then
  echo "stopping running experiment jobs:"; pgrep -af -- "$JOBS"
  stop_all
fi
set -e
python tests/test_core.py
python tests/test_access.py
python tests/test_brisc.py
python tests/test_dedup.py
mkdir -p "$MPR_RESULTS"
cd experiments
nohup python -u run_campaign.py "${@:2}" > "$MPR_RESULTS/scheduler.log" 2>&1 &
sleep 8
head -4 "$MPR_RESULTS/scheduler.log"
echo
echo "running in the background. Progress:"
echo "  bash run_experiments.sh status"
echo "  tail -f $MPR_RESULTS/scheduler.log"
