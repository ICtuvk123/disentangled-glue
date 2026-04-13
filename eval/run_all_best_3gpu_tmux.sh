#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/users/zhutianci/.conda/envs/scMRDR/bin/python}"

SESSION="${SESSION:-dglue-best-eval}"
GPU_MUTO="${GPU_MUTO:-0}"
GPU_YAO="${GPU_YAO:-1}"
GPU_BMMC="${GPU_BMMC:-2}"
N_JOBS="${N_JOBS:-4}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-20}"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is not installed or not on PATH" >&2
  exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 240 -y 60 -n monitor

tmux send-keys -t "$SESSION:monitor" "bash -lc '
while true; do
  clear
  echo \"=== disentangled-glue best eval monitor ===\"
  echo \"session:   $SESSION\"
  echo \"Muto GPU:  $GPU_MUTO\"
  echo \"Yao GPU:   $GPU_YAO\"
  echo \"BMMC GPU:  $GPU_BMMC\"
  echo \"N_JOBS:    $N_JOBS\"
  echo
  echo \"Recent logs:\"
  for f in \"$LOG_DIR\"/*.log; do
    [[ -f \"\$f\" ]] || continue
    echo \"----- \$(basename \"\$f\") -----\"
    tail -n 8 \"\$f\" || true
    echo
  done
  sleep $MONITOR_INTERVAL
done'" Enter

tmux new-window -t "$SESSION" -n muto
tmux send-keys -t "$SESSION:muto" "cd $REPO_ROOT && CUDA_VISIBLE_DEVICES=$GPU_MUTO NUMBA_CACHE_DIR=\${NUMBA_CACHE_DIR:-/tmp/numba-cache} MPLCONFIGDIR=\${MPLCONFIGDIR:-/tmp/mplconfig} $PYTHON_BIN $SCRIPT_DIR/run_best_muto.py --gpu $GPU_MUTO --n-jobs $N_JOBS 2>&1 | tee $LOG_DIR/muto_best.log; read" Enter

tmux new-window -t "$SESSION" -n yao
tmux send-keys -t "$SESSION:yao" "cd $REPO_ROOT && CUDA_VISIBLE_DEVICES=$GPU_YAO NUMBA_CACHE_DIR=\${NUMBA_CACHE_DIR:-/tmp/numba-cache} MPLCONFIGDIR=\${MPLCONFIGDIR:-/tmp/mplconfig} $PYTHON_BIN $SCRIPT_DIR/run_best_yao.py --gpu $GPU_YAO --n-jobs $N_JOBS 2>&1 | tee $LOG_DIR/yao_best.log; read" Enter

tmux new-window -t "$SESSION" -n bmmc
tmux send-keys -t "$SESSION:bmmc" "cd $REPO_ROOT && CUDA_VISIBLE_DEVICES=$GPU_BMMC NUMBA_CACHE_DIR=\${NUMBA_CACHE_DIR:-/tmp/numba-cache} MPLCONFIGDIR=\${MPLCONFIGDIR:-/tmp/mplconfig} $PYTHON_BIN $SCRIPT_DIR/run_best_bmmc_trimodal.py --gpu $GPU_BMMC --n-jobs $N_JOBS 2>&1 | tee $LOG_DIR/bmmc_best.log; read" Enter

tmux select-window -t "$SESSION:monitor"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Windows:"
echo "  monitor"
echo "  muto"
echo "  yao"
echo "  bmmc"
echo "Logs:"
echo "  $LOG_DIR/muto_best.log"
echo "  $LOG_DIR/yao_best.log"
echo "  $LOG_DIR/bmmc_best.log"
