#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/users/zhutianci/proj/disentangled-glue"
EVAL_DIR="$ROOT/eval"
PYTHON_BIN="/data1/users/zhutianci/.conda/envs/scMRDR/bin/python"

SESSION="${SESSION:-bmmc-best-origpp}"
GPU="${GPU:-2}"
N_JOBS="${N_JOBS:-4}"

OUT_ROOT="$EVAL_DIR/outputs/bmmc_trimodal_best_origpp"
RUN_NAME="baseline_t0088_sd48_pd8_bs0.75_li2.0_la0.3_bpr0.25_bpa1.0_bpp1.0"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
BENCH_DIR="$RUN_DIR/benchmarker_eval"
LOG_DIR="$EVAL_DIR/logs"
TRAIN_LOG="$LOG_DIR/bmmc_best_origpp_train.log"
EVAL_LOG="$LOG_DIR/bmmc_best_origpp_eval.log"

mkdir -p "$LOG_DIR" "$OUT_ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

CMD=$(cat <<EOF
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export NUMBA_CACHE_DIR=/tmp/numba-cache
export MPLCONFIGDIR=/tmp/mplconfig
echo "[train] \$(date '+%F %T')" | tee "$TRAIN_LOG"
"$PYTHON_BIN" "$EVAL_DIR/run_best_bmmc_trimodal.py" \
  --gpu "$GPU" \
  --n-jobs "$N_JOBS" \
  --output-dir "$OUT_ROOT" 2>&1 | tee -a "$TRAIN_LOG"
echo "[eval] \$(date '+%F %T')" | tee "$EVAL_LOG"
"$PYTHON_BIN" "$EVAL_DIR/eval_bmmc_like_muto_yao.py" \
  --run-dir "$RUN_DIR" \
  --feature-aligned "$ROOT/experiments/BMMC/s01_preprocessing/feature_aligned_sampled.h5ad" \
  --output-dir "$BENCH_DIR" \
  --tag "$RUN_NAME" \
  --n-jobs "$N_JOBS" 2>&1 | tee -a "$EVAL_LOG"
echo "[done] \$(date '+%F %T')" | tee -a "$EVAL_LOG"
EOF
)

tmux new-session -d -s "$SESSION" -n bmmc "$CMD"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Train log: $TRAIN_LOG"
echo "Eval log: $EVAL_LOG"
