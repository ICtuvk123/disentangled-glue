#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/users/zhutianci/.conda/envs/scMRDR/bin/python}"
SESSION="${SESSION:-yao-baseline-hpo}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-0}"
RANDOM_SEED="${RANDOM_SEED:-0}"
N_TRIALS_PER_PRESET="${N_TRIALS_PER_PRESET:-4}"
PRESETS="${PRESETS:-nb_hvg nb_all normal_hvg normal_all}"
LSI_METHODS="${LSI_METHODS:-raw_svd}"
SEARCH_DIR="${SEARCH_DIR:-$SCRIPT_DIR/../hparam_search_v4_yao_baseline_style_fast}"
N_JOBS="${N_JOBS:-10}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"

# Fast HPO defaults. Override via env when needed.
QUICK_SEARCH="${QUICK_SEARCH:-1}"
PRETRAIN_MAX_EPOCHS="${PRETRAIN_MAX_EPOCHS:-60}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-12}"
PRETRAIN_REDUCE_LR_PATIENCE="${PRETRAIN_REDUCE_LR_PATIENCE:-6}"
FINETUNE_MAX_EPOCHS="${FINETUNE_MAX_EPOCHS:-30}"
FINETUNE_PATIENCE="${FINETUNE_PATIENCE:-8}"
FINETUNE_REDUCE_LR_PATIENCE="${FINETUNE_REDUCE_LR_PATIENCE:-4}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed or not on PATH" >&2
    exit 1
fi

mkdir -p "$SEARCH_DIR"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
read -r -a PRESET_LIST <<< "$PRESETS"
read -r -a LSI_LIST <<< "$LSI_METHODS"
N_WORKERS="${#GPU_LIST[@]}"
TOTAL_CONFIGS=$(( ${#PRESET_LIST[@]} * ${#LSI_LIST[@]} * N_TRIALS_PER_PRESET ))

if [[ "$N_WORKERS" -lt 1 ]]; then
    echo "ERROR: no GPUs configured. Set GPUS like 0,1,2,3" >&2
    exit 1
fi

echo "Launching Yao baseline-style HPO"
echo "  session:      $SESSION"
echo "  gpus:         $GPUS"
echo "  workers:      $N_WORKERS"
echo "  presets:      $PRESETS"
echo "  lsi_methods:  $LSI_METHODS"
echo "  trials/preset:$N_TRIALS_PER_PRESET"
echo "  total configs:$TOTAL_CONFIGS"
echo "  search_dir:   $SEARCH_DIR"
echo "  fast budget:  pretrain ${PRETRAIN_MAX_EPOCHS}/${PRETRAIN_PATIENCE}/${PRETRAIN_REDUCE_LR_PATIENCE}, fine-tune ${FINETUNE_MAX_EPOCHS}/${FINETUNE_PATIENCE}/${FINETUNE_REDUCE_LR_PATIENCE}"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 240 -y 60 -n monitor

tmux send-keys -t "$SESSION:monitor" "cd $SCRIPT_DIR && bash -lc '
while true; do
  clear
  echo \"=== Yao baseline-style HPO monitor ===\"
  echo \"search_dir: $SEARCH_DIR\"
  echo \"workers: $N_WORKERS | gpus: $GPUS | total configs: $TOTAL_CONFIGS\"
  echo
  DONE=\$(find \"$SEARCH_DIR\" -path \"*/metrics.json\" 2>/dev/null | wc -l)
  echo \"completed trials: \$DONE / $TOTAL_CONFIGS\"
  echo
  $PYTHON_BIN - <<\"PY\"
import json
from pathlib import Path
import pandas as pd

search_dir = Path(r\"$SEARCH_DIR\")
summary_path = search_dir / \"summary.tsv\"
if summary_path.exists():
    df = pd.read_csv(summary_path, sep=\"\\t\")
else:
    rows = []
    for metrics_path in sorted(search_dir.glob(\"t*/metrics.json\")):
        config_path = metrics_path.parent / \"config.json\"
        if not config_path.exists():
            continue
        rows.append({
            **json.loads(config_path.read_text()),
            **json.loads(metrics_path.read_text()),
        })
    df = pd.DataFrame(rows)

if df.empty:
    print(\"No completed trials yet.\")
else:
    cols = [
        \"trial_id\",
        \"preset\",
        \"feature_space\",
        \"lsi_method\",
        \"Total\",
        \"Bio conservation\",
        \"Batch correction\",
        \"Modality integration\",
        \"shared_dim\",
        \"private_dim\",
        \"batch_embed_dim\",
        \"beta_shared\",
        \"lam_iso\",
        \"lam_align\",
        \"beta_private_rna\",
        \"beta_private_atac\",
        \"dropout\",
        \"lr\",
    ]
    cols = [c for c in cols if c in df.columns]
    if \"Total\" in df.columns:
        df = df.sort_values(\"Total\", ascending=False, na_position=\"last\")
    print(df[cols].head(12).to_string(index=False, float_format=lambda x: f\"{x:.4f}\"))
PY
  echo
  echo \"latest metric files:\"
  find \"$SEARCH_DIR\" -path \"*/metrics.tsv\" -printf \"%TY-%Tm-%Td %TH:%TM:%TS %p\\n\" 2>/dev/null | sort | tail -n 5
  sleep $MONITOR_INTERVAL
done'" Enter

for idx in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$idx]}"
    win="gpu${gpu}"
    tmux new-window -t "$SESSION" -n "$win"
    cmd="cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$gpu NUMBA_CACHE_DIR=\${NUMBA_CACHE_DIR:-/tmp/numba-cache} MPLCONFIGDIR=\${MPLCONFIGDIR:-/tmp/mplconfig} \
$PYTHON_BIN yao_hparam_search_baseline_style.py \
  --gpu $gpu \
  --seed $SEED \
  --random-seed $RANDOM_SEED \
  --n-gpus $N_WORKERS \
  --gpu-id $idx \
  --n-trials-per-preset $N_TRIALS_PER_PRESET \
  --presets $PRESETS \
  --lsi-methods $LSI_METHODS \
  --search-dir $SEARCH_DIR \
  --n-jobs $N_JOBS \
  --pretrain-max-epochs $PRETRAIN_MAX_EPOCHS \
  --pretrain-patience $PRETRAIN_PATIENCE \
  --pretrain-reduce-lr-patience $PRETRAIN_REDUCE_LR_PATIENCE \
  --finetune-max-epochs $FINETUNE_MAX_EPOCHS \
  --finetune-patience $FINETUNE_PATIENCE \
  --finetune-reduce-lr-patience $FINETUNE_REDUCE_LR_PATIENCE"
    if [[ "$QUICK_SEARCH" != "0" ]]; then
        cmd="$cmd --quick-search"
    fi
    cmd="$cmd 2>&1 | tee $SEARCH_DIR/worker${idx}.log; read"
    tmux send-keys -t "$SESSION:$win" "$cmd" Enter
done

tmux select-window -t "$SESSION:monitor"

echo "Started tmux session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"
echo "Monitor window refreshes every ${MONITOR_INTERVAL}s."
