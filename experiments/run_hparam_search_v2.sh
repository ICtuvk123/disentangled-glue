#!/bin/bash
# Launch hparam_search_v2.py on GPUs 1, 2, 3 in parallel via tmux.
# GPU 0 is reserved for the ongoing hparam_search.py run.

set -e

SESSION="hparam_v2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(conda run -n scMRDR which python 2>/dev/null || which python)"

N_TRIALS=${1:-60}
SEED=${2:-42}

echo "Launching ${N_TRIALS} trials (seed=${SEED}) on GPUs 1,2,3 in tmux session '${SESSION}'"

# kill existing session if any
tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session  -d -s "${SESSION}" -n "gpu0" \
    "CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${SCRIPT_DIR}/hparam_search_v2.py \
        --n-trials ${N_TRIALS} --seed ${SEED} --n-gpus 3 --gpu-id 0 \
        2>&1 | tee ${SCRIPT_DIR}/../hparam_search_v2/worker0.log; read"

tmux new-window      -t "${SESSION}" -n "gpu2" \
    "CUDA_VISIBLE_DEVICES=2 ${PYTHON} ${SCRIPT_DIR}/hparam_search_v2.py \
        --n-trials ${N_TRIALS} --seed ${SEED} --n-gpus 3 --gpu-id 1 \
        2>&1 | tee ${SCRIPT_DIR}/../hparam_search_v2/worker1.log; read"

tmux new-window      -t "${SESSION}" -n "gpu3" \
    "CUDA_VISIBLE_DEVICES=3 ${PYTHON} ${SCRIPT_DIR}/hparam_search_v2.py \
        --n-trials ${N_TRIALS} --seed ${SEED} --n-gpus 3 --gpu-id 2 \
        2>&1 | tee ${SCRIPT_DIR}/../hparam_search_v2/worker2.log; read"

echo "Started. Attach with:  tmux attach -t ${SESSION}"
echo "Switch windows:        Ctrl-b  n / p"
echo "Detach:                Ctrl-b  d"
