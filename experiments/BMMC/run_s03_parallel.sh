#!/bin/bash
# Launch hyperparameter search across 4 GPUs in parallel tmux windows.
# Usage: bash run_s03_parallel.sh [--resume]

BASE=/data1/users/zhutianci/proj/disentangled-glue/experiments/BMMC
EXTRA_ARGS="$*"

OUTDIR="s05_sweep"
GRID="--beta-shared 0.25 0.5 1.0 1.5 2.0 \
      --lam-iso 0.1 0.5 1.0 2.0 4.0 \
      --private-dims 5 10 20 30"
COMMON="--rna s01_preprocessing/RNA_counts_qc_sampled.h5ad \
        --atac s01_preprocessing/ATAC_counts_qc_sampled.h5ad \
        --prot s01_preprocessing/protein_counts_qc_sampled.h5ad \
        --gtf gencode.v38.chr_patch_hapl_scaff.annotation.gtf \
        --protein-gene-map s01_preprocessing/protein_gene_map.tsv \
        --output-dir ${OUTDIR} --n-gpus 4 ${GRID}"

tmux kill-session -t hparam 2>/dev/null; tmux new-session -d -s hparam -x 220 -y 50
for GPU_ID in 0 1 2 3; do
    if [ $GPU_ID -gt 0 ]; then
        tmux new-window -t hparam
    fi
    tmux rename-window -t hparam "gpu${GPU_ID}"
    tmux send-keys -t hparam "cd $BASE && conda activate scMRDR && CUDA_VISIBLE_DEVICES=${GPU_ID} python s03_hparam_search.py $COMMON --gpu-id ${GPU_ID} ${EXTRA_ARGS} 2>&1 | tee ${OUTDIR}_gpu${GPU_ID}.log" Enter
done

echo "Started 4 workers. Attach with: tmux attach -t hparam"
echo "Switch windows: Ctrl+B then 0/1/2/3"
echo "Detach: Ctrl+B then D"
