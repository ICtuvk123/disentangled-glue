#!/bin/bash
# Launch s06 hyperparameter sweep across 4 GPUs in parallel tmux windows.
# Grid: shared_dims x private_dims x beta_shared x lam_iso = 4x4x3x2 = 96 runs
# Usage: bash run_s06_parallel.sh [--resume]

BASE=/data1/users/zhutianci/proj/disentangled-glue/experiments/BMMC
EXTRA_ARGS="$*"

OUTDIR="s06_sweep"
GRID="--shared-dims 30 40 50 60 \
      --private-dims 4 5 6 8 \
      --beta-shared 0.75 1.0 1.25 \
      --lam-iso 0.5 1.0 \
      --beta-private-rna 1.0 \
      --beta-private-atac 1.0 \
      --beta-private-prot 1.0"
COMMON="--rna s01_preprocessing/RNA_counts_qc_sampled.h5ad \
        --atac s01_preprocessing/ATAC_counts_qc_sampled.h5ad \
        --prot s01_preprocessing/protein_counts_qc_sampled.h5ad \
        --gtf gencode.v38.chr_patch_hapl_scaff.annotation.gtf \
        --protein-gene-map s01_preprocessing/protein_gene_map.tsv \
        --output-dir ${OUTDIR} --n-gpus 4 ${GRID}"

tmux kill-session -t s06 2>/dev/null; tmux new-session -d -s s06 -x 220 -y 50
for GPU_ID in 0 1 2 3; do
    if [ $GPU_ID -gt 0 ]; then
        tmux new-window -t s06
    fi
    tmux rename-window -t s06 "gpu${GPU_ID}"
    tmux send-keys -t s06 "cd $BASE && conda activate scMRDR && CUDA_VISIBLE_DEVICES=${GPU_ID} python s03_hparam_search.py $COMMON --gpu-id ${GPU_ID} ${EXTRA_ARGS} 2>&1 | tee ${OUTDIR}_gpu${GPU_ID}.log" Enter
done

echo "Started 4 workers in tmux session 's06'."
echo "Attach with: tmux attach -t s06"
echo "Switch windows: Ctrl+B then 0/1/2/3"
echo "Detach: Ctrl+B then D"
