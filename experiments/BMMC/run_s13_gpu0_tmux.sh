#!/usr/bin/env bash
set -euo pipefail

SESSION="s13_gpu0"
ROOT="/data1/users/zhutianci/proj/disentangled-glue/experiments/BMMC"
PYTHON="/data1/users/zhutianci/.conda/envs/scMRDR/bin/python"
BEDTOOLS="/data1/users/zhutianci/.conda/envs/scMRDR/bin/bedtools"
OUTDIR="$ROOT/s13_rna_atac_search"

RNA="$ROOT/s01_preprocessing/RNA_counts_qc_sampled.h5ad"
ATAC="$ROOT/s01_preprocessing/ATAC_counts_qc_sampled.h5ad"
GTF="$ROOT/gencode.v38.chr_patch_hapl_scaff.annotation.gtf"

mkdir -p "$OUTDIR"

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -n gpu0

tmux send-keys -t "$SESSION:gpu0" "cd $ROOT && \
CUDA_VISIBLE_DEVICES=0 \
NUMBA_CACHE_DIR=/tmp/numba-cache \
MPLCONFIGDIR=/tmp/mplconfig \
$PYTHON s13_rna_atac_search.py \
  --rna $RNA \
  --atac $ATAC \
  --gtf $GTF \
  --output-dir $OUTDIR \
  --bedtools $BEDTOOLS \
  --batch-key batch \
  --cell-type-key celltype \
  --domain-key domain \
  --n-trials 100 \
  --n-gpus 4 \
  --gpu-id 0 \
  --seed 42 \
  --resume 2>&1 | tee $OUTDIR/gpu0.log" C-m

tmux attach -t "$SESSION"
