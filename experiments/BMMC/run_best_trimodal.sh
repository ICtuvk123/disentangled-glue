#!/bin/bash
cd /data1/users/zhutianci/proj/disentangled-glue/experiments/BMMC
export CUDA_VISIBLE_DEVICES=0
python s02_glue.py \
    --rna s01_preprocessing/RNA_counts_qc_sampled.h5ad \
    --atac s01_preprocessing/ATAC_counts_qc_sampled.h5ad \
    --prot s01_preprocessing/protein_counts_qc_sampled.h5ad \
    --gtf gencode.v38.chr_patch_hapl_scaff.annotation.gtf \
    --protein-gene-map s01_preprocessing/protein_gene_map.tsv \
    --model disentangled \
    --shared-dim 24 \
    --private-dim 4 \
    --beta-shared 0.75 \
    --lam-iso 1.0 \
    --lam-align 0.03 \
    --beta-private-rna 1.0 \
    --beta-private-atac 1.0 \
    --beta-private-prot 1.0 \
    --align-support \
    --align-support-k 10 \
    --align-support-strategy soft \
    --align-support-min-weight 0.2 \
    --output-dir s02_best_run \
    2>&1 | tee s02_best_run.log
