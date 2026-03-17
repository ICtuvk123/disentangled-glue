#!/bin/bash
cd /data1/users/zhutianci/proj/disentangled-glue/experiments/BMMC
export CUDA_VISIBLE_DEVICES=0
python s02_run_and_compare.py --rna s01_preprocessing/RNA_counts_qc_sampled.h5ad --atac s01_preprocessing/ATAC_counts_qc_sampled.h5ad --prot s01_preprocessing/protein_counts_qc_sampled.h5ad --gtf gencode.v38.chr_patch_hapl_scaff.annotation.gtf --protein-gene-map s01_preprocessing/protein_gene_map.tsv --output-dir s02_compare 2>&1 | tee s02.log
