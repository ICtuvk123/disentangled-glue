#!/bin/bash
set -e
exec > >(tee -a pipeline.log) 2>&1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GTF="gencode.v38.chr_patch_hapl_scaff.annotation.gtf"
MULTIOME="GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
CITE="GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
HGNC="hgnc_complete_set.txt"

# Bypass pynvml GPU enumeration bug by pre-setting the GPU
export CUDA_VISIBLE_DEVICES=0

echo "=== Step 1: Preprocessing (sampled only) ==="
python s01_preprocessing.py --multiome "$MULTIOME" --cite "$CITE" --gtf "$GTF" --hgnc "$HGNC" --output-dir s01_preprocessing --sampled-only

echo "=== Step 2: Train & compare ==="
python s02_run_and_compare.py --rna s01_preprocessing/RNA_counts_qc_sampled.h5ad --atac s01_preprocessing/ATAC_counts_qc_sampled.h5ad --prot s01_preprocessing/protein_counts_qc_sampled.h5ad --gtf "$GTF" --protein-gene-map s01_preprocessing/protein_gene_map.tsv --output-dir s02_compare

echo "=== Done! Results: s02_compare/comparison.png and s02_compare/metrics.tsv ==="
