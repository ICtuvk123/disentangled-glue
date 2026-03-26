#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

GTF="${GTF:-$SCRIPT_DIR/gencode.v38.chr_patch_hapl_scaff.annotation.gtf}"
MULTIOME="${MULTIOME:-$SCRIPT_DIR/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad}"
CITE="${CITE:-$SCRIPT_DIR/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad}"
HGNC="${HGNC:-$SCRIPT_DIR/hgnc_complete_set.txt}"

PREP_DIR="${PREP_DIR:-$SCRIPT_DIR/s01_preprocessing_full}"
PP_DIR="${PP_DIR:-$SCRIPT_DIR/s08_full_runs/preprocessed}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/s08_full_runs/models}"
EVAL_SCMRDR_DIR="${EVAL_SCMRDR_DIR:-$SCRIPT_DIR/s08_full_runs/eval_scmrdr}"
EVAL_SCIB_DIR="${EVAL_SCIB_DIR:-$SCRIPT_DIR/s08_full_runs/eval_scib}"

RNA="$PREP_DIR/RNA_counts_qc.h5ad"
ATAC="$PREP_DIR/ATAC_counts_qc.h5ad"
PROT="$PREP_DIR/protein_counts_qc.h5ad"
PROTEIN_GENE_MAP="$PREP_DIR/protein_gene_map.tsv"
FEATURE_ALIGNED="$PREP_DIR/feature_aligned.h5ad"

mkdir -p "$RUN_ROOT" "$EVAL_SCMRDR_DIR" "$EVAL_SCIB_DIR"

if [[ ! -f "$RNA" || ! -f "$ATAC" || ! -f "$PROT" || ! -f "$FEATURE_ALIGNED" ]]; then
    echo "=== Step 1: full preprocessing ==="
    "$PYTHON_BIN" s01_preprocessing.py \
        --multiome "$MULTIOME" \
        --cite "$CITE" \
        --gtf "$GTF" \
        --hgnc "$HGNC" \
        --output-dir "$PREP_DIR"
fi

if [[ ! -f "$PP_DIR/guidance.graphml.gz" ]]; then
    echo "=== Step 2: shared graph preprocessing ==="
    "$PYTHON_BIN" s02_glue.py \
        --model disentangled \
        --rna "$RNA" \
        --atac "$ATAC" \
        --prot "$PROT" \
        --gtf "$GTF" \
        --protein-gene-map "$PROTEIN_GENE_MAP" \
        --batch-key batch \
        --output-dir "$PP_DIR" \
        --preprocess-only
fi

run_and_eval() {
    local tag="$1"
    shift
    local run_dir="$RUN_ROOT/$tag"

    if [[ ! -f "$run_dir/combined_glue.h5ad" ]]; then
        echo "=== Training $tag ==="
        "$PYTHON_BIN" s02_glue.py \
            --rna "$RNA" \
            --atac "$ATAC" \
            --prot "$PROT" \
            --gtf "$GTF" \
            --protein-gene-map "$PROTEIN_GENE_MAP" \
            --batch-key batch \
            --output-dir "$run_dir" \
            --preprocessed-dir "$PP_DIR" \
            "$@"
    fi

    echo "=== Evaluating $tag with strict scMRDR workflow (RNA/ATAC only) ==="
    "$PYTHON_BIN" s08_eval_scmrdr.py \
        --run-dir "$run_dir" \
        --rna "$RNA" \
        --atac "$ATAC" \
        --output-dir "$EVAL_SCMRDR_DIR" \
        --tag "$tag" \
        --cell-type-key cell_type \
        --domain-key domain

    echo "=== Evaluating $tag with scib-metrics + fixed PCR ==="
    "$PYTHON_BIN" s06_eval.py \
        --run-dir "$run_dir" \
        --feature-aligned "$FEATURE_ALIGNED" \
        --enable-pcr \
        --output-dir "$EVAL_SCIB_DIR" \
        --tag "$tag" \
        --cell-type-key celltype \
        --batch-key batch \
        --domain-key domain \
        --n-jobs 8 \
        --no-show
}

run_and_eval \
    "scglue_reference_full" \
    --model scglue

run_and_eval \
    "disentangled_baseline_best_full" \
    --model disentangled \
    --shared-dim 30 \
    --private-dim 8 \
    --beta-shared 1.0 \
    --lam-iso 1.0 \
    --lam-align 0.03 \
    --beta-private-rna 1.0 \
    --beta-private-atac 1.0 \
    --beta-private-prot 1.0

run_and_eval \
    "disentangled_support_best_full" \
    --model disentangled \
    --shared-dim 30 \
    --private-dim 8 \
    --beta-shared 1.25 \
    --lam-iso 1.0 \
    --lam-align 0.05 \
    --beta-private-rna 1.0 \
    --beta-private-atac 1.0 \
    --beta-private-prot 1.0 \
    --align-support \
    --align-support-k 15 \
    --align-support-strategy soft \
    --align-support-min-weight 0.05

echo "=== Done ==="
echo "Models: $RUN_ROOT"
echo "scMRDR-style metrics: $EVAL_SCMRDR_DIR"
echo "scib metrics with fixed PCR: $EVAL_SCIB_DIR"
