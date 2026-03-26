#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SOURCE_RUN="${SOURCE_RUN:-$SCRIPT_DIR/s06_sweep/run_023}"
SESSION_NAME="${SESSION_NAME:-bmmc-align-support}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-$SCRIPT_DIR/align_support_runs/$RUN_TAG}"
LOG_FILE="$BASE_OUT/session.log"

GPU="${GPU:-0}"
CELL_TYPE_KEY="${CELL_TYPE_KEY:-celltype}"
BATCH_KEY="${BATCH_KEY:-batch}"
DOMAIN_KEY="${DOMAIN_KEY:-domain}"
N_JOBS="${N_JOBS:-8}"

RNA="${RNA:-$SCRIPT_DIR/s01_preprocessing/RNA_counts_qc_sampled.h5ad}"
ATAC="${ATAC:-$SCRIPT_DIR/s01_preprocessing/ATAC_counts_qc_sampled.h5ad}"
PROT="${PROT:-$SCRIPT_DIR/s01_preprocessing/protein_counts_qc_sampled.h5ad}"
GTF="${GTF:-$SCRIPT_DIR/gencode.v38.chr_patch_hapl_scaff.annotation.gtf}"
PROTEIN_GENE_MAP="${PROTEIN_GENE_MAP:-$SCRIPT_DIR/s01_preprocessing/protein_gene_map.tsv}"

ALIGN_SUPPORT_K="${ALIGN_SUPPORT_K:-15}"
ALIGN_SUPPORT_STRATEGY="${ALIGN_SUPPORT_STRATEGY:-soft}"
ALIGN_SUPPORT_MIN_WEIGHT="${ALIGN_SUPPORT_MIN_WEIGHT:-0.05}"

NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

mkdir -p "$BASE_OUT"

if [[ -z "${TMUX:-}" && "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not installed or not on PATH" >&2
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists" >&2
        echo "Use a different SESSION_NAME or kill the existing session first" >&2
        exit 1
    fi
    tmux new-session -d -s "$SESSION_NAME" \
        "cd '$SCRIPT_DIR' && \
         RUN_INSIDE_TMUX=1 \
         SOURCE_RUN='$SOURCE_RUN' \
         SESSION_NAME='$SESSION_NAME' \
         RUN_TAG='$RUN_TAG' \
         BASE_OUT='$BASE_OUT' \
         GPU='$GPU' \
         CELL_TYPE_KEY='$CELL_TYPE_KEY' \
         BATCH_KEY='$BATCH_KEY' \
         DOMAIN_KEY='$DOMAIN_KEY' \
         N_JOBS='$N_JOBS' \
         RNA='$RNA' \
         ATAC='$ATAC' \
         PROT='$PROT' \
         GTF='$GTF' \
         PROTEIN_GENE_MAP='$PROTEIN_GENE_MAP' \
         ALIGN_SUPPORT_K='$ALIGN_SUPPORT_K' \
         ALIGN_SUPPORT_STRATEGY='$ALIGN_SUPPORT_STRATEGY' \
         ALIGN_SUPPORT_MIN_WEIGHT='$ALIGN_SUPPORT_MIN_WEIGHT' \
         NUMBA_CACHE_DIR='$NUMBA_CACHE_DIR' \
         MPLCONFIGDIR='$MPLCONFIGDIR' \
         bash '$0'"
    tmux new-window -t "$SESSION_NAME" -n metrics \
        "bash -lc 'while true; do clear; \
         if [[ -f \"$BASE_OUT/compare/metric_summary.tsv\" ]]; then \
             echo \"Summary: $BASE_OUT/compare/metric_summary.tsv\"; \
             echo; \
             cat \"$BASE_OUT/compare/metric_summary.tsv\"; \
         elif [[ -f \"$LOG_FILE\" ]]; then \
             echo \"Waiting for metrics...\"; \
             echo \"Log: $LOG_FILE\"; \
             echo; \
             tail -n 40 \"$LOG_FILE\"; \
         else \
             echo \"Waiting for runner to start...\"; \
         fi; \
         sleep 15; \
         done'"
    echo "Created tmux session: $SESSION_NAME"
    echo "Output directory: $BASE_OUT"
    tmux attach -t "$SESSION_NAME"
    exit 0
fi

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Session: $SESSION_NAME"
echo "Base output: $BASE_OUT"
echo "Source run: $SOURCE_RUN"
echo "GPU: $GPU"

for path in "$RNA" "$ATAC" "$PROT" "$GTF"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

if [[ ! -f "$SOURCE_RUN/hparams.json" ]]; then
    echo "Missing hparams.json under SOURCE_RUN: $SOURCE_RUN" >&2
    exit 1
fi

eval "$(python - "$SOURCE_RUN/hparams.json" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    h = json.load(fh)

keys = [
    "shared_dim",
    "private_dim",
    "beta_shared",
    "lam_iso",
    "beta_private_rna",
    "beta_private_atac",
    "beta_private_prot",
]
for key in keys:
    value = h[key]
    shell_key = key.upper()
    print(f"{shell_key}={shlex.quote(str(value))}")
PY
)"

echo "Loaded hyperparameters from $SOURCE_RUN/hparams.json"
echo "  shared_dim=$SHARED_DIM"
echo "  private_dim=$PRIVATE_DIM"
echo "  beta_shared=$BETA_SHARED"
echo "  lam_iso=$LAM_ISO"
echo "  beta_private_rna=$BETA_PRIVATE_RNA"
echo "  beta_private_atac=$BETA_PRIVATE_ATAC"
echo "  beta_private_prot=$BETA_PRIVATE_PROT"

export CUDA_VISIBLE_DEVICES="$GPU"
export NUMBA_CACHE_DIR
export MPLCONFIGDIR

BASELINE_RUN="$BASE_OUT/baseline"
SUPPORT_RUN="$BASE_OUT/support"
BASELINE_EVAL="$BASE_OUT/baseline_eval"
SUPPORT_EVAL="$BASE_OUT/support_eval"
COMPARE_DIR="$BASE_OUT/compare"

mkdir -p "$BASELINE_RUN" "$SUPPORT_RUN" "$BASELINE_EVAL" "$SUPPORT_EVAL" "$COMPARE_DIR"

common_train_args=(
    python s02_glue.py
    --model disentangled
    --rna "$RNA"
    --atac "$ATAC"
    --prot "$PROT"
    --gtf "$GTF"
    --shared-dim "$SHARED_DIM"
    --private-dim "$PRIVATE_DIM"
    --beta-shared "$BETA_SHARED"
    --lam-iso "$LAM_ISO"
    --beta-private-rna "$BETA_PRIVATE_RNA"
    --beta-private-atac "$BETA_PRIVATE_ATAC"
    --beta-private-prot "$BETA_PRIVATE_PROT"
)

if [[ -n "$PROTEIN_GENE_MAP" ]]; then
    common_train_args+=(--protein-gene-map "$PROTEIN_GENE_MAP")
fi
if [[ -n "$BATCH_KEY" ]]; then
    common_train_args+=(--batch-key "$BATCH_KEY")
fi

common_eval_args=(
    --cell-type-key "$CELL_TYPE_KEY"
    --domain-key "$DOMAIN_KEY"
    --n-jobs "$N_JOBS"
    --no-show
)
if [[ -n "$BATCH_KEY" ]]; then
    common_eval_args+=(--batch-key "$BATCH_KEY")
fi

echo
echo "=== Baseline training ==="
"${common_train_args[@]}" --output-dir "$BASELINE_RUN"

echo
echo "=== Baseline evaluation ==="
python s06_eval.py \
    --run-dir "$BASELINE_RUN" \
    --preprocessed-dir "$BASELINE_RUN" \
    --output-dir "$BASELINE_EVAL" \
    --tag baseline \
    "${common_eval_args[@]}"

echo
echo "=== Support-weighted training ==="
"${common_train_args[@]}" \
    --output-dir "$SUPPORT_RUN" \
    --align-support \
    --align-support-k "$ALIGN_SUPPORT_K" \
    --align-support-strategy "$ALIGN_SUPPORT_STRATEGY" \
    --align-support-min-weight "$ALIGN_SUPPORT_MIN_WEIGHT"

echo
echo "=== Support-weighted evaluation ==="
python s06_eval.py \
    --run-dir "$SUPPORT_RUN" \
    --preprocessed-dir "$SUPPORT_RUN" \
    --output-dir "$SUPPORT_EVAL" \
    --tag support \
    "${common_eval_args[@]}"

echo
echo "=== Metrics summary ==="
python - "$BASELINE_EVAL/baseline_unscaled.csv" "$SUPPORT_EVAL/support_unscaled.csv" "$COMPARE_DIR/metric_summary.tsv" <<'PY'
from pathlib import Path
import pandas as pd
import sys

baseline_path = Path(sys.argv[1])
support_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

baseline = pd.read_csv(baseline_path, index_col=0).iloc[0]
support = pd.read_csv(support_path, index_col=0).iloc[0]

summary = pd.DataFrame(
    [baseline, support, support - baseline],
    index=["baseline", "support", "delta_support_minus_baseline"],
)
summary.to_csv(out_path, sep="\t", float_format="%.6f")
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
print()
print(f"Saved summary -> {out_path}")
PY

echo
echo "Done."
echo "Log: $LOG_FILE"
echo "Summary: $COMPARE_DIR/metric_summary.tsv"
