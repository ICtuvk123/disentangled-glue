#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SOURCE_RUN="${SOURCE_RUN:-$SCRIPT_DIR/s06_sweep/run_023}"
SOURCE_PREPROCESSED_DIR="${SOURCE_PREPROCESSED_DIR:-$(dirname "$SOURCE_RUN")/preprocessed}"
SESSION_NAME="${SESSION_NAME:-bmmc-align-support-4gpu}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-$SCRIPT_DIR/align_support_4gpu_runs/$RUN_TAG}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RNA="${RNA:-$SCRIPT_DIR/s01_preprocessing/RNA_counts_qc_sampled.h5ad}"
ATAC="${ATAC:-$SCRIPT_DIR/s01_preprocessing/ATAC_counts_qc_sampled.h5ad}"
PROT="${PROT:-$SCRIPT_DIR/s01_preprocessing/protein_counts_qc_sampled.h5ad}"
GTF="${GTF:-$SCRIPT_DIR/gencode.v38.chr_patch_hapl_scaff.annotation.gtf}"
PROTEIN_GENE_MAP="${PROTEIN_GENE_MAP:-$SCRIPT_DIR/s01_preprocessing/protein_gene_map.tsv}"

CELL_TYPE_KEY="${CELL_TYPE_KEY:-celltype}"
BATCH_KEY="${BATCH_KEY:-batch}"
DOMAIN_KEY="${DOMAIN_KEY:-domain}"
N_JOBS="${N_JOBS:-8}"

NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
RESUME="${RESUME:-1}"

worker_strategy_name() {
    case "$1" in
        0) echo "soft_k15_w005" ;;
        1) echo "soft_k30_w005" ;;
        2) echo "soft_k15_w001" ;;
        3) echo "hard_k15" ;;
        *) echo "unknown" >&2; return 1 ;;
    esac
}

worker_strategy_flags() {
    case "$1" in
        0) echo "--align-support --align-support-k 15 --align-support-strategy soft --align-support-min-weight 0.05" ;;
        1) echo "--align-support --align-support-k 30 --align-support-strategy soft --align-support-min-weight 0.05" ;;
        2) echo "--align-support --align-support-k 15 --align-support-strategy soft --align-support-min-weight 0.01" ;;
        3) echo "--align-support --align-support-k 15 --align-support-strategy hard --align-support-min-weight 0.05" ;;
        *) echo "unknown" >&2; return 1 ;;
    esac
}

load_hparams() {
    eval "$("$PYTHON_BIN" - "$SOURCE_RUN/hparams.json" <<'PY'
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
    print(f"{key.upper()}={shlex.quote(str(h[key]))}")
PY
)"
}

run_eval() {
    local run_dir="$1"
    local output_dir="$2"
    local tag="$3"
    local preprocessed_dir="$4"

    local cmd=(
        "$PYTHON_BIN" s06_eval.py
        --run-dir "$run_dir"
        --output-dir "$output_dir"
        --tag "$tag"
        --cell-type-key "$CELL_TYPE_KEY"
        --domain-key "$DOMAIN_KEY"
        --n-jobs "$N_JOBS"
        --no-show
    )
    if [[ -n "$BATCH_KEY" ]]; then
        cmd+=(--batch-key "$BATCH_KEY")
    fi
    if [[ -n "$preprocessed_dir" && -d "$preprocessed_dir" ]]; then
        cmd+=(--preprocessed-dir "$preprocessed_dir")
    fi
    "${cmd[@]}"
}

baseline_mode() {
    mkdir -p "$BASE_OUT/baseline_eval"
    local log_file="$BASE_OUT/baseline_eval.log"
    exec > >(tee -a "$log_file") 2>&1
    export NUMBA_CACHE_DIR MPLCONFIGDIR

    echo "Baseline source run: $SOURCE_RUN"
    if [[ ! -f "$SOURCE_RUN/combined_glue.h5ad" ]]; then
        echo "Missing baseline combined_glue.h5ad under $SOURCE_RUN" >&2
        exit 1
    fi
    if [[ "$RESUME" == "1" && -f "$BASE_OUT/baseline_eval/baseline_unscaled.csv" ]]; then
        echo "Baseline evaluation already exists, skipping"
        exit 0
    fi
    run_eval "$SOURCE_RUN" "$BASE_OUT/baseline_eval" "baseline" "$SOURCE_PREPROCESSED_DIR"
}

worker_mode() {
    local worker_id="$1"
    local gpu_id="$1"
    local name
    name="$(worker_strategy_name "$worker_id")"
    local flags
    flags="$(worker_strategy_flags "$worker_id")"
    local run_dir="$BASE_OUT/$name"
    local eval_dir="$BASE_OUT/${name}_eval"
    local log_file="$BASE_OUT/${name}.log"

    mkdir -p "$run_dir" "$eval_dir"
    exec > >(tee -a "$log_file") 2>&1

    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export NUMBA_CACHE_DIR MPLCONFIGDIR

    load_hparams

    echo "Worker: $worker_id"
    echo "GPU: $gpu_id"
    echo "Strategy: $name"
    echo "Flags: $flags"

    if [[ "$RESUME" != "1" || ! -f "$run_dir/combined_glue.h5ad" ]]; then
        common_train_args=(
            "$PYTHON_BIN" s02_glue.py
            --model disentangled
            --rna "$RNA"
            --atac "$ATAC"
            --prot "$PROT"
            --gtf "$GTF"
            --output-dir "$run_dir"
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
        read -r -a strategy_args <<<"$flags"
        "${common_train_args[@]}" "${strategy_args[@]}"
    else
        echo "Training output already exists for $name, skipping training"
    fi

    if [[ "$RESUME" != "1" || ! -f "$eval_dir/${name}_unscaled.csv" ]]; then
        run_eval "$run_dir" "$eval_dir" "$name" "$run_dir"
    else
        echo "Evaluation output already exists for $name, skipping evaluation"
    fi
}

summary_mode() {
    local log_file="$BASE_OUT/summary.log"
    mkdir -p "$BASE_OUT/compare"
    exec > >(tee -a "$log_file") 2>&1

    while true; do
        clear
        echo "Base output: $BASE_OUT"
        echo
        echo "Expected strategies:"
        for worker_id in 0 1 2 3; do
            echo "  $(worker_strategy_name "$worker_id")"
        done
        echo

        "$PYTHON_BIN" - "$BASE_OUT" <<'PY'
from pathlib import Path
import pandas as pd
import sys

base_out = Path(sys.argv[1])
compare_dir = base_out / "compare"
compare_dir.mkdir(parents=True, exist_ok=True)

paths = {
    "baseline": base_out / "baseline_eval" / "baseline_unscaled.csv",
    "soft_k15_w005": base_out / "soft_k15_w005_eval" / "soft_k15_w005_unscaled.csv",
    "soft_k30_w005": base_out / "soft_k30_w005_eval" / "soft_k30_w005_unscaled.csv",
    "soft_k15_w001": base_out / "soft_k15_w001_eval" / "soft_k15_w001_unscaled.csv",
    "hard_k15": base_out / "hard_k15_eval" / "hard_k15_unscaled.csv",
}

available = {}
missing = []
for key, path in paths.items():
    if path.exists():
        available[key] = pd.read_csv(path, index_col=0).iloc[0]
    else:
        missing.append(f"{key}: {path}")

if missing:
    print("Waiting for runs:")
    for item in missing:
        print(f"  {item}")
    print()
else:
    baseline = available["baseline"]
    numeric_cols = baseline.index[pd.to_numeric(baseline, errors="coerce").notna()]
    rows = [baseline]
    index = ["baseline"]
    for key in ["soft_k15_w005", "soft_k30_w005", "soft_k15_w001", "hard_k15"]:
        rows.append(available[key])
        index.append(key)
    for key in ["soft_k15_w005", "soft_k30_w005", "soft_k15_w001", "hard_k15"]:
        delta = available[key].copy()
        delta.loc[numeric_cols] = (
            pd.to_numeric(available[key].loc[numeric_cols], errors="coerce")
            - pd.to_numeric(baseline.loc[numeric_cols], errors="coerce")
        )
        delta.loc[~delta.index.isin(numeric_cols)] = ""
        rows.append(delta)
        index.append(f"delta:{key}-baseline")
    summary = pd.DataFrame(rows, index=index)
    summary_path = compare_dir / "metric_summary.tsv"
    summary.to_csv(summary_path, sep="\t", float_format="%.6f")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Saved summary -> {summary_path}")
    print()

status_rows = []
for key, path in paths.items():
    status_rows.append({"run": key, "ready": path.exists(), "csv": str(path)})
status = pd.DataFrame(status_rows)
status_path = compare_dir / "status.tsv"
status.to_csv(status_path, sep="\t", index=False)
print(status.to_string(index=False))
PY
        sleep 20
    done
}

launch_tmux() {
    mkdir -p "$BASE_OUT"
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not installed or not on PATH" >&2
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists" >&2
        echo "Use a different SESSION_NAME or kill the existing session first" >&2
        exit 1
    fi

    tmux new-session -d -s "$SESSION_NAME" -n baseline \
        "cd '$SCRIPT_DIR' && MODE=baseline SOURCE_RUN='$SOURCE_RUN' SOURCE_PREPROCESSED_DIR='$SOURCE_PREPROCESSED_DIR' SESSION_NAME='$SESSION_NAME' RUN_TAG='$RUN_TAG' BASE_OUT='$BASE_OUT' PYTHON_BIN='$PYTHON_BIN' RNA='$RNA' ATAC='$ATAC' PROT='$PROT' GTF='$GTF' PROTEIN_GENE_MAP='$PROTEIN_GENE_MAP' CELL_TYPE_KEY='$CELL_TYPE_KEY' BATCH_KEY='$BATCH_KEY' DOMAIN_KEY='$DOMAIN_KEY' N_JOBS='$N_JOBS' NUMBA_CACHE_DIR='$NUMBA_CACHE_DIR' MPLCONFIGDIR='$MPLCONFIGDIR' RESUME='$RESUME' bash '$0'"

    for worker_id in 0 1 2 3; do
        tmux new-window -t "$SESSION_NAME" -n "gpu${worker_id}" \
            "cd '$SCRIPT_DIR' && MODE=worker WORKER_ID='$worker_id' SOURCE_RUN='$SOURCE_RUN' SOURCE_PREPROCESSED_DIR='$SOURCE_PREPROCESSED_DIR' SESSION_NAME='$SESSION_NAME' RUN_TAG='$RUN_TAG' BASE_OUT='$BASE_OUT' PYTHON_BIN='$PYTHON_BIN' RNA='$RNA' ATAC='$ATAC' PROT='$PROT' GTF='$GTF' PROTEIN_GENE_MAP='$PROTEIN_GENE_MAP' CELL_TYPE_KEY='$CELL_TYPE_KEY' BATCH_KEY='$BATCH_KEY' DOMAIN_KEY='$DOMAIN_KEY' N_JOBS='$N_JOBS' NUMBA_CACHE_DIR='$NUMBA_CACHE_DIR' MPLCONFIGDIR='$MPLCONFIGDIR' RESUME='$RESUME' bash '$0'"
    done

    tmux new-window -t "$SESSION_NAME" -n summary \
        "cd '$SCRIPT_DIR' && MODE=summary SOURCE_RUN='$SOURCE_RUN' SOURCE_PREPROCESSED_DIR='$SOURCE_PREPROCESSED_DIR' SESSION_NAME='$SESSION_NAME' RUN_TAG='$RUN_TAG' BASE_OUT='$BASE_OUT' PYTHON_BIN='$PYTHON_BIN' RNA='$RNA' ATAC='$ATAC' PROT='$PROT' GTF='$GTF' PROTEIN_GENE_MAP='$PROTEIN_GENE_MAP' CELL_TYPE_KEY='$CELL_TYPE_KEY' BATCH_KEY='$BATCH_KEY' DOMAIN_KEY='$DOMAIN_KEY' N_JOBS='$N_JOBS' NUMBA_CACHE_DIR='$NUMBA_CACHE_DIR' MPLCONFIGDIR='$MPLCONFIGDIR' RESUME='$RESUME' bash '$0'"

    echo "Created tmux session: $SESSION_NAME"
    echo "Baseline source run: $SOURCE_RUN"
    echo "Output directory: $BASE_OUT"
    echo "Windows: baseline, gpu0, gpu1, gpu2, gpu3, summary"
    tmux attach -t "$SESSION_NAME"
}

main() {
    case "${MODE:-launch}" in
        launch) launch_tmux ;;
        baseline) baseline_mode ;;
        worker) worker_mode "${WORKER_ID:?missing WORKER_ID}" ;;
        summary) summary_mode ;;
        *) echo "Unknown MODE=${MODE:-}" >&2; exit 1 ;;
    esac
}

main "$@"
