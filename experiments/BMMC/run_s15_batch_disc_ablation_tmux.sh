#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "$0")"
PYTHON_BIN="${PYTHON_BIN:-/data1/users/zhutianci/.conda/envs/scMRDR/bin/python}"
BEDTOOLS="${BEDTOOLS:-/data1/users/zhutianci/.conda/envs/scMRDR/bin/bedtools}"
SESSION="${SESSION:-bmmc-s15-batchdisc}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-$SCRIPT_DIR/s15_batch_disc_ablation/$RUN_TAG}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$BASE_OUT/preprocessed}"
LOG_DIR="$BASE_OUT/logs"
EVAL_DIR="$BASE_OUT/eval"

RNA="${RNA:-$SCRIPT_DIR/s01_preprocessing/RNA_counts_qc_sampled.h5ad}"
ATAC="${ATAC:-$SCRIPT_DIR/s01_preprocessing/ATAC_counts_qc_sampled.h5ad}"
GTF="${GTF:-$SCRIPT_DIR/gencode.v38.chr_patch_hapl_scaff.annotation.gtf}"
FEATURE_ALIGNED="${FEATURE_ALIGNED:-$SCRIPT_DIR/s01_preprocessing/feature_aligned_sampled.h5ad}"

N_GPUS="${N_GPUS:-4}"
LAM_BATCH_VALUES="${LAM_BATCH_VALUES:-0,0.02,0.05,0.10,0.20}"
STARTUP_STAGGER_SEC="${STARTUP_STAGGER_SEC:-20}"
BETWEEN_RUN_SEC="${BETWEEN_RUN_SEC:-5}"
USE_IONICE="${USE_IONICE:-1}"
ENABLE_FORMAL_EVAL="${ENABLE_FORMAL_EVAL:-0}"
EVAL_N_JOBS="${EVAL_N_JOBS:-2}"
RESUME="${RESUME:-1}"

BATCH_KEY="${BATCH_KEY:-batch}"
CELL_TYPE_KEY="${CELL_TYPE_KEY:-celltype}"
DOMAIN_KEY="${DOMAIN_KEY:-domain}"
SHARED_BATCHES="${SHARED_BATCHES:-1}"

SHARED_DIM="${SHARED_DIM:-32}"
PRIVATE_DIM="${PRIVATE_DIM:-4}"
BETA_SHARED="${BETA_SHARED:-0.75}"
LAM_ISO="${LAM_ISO:-2.0}"
LAM_ALIGN="${LAM_ALIGN:-0.03}"
BETA_PRIVATE_RNA="${BETA_PRIVATE_RNA:-1.0}"
BETA_PRIVATE_ATAC="${BETA_PRIVATE_ATAC:-0.5}"

ENABLE_SUPPORT="${ENABLE_SUPPORT:-0}"
ALIGN_SUPPORT_K="${ALIGN_SUPPORT_K:-15}"
ALIGN_SUPPORT_STRATEGY="${ALIGN_SUPPORT_STRATEGY:-soft}"
ALIGN_SUPPORT_MIN_WEIGHT="${ALIGN_SUPPORT_MIN_WEIGHT:-0.05}"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

PREPROCESS_FILES=(rna_pp.h5ad atac_pp.h5ad guidance.graphml.gz)

trim() {
    local value="$1"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    printf '%s' "$value"
}

run_name() {
    local lam_batch="$1"
    local support_suffix=""
    if [[ "$ENABLE_SUPPORT" == "1" ]]; then
        support_suffix="_sk${ALIGN_SUPPORT_K}_${ALIGN_SUPPORT_STRATEGY:0:1}_mw${ALIGN_SUPPORT_MIN_WEIGHT}"
    fi
    printf 'baseline_best_sd%s_pd%s_bs%s_li%s_la%s_lb%s_bpr%s_bpa%s_sb%s%s' \
        "$SHARED_DIM" "$PRIVATE_DIM" "$BETA_SHARED" "$LAM_ISO" "$LAM_ALIGN" \
        "$lam_batch" "$BETA_PRIVATE_RNA" "$BETA_PRIVATE_ATAC" "$SHARED_BATCHES" "$support_suffix"
}

check_inputs() {
    local missing=0
    for f in "$RNA" "$ATAC" "$GTF"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: missing file: $f" >&2
            missing=1
        fi
    done
    if [[ "$ENABLE_FORMAL_EVAL" == "1" && ! -f "$FEATURE_ALIGNED" ]]; then
        echo "ERROR: feature aligned file not found: $FEATURE_ALIGNED" >&2
        missing=1
    fi
    if [[ "$missing" != "0" ]]; then
        exit 1
    fi
}

preprocess_ready() {
    local f
    for f in "${PREPROCESS_FILES[@]}"; do
        [[ -f "$PREPROCESSED_DIR/$f" ]] || return 1
    done
}

preprocess_once() {
    mkdir -p "$PREPROCESSED_DIR"
    if preprocess_ready; then
        echo "Reusing preprocessing from $PREPROCESSED_DIR"
        return 0
    fi
    echo "Preparing shared preprocessing under $PREPROCESSED_DIR"
    "$PYTHON_BIN" "$SCRIPT_DIR/s02_glue_rna_atac.py" \
        --model disentangled \
        --rna "$RNA" \
        --atac "$ATAC" \
        --gtf "$GTF" \
        --output-dir "$PREPROCESSED_DIR" \
        --bedtools "$BEDTOOLS" \
        --preprocess-only
}

formal_eval() {
    local run_dir="$1"
    local tag="$2"
    mkdir -p "$EVAL_DIR"
    "$PYTHON_BIN" "$SCRIPT_DIR/s06_eval.py" \
        --run-dir "$run_dir" \
        --output-dir "$EVAL_DIR" \
        --tag "$tag" \
        --feature-aligned "$FEATURE_ALIGNED" \
        --enable-pcr \
        --domain-key "$DOMAIN_KEY" \
        --cell-type-key "$CELL_TYPE_KEY" \
        --batch-key "$BATCH_KEY" \
        --n-jobs "$EVAL_N_JOBS" \
        --no-show
}

worker_mode() {
    local worker_id="$1"
    mkdir -p "$BASE_OUT" "$LOG_DIR" "$EVAL_DIR"
    exec > >(tee -a "$LOG_DIR/gpu${worker_id}.log") 2>&1

    export CUDA_VISIBLE_DEVICES="$worker_id"

    IFS=',' read -r -a lam_batches <<< "$LAM_BATCH_VALUES"
    local first_run=1
    local idx lam_batch name run_dir status_file failed_file started_file rc sleep_s

    echo "Worker $worker_id starting on GPU $worker_id"
    echo "LAM_BATCH_VALUES=$LAM_BATCH_VALUES"
    echo "BASE_OUT=$BASE_OUT"

    for idx in "${!lam_batches[@]}"; do
        if (( idx % N_GPUS != worker_id )); then
            continue
        fi
        lam_batch="$(trim "${lam_batches[$idx]}")"
        name="$(run_name "$lam_batch")"
        run_dir="$BASE_OUT/$name"
        status_file="$run_dir/.done"
        failed_file="$run_dir/.failed"
        started_file="$run_dir/.started"

        mkdir -p "$run_dir"
        "$PYTHON_BIN" - "$run_dir/ablation_config.json" <<PY
import json, sys
cfg = {
    "shared_dim": "$SHARED_DIM",
    "private_dim": "$PRIVATE_DIM",
    "beta_shared": "$BETA_SHARED",
    "lam_iso": "$LAM_ISO",
    "lam_align": "$LAM_ALIGN",
    "lam_batch": "$lam_batch",
    "beta_private_rna": "$BETA_PRIVATE_RNA",
    "beta_private_atac": "$BETA_PRIVATE_ATAC",
    "shared_batches": bool(int("$SHARED_BATCHES")),
    "enable_support": bool(int("$ENABLE_SUPPORT")),
    "align_support_k": "$ALIGN_SUPPORT_K",
    "align_support_strategy": "$ALIGN_SUPPORT_STRATEGY",
    "align_support_min_weight": "$ALIGN_SUPPORT_MIN_WEIGHT",
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
PY

        if [[ "$RESUME" == "1" && -f "$run_dir/combined_glue.h5ad" && -f "$status_file" ]]; then
            echo "[$name] training already completed, skipping"
        else
            rm -f "$failed_file"
            date > "$started_file"
            if [[ "$first_run" == "1" && "$STARTUP_STAGGER_SEC" -gt 0 ]]; then
                sleep_s=$(( worker_id * STARTUP_STAGGER_SEC ))
                if (( sleep_s > 0 )); then
                    echo "[$name] staggering startup by ${sleep_s}s to reduce I/O burst"
                    sleep "$sleep_s"
                fi
                first_run=0
            fi

            cmd=(
                "$PYTHON_BIN" "$SCRIPT_DIR/s02_glue_rna_atac.py"
                --model disentangled
                --rna "$RNA"
                --atac "$ATAC"
                --gtf "$GTF"
                --preprocessed-dir "$PREPROCESSED_DIR"
                --output-dir "$run_dir"
                --shared-dim "$SHARED_DIM"
                --private-dim "$PRIVATE_DIM"
                --beta-shared "$BETA_SHARED"
                --lam-iso "$LAM_ISO"
                --lam-align "$LAM_ALIGN"
                --lam-batch "$lam_batch"
                --beta-private-rna "$BETA_PRIVATE_RNA"
                --beta-private-atac "$BETA_PRIVATE_ATAC"
                --bedtools "$BEDTOOLS"
                --batch-key "$BATCH_KEY"
                --skip-modality-h5ad
            )
            if [[ "$SHARED_BATCHES" == "1" ]]; then
                cmd+=(--shared-batches)
            fi
            if [[ "$ENABLE_SUPPORT" == "1" ]]; then
                cmd+=(
                    --align-support
                    --align-support-k "$ALIGN_SUPPORT_K"
                    --align-support-strategy "$ALIGN_SUPPORT_STRATEGY"
                    --align-support-min-weight "$ALIGN_SUPPORT_MIN_WEIGHT"
                )
            fi
            if [[ "$USE_IONICE" == "1" ]] && command -v ionice >/dev/null 2>&1; then
                cmd=(ionice -c2 -n7 "${cmd[@]}")
            fi
            if command -v nice >/dev/null 2>&1; then
                cmd=(nice -n 5 "${cmd[@]}")
            fi

            echo "[$name] starting training at $(date '+%F %T')"
            set +e
            "${cmd[@]}"
            rc=$?
            set -e
            if [[ "$rc" -ne 0 ]]; then
                echo "[$name] training failed with exit $rc"
                date > "$failed_file"
                continue
            fi
            date > "$status_file"
            echo "[$name] training completed at $(date '+%F %T')"
        fi

        if [[ "$ENABLE_FORMAL_EVAL" == "1" ]]; then
            if [[ "$RESUME" == "1" && -f "$EVAL_DIR/${name}_unscaled.csv" ]]; then
                echo "[$name] formal eval already exists, skipping"
            else
                echo "[$name] starting formal eval"
                set +e
                formal_eval "$run_dir" "$name"
                rc=$?
                set -e
                if [[ "$rc" -ne 0 ]]; then
                    echo "[$name] formal eval failed with exit $rc"
                fi
            fi
        fi

        if [[ "$BETWEEN_RUN_SEC" -gt 0 ]]; then
            sleep "$BETWEEN_RUN_SEC"
        fi
    done

    echo "Worker $worker_id finished"
}

monitor_mode() {
    mkdir -p "$BASE_OUT" "$LOG_DIR" "$EVAL_DIR"
    while true; do
        clear
        echo '=== s15 shared-batch-discriminator ablation ==='
        echo "BASE_OUT: $BASE_OUT"
        echo "PREPROCESSED_DIR: $PREPROCESSED_DIR"
        echo "LAM_BATCH_VALUES: $LAM_BATCH_VALUES"
        echo "Config: sd=$SHARED_DIM pd=$PRIVATE_DIM bs=$BETA_SHARED li=$LAM_ISO la=$LAM_ALIGN bpr=$BETA_PRIVATE_RNA bpa=$BETA_PRIVATE_ATAC sb=$SHARED_BATCHES support=$ENABLE_SUPPORT"
        echo
        "$PYTHON_BIN" - "$BASE_OUT" "$EVAL_DIR" "$LAM_BATCH_VALUES" "$SHARED_DIM" "$PRIVATE_DIM" "$BETA_SHARED" "$LAM_ISO" "$LAM_ALIGN" "$BETA_PRIVATE_RNA" "$BETA_PRIVATE_ATAC" "$SHARED_BATCHES" "$ENABLE_SUPPORT" "$ALIGN_SUPPORT_K" "$ALIGN_SUPPORT_STRATEGY" "$ALIGN_SUPPORT_MIN_WEIGHT" <<'PY'
from pathlib import Path
import sys
import pandas as pd

base_out = Path(sys.argv[1])
eval_dir = Path(sys.argv[2])
lam_batches = [item.strip() for item in sys.argv[3].split(',') if item.strip()]
shared_dim, private_dim, beta_shared, lam_iso, lam_align = sys.argv[4:9]
bpr, bpa, shared_batches = sys.argv[9:12]
enable_support = sys.argv[12]
ask, astrategy, aminw = sys.argv[13:16]

rows = []
for lam_batch in lam_batches:
    suffix = ''
    if enable_support == '1':
        suffix = f'_sk{ask}_{astrategy[0]}_mw{aminw}'
    name = f'baseline_best_sd{shared_dim}_pd{private_dim}_bs{beta_shared}_li{lam_iso}_la{lam_align}_lb{lam_batch}_bpr{bpr}_bpa{bpa}_sb{shared_batches}{suffix}'
    run_dir = base_out / name
    status = 'pending'
    if (run_dir / '.failed').exists():
        status = 'failed'
    elif (run_dir / '.done').exists():
        status = 'done'
    elif (run_dir / '.started').exists():
        status = 'running'
    combined = (run_dir / 'combined_glue.h5ad').exists()
    total = None
    csv_path = eval_dir / f'{name}_unscaled.csv'
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, index_col=0)
            scores = df.get('X_embed', df.iloc[:, 0])
            total = float(scores.get('Total')) if 'Total' in scores.index else None
        except Exception:
            total = None
    rows.append({
        'lam_batch': lam_batch,
        'status': status,
        'combined_glue': combined,
        'formal_total': total,
        'run_dir': str(run_dir),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False, justify='left', col_space=12))
PY
        echo
        echo 'Logs:'
        local gpu_id log_file
        for gpu_id in $(seq 0 $((N_GPUS - 1))); do
            log_file="$LOG_DIR/gpu${gpu_id}.log"
            if [[ -f "$log_file" ]]; then
                echo "--- gpu${gpu_id}.log (tail -5) ---"
                tail -n 5 "$log_file"
            else
                echo "--- gpu${gpu_id}.log: not started ---"
            fi
            echo
        done
        sleep 30
    done
}

launch_mode() {
    local gpu_id worker_cmd monitor_cmd
    check_inputs
    mkdir -p "$BASE_OUT" "$LOG_DIR" "$EVAL_DIR"
    preprocess_once

    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -x 240 -y 60 -n gpu0

    for gpu_id in $(seq 0 $((N_GPUS - 1))); do
        if [[ "$gpu_id" != "0" ]]; then
            tmux new-window -t "$SESSION" -n "gpu${gpu_id}"
        fi
        worker_cmd="cd $SCRIPT_DIR && BASE_OUT=$BASE_OUT PREPROCESSED_DIR=$PREPROCESSED_DIR LOG_DIR=$LOG_DIR EVAL_DIR=$EVAL_DIR PYTHON_BIN=$PYTHON_BIN BEDTOOLS=$BEDTOOLS RNA=$RNA ATAC=$ATAC GTF=$GTF FEATURE_ALIGNED=$FEATURE_ALIGNED N_GPUS=$N_GPUS LAM_BATCH_VALUES=$LAM_BATCH_VALUES STARTUP_STAGGER_SEC=$STARTUP_STAGGER_SEC BETWEEN_RUN_SEC=$BETWEEN_RUN_SEC USE_IONICE=$USE_IONICE ENABLE_FORMAL_EVAL=$ENABLE_FORMAL_EVAL EVAL_N_JOBS=$EVAL_N_JOBS RESUME=$RESUME BATCH_KEY=$BATCH_KEY CELL_TYPE_KEY=$CELL_TYPE_KEY DOMAIN_KEY=$DOMAIN_KEY SHARED_BATCHES=$SHARED_BATCHES SHARED_DIM=$SHARED_DIM PRIVATE_DIM=$PRIVATE_DIM BETA_SHARED=$BETA_SHARED LAM_ISO=$LAM_ISO LAM_ALIGN=$LAM_ALIGN BETA_PRIVATE_RNA=$BETA_PRIVATE_RNA BETA_PRIVATE_ATAC=$BETA_PRIVATE_ATAC ENABLE_SUPPORT=$ENABLE_SUPPORT ALIGN_SUPPORT_K=$ALIGN_SUPPORT_K ALIGN_SUPPORT_STRATEGY=$ALIGN_SUPPORT_STRATEGY ALIGN_SUPPORT_MIN_WEIGHT=$ALIGN_SUPPORT_MIN_WEIGHT NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR MPLCONFIGDIR=$MPLCONFIGDIR OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS bash $SELF_PATH worker $gpu_id"
        tmux send-keys -t "$SESSION:gpu${gpu_id}" "$worker_cmd" Enter
    done

    tmux new-window -t "$SESSION" -n monitor
    monitor_cmd="cd $SCRIPT_DIR && BASE_OUT=$BASE_OUT PREPROCESSED_DIR=$PREPROCESSED_DIR LOG_DIR=$LOG_DIR EVAL_DIR=$EVAL_DIR PYTHON_BIN=$PYTHON_BIN N_GPUS=$N_GPUS LAM_BATCH_VALUES=$LAM_BATCH_VALUES SHARED_DIM=$SHARED_DIM PRIVATE_DIM=$PRIVATE_DIM BETA_SHARED=$BETA_SHARED LAM_ISO=$LAM_ISO LAM_ALIGN=$LAM_ALIGN BETA_PRIVATE_RNA=$BETA_PRIVATE_RNA BETA_PRIVATE_ATAC=$BETA_PRIVATE_ATAC SHARED_BATCHES=$SHARED_BATCHES ENABLE_SUPPORT=$ENABLE_SUPPORT ALIGN_SUPPORT_K=$ALIGN_SUPPORT_K ALIGN_SUPPORT_STRATEGY=$ALIGN_SUPPORT_STRATEGY ALIGN_SUPPORT_MIN_WEIGHT=$ALIGN_SUPPORT_MIN_WEIGHT bash $SELF_PATH monitor"
    tmux send-keys -t "$SESSION:monitor" "$monitor_cmd" Enter

    tmux select-window -t "$SESSION:monitor"
    echo "Started tmux session '$SESSION'"
    echo "Attach with: tmux attach -t $SESSION"
    echo "Outputs: $BASE_OUT"
}

MODE="${1:-launch}"
case "$MODE" in
    launch)
        launch_mode
        ;;
    worker)
        worker_mode "${2:?worker id required}"
        ;;
    monitor)
        monitor_mode
        ;;
    *)
        echo "Usage: $0 [launch|worker <gpu_id>|monitor]" >&2
        exit 1
        ;;
esac
