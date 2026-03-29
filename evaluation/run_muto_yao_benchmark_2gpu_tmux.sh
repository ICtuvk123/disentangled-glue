#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_BASE="$SCRIPT_DIR/config/config.yaml"
GENERATED_DIR="${GENERATED_DIR:-$SCRIPT_DIR/config/generated}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/muto_yao_benchmark/$RUN_TAG}"
SESSION_NAME="${SESSION_NAME:-muto-yao-benchmark}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SNAKEMAKE_BIN="${SNAKEMAKE_BIN:-snakemake}"
PROFILE="${PROFILE:-profiles/local}"
USE_PROFILE="${USE_PROFILE:-0}"
BENCHMARK_METHODS="${BENCHMARK_METHODS:-GLUE}"
NOLOCK="${NOLOCK:-1}"
GLUE_SEED_COUNT="${GLUE_SEED_COUNT:-1}"
GLUE_DIM="${GLUE_DIM:-}"
GLUE_ALT_DIM="${GLUE_ALT_DIM:-}"
GLUE_HIDDEN_DEPTH="${GLUE_HIDDEN_DEPTH:-}"
GLUE_HIDDEN_DIM="${GLUE_HIDDEN_DIM:-}"
GLUE_DROPOUT="${GLUE_DROPOUT:-}"
GLUE_LAM_GRAPH="${GLUE_LAM_GRAPH:-}"
GLUE_LAM_ALIGN="${GLUE_LAM_ALIGN:-}"
GLUE_NEG_SAMPLES="${GLUE_NEG_SAMPLES:-}"

MUTO_GPU="${MUTO_GPU:-0}"
YAO_GPU="${YAO_GPU:-1}"
MUTO_JOBS="${MUTO_JOBS:-8}"
YAO_JOBS="${YAO_JOBS:-8}"
SUMMARY_JOBS="${SUMMARY_JOBS:-2}"

NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

slugify_dataset() {
    case "$1" in
        Muto-2021) echo "muto_2021" ;;
        Yao-2021) echo "yao_2021" ;;
        *)
            echo "Unsupported dataset: $1" >&2
            return 1
            ;;
    esac
}

dataset_config_path() {
    local slug
    slug="$(slugify_dataset "$1")"
    echo "$GENERATED_DIR/${slug}.benchmark.yaml"
}

dataset_targets_path() {
    local slug
    slug="$(slugify_dataset "$1")"
    echo "$GENERATED_DIR/${slug}.benchmark.targets.txt"
}

done_flag_path() {
    local slug
    slug="$(slugify_dataset "$1")"
    echo "$LOG_DIR/${slug}.done"
}

failed_flag_path() {
    local slug
    slug="$(slugify_dataset "$1")"
    echo "$LOG_DIR/${slug}.failed"
}

dataset_log_path() {
    local slug
    slug="$(slugify_dataset "$1")"
    echo "$LOG_DIR/${slug}.log"
}

combined_config_path() {
    echo "$GENERATED_DIR/muto_yao.benchmark.yaml"
}

write_generated_files() {
    mkdir -p "$GENERATED_DIR" "$LOG_DIR"

    BENCHMARK_METHODS="$BENCHMARK_METHODS" \
    GLUE_SEED_COUNT="$GLUE_SEED_COUNT" \
    GLUE_DIM="$GLUE_DIM" \
    GLUE_ALT_DIM="$GLUE_ALT_DIM" \
    GLUE_HIDDEN_DEPTH="$GLUE_HIDDEN_DEPTH" \
    GLUE_HIDDEN_DIM="$GLUE_HIDDEN_DIM" \
    GLUE_DROPOUT="$GLUE_DROPOUT" \
    GLUE_LAM_GRAPH="$GLUE_LAM_GRAPH" \
    GLUE_LAM_ALIGN="$GLUE_LAM_ALIGN" \
    GLUE_NEG_SAMPLES="$GLUE_NEG_SAMPLES" \
    "$PYTHON_BIN" - <<'PY' "$CONFIG_BASE" "$GENERATED_DIR"
import importlib.util
import os
import pathlib
import sys
import yaml

config_base = pathlib.Path(sys.argv[1])
generated_dir = pathlib.Path(sys.argv[2])
generated_dir.mkdir(parents=True, exist_ok=True)

base_cfg = yaml.safe_load(config_base.read_text())
datasets_keep = ["Muto-2021", "Yao-2021"]
methods_keep = [item.strip() for item in os.environ["BENCHMARK_METHODS"].split(",") if item.strip()]
if not methods_keep:
    raise ValueError("BENCHMARK_METHODS must not be empty")
glue_seed_count = int(os.environ["GLUE_SEED_COUNT"])

workflow_utils_path = config_base.parent.parent / "workflow" / "utils.py"
spec = importlib.util.spec_from_file_location("workflow_utils", workflow_utils_path)
workflow_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow_utils)

def benchmark_cfg(dataset_names):
    cfg = yaml.safe_load(config_base.read_text())
    cfg["use"] = ["benchmark"]
    cfg["dataset"] = {
        key: value for key, value in cfg["dataset"].items()
        if key in dataset_names
    }
    cfg["benchmark"]["seed"] = glue_seed_count
    cfg["benchmark"]["method"] = {
        key: value for key, value in cfg["benchmark"]["method"].items()
        if key in methods_keep
    }
    missing_methods = [item for item in methods_keep if item not in cfg["benchmark"]["method"]]
    if missing_methods:
        raise KeyError(f"Unknown benchmark methods: {missing_methods}")
    if "GLUE" in cfg["benchmark"]["method"]:
        glue_cfg = cfg["benchmark"]["method"]["GLUE"] or {}
        overrides = {
            "dim": os.environ["GLUE_DIM"],
            "alt_dim": os.environ["GLUE_ALT_DIM"],
            "hidden_depth": os.environ["GLUE_HIDDEN_DEPTH"],
            "hidden_dim": os.environ["GLUE_HIDDEN_DIM"],
            "dropout": os.environ["GLUE_DROPOUT"],
            "lam_graph": os.environ["GLUE_LAM_GRAPH"],
            "lam_align": os.environ["GLUE_LAM_ALIGN"],
            "neg_samples": os.environ["GLUE_NEG_SAMPLES"],
        }
        casters = {
            "dim": int,
            "alt_dim": int,
            "hidden_depth": int,
            "hidden_dim": int,
            "dropout": float,
            "lam_graph": float,
            "lam_align": float,
            "neg_samples": int,
        }
        for key, val in overrides.items():
            if val != "":
                glue_cfg[key] = casters[key](val)
        cfg["benchmark"]["method"]["GLUE"] = glue_cfg
    return cfg

def write_cfg(name, cfg):
    out = generated_dir / name
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out

def write_targets(name, cfg):
    merged = {
        "dataset": cfg["dataset"],
        "timeout": cfg["timeout"],
        **cfg["benchmark"]
    }
    targets = workflow_utils.target_files(
        workflow_utils.target_directories(merged)
    )
    out = generated_dir / name
    out.write_text("\n".join(str(item) for item in targets) + "\n")
    return out

combined = benchmark_cfg(datasets_keep)
write_cfg("muto_yao.benchmark.yaml", combined)

for dataset in datasets_keep:
    slug = dataset.lower().replace("-", "_")
    cfg = benchmark_cfg([dataset])
    write_cfg(f"{slug}.benchmark.yaml", cfg)
    write_targets(f"{slug}.benchmark.targets.txt", cfg)
PY
}

run_worker() {
    local dataset="$1"
    local gpu="$2"
    local jobs="$3"
    local cfg target_file log_file done_file failed_file

    cfg="$(dataset_config_path "$dataset")"
    target_file="$(dataset_targets_path "$dataset")"
    log_file="$(dataset_log_path "$dataset")"
    done_file="$(done_flag_path "$dataset")"
    failed_file="$(failed_flag_path "$dataset")"

    rm -f "$done_file" "$failed_file"
    mkdir -p "$LOG_DIR"
    exec > >(tee -a "$log_file") 2>&1

    echo "Dataset: $dataset"
    echo "GPU: $gpu"
    echo "Jobs: $jobs"
    echo "Config: $cfg"
    echo "Targets: $target_file"
    echo

    mapfile -t TARGETS < "$target_file"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export NUMBA_CACHE_DIR MPLCONFIGDIR XDG_CACHE_HOME
    mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

    local -a snakemake_args
    snakemake_args=(-j "$jobs" --resources gpu=1)
    if [[ "$NOLOCK" == "1" ]]; then
        snakemake_args+=(--nolock)
    fi
    if [[ "$USE_PROFILE" == "1" ]]; then
        snakemake_args+=(--profile "$PROFILE")
    else
        snakemake_args+=(--printshellcmds --keep-going)
    fi

    if "$SNAKEMAKE_BIN" "${snakemake_args[@]}" "${TARGETS[@]}" --configfile "$cfg"; then
        touch "$done_file"
        echo
        echo "Finished dataset worker: $dataset"
    else
        touch "$failed_file"
        echo
        echo "Dataset worker failed: $dataset" >&2
        exit 1
    fi
}

run_summary() {
    local combined_cfg summary_log
    combined_cfg="$(combined_config_path)"
    summary_log="$LOG_DIR/summary.log"

    mkdir -p "$LOG_DIR"
    exec > >(tee -a "$summary_log") 2>&1
    export NUMBA_CACHE_DIR MPLCONFIGDIR XDG_CACHE_HOME
    mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

    echo "Waiting for dataset workers to finish..."
    while true; do
        if [[ -f "$(failed_flag_path "Muto-2021")" || -f "$(failed_flag_path "Yao-2021")" ]]; then
            echo "Detected failed worker. Summary aborted." >&2
            exit 1
        fi
        if [[ -f "$(done_flag_path "Muto-2021")" && -f "$(done_flag_path "Yao-2021")" ]]; then
            break
        fi
        date
        echo "  Muto done: $([[ -f "$(done_flag_path "Muto-2021")" ]] && echo yes || echo no)"
        echo "  Yao  done: $([[ -f "$(done_flag_path "Yao-2021")" ]] && echo yes || echo no)"
        sleep 30
    done

    echo
    echo "Both dataset workers finished. Running combined benchmark summary..."
    local -a snakemake_args
    snakemake_args=(-j "$SUMMARY_JOBS")
    if [[ "$NOLOCK" == "1" ]]; then
        snakemake_args+=(--nolock)
    fi
    if [[ "$USE_PROFILE" == "1" ]]; then
        snakemake_args+=(--profile "$PROFILE")
    else
        snakemake_args+=(--printshellcmds --keep-going)
    fi
    "$SNAKEMAKE_BIN" "${snakemake_args[@]}" --configfile "$combined_cfg"
    echo
    echo "Combined summary finished."
    echo "Main table: $SCRIPT_DIR/results/benchmark.csv"
}

launch_tmux() {
    write_generated_files

    need_cmd tmux
    need_cmd "$PYTHON_BIN"
    need_cmd "$SNAKEMAKE_BIN"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists" >&2
        echo "Use a different SESSION_NAME or kill the existing session first." >&2
        exit 1
    fi

    tmux new-session -d -s "$SESSION_NAME" -x 240 -y 60 -n muto \
        "cd '$SCRIPT_DIR' && LOG_DIR='$LOG_DIR' GENERATED_DIR='$GENERATED_DIR' PYTHON_BIN='$PYTHON_BIN' SNAKEMAKE_BIN='$SNAKEMAKE_BIN' PROFILE='$PROFILE' USE_PROFILE='$USE_PROFILE' BENCHMARK_METHODS='$BENCHMARK_METHODS' NOLOCK='$NOLOCK' GLUE_SEED_COUNT='$GLUE_SEED_COUNT' GLUE_DIM='$GLUE_DIM' GLUE_ALT_DIM='$GLUE_ALT_DIM' GLUE_HIDDEN_DEPTH='$GLUE_HIDDEN_DEPTH' GLUE_HIDDEN_DIM='$GLUE_HIDDEN_DIM' GLUE_DROPOUT='$GLUE_DROPOUT' GLUE_LAM_GRAPH='$GLUE_LAM_GRAPH' GLUE_LAM_ALIGN='$GLUE_LAM_ALIGN' GLUE_NEG_SAMPLES='$GLUE_NEG_SAMPLES' NUMBA_CACHE_DIR='$NUMBA_CACHE_DIR' MPLCONFIGDIR='$MPLCONFIGDIR' XDG_CACHE_HOME='$XDG_CACHE_HOME' bash '$0' worker 'Muto-2021' '$MUTO_GPU' '$MUTO_JOBS'"

    tmux new-window -t "$SESSION_NAME" -n yao \
        "cd '$SCRIPT_DIR' && LOG_DIR='$LOG_DIR' GENERATED_DIR='$GENERATED_DIR' PYTHON_BIN='$PYTHON_BIN' SNAKEMAKE_BIN='$SNAKEMAKE_BIN' PROFILE='$PROFILE' USE_PROFILE='$USE_PROFILE' BENCHMARK_METHODS='$BENCHMARK_METHODS' NOLOCK='$NOLOCK' GLUE_SEED_COUNT='$GLUE_SEED_COUNT' GLUE_DIM='$GLUE_DIM' GLUE_ALT_DIM='$GLUE_ALT_DIM' GLUE_HIDDEN_DEPTH='$GLUE_HIDDEN_DEPTH' GLUE_HIDDEN_DIM='$GLUE_HIDDEN_DIM' GLUE_DROPOUT='$GLUE_DROPOUT' GLUE_LAM_GRAPH='$GLUE_LAM_GRAPH' GLUE_LAM_ALIGN='$GLUE_LAM_ALIGN' GLUE_NEG_SAMPLES='$GLUE_NEG_SAMPLES' NUMBA_CACHE_DIR='$NUMBA_CACHE_DIR' MPLCONFIGDIR='$MPLCONFIGDIR' XDG_CACHE_HOME='$XDG_CACHE_HOME' bash '$0' worker 'Yao-2021' '$YAO_GPU' '$YAO_JOBS'"

    tmux new-window -t "$SESSION_NAME" -n summary \
        "cd '$SCRIPT_DIR' && LOG_DIR='$LOG_DIR' GENERATED_DIR='$GENERATED_DIR' PYTHON_BIN='$PYTHON_BIN' SNAKEMAKE_BIN='$SNAKEMAKE_BIN' PROFILE='$PROFILE' USE_PROFILE='$USE_PROFILE' BENCHMARK_METHODS='$BENCHMARK_METHODS' NOLOCK='$NOLOCK' GLUE_SEED_COUNT='$GLUE_SEED_COUNT' GLUE_DIM='$GLUE_DIM' GLUE_ALT_DIM='$GLUE_ALT_DIM' GLUE_HIDDEN_DEPTH='$GLUE_HIDDEN_DEPTH' GLUE_HIDDEN_DIM='$GLUE_HIDDEN_DIM' GLUE_DROPOUT='$GLUE_DROPOUT' GLUE_LAM_GRAPH='$GLUE_LAM_GRAPH' GLUE_LAM_ALIGN='$GLUE_LAM_ALIGN' GLUE_NEG_SAMPLES='$GLUE_NEG_SAMPLES' NUMBA_CACHE_DIR='$NUMBA_CACHE_DIR' MPLCONFIGDIR='$MPLCONFIGDIR' XDG_CACHE_HOME='$XDG_CACHE_HOME' SUMMARY_JOBS='$SUMMARY_JOBS' bash '$0' summary"

    echo "Started tmux session '$SESSION_NAME'."
    echo "Windows: muto, yao, summary"
    echo "Log dir: $LOG_DIR"
    echo "GPUs: Muto->$MUTO_GPU, Yao->$YAO_GPU"
    echo "Attach with: tmux attach -t $SESSION_NAME"
}

usage() {
    cat <<'EOF'
Usage:
  run_muto_yao_benchmark_2gpu_tmux.sh launch
  run_muto_yao_benchmark_2gpu_tmux.sh worker <dataset> <gpu_id> <jobs>
  run_muto_yao_benchmark_2gpu_tmux.sh summary

Environment overrides:
  SESSION_NAME, LOG_DIR, GENERATED_DIR, PYTHON_BIN, SNAKEMAKE_BIN, PROFILE
  USE_PROFILE, NOLOCK, NUMBA_CACHE_DIR, MPLCONFIGDIR, XDG_CACHE_HOME
  BENCHMARK_METHODS
  GLUE_SEED_COUNT
  GLUE_DIM, GLUE_ALT_DIM, GLUE_HIDDEN_DEPTH, GLUE_HIDDEN_DIM
  GLUE_DROPOUT, GLUE_LAM_GRAPH, GLUE_LAM_ALIGN, GLUE_NEG_SAMPLES
  MUTO_GPU, YAO_GPU, MUTO_JOBS, YAO_JOBS, SUMMARY_JOBS
EOF
}

main() {
    local mode="${1:-launch}"
    case "$mode" in
        -h|--help|help)
            usage
            ;;
        launch)
            launch_tmux
            ;;
        worker)
            write_generated_files
            run_worker "$2" "$3" "$4"
            ;;
        summary)
            write_generated_files
            run_summary
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
