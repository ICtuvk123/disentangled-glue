#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SESSION_NAME="${SESSION_NAME:-bmmc-preserved-full}"
PYTHON_BIN="${PYTHON_BIN:-/data1/users/zhutianci/.conda/envs/scMRDR/bin/python}"

MULTIOME="${MULTIOME:-$SCRIPT_DIR/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad}"
CITE="${CITE:-$SCRIPT_DIR/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad}"
GTF="${GTF:-$SCRIPT_DIR/gencode.v38.chr_patch_hapl_scaff.annotation.gtf}"
HGNC="${HGNC:-$SCRIPT_DIR/hgnc_complete_set.txt}"

PREP_DIR="${PREP_DIR:-$SCRIPT_DIR/s01_preprocessing_full}"
PP_DIR="${PP_DIR:-$SCRIPT_DIR/s10_preserved_full_runs/preprocessed}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/s10_preserved_full_runs/models}"
EVAL_ROOT="${EVAL_ROOT:-$SCRIPT_DIR/s10_preserved_full_runs/eval_true_scmrdr}"
SUMMARY_TSV="${SUMMARY_TSV:-$SCRIPT_DIR/s10_preserved_full_runs/summary.tsv}"

NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
N_JOBS="${N_JOBS:-8}"

SLOT1_GPU="${SLOT1_GPU:-0}"
SLOT2_GPU="${SLOT2_GPU:-1}"
SLOT4_GPU="${SLOT4_GPU:-2}"

export NUMBA_CACHE_DIR MPLCONFIGDIR

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

need_cmd tmux
"$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import networkx  # noqa: F401
PY

mkdir -p "$RUN_ROOT" "$EVAL_ROOT" "$PP_DIR"

echo "== Step 1/3: ensure full preprocessing =="
if [[ ! -f "$PREP_DIR/feature_aligned.h5ad" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/s01_preprocessing.py" \
        --multiome "$MULTIOME" \
        --cite "$CITE" \
        --gtf "$GTF" \
        --hgnc "$HGNC" \
        --output-dir "$PREP_DIR"
else
    echo "Reusing full preprocessing from $PREP_DIR"
fi

echo "== Step 2/3: validate shared preprocessing =="
pp_valid=0
if [[ -f "$PP_DIR/rna_pp.h5ad" && -f "$PP_DIR/atac_pp.h5ad" && -f "$PP_DIR/prot_pp.h5ad" && -f "$PP_DIR/guidance.graphml.gz" ]]; then
    if "$PYTHON_BIN" - <<PY
from pathlib import Path
import networkx as nx
pp = Path(r"$PP_DIR")
nx.read_graphml(pp / "guidance.graphml.gz")
print("Shared preprocessing is valid.")
PY
    then
        pp_valid=1
    else
        echo "Shared preprocessing is corrupted. Removing cached files under $PP_DIR"
        rm -f "$PP_DIR/rna_pp.h5ad" "$PP_DIR/atac_pp.h5ad" "$PP_DIR/prot_pp.h5ad" "$PP_DIR/guidance.graphml.gz"
    fi
fi

if [[ "$pp_valid" != "1" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/s02_glue.py" \
        --model disentangled \
        --rna "$PREP_DIR/RNA_counts_qc.h5ad" \
        --atac "$PREP_DIR/ATAC_counts_qc.h5ad" \
        --prot "$PREP_DIR/protein_counts_qc.h5ad" \
        --gtf "$GTF" \
        --protein-gene-map "$PREP_DIR/protein_gene_map.tsv" \
        --batch-key batch \
        --output-dir "$PP_DIR" \
        --preprocess-only
fi

echo "== Step 3/3: launch tmux workers =="
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -x 240 -y 60 -n summary

tmux send-keys -t "$SESSION_NAME:summary" "cd $SCRIPT_DIR && bash -lc '
while true; do
  clear
  echo \"Session: $SESSION_NAME\"
  echo \"Run root: $RUN_ROOT\"
  echo \"Eval root: $EVAL_ROOT\"
  echo
  $PYTHON_BIN - <<\"PY\"
from pathlib import Path
import csv
import pandas as pd

run_root = Path(\"$RUN_ROOT\")
eval_root = Path(\"$EVAL_ROOT\")

print(\"Training runs:\")
run_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()]) if run_root.exists() else []
if not run_dirs:
    print(\"  none\")
else:
    for path in run_dirs:
        ready = (path / \"combined_glue.h5ad\").exists()
        status = \"ready\" if ready else \"running\"
        print(\"  {:>7}  {}\".format(status, path.name))

print()
print(\"Evaluation results:\")
rows = []
for csv_path in sorted(eval_root.glob(\"*/*_unscaled.csv\")):
    with csv_path.open(\"r\", encoding=\"utf-8\", newline=\"\") as fh:
        row = next(csv.DictReader(fh))
    rows.append({
        \"tag\": csv_path.parent.name,
        \"Total\": float(row[\"Total\"]),
        \"Bio\": float(row[\"Bio conservation\"]),
        \"Batch\": float(row[\"Batch correction\"]),
        \"Modality\": float(row[\"Modality integration\"]),
    })
if not rows:
    print(\"  none\")
else:
    df = pd.DataFrame(rows).sort_values(\"Total\", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f\"{x:.4f}\"))
PY
  sleep 30
done'" Enter

tmux new-window -t "$SESSION_NAME" -n slot1-gpu${SLOT1_GPU}
tmux send-keys -t "$SESSION_NAME:slot1-gpu${SLOT1_GPU}" "cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$SLOT1_GPU NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR MPLCONFIGDIR=$MPLCONFIGDIR $PYTHON_BIN s10_try_preserved_full_models.py --slots 1 --prep-dir $PREP_DIR --pp-dir $PP_DIR --run-root $RUN_ROOT --eval-root $EVAL_ROOT --summary-tsv $RUN_ROOT/slot_1.summary.tsv --multiome $MULTIOME --cite $CITE --gtf $GTF --hgnc $HGNC --python-bin $PYTHON_BIN --batch-key batch --cell-type-key celltype --domain-key domain --n-jobs $N_JOBS 2>&1 | tee $RUN_ROOT/slot1_gpu${SLOT1_GPU}.log" Enter

tmux new-window -t "$SESSION_NAME" -n slot2-gpu${SLOT2_GPU}
tmux send-keys -t "$SESSION_NAME:slot2-gpu${SLOT2_GPU}" "cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$SLOT2_GPU NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR MPLCONFIGDIR=$MPLCONFIGDIR $PYTHON_BIN s10_try_preserved_full_models.py --slots 2 --prep-dir $PREP_DIR --pp-dir $PP_DIR --run-root $RUN_ROOT --eval-root $EVAL_ROOT --summary-tsv $RUN_ROOT/slot_2.summary.tsv --multiome $MULTIOME --cite $CITE --gtf $GTF --hgnc $HGNC --python-bin $PYTHON_BIN --batch-key batch --cell-type-key celltype --domain-key domain --n-jobs $N_JOBS 2>&1 | tee $RUN_ROOT/slot2_gpu${SLOT2_GPU}.log" Enter

tmux new-window -t "$SESSION_NAME" -n slot4-gpu${SLOT4_GPU}
tmux send-keys -t "$SESSION_NAME:slot4-gpu${SLOT4_GPU}" "cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$SLOT4_GPU NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR MPLCONFIGDIR=$MPLCONFIGDIR $PYTHON_BIN s10_try_preserved_full_models.py --slots 4 --prep-dir $PREP_DIR --pp-dir $PP_DIR --run-root $RUN_ROOT --eval-root $EVAL_ROOT --summary-tsv $RUN_ROOT/slot_4.summary.tsv --multiome $MULTIOME --cite $CITE --gtf $GTF --hgnc $HGNC --python-bin $PYTHON_BIN --batch-key batch --cell-type-key celltype --domain-key domain --n-jobs $N_JOBS 2>&1 | tee $RUN_ROOT/slot4_gpu${SLOT4_GPU}.log" Enter

echo "Started tmux session '$SESSION_NAME'."
echo "Windows: summary, slot1-gpu${SLOT1_GPU}, slot2-gpu${SLOT2_GPU}, slot4-gpu${SLOT4_GPU}"
echo "GPUs in use: ${SLOT1_GPU}, ${SLOT2_GPU}, ${SLOT4_GPU}"
echo "Attach with: tmux attach -t $SESSION_NAME"
