#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/users/zhutianci/.conda/envs/scMRDR/bin/python}"

SESSION="${SESSION:-bmmc-s11-long}"
N_GPUS="${N_GPUS:-4}"
TRIALS_PER_GPU="${TRIALS_PER_GPU:-80}"
N_TRIALS="${N_TRIALS:-$((N_GPUS * TRIALS_PER_GPU))}"
SEED="${SEED:-42}"
REFRESH_SECS="${REFRESH_SECS:-60}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SEARCH_ROOT="${SEARCH_ROOT:-$SCRIPT_DIR/s11_long_search}"
OUTPUT_DIR="${OUTPUT_DIR:-$SEARCH_ROOT/$RUN_TAG}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$SEARCH_ROOT/preprocessed}"

RNA="${RNA:-$SCRIPT_DIR/s01_preprocessing/RNA_counts_qc_sampled.h5ad}"
ATAC="${ATAC:-$SCRIPT_DIR/s01_preprocessing/ATAC_counts_qc_sampled.h5ad}"
PROT="${PROT:-$SCRIPT_DIR/s01_preprocessing/protein_counts_qc_sampled.h5ad}"
GTF="${GTF:-$SCRIPT_DIR/gencode.v38.chr_patch_hapl_scaff.annotation.gtf}"
PROTEIN_GENE_MAP="${PROTEIN_GENE_MAP:-$SCRIPT_DIR/s01_preprocessing/protein_gene_map.tsv}"
BEDTOOLS="${BEDTOOLS:-/data1/users/zhutianci/.conda/envs/scMRDR/bin/bedtools}"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
PREPROCESS_FILES=(rna_pp.h5ad atac_pp.h5ad prot_pp.h5ad guidance.graphml.gz)

for f in "$RNA" "$ATAC" "$PROT" "$GTF" "$PROTEIN_GENE_MAP" "$BEDTOOLS"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file: $f" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR" "$PREPROCESSED_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists." >&2
    echo "Use a different SESSION, or run: tmux kill-session -t $SESSION" >&2
    exit 1
fi

cat > "$OUTPUT_DIR/run_config.env" <<EOF
SESSION=$SESSION
N_GPUS=$N_GPUS
TRIALS_PER_GPU=$TRIALS_PER_GPU
N_TRIALS=$N_TRIALS
SEED=$SEED
RUN_TAG=$RUN_TAG
OUTPUT_DIR=$OUTPUT_DIR
PREPROCESSED_DIR=$PREPROCESSED_DIR
RNA=$RNA
ATAC=$ATAC
PROT=$PROT
GTF=$GTF
PROTEIN_GENE_MAP=$PROTEIN_GENE_MAP
BEDTOOLS=$BEDTOOLS
PYTHON_BIN=$PYTHON_BIN
NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR
MPLCONFIGDIR=$MPLCONFIGDIR
EOF

PREPROCESS_READY=1
for f in "${PREPROCESS_FILES[@]}"; do
    if [[ ! -f "$PREPROCESSED_DIR/$f" ]]; then
        PREPROCESS_READY=0
        break
    fi
done

if [[ "$PREPROCESS_READY" -eq 1 ]]; then
    if ! env \
        NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" \
        MPLCONFIGDIR="$MPLCONFIGDIR" \
        "$PYTHON_BIN" - <<PY
from pathlib import Path
import anndata as ad
import networkx as nx

pp = Path(r"$PREPROCESSED_DIR")
for name in ("rna_pp.h5ad", "atac_pp.h5ad", "prot_pp.h5ad"):
    adata = ad.read_h5ad(pp / name)
    _ = adata.shape
    _ = adata.X[:1, :1]
nx.read_graphml(pp / "guidance.graphml.gz")
print("preprocessed cache validation passed")
PY
    then
        echo "Detected unreadable cached preprocessing under: $PREPROCESSED_DIR"
        echo "Removing stale cache files and rebuilding."
        rm -f \
          "$PREPROCESSED_DIR/rna_pp.h5ad" \
          "$PREPROCESSED_DIR/atac_pp.h5ad" \
          "$PREPROCESSED_DIR/prot_pp.h5ad" \
          "$PREPROCESSED_DIR/guidance.graphml.gz"
        PREPROCESS_READY=0
    fi
fi

if [[ "$PREPROCESS_READY" -eq 0 ]]; then
    echo "Bootstrapping shared preprocessing into: $PREPROCESSED_DIR"
    env \
      NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" \
      MPLCONFIGDIR="$MPLCONFIGDIR" \
      "$PYTHON_BIN" "$SCRIPT_DIR/s02_glue.py" \
      --rna "$RNA" \
      --atac "$ATAC" \
      --prot "$PROT" \
      --gtf "$GTF" \
      --protein-gene-map "$PROTEIN_GENE_MAP" \
      --bedtools "$BEDTOOLS" \
      --output-dir "$PREPROCESSED_DIR" \
      --preprocess-only \
      2>&1 | tee "$OUTPUT_DIR/preprocess.log"
else
    echo "Reusing existing preprocessing from: $PREPROCESSED_DIR"
fi

tmux new-session -d -s "$SESSION" -n gpu0 -x 240 -y 60

for (( i=0; i<N_GPUS; i++ )); do
    WIN="gpu${i}"
    if [[ "$i" -gt 0 ]]; then
        tmux new-window -t "$SESSION" -n "$WIN"
    fi

    LOG_FILE="$OUTPUT_DIR/gpu${i}.log"
    CMD="cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$i NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR MPLCONFIGDIR=$MPLCONFIGDIR \
$PYTHON_BIN s11_hparam_search.py \
  --rna $RNA \
  --atac $ATAC \
  --prot $PROT \
  --gtf $GTF \
  --protein-gene-map $PROTEIN_GENE_MAP \
  --bedtools $BEDTOOLS \
  --preprocessed-dir $PREPROCESSED_DIR \
  --output-dir $OUTPUT_DIR \
  --batch-key batch \
  --cell-type-key celltype \
  --domain-key domain \
  --n-trials $N_TRIALS \
  --n-gpus $N_GPUS \
  --gpu-id $i \
  --seed $SEED \
  --resume \
  2>&1 | tee $LOG_FILE"
    tmux send-keys -t "$SESSION:$WIN" "$CMD" Enter
done

tmux new-window -t "$SESSION" -n monitor
tmux send-keys -t "$SESSION:monitor" "
START_TS=\$(date +%s)
while true; do
  clear
  NOW_TS=\$(date +%s)
  ELAPSED_H=\$(( (NOW_TS - START_TS) / 3600 ))
  ELAPSED_M=\$(( ((NOW_TS - START_TS) % 3600) / 60 ))
  echo '=== BMMC trimodal s11 long search ==='
  echo \"Session: $SESSION\"
  echo \"Output:  $OUTPUT_DIR\"
  echo \"Elapsed: \${ELAPSED_H}h \${ELAPSED_M}m\"
  echo
  DONE=\$(find $OUTPUT_DIR -name 'metrics.json' 2>/dev/null | wc -l)
  RUNS=\$(find $OUTPUT_DIR -maxdepth 1 -mindepth 1 -type d | wc -l)
  echo \"Completed trials: \$DONE / $N_TRIALS\"
  echo \"Run directories:   \$RUNS\"
  echo
  $PYTHON_BIN - <<'PY'
import json
import pandas as pd
from pathlib import Path

out = Path('$OUTPUT_DIR')
rows = []
for mf in out.glob('*/metrics.json'):
    hf = mf.parent / 'hparams.json'
    if not hf.exists():
        continue
    with mf.open() as f:
        m = json.load(f)
    with hf.open() as f:
        h = json.load(f)
    rows.append({**h, **m, 'run': mf.parent.name})

if not rows:
    print('No completed trials yet.')
else:
    df = pd.DataFrame(rows)
    cols = [
        'run', 'trial_id', 'mode', 'Total',
        'Bio conservation', 'Batch correction', 'Modality integration',
        'shared_dim', 'private_dim', 'lam_align', 'beta_shared', 'lam_iso',
        'beta_private_rna', 'beta_private_atac', 'beta_private_prot',
        'align_support_k', 'align_support_strategy', 'align_support_min_weight',
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values('Total', ascending=False)
    print(df.head(12).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
PY
  echo
  echo 'Recent worker tail:'
  tail -n 8 $OUTPUT_DIR/gpu*.log 2>/dev/null || true
  sleep $REFRESH_SECS
done" Enter

tmux select-window -t "$SESSION:gpu0"

echo "Started tmux session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"
echo "Logs and runs are under: $OUTPUT_DIR"
echo
echo "Recommended baseline for >10h:"
echo "  SESSION=$SESSION TRIALS_PER_GPU=$TRIALS_PER_GPU N_GPUS=$N_GPUS $0"
echo
echo "To resume the same search later, reuse:"
echo "  SESSION=<new_session> OUTPUT_DIR=$OUTPUT_DIR PREPROCESSED_DIR=$PREPROCESSED_DIR N_TRIALS=$N_TRIALS N_GPUS=$N_GPUS SEED=$SEED $0"
