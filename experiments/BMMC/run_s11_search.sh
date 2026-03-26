#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/users/zhutianci/.conda/envs/scMRDR/bin/python}"
SESSION="${SESSION:-bmmc-s11-search}"
N_GPUS="${N_GPUS:-4}"
N_TRIALS="${N_TRIALS:-200}"
SEED="${SEED:-42}"

RNA="$SCRIPT_DIR/s01_preprocessing/RNA_counts_qc_sampled.h5ad"
ATAC="$SCRIPT_DIR/s01_preprocessing/ATAC_counts_qc_sampled.h5ad"
PROT="$SCRIPT_DIR/s01_preprocessing/protein_counts_qc_sampled.h5ad"
GTF="$SCRIPT_DIR/gencode.v38.chr_patch_hapl_scaff.annotation.gtf"
PROTEIN_GENE_MAP="$SCRIPT_DIR/s01_preprocessing/protein_gene_map.tsv"
PREPROCESSED_DIR="$SCRIPT_DIR/s06_sweep/preprocessed"
OUTPUT_DIR="$SCRIPT_DIR/s11_hparam_search"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

# Sanity checks
for f in "$RNA" "$ATAC" "$PROT" "$GTF" "$PROTEIN_GENE_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file: $f" >&2
        exit 1
    fi
done
if [[ ! -f "$PREPROCESSED_DIR/guidance.graphml.gz" ]]; then
    echo "ERROR: preprocessed dir not found or incomplete: $PREPROCESSED_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Kill old session if exists
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

# One window per GPU
for (( i=0; i<N_GPUS; i++ )); do
    WIN="gpu${i}"
    tmux new-window -t "$SESSION" -n "$WIN"
    CMD="cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$i NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR MPLCONFIGDIR=$MPLCONFIGDIR \
$PYTHON_BIN s11_hparam_search.py \
  --rna  $RNA \
  --atac $ATAC \
  --prot $PROT \
  --gtf  $GTF \
  --protein-gene-map $PROTEIN_GENE_MAP \
  --preprocessed-dir $PREPROCESSED_DIR \
  --output-dir $OUTPUT_DIR \
  --batch-key batch --cell-type-key celltype --domain-key domain \
  --n-trials $N_TRIALS --n-gpus $N_GPUS --gpu-id $i \
  --seed $SEED --resume \
  2>&1 | tee $OUTPUT_DIR/gpu${i}.log"
    tmux send-keys -t "$SESSION:$WIN" "$CMD" Enter
done

# Summary monitor window
tmux new-window -t "$SESSION" -n monitor
tmux send-keys -t "$SESSION:monitor" "
while true; do
  clear
  echo '=== s11 hparam search ==='
  echo \"Output: $OUTPUT_DIR\"
  echo
  DONE=\$(find $OUTPUT_DIR -name 'metrics.json' 2>/dev/null | wc -l)
  echo \"Completed trials: \$DONE / $N_TRIALS\"
  echo
  $PYTHON_BIN - <<'PY'
import json, pandas as pd
from pathlib import Path
out = Path('$OUTPUT_DIR')
rows = []
for mf in out.glob('*/metrics.json'):
    hf = mf.parent / 'hparams.json'
    if not hf.exists(): continue
    with mf.open() as f: m = json.load(f)
    with hf.open() as f: h = json.load(f)
    rows.append({**h, **m})
if not rows:
    print('No completed trials yet.')
else:
    df = pd.DataFrame(rows)
    cols = ['trial_id','mode','Total','Bio conservation','Batch correction','Modality integration',
            'shared_dim','private_dim','lam_align','beta_shared','lam_iso',
            'beta_private_rna','beta_private_atac','beta_private_prot']
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values('Total', ascending=False)
    print(df.head(10).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
PY
  sleep 60
done" Enter

# Select first GPU window
tmux select-window -t "$SESSION:gpu0"

echo "Started tmux session '$SESSION' with $N_GPUS GPU workers."
echo "Attach: tmux attach -t $SESSION"
echo "Monitor window shows top results, refreshes every 60s."
