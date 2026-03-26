#!/bin/bash
set -euo pipefail

BASE=/data1/users/zhutianci/proj/disentangled-glue/experiments/BMMC
SESSION="${SESSION:-s07}"
OUTDIR="${OUTDIR:-s07_support_search}"
EXTRA_ARGS="$*"
PYTHON_BIN="${PYTHON_BIN:-/data1/users/zhutianci/.conda/envs/scMRDR/bin/python}"

COMMON="--rna s01_preprocessing/RNA_counts_qc_sampled.h5ad \
        --atac s01_preprocessing/ATAC_counts_qc_sampled.h5ad \
        --prot s01_preprocessing/protein_counts_qc_sampled.h5ad \
        --gtf gencode.v38.chr_patch_hapl_scaff.annotation.gtf \
        --protein-gene-map s01_preprocessing/protein_gene_map.tsv \
        --source-run s06_sweep/run_023 \
        --output-dir ${OUTDIR} \
        --n-gpus 4 \
        --shared-dims 24 30 36 \
        --private-dims 6 8 12 \
        --beta-shared 1.0 1.25 \
        --lam-iso 1.0 \
        --lam-align 0.03 0.05 \
        --align-support-k 15 \
        --align-support-strategy soft \
        --align-support-min-weight 0.05"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 240 -y 60 -n summary
tmux send-keys -t "$SESSION:summary" "cd $BASE && bash -lc '
while true; do
  clear
  echo \"Output: $OUTDIR\"
  echo \"Python: $PYTHON_BIN\"
  echo
  $PYTHON_BIN - <<\"PY\"
from pathlib import Path
import json
import pandas as pd

outdir = Path(\"$BASE/$OUTDIR\")
rows = []
for metrics_file in sorted(outdir.glob(\"*/metrics.json\")):
    run_dir = metrics_file.parent
    hparams_file = run_dir / \"hparams.json\"
    if not hparams_file.exists():
        continue
    with hparams_file.open() as fh:
        hp = json.load(fh)
    with metrics_file.open() as fh:
        metrics = json.load(fh)
    row = {**hp, **metrics, \"run_name\": run_dir.name}
    rows.append(row)

if not rows:
    print(\"No completed runs yet.\")
else:
    df = pd.DataFrame(rows)
    cols = [
        \"run_name\", \"mode\", \"shared_dim\", \"private_dim\", \"beta_shared\",
        \"lam_iso\", \"lam_align\", \"Total\", \"Bio conservation\",
        \"Batch correction\", \"Modality integration\"
    ]
    present = [c for c in cols if c in df.columns]
    df = df[present].sort_values(\"Total\", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f\"{x:.4f}\"))
    print()
    if {\"mode\", \"Total\"}.issubset(df.columns):
        best = df.groupby(\"mode\", as_index=False).first()
        print(\"Best per mode:\")
        print(best[[c for c in present if c in best.columns]].to_string(index=False, float_format=lambda x: f\"{x:.4f}\"))
PY
  sleep 30
done'" Enter

for GPU_ID in 0 1 2 3; do
    tmux new-window -t "$SESSION" -n "gpu${GPU_ID}"
    tmux send-keys -t "$SESSION:gpu${GPU_ID}" "cd $BASE && CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_BIN s07_support_vs_baseline_search.py $COMMON --gpu-id ${GPU_ID} --resume ${EXTRA_ARGS} 2>&1 | tee ${OUTDIR}_gpu${GPU_ID}.log" Enter
done

echo "Started matched baseline-vs-support sweep in tmux session '$SESSION'."
echo "Grid per mode: 3 x 3 x 2 x 1 x 2 = 36 configs"
echo "Modes: baseline + support = 72 total runs"
echo "Python: $PYTHON_BIN"
echo "Attach with: tmux attach -t $SESSION"
echo "Windows: summary, gpu0, gpu1, gpu2, gpu3"
