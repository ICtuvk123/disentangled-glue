#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/users/zhutianci/proj/disentangled-glue"
EVAL_DIR="$ROOT/eval"
BMMC_DIR="$ROOT/experiments/BMMC"
PYTHON_BIN="/data1/users/zhutianci/.conda/envs/scMRDR/bin/python"

SESSION="${SESSION:-bmmc-t0088-origcode-rerun}"
GPU="${GPU:-2}"
N_JOBS="${N_JOBS:-4}"

RUN_NAME="baseline_t0088_sd48_pd8_bs0.75_li2.0_la0.3_bpr0.25_bpa1.0_bpp1.0"
OUT_ROOT="$EVAL_DIR/outputs/bmmc_origcode_rerun"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
PP_SRC="$BMMC_DIR/s11_long_search/preprocessed"
PP_COMPAT="$OUT_ROOT/preprocessed_from_s11_long_search_compat"
BENCH_DIR="$RUN_DIR/benchmarker_eval"

LOG_DIR="$EVAL_DIR/logs"
PREP_LOG="$LOG_DIR/bmmc_t0088_origcode_prepare.log"
TRAIN_LOG="$LOG_DIR/bmmc_t0088_origcode_train.log"
EVAL_LOG="$LOG_DIR/bmmc_t0088_origcode_eval.log"

mkdir -p "$LOG_DIR" "$OUT_ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

CMD=$(cat <<EOF
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export NUMBA_CACHE_DIR=/tmp/numba-cache
export MPLCONFIGDIR=/tmp/mplconfig

echo "[prepare] \$(date '+%F %T')" | tee "$PREP_LOG"
"$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$PREP_LOG"
from pathlib import Path
import json
import shutil
import h5py

src = Path("$PP_SRC")
dst = Path("$PP_COMPAT")
dst.mkdir(parents=True, exist_ok=True)
for name in ("rna_pp.h5ad", "atac_pp.h5ad", "guidance.graphml.gz"):
    src_path = src / name
    dst_path = dst / name
    if not dst_path.exists():
        dst_path.symlink_to(src_path)
prot_src = src / "prot_pp.h5ad"
prot_dst = dst / "prot_pp.h5ad"
if not prot_dst.exists():
    shutil.copy2(prot_src, prot_dst)
    with h5py.File(prot_dst, "r+") as f:
        if "/uns/pca/params/mask_var" in f:
            del f["/uns/pca/params/mask_var"]
(dst / "compat_info.json").write_text(json.dumps({
    "source_preprocessed_dir": str(src),
    "patch": "Removed /uns/pca/params/mask_var with encoding-type=null from prot_pp.h5ad",
}, indent=2))
print(dst)
PY

echo "[train] \$(date '+%F %T')" | tee "$TRAIN_LOG"
"$PYTHON_BIN" "$BMMC_DIR/s02_glue.py" \
  --model disentangled \
  --rna "$BMMC_DIR/s01_preprocessing/RNA_counts_qc_sampled.h5ad" \
  --atac "$BMMC_DIR/s01_preprocessing/ATAC_counts_qc_sampled.h5ad" \
  --prot "$BMMC_DIR/s01_preprocessing/protein_counts_qc_sampled.h5ad" \
  --gtf "$BMMC_DIR/gencode.v38.chr_patch_hapl_scaff.annotation.gtf" \
  --preprocessed-dir "$PP_COMPAT" \
  --output-dir "$RUN_DIR" \
  --shared-dim 48 \
  --private-dim 8 \
  --beta-shared 0.75 \
  --lam-iso 2.0 \
  --lam-align 0.3 \
  --beta-private-rna 0.25 \
  --beta-private-atac 1.0 \
  --beta-private-prot 1.0 \
  --batch-key batch 2>&1 | tee -a "$TRAIN_LOG"

echo "[eval] \$(date '+%F %T')" | tee "$EVAL_LOG"
"$PYTHON_BIN" "$EVAL_DIR/eval_bmmc_like_muto_yao.py" \
  --run-dir "$RUN_DIR" \
  --feature-aligned "$BMMC_DIR/s01_preprocessing/feature_aligned_sampled.h5ad" \
  --output-dir "$BENCH_DIR" \
  --tag "$RUN_NAME" \
  --n-jobs "$N_JOBS" 2>&1 | tee -a "$EVAL_LOG"

echo "[done] \$(date '+%F %T')" | tee -a "$EVAL_LOG"
EOF
)

tmux new-session -d -s "$SESSION" -n bmmc "$CMD"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Prepare log: $PREP_LOG"
echo "Train log: $TRAIN_LOG"
echo "Eval log: $EVAL_LOG"
