#!/bin/bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-bmmc-preserved-full}"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
pkill -f 's10_try_preserved_full_models.py|run_preserved_full_tmux.sh|s02_glue.py --model disentangled --rna .*s01_preprocessing_full|s08_eval_scmrdr.py --run-dir experiments/BMMC/s04_sweep/best' 2>/dev/null || true

echo "Stopped session '$SESSION_NAME' and related preserved-full processes."
