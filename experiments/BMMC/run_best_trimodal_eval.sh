#!/bin/bash
cd /data1/users/zhutianci/proj/disentangled-glue/experiments/BMMC
python s06_eval.py \
    --run-dir s02_best_run \
    --feature-aligned s01_preprocessing/feature_aligned_sampled.h5ad \
    --enable-pcr \
    --domain-key domain \
    --cell-type-key celltype \
    --batch-key batch \
    --output-dir s02_best_run_eval \
    2>&1 | tee s02_best_run_eval.log
