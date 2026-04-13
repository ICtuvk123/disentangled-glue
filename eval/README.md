# Eval Scripts

This directory contains non-search reproduction scripts for the best configs
from three existing hyperparameter searches. Each script fixes the best
hyperparameters, runs training, runs evaluation, and writes outputs under
`eval/outputs/`.

## Scripts

- `run_best_muto.py`
  - Source search: `hparam_search_v3`
  - Dataset: `Muto`
  - Best config: `t0018_sd32_pd8_be4_bs1.5_li0.5_la0.05_bpr0.5_bpa0.5_sb0`
  - Evaluation: same `Benchmarker2` pipeline used in the original search

- `run_best_yao.py`
  - Source search: `hparam_search_v4_yao_baseline_style_fast`
  - Dataset: `Yao-2021`
  - Best config: `t0006_nb_all_raw_svd_sd64_pd16_be8_bs0.75_li2.0_la0.02_bpr0.1_bpa0.5_do0.2_lr0.002`
  - Evaluation: same `Benchmarker2` pipeline used in the original search
  - Training budget: matches the original `fast` search budget

- `run_best_bmmc_trimodal.py`
  - Source search: `experiments/BMMC/s11_long_search/20260329_204614`
  - Dataset: `BMMC`
  - Best config: `baseline_t0088_sd48_pd8_bs0.75_li2.0_la0.3_bpr0.25_bpa1.0_bpp1.0`
  - Evaluation:
    - search-style quick metrics (`metrics.json`)
    - standard scMRDR-style evaluation (`standard_eval/*_unscaled.csv` and `standard_metrics.json`)

## Usage

Run from the repo root or any working directory:

```bash
/data1/users/zhutianci/.conda/envs/scMRDR/bin/python \
  /data1/users/zhutianci/proj/disentangled-glue/eval/run_best_muto.py

/data1/users/zhutianci/.conda/envs/scMRDR/bin/python \
  /data1/users/zhutianci/proj/disentangled-glue/eval/run_best_yao.py

/data1/users/zhutianci/.conda/envs/scMRDR/bin/python \
  /data1/users/zhutianci/proj/disentangled-glue/eval/run_best_bmmc_trimodal.py
```

Useful overrides:

- `--output-dir`: change where outputs are written
- `--gpu`: set `CUDA_VISIBLE_DEVICES`
- `--n-jobs`: neighbor-search / evaluation parallelism

Yao also supports:

- `--data-dir`
- `--feature-aligned`
- `--cache-dir`
- `--pretrain-max-epochs`, `--pretrain-patience`, `--pretrain-reduce-lr-patience`
- `--finetune-max-epochs`, `--finetune-patience`, `--finetune-reduce-lr-patience`

BMMC also supports:

- `--rna`, `--atac`, `--prot`, `--gtf`
- `--protein-gene-map`
- `--feature-aligned`
- `--preprocessed-dir`
- `--python-bin`
- `--resume`

## Outputs

Each script creates a subdirectory named after the reproduced trial tag under
its output root. Typical outputs include:

- `config.json` or `best_config.json`
- `metrics.json`
- `metrics.tsv`
- `summary.tsv`
- `embedding.npy` or `combined_glue.h5ad`
- for BMMC standard eval: `standard_eval/*_unscaled.csv`, `standard_metrics.json`
