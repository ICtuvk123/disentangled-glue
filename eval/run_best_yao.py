#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "yao_best"

BEST_CFG = {
    "trial_id": 6,
    "shared_dim": 64,
    "private_dim": 16,
    "batch_embed_dim": 8,
    "beta_shared": 0.75,
    "lam_iso": 2.0,
    "lam_align": 0.02,
    "beta_private_rna": 0.1,
    "beta_private_atac": 0.5,
    "dropout": 0.2,
    "lr": 0.002,
    "preset": "nb_all",
    "lsi_method": "raw_svd",
    "trial_in_bucket": 2,
    "feature_space": "all",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the best Yao config from hparam_search_v4_yao_baseline_style_fast."
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "dataset" / "Yao",
    )
    parser.add_argument(
        "--feature-aligned",
        type=Path,
        default=REPO_ROOT / "data" / "dataset" / "Yao" / "feature_aligned.h5ad",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "yao_baseline_style_cache",
    )
    parser.add_argument(
        "--bedtools",
        default="/data1/users/zhutianci/.conda/envs/scMRDR/bin/bedtools",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--lsi-n-components", type=int, default=100)
    parser.add_argument("--lsi-n-iter", type=int, default=15)
    parser.add_argument("--pretrain-max-epochs", type=int, default=60)
    parser.add_argument("--pretrain-patience", type=int, default=12)
    parser.add_argument("--pretrain-reduce-lr-patience", type=int, default=6)
    parser.add_argument("--finetune-max-epochs", type=int, default=30)
    parser.add_argument("--finetune-patience", type=int, default=8)
    parser.add_argument("--finetune-reduce-lr-patience", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

    import yao_hparam_search_baseline_style as yao_hpo

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.search_dir = args.output_dir
    yao_hpo.setup_environment(args)

    training_schedule = {
        "pretrain": {
            "max_epochs": args.pretrain_max_epochs,
            "patience": args.pretrain_patience,
            "reduce_lr_patience": args.pretrain_reduce_lr_patience,
        },
        "finetune": {
            "max_epochs": args.finetune_max_epochs,
            "patience": args.finetune_patience,
            "reduce_lr_patience": args.finetune_reduce_lr_patience,
        },
    }

    print("Preparing Yao cached inputs...")
    rna, atac, guidance = yao_hpo.preprocess_cached(
        args,
        feature_space=BEST_CFG["feature_space"],
        lsi_method=BEST_CFG["lsi_method"],
    )
    adata_eval = yao_hpo.load_eval_adata(args.feature_aligned)
    expected_n_obs = rna.n_obs + atac.n_obs
    if adata_eval.n_obs != expected_n_obs:
        raise ValueError(
            f"feature_aligned n_obs mismatch: {adata_eval.n_obs} vs {expected_n_obs}"
        )

    print("Training best Yao config:")
    print(json.dumps(BEST_CFG, indent=2))
    print(
        "Training schedule: "
        f"pretrain({args.pretrain_max_epochs}/{args.pretrain_patience}/{args.pretrain_reduce_lr_patience}) "
        f"finetune({args.finetune_max_epochs}/{args.finetune_patience}/{args.finetune_reduce_lr_patience})"
    )
    metrics = yao_hpo.run_trial(
        BEST_CFG,
        rna,
        atac,
        guidance,
        adata_eval,
        args.search_dir,
        args,
        training_schedule,
    )
    yao_hpo.write_live_summary(args.search_dir, announce=False)
    yao_hpo.write_summary([{**BEST_CFG, **metrics}], args.search_dir)

    run_dir = args.search_dir / yao_hpo.trial_name(BEST_CFG)
    (run_dir / "repro_info.json").write_text(
        json.dumps(
            {
                "dataset": "Yao-2021",
                "source_search_dir": str(REPO_ROOT / "hparam_search_v4_yao_baseline_style_fast"),
                "data_dir": str(args.data_dir.resolve()),
                "feature_aligned": str(args.feature_aligned.resolve()),
                "cache_dir": str(args.cache_dir.resolve()),
            },
            indent=2,
        )
    )

    print(f"\nRun directory: {run_dir}")
    if metrics:
        print(
            "Metrics: "
            f"Total={metrics.get('Total', float('nan')):.4f} "
            f"Bio={metrics.get('Bio conservation', float('nan')):.4f} "
            f"Batch={metrics.get('Batch correction', float('nan')):.4f} "
            f"Modality={metrics.get('Modality integration', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
