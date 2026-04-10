"""
Random hyperparameter search for DisentangledSCGLUEModel on the Muto dataset.
Search space aligned with BMMC s01_hparam_search_scmrdr.py.

Each trial:
1. Train DisentangledSCGLUEModel with sampled config
2. Save embedding.npy
3. Evaluate inline with Benchmarker2 and save metrics.json
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
import json
import argparse
from itertools import chain
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

DGLUE_ROOT  = Path("/data1/users/zhutianci/proj/disentangled-glue")
SCMRDR_ROOT = Path("/data1/users/zhutianci/proj/scMRDR")
sys.path.insert(0, str(DGLUE_ROOT))
sys.path.insert(0, str(SCMRDR_ROOT / "experiments" / "plots"))

from scglue.models import configure_dataset, fit_SCGLUE
from scglue.models.scglue import DisentangledSCGLUEModel
from metrics import Benchmarker2, BioConservation2, BatchCorrection2, ModalityIntegration2

# ── search space (aligned with BMMC s01_hparam_search_scmrdr.py) ─────────────
SEARCH_SPACE = {
    "mode":                    ["baseline", "support"],
    "shared_dim":              [24, 32, 48, 64],
    "private_dim":             [4, 8, 16],
    "batch_embed_dim":         [4, 8, 16],
    "beta_shared":             [0.5, 0.75, 1.0, 1.25, 1.5],
    "lam_iso":                 [0.5, 1.0, 2.0],
    "lam_align":               [0.03, 0.05, 0.10, 0.20, 0.30, 0.50],
    "beta_private_rna":        [0.25, 0.5, 1.0],
    "beta_private_atac":       [0.25, 0.5, 1.0],
    # support-mode only
    "align_support_k":         [10, 15, 20, 30],
    "align_support_strategy":  ["soft", "hard"],
    "align_support_min_weight":[0.01, 0.05, 0.10, 0.20],
}

FIXED = {
    "lam_data":    1.0,
    "lam_graph":   0.02,
    "lr":          2e-3,
    "shared_batches": False,
}

SEARCH_DIR = DGLUE_ROOT / "hparam_search_v2"


def sample_config(rng: np.random.Generator, trial_id: int) -> dict:
    cfg = {"trial_id": trial_id}
    for key, choices in SEARCH_SPACE.items():
        cfg[key] = choices[rng.integers(len(choices))]
    return cfg


def trial_name(cfg: dict) -> str:
    suffix = ""
    if cfg["mode"] == "support":
        suffix = (
            f"_sk{cfg['align_support_k']}"
            f"_{cfg['align_support_strategy'][0]}"
            f"_mw{cfg['align_support_min_weight']}"
        )
    return (
        f"{cfg['mode']}_t{cfg['trial_id']:04d}"
        f"_sd{cfg['shared_dim']}"
        f"_pd{cfg['private_dim']}"
        f"_be{cfg['batch_embed_dim']}"
        f"_bs{cfg['beta_shared']}"
        f"_li{cfg['lam_iso']}"
        f"_la{cfg['lam_align']}"
        f"_bpr{cfg['beta_private_rna']}"
        f"_bpa{cfg['beta_private_atac']}"
        f"_sb{int(FIXED['shared_batches'])}"
        f"{suffix}"
    )


def run_trial(cfg: dict, rna, atac, guidance_hvf, adata_eval) -> dict:
    tag = trial_name(cfg)
    out_dir = SEARCH_DIR / tag
    npy_path = out_dir / "embedding.npy"
    metrics_path = out_dir / "metrics.json"

    # skip if already done
    if npy_path.exists() and metrics_path.exists():
        print(f"  Already done, skipping.")
        with metrics_path.open() as f:
            return json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    # align_support_kws for support mode
    align_support_kws = None
    if cfg["mode"] == "support":
        align_support_kws = {
            "n_neighbors": cfg["align_support_k"],
            "strategy":    cfg["align_support_strategy"],
            "min_weight":  cfg["align_support_min_weight"],
        }

    compile_kws = {
        "lam_data":    FIXED["lam_data"],
        "lam_graph":   FIXED["lam_graph"],
        "lam_align":   cfg["lam_align"],
        "beta_shared": cfg["beta_shared"],
        "lam_iso":     cfg["lam_iso"],
        "beta_private": {
            "rna":  cfg["beta_private_rna"],
            "atac": cfg["beta_private_atac"],
        },
        "lr": FIXED["lr"],
    }

    init_kws = {
        "shared_dim":      cfg["shared_dim"],
        "private_dim":     cfg["private_dim"],
        "batch_embed_dim": cfg["batch_embed_dim"],
        "shared_batches":  FIXED["shared_batches"],
        "h_depth": 2,
        "h_dim":   256,
        "dropout": 0.2,
    }

    try:
        glue = fit_SCGLUE(
            {"rna": rna, "atac": atac},
            guidance_hvf,
            model=DisentangledSCGLUEModel,
            init_kws=init_kws,
            compile_kws=compile_kws,
            fit_kws={"directory": str(out_dir / "glue")},
            align_support_kws=align_support_kws,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return {}

    # save embedding
    rna.obsm["X_glue_shared"], _ = glue.encode_data("rna",  rna,  return_private=True)
    atac.obsm["X_glue_shared"], _ = glue.encode_data("atac", atac, return_private=True)
    emb = np.vstack([rna.obsm["X_glue_shared"], atac.obsm["X_glue_shared"]])
    np.save(str(npy_path), emb)

    # evaluate inline
    adata_eval.obsm[tag] = emb
    bm2 = Benchmarker2(
        adata_eval,
        batch_key="batch",
        label_key="cell_type",
        modality_key="modality",
        bio_conservation_metrics=BioConservation2(),
        batch_correction_metrics=BatchCorrection2(),
        modality_integration_metrics=ModalityIntegration2(),
        embedding_obsm_keys=[tag],
        n_jobs=10,
    )
    bm2.benchmark()
    df = bm2.get_results(min_max_scale=False)

    # extract scalar metrics
    metrics = {}
    for col in df.columns:
        val = df[col].iloc[0]
        if pd.notna(val):
            metrics[col] = float(val)

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.tsv", sep="\t", index=False)

    # clean up obsm to avoid memory accumulation
    del adata_eval.obsm[tag]

    return metrics


def write_summary(results: list[dict]) -> None:
    rows = []
    for cfg, metrics in results:
        if metrics:
            rows.append({**cfg, **metrics})
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols_first = [
        "trial_id", "mode", "Total", "Bio conservation",
        "Batch correction", "Modality integration",
        "shared_dim", "private_dim", "batch_embed_dim",
        "beta_shared", "lam_iso", "lam_align",
        "beta_private_rna", "beta_private_atac",
    ]
    cols_first = [c for c in cols_first if c in df.columns]
    rest = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + rest].sort_values("Total", ascending=False, na_position="last")
    summary_path = SEARCH_DIR / "summary.tsv"
    df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary written to {summary_path}")
    print(df[cols_first].head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    SEARCH_DIR.mkdir(exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    rna      = ad.read_h5ad(str(DGLUE_ROOT / "rna-pp.h5ad"))
    atac     = ad.read_h5ad(str(DGLUE_ROOT / "atac-pp.h5ad"))
    guidance = nx.read_graphml(str(DGLUE_ROOT / "guidance.graphml.gz"))

    configure_dataset(rna,  "NB", use_highly_variable=True,
                      use_layer="counts", use_rep="X_pca", use_batch="batch")
    configure_dataset(atac, "NB", use_highly_variable=True,
                      use_layer="counts", use_rep="X_lsi", use_batch="batch")

    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index,
    )).copy()

    # ── load eval base adata ──────────────────────────────────────────────────
    adata_eval = sc.read_h5ad(str(DGLUE_ROOT / "feature_aligned_trained.h5ad"))
    adata_eval.obs["modality"] = adata_eval.obs["modality"].astype(str)
    adata_eval.obs.loc[adata_eval.obs["modality"] == "0", "modality"] = "RNA"
    adata_eval.obs.loc[adata_eval.obs["modality"] == "1", "modality"] = "ATAC"

    # ── generate trials ───────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    configs = [sample_config(rng, i) for i in range(args.n_trials)]

    my_configs = [cfg for cfg in configs if cfg["trial_id"] % args.n_gpus == args.gpu_id]
    print(f"Total trials: {args.n_trials}  |  This worker (GPU {args.gpu_id}): {len(my_configs)} trials")
    results = []
    for i, cfg in enumerate(my_configs):
        tag = trial_name(cfg)
        print(f"\n[{i+1}/{len(my_configs)}] {tag}")
        metrics = run_trial(cfg, rna, atac, guidance_hvf, adata_eval)
        results.append((cfg, metrics))
        if metrics:
            print(
                f"  Total={metrics.get('Total', float('nan')):.4f}  "
                f"Bio={metrics.get('Bio conservation', float('nan')):.4f}  "
                f"Batch={metrics.get('Batch correction', float('nan')):.4f}  "
                f"Modality={metrics.get('Modality integration', float('nan')):.4f}"
            )

    write_summary(results)


if __name__ == "__main__":
    main()
