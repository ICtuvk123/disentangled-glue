#!/usr/bin/env python
"""
Unified random hyperparameter search for disentangled SCGLUE.

Searches over all model hyperparameters (shared_dim, private_dim, beta_shared,
lam_iso, lam_align, beta_private_*) and support-mode hyperparameters
(align_support_k, align_support_strategy, align_support_min_weight) jointly.
Mode (baseline vs support) is also a searchable parameter.

Multi-GPU: launch one process per GPU with --gpu-id 0..N-1 --n-gpus N.
Each worker picks up trials where trial_id % n_gpus == gpu_id.

Example (4 GPUs, 80 trials):
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python s11_hparam_search.py \\
      --rna RNA_counts_qc_sampled.h5ad \\
      --atac ATAC_counts_qc_sampled.h5ad \\
      --prot protein_counts_qc_sampled.h5ad \\
      --gtf gencode.v38.chr_patch_hapl_scaff.annotation.gtf \\
      --protein-gene-map s01_preprocessing/protein_gene_map.tsv \\
      --preprocessed-dir s07_support_search_fix/preprocessed \\
      --output-dir s11_hparam_search \\
      --n-trials 80 --n-gpus 4 --gpu-id $i --resume &
  done
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scib_metrics
from scib_metrics.nearest_neighbors import pynndescent

from scglue.metrics import (
    avg_silhouette_width,
    avg_silhouette_width_batch,
    graph_connectivity,
    mean_average_precision,
    normalized_mutual_info,
    seurat_alignment_score,
)

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict[str, list] = {
    "mode":                    ["baseline", "support"],
    "shared_dim":              [24, 30, 48],
    "private_dim":             [4, 8, 16],
    "beta_shared":             [0.5, 0.75, 1.0, 1.25, 1.5],
    "lam_iso":                 [0.5, 1.0, 2.0],
    "lam_align":               [0.03, 0.05, 0.10, 0.20, 0.30, 0.50],
    "beta_private_rna":        [0.25, 0.5, 1.0],
    "beta_private_atac":       [0.25, 0.5, 1.0],
    "beta_private_prot":       [0.25, 0.5, 1.0],
    # support-mode params (ignored when mode == "baseline")
    "align_support_k":         [10, 15, 20, 30],
    "align_support_strategy":  ["soft", "hard"],
    "align_support_min_weight":[0.01, 0.05, 0.10, 0.20],
}


def sample_config(rng: np.random.Generator, trial_id: int) -> dict:
    cfg: dict = {"trial_id": trial_id}
    for key, choices in SEARCH_SPACE.items():
        cfg[key] = choices[rng.integers(len(choices))]
    return cfg


def generate_trials(n_trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [sample_config(rng, i) for i in range(n_trials)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rna",  required=True)
    p.add_argument("--atac", required=True)
    p.add_argument("--prot", required=True)
    p.add_argument("--gtf",  required=True)
    p.add_argument("--protein-gene-map", default=None)
    p.add_argument("--bedtools", default=None)
    p.add_argument("--preprocessed-dir", default=None,
                   help="Reuse existing preprocessing (skips preprocess step)")
    p.add_argument("--output-dir", default="s11_hparam_search")
    p.add_argument("--batch-key",     default="batch")
    p.add_argument("--cell-type-key", default="celltype")
    p.add_argument("--domain-key",    default="domain")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--n-gpus",  type=int, default=1)
    p.add_argument("--gpu-id",  type=int, default=0)
    p.add_argument("--resume",  action="store_true",
                   help="Skip runs that already have metrics.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Print configs without running anything")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trial_name(cfg: dict) -> str:
    sup = ""
    if cfg["mode"] == "support":
        sup = (
            f"_sk{cfg['align_support_k']}"
            f"_{cfg['align_support_strategy'][0]}"
            f"_mw{cfg['align_support_min_weight']}"
        )
    return (
        f"{cfg['mode']}_t{cfg['trial_id']:04d}"
        f"_sd{cfg['shared_dim']}"
        f"_pd{cfg['private_dim']}"
        f"_bs{cfg['beta_shared']}"
        f"_li{cfg['lam_iso']}"
        f"_la{cfg['lam_align']}"
        f"_bpr{cfg['beta_private_rna']}"
        f"_bpa{cfg['beta_private_atac']}"
        f"_bpp{cfg['beta_private_prot']}"
        f"{sup}"
    )


def preprocess_once(args: argparse.Namespace, preprocess_dir: Path) -> bool:
    if (preprocess_dir / "guidance.graphml.gz").exists():
        print(f"Reusing shared preprocessing from {preprocess_dir}")
        return True
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "s02_glue.py"),
        "--rna",  args.rna,
        "--atac", args.atac,
        "--prot", args.prot,
        "--gtf",  args.gtf,
        "--output-dir", str(preprocess_dir),
        "--preprocess-only",
    ]
    if args.protein_gene_map:
        cmd += ["--protein-gene-map", args.protein_gene_map]
    if args.bedtools:
        cmd += ["--bedtools", args.bedtools]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: preprocessing failed (exit {exc.returncode})")
        return False
    return True


def train_one(cfg: dict, args: argparse.Namespace,
              run_dir: Path, preprocess_dir: Path) -> bool:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "s02_glue.py"),
        "--model", "disentangled",
        "--rna",  args.rna,
        "--atac", args.atac,
        "--prot", args.prot,
        "--gtf",  args.gtf,
        "--preprocessed-dir", str(preprocess_dir),
        "--output-dir",       str(run_dir),
        "--shared-dim",       str(cfg["shared_dim"]),
        "--private-dim",      str(cfg["private_dim"]),
        "--beta-shared",      str(cfg["beta_shared"]),
        "--lam-iso",          str(cfg["lam_iso"]),
        "--lam-align",        str(cfg["lam_align"]),
        "--beta-private-rna", str(cfg["beta_private_rna"]),
        "--beta-private-atac",str(cfg["beta_private_atac"]),
        "--beta-private-prot",str(cfg["beta_private_prot"]),
    ]
    if args.protein_gene_map:
        cmd += ["--protein-gene-map", args.protein_gene_map]
    if args.bedtools:
        cmd += ["--bedtools", args.bedtools]
    if args.batch_key:
        cmd += ["--batch-key", args.batch_key]
    if cfg["mode"] == "support":
        cmd += [
            "--align-support",
            "--align-support-k",          str(cfg["align_support_k"]),
            "--align-support-strategy",   cfg["align_support_strategy"],
            "--align-support-min-weight", str(cfg["align_support_min_weight"]),
        ]
    print("=" * 80)
    print(trial_name(cfg))
    print("=" * 80)
    t0 = time.time()
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: training failed (exit {exc.returncode})")
        return False
    print(f"Training done in {(time.time() - t0) / 60:.1f} min")
    return True


def quick_eval_one(cfg: dict, args: argparse.Namespace, run_dir: Path) -> dict:
    adata = sc.read(run_dir / "combined_glue.h5ad")
    x  = adata.obsm["X_glue"].astype(np.float32)
    ct = adata.obs[args.cell_type_key].to_numpy().astype(str)
    domain = adata.obs[args.domain_key].to_numpy().astype(str)

    metrics: dict = {
        "NMI":          normalized_mutual_info(x, ct),
        "ARI_MAP":      mean_average_precision(x, ct),
        "ASW_celltype": avg_silhouette_width(x, ct),
        "Graph_conn":   graph_connectivity(x, ct),
        "Seurat_domain":seurat_alignment_score(x, domain),
    }
    if args.batch_key and args.batch_key in adata.obs:
        batch = adata.obs[args.batch_key].to_numpy().astype(str)
        metrics["ASW_batch"]    = avg_silhouette_width_batch(x, batch, ct)
        metrics["Seurat_batch"] = seurat_alignment_score(x, batch)

    # kBET (50-NN) and iLISI (90-NN) via scib_metrics
    nn50  = pynndescent(x, n_neighbors=50,  n_jobs=4)
    nn90  = pynndescent(x, n_neighbors=90,  n_jobs=4)

    if args.batch_key and args.batch_key in adata.obs:
        batch = adata.obs[args.batch_key].to_numpy().astype(str)
        metrics["kBET_batch"]  = float(scib_metrics.kbet_per_label(nn50, batch, ct))
        metrics["iLISI_batch"] = float(scib_metrics.ilisi_knn(nn90, batch))

    metrics["kBET_modality"]  = float(scib_metrics.kbet_per_label(nn50, domain, ct))
    metrics["iLISI_modality"] = float(scib_metrics.ilisi_knn(nn90, domain))

    bio   = float(np.mean([metrics["NMI"], metrics["ARI_MAP"],
                            metrics["ASW_celltype"], metrics["Graph_conn"]]))
    batch_score = float(np.mean([v for k, v in metrics.items()
                                  if k in ("ASW_batch", "Seurat_batch",
                                           "kBET_batch", "iLISI_batch")]))
    integ = float(np.mean([metrics["Seurat_domain"],
                            metrics["kBET_modality"],
                            metrics["iLISI_modality"]]))

    metrics["Bio conservation"]    = bio
    metrics["Batch correction"]    = batch_score
    metrics["Modality integration"]= integ
    metrics["Total"]               = (bio + batch_score + integ) / 3
    return metrics


def save_artifacts(run_dir: Path, cfg: dict, metrics: dict) -> None:
    with (run_dir / "hparams.json").open("w") as fh:
        json.dump(cfg, fh, indent=2)
    with (run_dir / "metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


def write_summary(out_dir: Path, trials: list[dict]) -> None:
    rows = []
    for cfg in trials:
        run_dir = out_dir / trial_name(cfg)
        mf = run_dir / "metrics.json"
        if not mf.exists():
            continue
        with mf.open() as fh:
            m = json.load(fh)
        rows.append({**cfg, **m})
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols_first = ["trial_id", "mode", "Total", "Bio conservation",
                  "Batch correction", "Modality integration"]
    rest = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + rest].sort_values("Total", ascending=False)
    summary_path = out_dir / "summary.tsv"
    df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary written to {summary_path}")
    print(df[cols_first].head(10).to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = generate_trials(args.n_trials, args.seed)

    if args.dry_run:
        for cfg in trials:
            if cfg["trial_id"] % args.n_gpus == args.gpu_id:
                print(trial_name(cfg))
        return

    # Preprocessing
    if args.preprocessed_dir:
        preprocess_dir = Path(args.preprocessed_dir)
    else:
        preprocess_dir = out_dir / "preprocessed"
    if not preprocess_once(args, preprocess_dir):
        sys.exit(1)

    ran = 0
    for cfg in trials:
        if cfg["trial_id"] % args.n_gpus != args.gpu_id:
            continue

        name    = trial_name(cfg)
        run_dir = out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.resume and (run_dir / "metrics.json").exists():
            print(f"Skipping completed trial: {name}")
            ran += 1
            continue

        ok = train_one(cfg, args, run_dir, preprocess_dir)
        if ok:
            metrics = quick_eval_one(cfg, args, run_dir)
            save_artifacts(run_dir, cfg, metrics)
            ran += 1
            print(f"  Total={metrics['Total']:.4f}  "
                  f"Bio={metrics['Bio conservation']:.4f}  "
                  f"Batch={metrics['Batch correction']:.4f}  "
                  f"Modality={metrics['Modality integration']:.4f}")
        else:
            print(f"Trial failed: {name}")

    write_summary(out_dir, trials)
    print(f"\nWorker {args.gpu_id} finished {ran} trials")


if __name__ == "__main__":
    main()
