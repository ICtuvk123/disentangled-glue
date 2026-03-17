#!/usr/bin/env python
"""
Grid search over disentangled SCGLUE hyperparameters.

Searched dimensions
-------------------
  beta_private_rna  – private KL weight for RNA
  beta_private_atac – private KL weight for ATAC
  beta_private_prot – private KL weight for protein
  shared_dim        – shared latent dimensionality
  private_dim       – per-modality private latent dimensionality
                      (latent_dim = shared_dim + private_dim)

Usage
-----
    python s03_hparam_search.py \\
        --rna  s01_preprocessing/RNA_counts_qc_sampled.h5ad \\
        --atac s01_preprocessing/ATAC_counts_qc_sampled.h5ad \\
        --prot s01_preprocessing/protein_counts_qc_sampled.h5ad \\
        --gtf  gencode.v38.chr_patch_hapl_scaff.annotation.gtf \\
        --protein-gene-map s01_preprocessing/protein_gene_map.tsv \\
        --output-dir s03_hparam_search

    # Narrow the grid explicitly
    python s03_hparam_search.py ... \\
        --beta-private-rna  0.5 1.0 2.0 \\
        --beta-private-atac 0.5 1.0 2.0 \\
        --beta-private-prot 0.5 1.0 2.0 4.0 \\
        --shared-dims 30 50 \\
        --private-dims 10 20

Outputs
-------
    s03_hparam_search/
        results.tsv          – one row per configuration, sorted by Overall score
        best/                – model artefacts for the best configuration
        run_<id>/            – model artefacts for every configuration
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from scglue.metrics import (
    avg_silhouette_width,
    avg_silhouette_width_batch,
    graph_connectivity,
    mean_average_precision,
    normalized_mutual_info,
    seurat_alignment_score,
)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data inputs (forwarded to s02_glue.py)
    p.add_argument("--rna",  required=True)
    p.add_argument("--atac", required=True)
    p.add_argument("--prot", required=True)
    p.add_argument("--gtf",  required=True)
    p.add_argument("--protein-gene-map", default=None)
    p.add_argument("--bedtools", default=None)
    p.add_argument("--batch-key", default=None)
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--domain-key", default="domain")
    p.add_argument("--output-dir", default="s03_hparam_search")

    p.add_argument("--random-seed", type=int, default=0)

    # Search grid — each argument accepts one or more values.
    # Defaults are a small exploratory sweep (12 configs).
    # Core questions: does lower beta_shared fix shared-KL collapse?
    #                 does lam_iso help cross-modal alignment?
    # private_dim and per-modality betas are fixed for now.
    p.add_argument("--beta-shared", type=float, nargs="+",
                   default=[1.0, 2.0, 4.0],
                   help="Candidate shared KL weights")
    p.add_argument("--lam-iso", type=float, nargs="+",
                   default=[0.0, 0.1, 1.0],
                   help="Candidate isometric loss weights")
    p.add_argument("--beta-private-rna",  type=float, nargs="+",
                   default=[1.0],
                   help="Candidate beta_private values for RNA")
    p.add_argument("--beta-private-atac", type=float, nargs="+",
                   default=[1.0],
                   help="Candidate beta_private values for ATAC")
    p.add_argument("--beta-private-prot", type=float, nargs="+",
                   default=[1.0],
                   help="Candidate beta_private values for protein")
    p.add_argument("--shared-dims", type=int, nargs="+", default=[50],
                   help="Candidate shared latent dimensionalities")
    p.add_argument("--private-dims", type=int, nargs="+",
                   default=[20],
                   help="Candidate private latent dimensionalities")

    # Control
    p.add_argument("--resume", action="store_true",
                   help="Skip configs whose run directory already contains "
                        "combined_glue.h5ad")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the full grid and exit without training")
    p.add_argument("--n-gpus", type=int, default=1,
                   help="Total number of parallel GPU workers")
    p.add_argument("--gpu-id", type=int, default=0,
                   help="Index of this worker (0-indexed); runs configs where run_id %% n_gpus == gpu_id")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Grid helpers
# ──────────────────────────────────────────────────────────────────────

def build_grid(args) -> list[dict]:
    """Return a list of hyperparameter configs (dicts).

    Dim configs (shared_dim, private_dim) are the outermost loops so that
    each GPU worker (which takes every n_gpus-th run_id) sees all dim
    combinations rather than being stuck on one architecture.
    """
    combos = list(itertools.product(
        args.shared_dims,
        args.private_dims,
        args.beta_shared,
        args.lam_iso,
        args.beta_private_rna,
        args.beta_private_atac,
        args.beta_private_prot,
    ))
    configs = []
    for i, (sh_dim, pr_dim, bs, li, bp_rna, bp_atac, bp_prot) in enumerate(combos):
        configs.append({
            "run_id":            i,
            "shared_dim":        sh_dim,
            "private_dim":       pr_dim,
            "beta_shared":       bs,
            "lam_iso":           li,
            "beta_private_rna":  bp_rna,
            "beta_private_atac": bp_atac,
            "beta_private_prot": bp_prot,
        })
    return configs


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def preprocess_once(args, preprocess_dir: Path) -> bool:
    """Run s02_glue.py --preprocess-only once and cache results."""
    if (preprocess_dir / "guidance.graphml.gz").exists():
        print(f"  Reusing cached preprocessing from {preprocess_dir}")
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
    print(f"\nRunning preprocessing → {preprocess_dir}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: preprocessing failed with exit code {exc.returncode}")
        return False
    return True


def train_one(cfg: dict, args, run_dir: Path, preprocess_dir: Path) -> bool:
    """
    Call s02_glue.py for one configuration.
    Returns True on success, False on failure.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "s02_glue.py"),
        "--model", "disentangled",
        "--rna",  args.rna,
        "--atac", args.atac,
        "--prot", args.prot,
        "--gtf",  args.gtf,
        "--preprocessed-dir", str(preprocess_dir),
        "--output-dir", str(run_dir),
        "--shared-dim",  str(cfg["shared_dim"]),
        "--private-dim", str(cfg["private_dim"]),
        "--beta-shared", str(cfg["beta_shared"]),
        "--lam-iso",     str(cfg["lam_iso"]),
        "--beta-private-rna",  str(cfg["beta_private_rna"]),
        "--beta-private-atac", str(cfg["beta_private_atac"]),
        "--beta-private-prot", str(cfg["beta_private_prot"]),
        "--random-seed", str(args.random_seed),
    ]
    if args.protein_gene_map:
        cmd += ["--protein-gene-map", args.protein_gene_map]
    if args.bedtools:
        cmd += ["--bedtools", args.bedtools]
    if args.batch_key:
        cmd += ["--batch-key", args.batch_key]

    print(f"\n{'='*60}")
    print(f"  Run {cfg['run_id']:>3d}  |  "
          f"bp_rna={cfg['beta_private_rna']}  bp_atac={cfg['beta_private_atac']}  "
          f"bp_prot={cfg['beta_private_prot']}  "
          f"shared_dim={cfg['shared_dim']}  private_dim={cfg['private_dim']}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: run {cfg['run_id']} failed with exit code {exc.returncode}")
        return False
    print(f"  Finished in {(time.time()-t0)/60:.1f} min")
    return True


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_one(combined_path: Path, cell_type_key: str,
                 domain_key: str, batch_key: str | None) -> dict:
    adata = sc.read(combined_path)
    x  = adata.obsm["X_glue"]
    ct = adata.obs[cell_type_key].to_numpy().astype(str)
    domain = adata.obs[domain_key].to_numpy().astype(str)

    res = {
        "NMI":             normalized_mutual_info(x, ct),
        "ARI_MAP":         mean_average_precision(x, ct),
        "ASW_celltype":    avg_silhouette_width(x, ct),
        "Graph_conn":      graph_connectivity(x, ct),
        "Seurat_domain":   seurat_alignment_score(x, domain),
    }
    if batch_key and batch_key in adata.obs:
        batch = adata.obs[batch_key].to_numpy().astype(str)
        res["ASW_batch"]       = avg_silhouette_width_batch(x, batch, ct)
        res["Seurat_batch"]    = seurat_alignment_score(x, batch)

    bio   = np.mean([res["NMI"], res["ARI_MAP"], res["ASW_celltype"], res["Graph_conn"]])
    integ = res.get("Seurat_batch", res["Seurat_domain"])
    res["Bio_avg"]     = bio
    res["Integration"] = integ
    res["Overall"]     = 0.6 * bio + 0.4 * integ
    return res


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    grid = build_grid(args)
    grid = [cfg for cfg in grid if cfg["run_id"] % args.n_gpus == args.gpu_id]
    print(f"Grid size: {len(grid)} configurations (gpu_id={args.gpu_id}/{args.n_gpus})")

    if args.dry_run:
        df = pd.DataFrame(grid)
        print(df.to_string(index=False))
        return

    # Preprocessing is shared across all runs — do it once on gpu_id==0,
    # other workers wait until the cache is ready.
    preprocess_dir = base / "preprocessed"
    if args.gpu_id == 0:
        ok = preprocess_once(args, preprocess_dir)
        if not ok:
            return
    else:
        import time
        print(f"  gpu_id={args.gpu_id}: waiting for preprocessing by gpu_id=0 ...")
        while not (preprocess_dir / "guidance.graphml.gz").exists():
            time.sleep(10)
        print(f"  gpu_id={args.gpu_id}: preprocessing ready, starting grid.")

    rows = []
    for cfg in grid:
        run_dir = base / f"run_{cfg['run_id']:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        combined = run_dir / "combined_glue.h5ad"
        if args.resume and combined.exists():
            print(f"  run {cfg['run_id']:>3d}: resuming — combined_glue.h5ad exists")
        else:
            ok = train_one(cfg, args, run_dir, preprocess_dir)
            if not ok:
                continue

        if not combined.exists():
            print(f"  run {cfg['run_id']:>3d}: combined_glue.h5ad missing after training, skipping")
            continue

        # Persist config next to model artefacts
        with open(run_dir / "hparams.json", "w") as fh:
            json.dump(cfg, fh, indent=2)

        print(f"  Evaluating run {cfg['run_id']:>3d} ...")
        metrics = evaluate_one(combined, args.cell_type_key,
                               args.domain_key, args.batch_key)
        row = {**cfg, **metrics}
        rows.append(row)
        print(f"    Overall={metrics['Overall']:.4f}  Bio={metrics['Bio_avg']:.4f}  "
              f"Integ={metrics['Integration']:.4f}")

        # Save per-GPU results to avoid race condition when multiple workers
        # write to the same file simultaneously.
        gpu_tsv = base / f"results_gpu{args.gpu_id}.tsv"
        df = pd.DataFrame(rows).sort_values("Overall", ascending=False)
        df.to_csv(gpu_tsv, sep="\t", index=False, float_format="%.4f")

    if not rows:
        print("No successful runs to report.")
        return

    df = pd.DataFrame(rows).sort_values("Overall", ascending=False)
    gpu_tsv = base / f"results_gpu{args.gpu_id}.tsv"
    df.to_csv(gpu_tsv, sep="\t", index=False, float_format="%.4f")

    print(f"\n{'='*60}")
    print("Top 5 configurations:")
    print(df.head(5)[["run_id", "beta_shared", "beta_private_rna", "beta_private_atac",
                       "beta_private_prot", "shared_dim", "private_dim",
                       "Overall", "Bio_avg", "Integration"]].to_string(index=False))
    print(f"\nPer-GPU results -> {gpu_tsv}")

    # Copy best run artefacts
    best_id = int(df.iloc[0]["run_id"])
    best_src = base / f"run_{best_id:03d}"
    best_dst = base / "best"
    if best_dst.exists():
        import shutil
        shutil.rmtree(best_dst)
    import shutil
    shutil.copytree(best_src, best_dst)
    print(f"Best run ({best_id}) copied -> {best_dst}")


if __name__ == "__main__":
    main()
