#!/usr/bin/env python
"""
Train standard SCGLUE and disentangled SCGLUE on BMMC data, then compare
their embedding quality side-by-side.

Usage
-----
    python s02_run_and_compare.py \
        --rna s01_preprocessing/RNA_counts_qc_sampled.h5ad \
        --atac s01_preprocessing/ATAC_counts_qc_sampled.h5ad \
        --prot s01_preprocessing/protein_counts_qc_sampled.h5ad \
        --gtf gencode.v38.chr_patch_hapl_scaff.annotation.gtf \
        --protein-gene-map s01_preprocessing/protein_gene_map.tsv

Outputs
-------
    s02_compare/
        metrics.tsv          – per-model metrics table
        comparison.png       – side-by-side UMAP + bar chart
        scglue/              – standard model artifacts
        disentangled/        – disentangled model artifacts
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import scglue
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

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rna", required=True)
    p.add_argument("--atac", required=True)
    p.add_argument("--prot", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--protein-gene-map", default=None)
    p.add_argument("--bedtools", default=None,
                   help="Directory containing the bedtools binary")
    p.add_argument("--output-dir", default="s02_compare")
    p.add_argument("--cell-type-key", default="cell_type",
                   help="obs column for cell type labels")
    p.add_argument("--batch-key", default=None,
                   help="obs column for batch labels (also passed to GLUE)")
    p.add_argument("--domain-key", default="domain",
                   help="obs column marking modality origin")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training, only run evaluation on existing outputs")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

MODELS = ["scglue", "disentangled"]


def train_model(model_name: str, args, out_dir: Path) -> None:
    """Launch s02_glue.py as a subprocess for *model_name*."""
    cmd = [
        sys.executable, "s02_glue.py",
        "--model", model_name,
        "--rna", args.rna,
        "--atac", args.atac,
        "--prot", args.prot,
        "--gtf", args.gtf,
        "--output-dir", str(out_dir),
        "--umap",
    ]
    if args.protein_gene_map:
        cmd += ["--protein-gene-map", args.protein_gene_map]
    if args.bedtools:
        cmd += ["--bedtools", args.bedtools]
    if args.batch_key:
        cmd += ["--batch-key", args.batch_key]

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Output:   {out_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()
    subprocess.check_call(cmd)
    elapsed = time.time() - t0
    print(f"\n>> {model_name} finished in {elapsed/60:.1f} min\n")

# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(combined_path: Path, cell_type_key: str,
                   domain_key: str, batch_key: str | None) -> dict:
    """Compute integration metrics from a combined_glue.h5ad."""
    adata = sc.read(combined_path)
    x = adata.obsm["X_glue"]
    ct = adata.obs[cell_type_key].to_numpy().astype(str)
    domain = adata.obs[domain_key].to_numpy().astype(str)

    results = {
        "NMI":                normalized_mutual_info(x, ct),
        "ARI_MAP":            mean_average_precision(x, ct),
        "ASW_celltype":       avg_silhouette_width(x, ct),
        "Graph_conn":         graph_connectivity(x, ct),
        "Seurat_align_domain": seurat_alignment_score(x, domain),
    }

    if batch_key and batch_key in adata.obs:
        batch = adata.obs[batch_key].to_numpy().astype(str)
        results["ASW_batch"] = avg_silhouette_width_batch(x, batch, ct)
        results["Seurat_align_batch"] = seurat_alignment_score(x, batch)

    # Overall score (bio + integration, equal weight)
    bio = np.mean([results["NMI"], results["ARI_MAP"],
                    results["ASW_celltype"], results["Graph_conn"]])
    integ = results.get("Seurat_align_batch",
                        results["Seurat_align_domain"])
    results["Bio_avg"] = bio
    results["Integration"] = integ
    results["Overall"] = 0.6 * bio + 0.4 * integ

    return results

# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_comparison(model_dirs: dict[str, Path], metrics_df: pd.DataFrame,
                    cell_type_key: str, out_path: Path) -> None:
    """Side-by-side UMAP colored by cell type + bar chart of metrics."""
    n_models = len(model_dirs)
    fig, axes = plt.subplots(1, n_models + 1,
                             figsize=(7 * (n_models + 1), 6))

    # UMAPs
    for ax, (name, mdir) in zip(axes[:n_models], model_dirs.items()):
        combined = sc.read(mdir / "combined_glue.h5ad")
        if "X_umap" not in combined.obsm:
            sc.pp.neighbors(combined, n_pcs=50, use_rep="X_glue",
                            metric="cosine")
            sc.tl.umap(combined)
        sc.pl.umap(combined, color=cell_type_key, ax=ax, show=False,
                   title=name, frameon=False, legend_loc="none")

    # Bar chart
    ax_bar = axes[-1]
    plot_cols = [c for c in metrics_df.columns
                 if c not in ("model", "Bio_avg", "Integration", "Overall")]
    x_pos = np.arange(len(plot_cols))
    width = 0.8 / n_models
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        vals = [row[c] for c in plot_cols]
        ax_bar.bar(x_pos + i * width, vals, width, label=row["model"])
    ax_bar.set_xticks(x_pos + width * (n_models - 1) / 2)
    ax_bar.set_xticklabels(plot_cols, rotation=35, ha="right", fontsize=9)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Metrics comparison")
    ax_bar.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot -> {out_path}")

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    model_dirs = {m: base / m for m in MODELS}

    # --- Train both models sequentially ---
    if not args.skip_training:
        for name, mdir in model_dirs.items():
            if (mdir / "combined_glue.h5ad").exists():
                print(f">> {name}: combined_glue.h5ad already exists, skipping training")
                continue
            train_model(name, args, mdir)

    # --- Evaluate ---
    rows = []
    for name, mdir in model_dirs.items():
        combined = mdir / "combined_glue.h5ad"
        if not combined.exists():
            print(f"WARNING: {combined} not found, skipping {name}")
            continue
        print(f"\nEvaluating {name} ...")
        metrics = evaluate_model(combined, args.cell_type_key,
                                 args.domain_key, args.batch_key)
        metrics["model"] = name
        rows.append(metrics)
        print(f"  {metrics}")

    if not rows:
        print("ERROR: No models to evaluate. Check training output.")
        sys.exit(1)

    metrics_df = pd.DataFrame(rows)
    cols = ["model"] + [c for c in metrics_df.columns if c != "model"]
    metrics_df = metrics_df[cols]

    # --- Save results ---
    tsv_path = base / "metrics.tsv"
    metrics_df.to_csv(tsv_path, sep="\t", index=False, float_format="%.4f")
    print(f"\nSaved metrics -> {tsv_path}")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Plot ---
    plot_comparison(model_dirs, metrics_df, args.cell_type_key,
                    base / "comparison.png")


if __name__ == "__main__":
    main()
