#!/usr/bin/env python
"""
Matched local sweep of baseline vs support-weighted disentangled GLUE.

This script evaluates the same hyperparameter grid for two modes:
1. baseline disentangled GLUE
2. support-weighted disentangled GLUE

It is intended to answer the fair comparison question:
"Does support weighting still help after both methods are tuned on the same grid?"
"""

from __future__ import annotations

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rna", required=True)
    p.add_argument("--atac", required=True)
    p.add_argument("--prot", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--protein-gene-map", default=None)
    p.add_argument("--bedtools", default=None)
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--cell-type-key", default="celltype")
    p.add_argument("--domain-key", default="domain")
    p.add_argument("--source-run", default="s06_sweep/run_023")
    p.add_argument(
        "--preprocessed-dir",
        default=None,
        help="Shared preprocessed directory. Defaults to <source-run-parent>/preprocessed",
    )
    p.add_argument("--output-dir", default="s07_support_search")
    p.add_argument("--n-gpus", type=int, default=4)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "support"],
        choices=["baseline", "support"],
        help="Which methods to run on the matched grid",
    )
    p.add_argument("--shared-dims", type=int, nargs="+", default=[24, 30])
    p.add_argument("--private-dims", type=int, nargs="+", default=[8, 12])
    p.add_argument("--beta-shared", type=float, nargs="+", default=[1.0, 1.25])
    p.add_argument("--lam-iso", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--lam-align", type=float, nargs="+", default=[0.03, 0.05])
    p.add_argument("--align-support-k", type=int, default=15)
    p.add_argument(
        "--align-support-strategy",
        default="soft",
        choices=["soft", "hard"],
    )
    p.add_argument("--align-support-min-weight", type=float, default=0.05)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument(
        "--eval-mode",
        choices=["quick", "full"],
        default="quick",
        help=(
            "Evaluation mode. `quick` uses the original fast sweep metrics "
            "(recommended for the full grid). `full` runs s06_eval.py."
        ),
    )
    return p.parse_args()


def load_source_hparams(source_run: Path) -> dict:
    with (source_run / "hparams.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_grid(args: argparse.Namespace, source_hparams: dict) -> list[dict]:
    fallback = {
        "beta_private_rna": source_hparams["beta_private_rna"],
        "beta_private_atac": source_hparams["beta_private_atac"],
        "beta_private_prot": source_hparams["beta_private_prot"],
    }
    combos = list(
        itertools.product(
            args.shared_dims,
            args.private_dims,
            args.beta_shared,
            args.lam_iso,
            args.lam_align,
            args.modes,
        )
    )
    configs = []
    for run_id, (shared_dim, private_dim, beta_shared, lam_iso, lam_align, mode) in enumerate(combos):
        configs.append(
            {
                "run_id": run_id,
                "mode": mode,
                "shared_dim": shared_dim,
                "private_dim": private_dim,
                "beta_shared": beta_shared,
                "lam_iso": lam_iso,
                "lam_align": lam_align,
                **fallback,
            }
        )
    return configs


def preprocess_once(args: argparse.Namespace, preprocess_dir: Path) -> bool:
    if (preprocess_dir / "guidance.graphml.gz").exists():
        print(f"Reusing shared preprocessing from {preprocess_dir}")
        return True
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "s02_glue.py"),
        "--rna", args.rna,
        "--atac", args.atac,
        "--prot", args.prot,
        "--gtf", args.gtf,
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
        print(f"ERROR: preprocessing failed with exit code {exc.returncode}")
        return False
    return True


def config_name(cfg: dict) -> str:
    return (
        f"{cfg['mode']}_cfg{cfg['run_id']:03d}"
        f"_sd{cfg['shared_dim']}"
        f"_pd{cfg['private_dim']}"
        f"_bs{cfg['beta_shared']}"
        f"_li{cfg['lam_iso']}"
        f"_la{cfg['lam_align']}"
    )


def train_one(cfg: dict, args: argparse.Namespace, run_dir: Path, preprocess_dir: Path) -> bool:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "s02_glue.py"),
        "--model", "disentangled",
        "--rna", args.rna,
        "--atac", args.atac,
        "--prot", args.prot,
        "--gtf", args.gtf,
        "--preprocessed-dir", str(preprocess_dir),
        "--output-dir", str(run_dir),
        "--shared-dim", str(cfg["shared_dim"]),
        "--private-dim", str(cfg["private_dim"]),
        "--beta-shared", str(cfg["beta_shared"]),
        "--lam-iso", str(cfg["lam_iso"]),
        "--lam-align", str(cfg["lam_align"]),
        "--beta-private-rna", str(cfg["beta_private_rna"]),
        "--beta-private-atac", str(cfg["beta_private_atac"]),
        "--beta-private-prot", str(cfg["beta_private_prot"]),
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
            "--align-support-k", str(args.align_support_k),
            "--align-support-strategy", args.align_support_strategy,
            "--align-support-min-weight", str(args.align_support_min_weight),
        ]

    print("=" * 80)
    print(config_name(cfg))
    print("=" * 80)
    t0 = time.time()
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: training failed with exit code {exc.returncode}")
        return False
    print(f"Training finished in {(time.time() - t0) / 60:.1f} min")
    return True


def quick_eval_one(cfg: dict, args: argparse.Namespace, run_dir: Path) -> dict:
    adata = sc.read(run_dir / "combined_glue.h5ad")
    x = adata.obsm["X_glue"]
    ct = adata.obs[args.cell_type_key].to_numpy().astype(str)
    domain = adata.obs[args.domain_key].to_numpy().astype(str)

    metrics = {
        "NMI": normalized_mutual_info(x, ct),
        "ARI_MAP": mean_average_precision(x, ct),
        "ASW_celltype": avg_silhouette_width(x, ct),
        "Graph_conn": graph_connectivity(x, ct),
        "Seurat_domain": seurat_alignment_score(x, domain),
    }
    if args.batch_key and args.batch_key in adata.obs:
        batch = adata.obs[args.batch_key].to_numpy().astype(str)
        metrics["ASW_batch"] = avg_silhouette_width_batch(x, batch, ct)
        metrics["Seurat_batch"] = seurat_alignment_score(x, batch)

    bio = np.mean([
        metrics["NMI"],
        metrics["ARI_MAP"],
        metrics["ASW_celltype"],
        metrics["Graph_conn"],
    ])
    integ = metrics.get("Seurat_batch", metrics["Seurat_domain"])
    metrics["Bio_avg"] = bio
    metrics["Integration"] = integ
    metrics["Overall"] = 0.6 * bio + 0.4 * integ
    # Keep compatibility with the existing summary pane and downstream readers.
    metrics["Bio conservation"] = metrics["Bio_avg"]
    metrics["Modality integration"] = metrics["Integration"]
    metrics["Total"] = metrics["Overall"]
    return metrics


def full_eval_one(
    cfg: dict, args: argparse.Namespace, run_dir: Path, preprocess_dir: Path, eval_dir: Path
) -> bool:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "s06_eval.py"),
        "--run-dir", str(run_dir),
        "--preprocessed-dir", str(preprocess_dir),
        "--output-dir", str(eval_dir),
        "--tag", config_name(cfg),
        "--cell-type-key", args.cell_type_key,
        "--domain-key", args.domain_key,
        "--n-jobs", str(args.n_jobs),
        "--no-show",
    ]
    if args.batch_key:
        cmd += ["--batch-key", args.batch_key]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: evaluation failed with exit code {exc.returncode}")
        return False
    return True


def read_full_metrics(eval_dir: Path, cfg: dict) -> dict:
    csv_path = eval_dir / f"{config_name(cfg)}_unscaled.csv"
    row = pd.read_csv(csv_path, index_col=0).iloc[0]
    numeric = pd.to_numeric(row, errors="coerce").dropna().to_dict()
    return numeric


def save_run_artifacts(run_dir: Path, cfg: dict, metrics: dict) -> None:
    with (run_dir / "hparams.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


def save_quick_eval(eval_dir: Path, cfg: dict, metrics: dict) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    stem = config_name(cfg)
    df = pd.DataFrame([metrics])
    df.to_csv(eval_dir / f"{stem}_unscaled.csv", index=False)
    df.to_csv(eval_dir / f"{stem}_scaled.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    source_run = Path(args.source_run)
    preprocess_dir = (
        Path(args.preprocessed_dir)
        if args.preprocessed_dir
        else source_run.parent / "preprocessed"
    )

    source_hparams = load_source_hparams(source_run)
    configs = build_grid(args, source_hparams)

    print(f"Total configs: {len(configs)}")
    print(f"Worker {args.gpu_id}/{args.n_gpus} will run every {args.n_gpus}-th config")
    if args.dry_run:
        for cfg in configs:
            print(config_name(cfg))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if not preprocess_once(args, preprocess_dir):
        sys.exit(1)

    ran = 0
    for cfg in configs:
        if cfg["run_id"] % args.n_gpus != args.gpu_id:
            continue
        run_name = config_name(cfg)
        run_dir = out_dir / run_name
        eval_dir = out_dir / f"{run_name}_eval"
        run_dir.mkdir(parents=True, exist_ok=True)

        metric_file = run_dir / "metrics.json"
        combined_file = run_dir / "combined_glue.h5ad"
        eval_file = eval_dir / f"{run_name}_unscaled.csv"
        if args.resume and metric_file.exists():
            print(f"Skipping completed run: {run_name}")
            continue

        with (run_dir / "hparams.json").open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

        ok = True
        metrics = None
        if not (args.resume and combined_file.exists()):
            ok = train_one(cfg, args, run_dir, preprocess_dir)
        if ok:
            if args.eval_mode == "quick":
                metrics = quick_eval_one(cfg, args, run_dir)
                save_quick_eval(eval_dir, cfg, metrics)
            elif not (args.resume and eval_file.exists()):
                ok = full_eval_one(cfg, args, run_dir, preprocess_dir, eval_dir)
        if ok:
            if metrics is None:
                metrics = read_full_metrics(eval_dir, cfg)
            save_run_artifacts(run_dir, cfg, metrics)
            ran += 1
        else:
            print(f"Run failed: {run_name}")

    print(f"Worker {args.gpu_id} completed {ran} runs")


if __name__ == "__main__":
    main()
