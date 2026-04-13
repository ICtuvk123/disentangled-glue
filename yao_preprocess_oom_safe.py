#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OOM-safer preprocessing pipeline for the Yao-2021 RNA/ATAC dataset.

Main goals:
1. Keep the original logic as much as possible.
2. Avoid accidental dense conversion and unnecessary full-matrix copies.
3. Save intermediate results to disk instead of storing giant layers in memory.

Example:
python yao_preprocess_oom_safe.py \
  --rna /ailab/user/sunjianle-hdd/integration27/mop/Yao-2021-RNA.h5ad \
  --atac /ailab/user/sunjianle-hdd/integration27/mop/Yao-2021-ATAC.h5ad \
  --gtf /ailab/user/sunjianle-hdd/integration27/BMMC/gencode.vM10.chr_patch_hapl_scaff.annotation.gtf \
  --outdir /ailab/user/sunjianle-hdd/integration27/mop/Yao
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import scanpy as sc
import episcanpy as epi
from muon import atac as ac
from scipy import sparse

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# -----------------------------
# utilities
# -----------------------------

def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def mem_info(prefix: str = "") -> None:
    if psutil is None:
        return
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    used_gb = vm.used / (1024 ** 3)
    total_gb = vm.total / (1024 ** 3)
    tag = f"{prefix} " if prefix else ""
    print(
        f"[MEM] {tag}proc_rss={rss_gb:.2f} GB | system_used={used_gb:.2f}/{total_gb:.2f} GB",
        flush=True,
    )


def ensure_float32_csr(adata: ad.AnnData, name: str) -> ad.AnnData:
    log(f"Converting {name}.X to memory-friendlier format")
    if sparse.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    mem_info(f"after format cast for {name}")
    return adata



def drop_layers_and_raw(adata: ad.AnnData, keep_layers: Optional[Iterable[str]] = None) -> ad.AnnData:
    keep = set(keep_layers or [])
    if adata.raw is not None:
        adata.raw = None
    for key in list(adata.layers.keys()):
        if key not in keep:
            del adata.layers[key]
    gc.collect()
    return adata



def save_h5ad(adata: ad.AnnData, path: Path, compression: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing: {path}")
    if compression:
        adata.write_h5ad(path, compression=compression)
    else:
        adata.write_h5ad(path)



def maybe_plot_violin(
    adata: ad.AnnData,
    keys: list[str],
    outdir: Path,
    stem: str,
    enabled: bool,
) -> None:
    if not enabled:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(outdir)
    log(f"Saving QC violin plot: {stem}.png")
    sc.pl.violin(adata, keys, jitter=0.4, multi_panel=True, show=False, save=f"_{stem}.png")


# -----------------------------
# RNA
# -----------------------------

def process_rna(
    rna_path: Path,
    outdir: Path,
    compression: Optional[str],
    plot_qc: bool,
    do_filter: bool,
) -> Path:
    log(f"Reading RNA: {rna_path}")
    rna = sc.read_h5ad(rna_path)
    mem_info("after RNA load")

    rna.var_names_make_unique()
    rna = drop_layers_and_raw(rna)
    rna = ensure_float32_csr(rna, "RNA")

    # QC
    rna.var["mt"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    maybe_plot_violin(
        rna,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        outdir / "figures",
        "rna_qc",
        plot_qc,
    )

    if do_filter:
        log("Applying RNA filters")
        sc.pp.filter_genes(rna, min_cells=3)
        sc.pp.filter_cells(rna, min_genes=200)
        rna = rna[rna.obs["pct_counts_mt"] < 20, :].copy()
        mem_info("after RNA filtering")

    # batch
    if "Seq_batch" not in rna.obs.columns:
        raise KeyError("RNA obs does not contain 'Seq_batch'.")
    rna.obs["batch"] = rna.obs["Seq_batch"].astype("category")

    log("RNA normalize_total")
    sc.pp.normalize_total(rna)
    mem_info("after RNA normalize_total")

    log("RNA log1p")
    sc.pp.log1p(rna)
    mem_info("after RNA log1p")

    log("RNA highly_variable_genes")
    sc.pp.highly_variable_genes(
        rna,
        batch_key="batch",
        min_mean=0.02,
        max_mean=4,
        min_disp=0.5,
    )
    mem_info("after RNA HVG")

    n_hvg = int(np.sum(rna.var["highly_variable"].values))
    log(f"RNA HVG count: {n_hvg}")

    rna_out = outdir / "RNA_counts_qc.h5ad"
    save_h5ad(rna, rna_out, compression=compression)

    del rna
    gc.collect()
    mem_info("after RNA cleanup")
    return rna_out


# -----------------------------
# ATAC raw + tfidf/hvg
# -----------------------------

def process_atac_raw_and_tfidf(
    atac_path: Path,
    outdir: Path,
    compression: Optional[str],
    plot_qc: bool,
    do_filter: bool,
) -> tuple[Path, Path]:
    log(f"Reading ATAC: {atac_path}")
    atac = sc.read_h5ad(atac_path)
    mem_info("after ATAC load")

    atac.var_names_make_unique()
    atac = drop_layers_and_raw(atac)
    atac = ensure_float32_csr(atac, "ATAC")

    if "batch" not in atac.obs.columns:
        raise KeyError("ATAC obs does not contain 'batch'.")
    atac.obs["batch"] = atac.obs["batch"].astype("category")

    # QC
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    maybe_plot_violin(
        atac,
        ["total_counts", "n_genes_by_counts"],
        outdir / "figures",
        "atac_qc",
        plot_qc,
    )

    if do_filter:
        log("Applying ATAC filters")
        sc.pp.filter_genes(atac, min_cells=10)
        sc.pp.filter_cells(atac, min_genes=2000)
        atac = atac[atac.obs["n_genes_by_counts"] <= 15000, :].copy()
        mem_info("after ATAC filtering")

    # Save raw counts version for downstream gene activity.
    atac_raw_out = outdir / "ATAC_raw_qc.h5ad"
    save_h5ad(atac, atac_raw_out, compression=compression)

    log("ATAC TF-IDF")
    ac.pp.tfidf(atac, scale_factor=1e4)
    mem_info("after ATAC TFIDF")

    log("ATAC highly_variable_genes")
    sc.pp.highly_variable_genes(atac, batch_key="batch")
    mem_info("after ATAC HVG")

    n_hvg = int(np.sum(atac.var["highly_variable"].values))
    log(f"ATAC HVG count: {n_hvg}")

    atac_tfidf_out = outdir / "ATAC_tfidf_hvg.h5ad"
    save_h5ad(atac, atac_tfidf_out, compression=compression)

    del atac
    gc.collect()
    mem_info("after ATAC cleanup")
    return atac_raw_out, atac_tfidf_out


# -----------------------------
# gene activity
# -----------------------------

def process_gene_activity(
    atac_raw_path: Path,
    gtf_path: Path,
    outdir: Path,
    compression: Optional[str],
) -> Path:
    log(f"Reading raw-count ATAC for gene activity: {atac_raw_path}")
    atac = sc.read_h5ad(atac_raw_path)
    mem_info("after ATAC raw reload")

    atac = drop_layers_and_raw(atac)
    atac = ensure_float32_csr(atac, "ATAC raw for gene activity")

    log("Running episcanpy geneactivity")
    atac_gas = epi.tl.geneactivity(atac, str(gtf_path), annotation="HAVANA")
    mem_info("after geneactivity")

    # Avoid view warnings and duplicated variable names.
    atac_gas = atac_gas[:, ~atac_gas.var_names.duplicated()].copy()
    atac_gas.var_names_make_unique()
    atac_gas = drop_layers_and_raw(atac_gas)
    atac_gas = ensure_float32_csr(atac_gas, "ATAC gene activity")

    # Save raw gene activity counts version if needed later.
    atac_gas_raw_out = outdir / "ATAC_gas_raw.h5ad"
    save_h5ad(atac_gas, atac_gas_raw_out, compression=compression)

    log("ATAC gene activity normalize_total")
    sc.pp.normalize_total(atac_gas)
    mem_info("after atac_gas normalize_total")

    log("ATAC gene activity log1p")
    sc.pp.log1p(atac_gas)
    mem_info("after atac_gas log1p")

    log("ATAC gene activity highly_variable_genes")
    sc.pp.highly_variable_genes(
        atac_gas,
        batch_key="batch",
        min_mean=0.02,
        max_mean=4,
        min_disp=0.5,
    )
    mem_info("after atac_gas HVG")

    n_hvg = int(np.sum(atac_gas.var["highly_variable"].values))
    log(f"ATAC gene activity HVG count: {n_hvg}")

    atac_gas_out = outdir / "ATAC_gas.h5ad"
    save_h5ad(atac_gas, atac_gas_out, compression=compression)

    del atac, atac_gas
    gc.collect()
    mem_info("after ATAC gene activity cleanup")
    return atac_gas_out


# -----------------------------
# combine modalities
# -----------------------------

def combine_modalities(
    rna_path: Path,
    atac_gas_path: Path,
    outdir: Path,
    compression: Optional[str],
) -> tuple[Path, Path]:
    log(f"Reading processed RNA: {rna_path}")
    rna = sc.read_h5ad(rna_path)
    log(f"Reading processed ATAC gene activity: {atac_gas_path}")
    atac_gas = sc.read_h5ad(atac_gas_path)
    mem_info("after processed RNA/ATAC_GAS reload")

    rna = drop_layers_and_raw(rna)
    atac_gas = drop_layers_and_raw(atac_gas)
    rna = ensure_float32_csr(rna, "processed RNA")
    atac_gas = ensure_float32_csr(atac_gas, "processed ATAC GAS")

    rna_hvg = set(rna.var_names[rna.var["highly_variable"]].tolist())
    atac_hvg = set(atac_gas.var_names[atac_gas.var["highly_variable"]].tolist())
    common = set(rna.var_names) & set(atac_gas.var_names)
    feature_union = sorted((rna_hvg | atac_hvg) & common)
    feature_intersection = sorted((rna_hvg & atac_hvg) & common)

    log(f"feature_union size = {len(feature_union)}")
    log(f"feature_intersection size = {len(feature_intersection)}")

    adata = ad.concat([rna, atac_gas], join="inner", label="modality")
    mem_info("after modality concat")

    adata.uns["rna_hvg"] = sorted(rna_hvg)
    adata.uns["atac_hvg"] = sorted(atac_hvg)
    adata.uns["feature_union"] = feature_union
    adata.uns["feature_intersection"] = feature_intersection

    concat_out = outdir / "rna_atac_geneactivity_concat_inner.h5ad"
    save_h5ad(adata, concat_out, compression=compression)

    feature_aligned = adata[:, feature_union].copy()
    feature_aligned_out = outdir / "feature_aligned.h5ad"
    save_h5ad(feature_aligned, feature_aligned_out, compression=compression)

    del rna, atac_gas, adata, feature_aligned
    gc.collect()
    mem_info("after final cleanup")
    return concat_out, feature_aligned_out


# -----------------------------
# main
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOM-safer preprocessing for Yao RNA/ATAC data")
    parser.add_argument("--rna", type=Path, required=True, help="Path to Yao-2021-RNA.h5ad")
    parser.add_argument("--atac", type=Path, required=True, help="Path to Yao-2021-ATAC.h5ad")
    parser.add_argument("--gtf", type=Path, required=True, help="Path to GTF for gene activity")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        choices=[None, "gzip", "lzf"],
        help="Optional h5ad compression",
    )
    parser.add_argument(
        "--plot-qc",
        action="store_true",
        help="Save QC violin plots. Off by default to save memory/time.",
    )
    parser.add_argument(
        "--do-filter",
        action="store_true",
        help="Apply the commented filtering steps from your notebook. Off by default.",
    )
    return parser



def main() -> int:
    args = build_parser().parse_args()

    for p in [args.rna, args.atac, args.gtf]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    log("Starting pipeline")
    mem_info("startup")

    rna_out = process_rna(
        rna_path=args.rna,
        outdir=args.outdir,
        compression=args.compression,
        plot_qc=args.plot_qc,
        do_filter=args.do_filter,
    )

    atac_raw_out, _ = process_atac_raw_and_tfidf(
        atac_path=args.atac,
        outdir=args.outdir,
        compression=args.compression,
        plot_qc=args.plot_qc,
        do_filter=args.do_filter,
    )

    atac_gas_out = process_gene_activity(
        atac_raw_path=atac_raw_out,
        gtf_path=args.gtf,
        outdir=args.outdir,
        compression=args.compression,
    )

    concat_out, feature_aligned_out = combine_modalities(
        rna_path=rna_out,
        atac_gas_path=atac_gas_out,
        outdir=args.outdir,
        compression=args.compression,
    )

    log("Pipeline finished successfully")
    log(f"Processed RNA: {rna_out}")
    log(f"ATAC raw QC: {atac_raw_out}")
    log(f"ATAC gene activity: {atac_gas_out}")
    log(f"Concatenated common-feature file: {concat_out}")
    log(f"Final feature-aligned file: {feature_aligned_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        warn("Interrupted by user")
        raise
