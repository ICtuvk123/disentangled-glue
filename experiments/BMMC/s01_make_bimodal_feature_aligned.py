"""
Build a clean RNA+ATAC feature-aligned AnnData for bimodal evaluation.

This mirrors the RNA/ATAC subset of `s01_preprocessing.py` without
introducing protein features into the shared pre-integration baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rna",
        default="s01_preprocessing/RNA_counts_qc_sampled.h5ad",
        help="Path to sampled RNA counts h5ad",
    )
    parser.add_argument(
        "--atac-gas",
        default="s01_preprocessing/ATAC_gas_sampled.h5ad",
        help="Path to sampled ATAC gene-activity h5ad",
    )
    parser.add_argument(
        "--output",
        default="s01_preprocessing/feature_aligned_bimodal_sampled.h5ad",
        help="Output path for bimodal feature-aligned h5ad",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    rna_path = resolve_path(base_dir, args.rna)
    atac_gas_path = resolve_path(base_dir, args.atac_gas)
    output_path = resolve_path(base_dir, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rna = sc.read(rna_path)
    atac_gas = sc.read(atac_gas_path)

    genelist = rna.var.index[rna.var["highly_variable"]].tolist()
    peaklist = atac_gas.var.index[atac_gas.var["highly_variable"]].tolist()
    aligned_features = sorted(set(genelist) | set(peaklist))

    feature_aligned = ad.concat(
        [rna, atac_gas], join="outer", label="modality", keys=["rna", "atac"]
    )
    feature_aligned.uns["rna_hvg"] = genelist
    feature_aligned.uns["atac_hvg"] = peaklist
    feature_aligned.uns["rna_nz"] = sorted(set(aligned_features) & set(rna.var.index))
    feature_aligned.uns["atac_nz"] = sorted(set(aligned_features) & set(atac_gas.var.index))
    feature_aligned = feature_aligned[:, aligned_features].copy()
    feature_aligned.write_h5ad(output_path)


if __name__ == "__main__":
    main()
