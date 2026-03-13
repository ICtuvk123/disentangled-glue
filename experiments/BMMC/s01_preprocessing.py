import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from muon import atac as ac
from muon import prot as pt


FINAL_CELLTYPE_MAPPING = {
    "CD4+ T naive": "CD4 T",
    "CD4+ T activated": "CD4 T",
    "CD4+ T activated integrinB7+": "CD4 T",
    "CD4+ T CD314+ CD45RA+": "CD4 T",
    "T reg": "CD4 T",
    "CD8+ T naive": "CD8 T",
    "CD8+ T naive CD127+ CD26- CD101-": "CD8 T",
    "CD8+ T CD49f+": "CD8 T",
    "CD8+ T TIGIT+ CD45RO+": "CD8 T",
    "CD8+ T CD57+ CD45RA+": "CD8 T",
    "CD8+ T CD69+ CD45RO+": "CD8 T",
    "CD8+ T TIGIT+ CD45RA+": "CD8 T",
    "CD8+ T CD69+ CD45RA+": "CD8 T",
    "CD8+ T CD57+ CD45RO+": "CD8 T",
    "CD8+ T": "CD8 T",
    "MAIT": "CD8 T",
    "gdT TCRVD2+": "CD8 T",
    "gdT CD158b+": "CD8 T",
    "dnT": "CD8 T",
    "Naive CD20+ B IGKC+": "B cell",
    "Naive CD20+ B IGKC-": "B cell",
    "Naive CD20+ B": "B cell",
    "B1 B IGKC+": "B cell",
    "B1 B IGKC-": "B cell",
    "B1 B": "B cell",
    "Transitional B": "B cell",
    "Plasmablast IGKC+": "Plasma cell",
    "Plasmablast IGKC-": "Plasma cell",
    "Plasma cell IGKC+": "Plasma cell",
    "Plasma cell IGKC-": "Plasma cell",
    "Plasma cell": "Plasma cell",
    "NK": "NK",
    "NK CD158e1+": "NK",
    "CD14+ Mono": "Mono",
    "CD16+ Mono": "Mono",
    "pDC": "DC",
    "cDC1": "DC",
    "cDC2": "DC",
    "ILC": "ILC",
    "ILC1": "ILC",
    "HSC": "Progenitor",
    "Lymph prog": "Progenitor",
    "G/M prog": "Progenitor",
    "MK/E prog": "Progenitor",
    "ID2-hi myeloid prog": "Progenitor",
    "T prog cycling": "Progenitor",
    "Erythroblast": "Erythroid",
    "Normoblast": "Erythroid",
    "Proerythroblast": "Erythroid",
    "Reticulocyte": "Erythroid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess BMMC multiome/CITE-seq data for GLUE experiments."
    )
    parser.add_argument(
        "--multiome",
        default="data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad",
        help="Relative or absolute path to the processed BMMC multiome h5ad",
    )
    parser.add_argument(
        "--cite",
        default="data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad",
        help="Relative or absolute path to the processed BMMC CITE-seq h5ad",
    )
    parser.add_argument(
        "--gtf",
        default="data/gencode.v38.primary_assembly.annotation.gtf",
        help="Relative or absolute path to the GTF used for ATAC gene activity",
    )
    parser.add_argument(
        "--hgnc",
        default="data/hgnc_complete_set.txt",
        help="Relative or absolute path to the HGNC mapping table",
    )
    parser.add_argument(
        "--output-dir",
        default="s01_preprocessing",
        help="Output directory",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.2,
        help="Sampling fraction for sampled outputs",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--sampled-only",
        action="store_true",
        help="Only generate the sampled RNA/ATAC/protein files needed by GLUE",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else base_dir / path


def subset_obsm(adata: ad.AnnData, keys: list[str]) -> None:
    adata.obsm = {key: adata.obsm[key] for key in keys if key in adata.obsm}


def qc_rna(rna: ad.AnnData) -> ad.AnnData:
    rna = rna.copy()
    rna.var["mt"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    sc.pp.filter_genes(rna, min_cells=3)
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.filter_cells(rna, max_genes=5000)
    sc.pp.filter_cells(rna, max_counts=15000)
    rna = rna[rna.obs["pct_counts_mt"] < 20, :].copy()
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(
        rna, batch_key="batch", min_mean=0.02, max_mean=4, min_disp=0.5
    )
    return rna


def qc_atac(atac: ad.AnnData) -> ad.AnnData:
    atac = atac.copy()
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_genes(atac, min_cells=10)
    sc.pp.filter_cells(atac, min_genes=2000)
    sc.pp.filter_cells(atac, max_genes=15000)
    atac.layers["counts"] = atac.X.copy()
    ac.pp.tfidf(atac, scale_factor=1e4)
    sc.pp.highly_variable_genes(
        atac, min_mean=0.05, max_mean=1.5, min_disp=0.5, batch_key="batch"
    )
    atac.var_names = [name.replace("-", ":", 1) for name in atac.var_names]
    return atac


def build_atac_gene_activity(atac: ad.AnnData, gtf: Path) -> ad.AnnData:
    import episcanpy as epi

    atac_for_ga = atac.copy()
    atac_for_ga.layers["normalized"] = atac_for_ga.X.copy()
    atac_for_ga.X = atac_for_ga.layers["counts"].copy()
    atac_gas = epi.tl.geneactivity(
        atac_for_ga, str(gtf), annotation="HAVANA"
    )
    atac_gas = atac_gas[:, ~atac_gas.var_names.duplicated()].copy()
    atac_gas.layers["counts"] = atac_gas.X.copy()
    ac.pp.tfidf(atac_gas, scale_factor=1e4)
    sc.pp.highly_variable_genes(atac_gas)
    return atac_gas


def process_protein(prot: ad.AnnData) -> ad.AnnData:
    prot = prot.copy()
    prot.layers["counts"] = prot.X.copy()
    pt.pp.clr(prot)
    return prot


def map_celltypes(*adatas: ad.AnnData) -> None:
    for adata in adatas:
        adata.obs["celltype"] = adata.obs["cell_type"].map(FINAL_CELLTYPE_MAPPING)


def build_protein_gene_map(prot: ad.AnnData, hgnc: pd.DataFrame) -> pd.DataFrame:
    hgnc = hgnc.copy()
    if {"Approved symbol", "Previous symbols", "Aliases"}.issubset(hgnc.columns):
        approved_col = "Approved symbol"
        previous_col = "Previous symbols"
        alias_col = "Aliases"
    elif {"symbol", "prev_symbol", "alias_symbol"}.issubset(hgnc.columns):
        approved_col = "symbol"
        previous_col = "prev_symbol"
        alias_col = "alias_symbol"
    else:
        raise ValueError(
            "HGNC table must contain either "
            "{'Approved symbol', 'Previous symbols', 'Aliases'} "
            "or {'symbol', 'prev_symbol', 'alias_symbol'}"
        )

    hgnc[[previous_col, alias_col]] = hgnc[[previous_col, alias_col]].fillna("")

    mapped_gene_names = []
    for cd_name in prot.var_names:
        mapped = cd_name
        for _, row in hgnc.iterrows():
            approved = row[approved_col]
            previous = [item.strip() for item in str(row[previous_col]).split("|")]
            aliases = [item.strip() for item in str(row[alias_col]).split("|")]
            if len(previous) == 1 and "," in previous[0]:
                previous = [item.strip() for item in previous[0].split(",")]
            if len(aliases) == 1 and "," in aliases[0]:
                aliases = [item.strip() for item in aliases[0].split(",")]
            if cd_name == approved or cd_name in previous or cd_name in aliases:
                mapped = approved
                break
        mapped_gene_names.append(mapped)

    return pd.DataFrame(
        {"protein": prot.var_names.to_list(), "gene": mapped_gene_names}
    )


def sample_adata(
    adata: ad.AnnData, frac: float, rng: np.random.Generator
) -> tuple[ad.AnnData, np.ndarray]:
    size = int(frac * adata.n_obs)
    indices = rng.choice(np.arange(adata.n_obs), size=size, replace=False)
    return adata[indices, :].copy(), indices


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = resolve_path(base_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    multiome_path = resolve_path(base_dir, args.multiome)
    cite_path = resolve_path(base_dir, args.cite)
    hgnc_path = resolve_path(base_dir, args.hgnc)
    gtf_path = resolve_path(base_dir, args.gtf)

    multiome = sc.read(multiome_path)
    multiome.X = multiome.layers["counts"].copy()
    multiome.obs["cellid"] = list(range(multiome.n_obs))

    rna_multiome = multiome[:, multiome.var.index[multiome.var.feature_types == "GEX"]].copy()
    atac = multiome[:, multiome.var.index[multiome.var.feature_types == "ATAC"]].copy()
    subset_obsm(rna_multiome, ["GEX_X_pca", "GEX_X_umap"])
    subset_obsm(atac, ["ATAC_gene_activity", "ATAC_lsi_full", "ATAC_lsi_red", "ATAC_umap"])

    rna_multiome = qc_rna(rna_multiome)
    atac = qc_atac(atac)

    rna_multiome.write_h5ad(output_dir / "RNA_multiome_qc.h5ad")
    atac.write_h5ad(output_dir / "ATAC_counts_qc.h5ad")

    cite = sc.read(cite_path)
    cite.X = cite.layers["counts"].copy()
    cite.obs["cellid"] = list(range(cite.n_obs))

    gex_count = len(cite.var.index[cite.var.feature_types == "GEX"])
    rna_cite = cite[:, :gex_count].copy()
    prot = cite[:, gex_count:].copy()
    subset_obsm(rna_cite, ["GEX_X_pca", "GEX_X_umap"])
    subset_obsm(prot, ["ADT_X_pca", "ADT_X_umap", "ADT_isotype_controls"])

    rna_cite = qc_rna(rna_cite)
    prot = process_protein(prot)

    rna = ad.concat([rna_multiome, rna_cite], join="outer")
    rna.write_h5ad(output_dir / "RNA_counts_qc.h5ad")
    rna_cite.write_h5ad(output_dir / "RNA_cite_qc.h5ad")
    prot.write_h5ad(output_dir / "protein_counts_qc.h5ad")

    map_celltypes(rna, atac, prot)

    hgnc_sep = "\t" if hgnc_path.suffix in {".txt", ".tsv"} else ","
    hgnc = pd.read_csv(hgnc_path, sep=hgnc_sep)
    protein_gene_map = build_protein_gene_map(prot, hgnc)
    protein_gene_map.to_csv(output_dir / "protein_gene_map.tsv", sep="\t", index=False)

    prot.var["cd_name"] = prot.var_names
    prot.var["gene_name"] = protein_gene_map["gene"].to_numpy()
    prot.var_names = prot.var["gene_name"].to_numpy()

    rna.write_h5ad(output_dir / "RNA_counts_qc.h5ad")
    atac.write_h5ad(output_dir / "ATAC_counts_qc.h5ad")
    prot.write_h5ad(output_dir / "protein_counts_qc.h5ad")

    rng = np.random.default_rng(args.random_seed)
    rna_sampled, _ = sample_adata(rna, args.sample_frac, rng)
    atac_sampled, atac_indices = sample_adata(atac, args.sample_frac, rng)
    prot_sampled, _ = sample_adata(prot, args.sample_frac, rng)

    rna_sampled.write_h5ad(output_dir / "RNA_counts_qc_sampled.h5ad")
    atac_sampled.write_h5ad(output_dir / "ATAC_counts_qc_sampled.h5ad")
    prot_sampled.write_h5ad(output_dir / "protein_counts_qc_sampled.h5ad")

    if args.sampled_only:
        return

    atac_gas = build_atac_gene_activity(atac, gtf_path)
    atac_gas.obs["celltype"] = atac_gas.obs["cell_type"].map(FINAL_CELLTYPE_MAPPING)
    atac_gas.write_h5ad(output_dir / "ATAC_gas.h5ad")

    sc.pp.highly_variable_genes(
        rna, batch_key="batch", min_mean=0.02, max_mean=4, min_disp=0.5
    )
    genelist = rna.var.index[rna.var["highly_variable"]].tolist()
    peaklist = atac_gas.var.index[atac_gas.var["highly_variable"]].tolist()
    aligned_features = list(set(genelist) | set(peaklist) | set(prot.var.index))

    feature_aligned = ad.concat([rna, atac_gas, prot], join="outer", label="modality")
    feature_aligned.uns["rna_hvg"] = genelist
    feature_aligned.uns["atac_hvg"] = peaklist
    feature_aligned.uns["prot_hvg"] = prot.var.index.tolist()
    feature_aligned.uns["rna_nz"] = list(set(aligned_features) & set(rna.var.index))
    feature_aligned.uns["atac_nz"] = list(set(aligned_features) & set(atac_gas.var.index))
    feature_aligned.uns["prot_nz"] = list(set(aligned_features) & set(prot.var.index))
    feature_aligned = feature_aligned[:, aligned_features].copy()
    feature_aligned.write_h5ad(output_dir / "feature_aligned.h5ad")

    atac_gas_sampled = atac_gas[atac_indices, :].copy()
    atac_gas_sampled.write_h5ad(output_dir / "ATAC_gas_sampled.h5ad")

    feature_aligned_sampled = ad.concat(
        [rna_sampled, atac_gas_sampled, prot_sampled], join="outer", label="modality"
    )
    sampled_genelist = rna_sampled.var.index[rna_sampled.var["highly_variable"]].tolist()
    sampled_peaklist = atac_gas_sampled.var.index[
        atac_gas_sampled.var["highly_variable"]
    ].tolist()
    sampled_aligned_features = list(
        set(sampled_genelist) | set(sampled_peaklist) | set(prot_sampled.var.index)
    )
    feature_aligned_sampled.uns["rna_hvg"] = sampled_genelist
    feature_aligned_sampled.uns["atac_hvg"] = sampled_peaklist
    feature_aligned_sampled.uns["prot_hvg"] = prot_sampled.var.index.tolist()
    feature_aligned_sampled.uns["rna_nz"] = list(
        set(sampled_aligned_features) & set(rna_sampled.var.index)
    )
    feature_aligned_sampled.uns["atac_nz"] = list(
        set(sampled_aligned_features) & set(atac_gas_sampled.var.index)
    )
    feature_aligned_sampled.uns["prot_nz"] = list(
        set(sampled_aligned_features) & set(prot_sampled.var.index)
    )
    feature_aligned_sampled = feature_aligned_sampled[:, sampled_aligned_features].copy()
    feature_aligned_sampled.write_h5ad(output_dir / "feature_aligned_sampled.h5ad")


if __name__ == "__main__":
    main()
