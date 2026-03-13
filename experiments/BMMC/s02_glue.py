import argparse
import os
from itertools import chain
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

import scglue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SCGLUE on BMMC RNA/ATAC/protein data."
    )
    parser.add_argument("--rna", required=True, help="Path to RNA h5ad")
    parser.add_argument("--atac", required=True, help="Path to ATAC h5ad")
    parser.add_argument("--prot", required=True, help="Path to protein h5ad")
    parser.add_argument("--gtf", required=True, help="Path to GTF annotation file")
    parser.add_argument(
        "--output-dir",
        default="s02_glue",
        help="Output directory for preprocessed data and model artifacts",
    )
    parser.add_argument(
        "--protein-gene-map",
        default=None,
        help="Optional TSV/CSV with two columns: protein,gene",
    )
    parser.add_argument(
        "--bedtools",
        default=None,
        help="Optional path to the bedtools executable",
    )
    parser.add_argument(
        "--rna-layer",
        default="counts",
        help="Layer used by GLUE for RNA counts",
    )
    parser.add_argument(
        "--atac-layer",
        default="counts",
        help="Layer used by GLUE for ATAC counts",
    )
    parser.add_argument(
        "--prot-layer",
        default="counts",
        help="Layer used by GLUE for protein counts",
    )
    parser.add_argument(
        "--prot-model",
        default="NBMixture",
        choices=["NB", "NBMixture"],
        help="Probabilistic model for the protein modality",
    )
    parser.add_argument(
        "--model",
        default="disentangled",
        choices=["scglue", "paired", "disentangled"],
        help="SCGLUE variant to train",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Use PairedSCGLUEModel and obs_names to align matched cells",
    )
    parser.add_argument(
        "--shared-batches",
        action="store_true",
        help="Assume batch labels are shared across modalities",
    )
    parser.add_argument(
        "--batch-key",
        default=None,
        help="Optional obs column used as batch label",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used for model initialization",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=50,
        help="Total latent dimensionality",
    )
    parser.add_argument(
        "--shared-dim",
        type=int,
        default=30,
        help="Shared latent dimensionality for the disentangled model",
    )
    parser.add_argument(
        "--beta-shared",
        type=float,
        default=4.0,
        help="Shared KL weight for the disentangled model",
    )
    parser.add_argument(
        "--beta-private",
        type=float,
        default=1.0,
        help="Private KL weight for the disentangled model",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Compute neighbors and UMAP for the combined embedding",
    )
    return parser.parse_args()


def read_adata(path: str) -> ad.AnnData:
    return sc.read(path)


def ensure_layer(adata: ad.AnnData, layer: str) -> None:
    if layer not in adata.layers:
        adata.layers[layer] = adata.X.copy()


def ensure_obs_names_unique(adatas: dict[str, ad.AnnData]) -> None:
    for key, adata in adatas.items():
        if not adata.obs_names.is_unique:
            raise ValueError(f"{key}.obs_names are not unique")


def deduplicate_vars(adata: ad.AnnData) -> ad.AnnData:
    return adata[:, ~adata.var_names.duplicated(keep="first")].copy()


def ensure_hvg(adata: ad.AnnData, modality: str) -> None:
    if "highly_variable" not in adata.var:
        raise ValueError(f"{modality}.var['highly_variable'] is required")


def auto_mark_rna_hvg(rna: ad.AnnData) -> None:
    if "highly_variable" in rna.var:
        return
    tmp = rna.copy()
    if "counts" in tmp.layers:
        tmp.X = tmp.layers["counts"].copy()
    sc.pp.normalize_total(tmp)
    sc.pp.log1p(tmp)
    hvg_kws = {"min_mean": 0.02, "max_mean": 4, "min_disp": 0.5}
    if "batch" in tmp.obs:
        hvg_kws["batch_key"] = "batch"
    sc.pp.highly_variable_genes(tmp, **hvg_kws)
    rna.var["highly_variable"] = tmp.var["highly_variable"].to_numpy()


def prepare_representations(rna: ad.AnnData, atac: ad.AnnData, prot: ad.AnnData) -> None:
    if "X_pca" not in rna.obsm:
        sc.tl.pca(rna, n_comps=100, svd_solver="auto")
    if "X_pca" not in prot.obsm:
        sc.tl.pca(prot, n_comps=100, svd_solver="auto")
    if "X_lsi" not in atac.obsm:
        scglue.data.lsi(atac, n_components=100, n_iter=15)


def parse_peak_coordinates(atac: ad.AnnData) -> None:
    atac.var_names = [name.replace("-", ":", 1) for name in atac.var_names]
    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda item: item[0])
    atac.var["chromStart"] = split.map(lambda item: item[1]).astype(int)
    atac.var["chromEnd"] = split.map(lambda item: item[2]).astype(int)


def remove_duplicate_var_columns(adata: ad.AnnData) -> None:
    adata.var = adata.var.loc[:, ~adata.var.columns.duplicated(keep="first")]


def filter_missing_coordinates(adata: ad.AnnData) -> ad.AnnData:
    keep = ~adata.var[["chrom", "chromStart", "chromEnd"]].isna().any(axis=1)
    return adata[:, keep].copy()


def load_protein_gene_map(path: str) -> dict[str, str]:
    sep = "\t" if path.endswith(".tsv") else ","
    mapping = pd.read_csv(path, sep=sep)
    if mapping.shape[1] < 2:
        raise ValueError("protein_gene_map must have at least two columns")
    mapping = mapping.iloc[:, :2]
    mapping.columns = ["protein", "gene"]
    return dict(zip(mapping["protein"], mapping["gene"]))


def build_guidance_graph(
    rna: ad.AnnData,
    atac: ad.AnnData,
    prot: ad.AnnData,
    args: argparse.Namespace,
) -> nx.MultiDiGraph:
    scglue.data.get_gene_annotation(rna, gtf=args.gtf, gtf_by="gene_name")
    remove_duplicate_var_columns(rna)
    parse_peak_coordinates(atac)
    remove_duplicate_var_columns(atac)

    rna = filter_missing_coordinates(rna)
    atac = filter_missing_coordinates(atac)

    if args.protein_gene_map:
        g_rna_atac = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
        prot_map = load_protein_gene_map(args.protein_gene_map)
        g_rna_prot = scglue.genomics.generate_prot_guidance_graph(rna, prot, prot_map)
        guidance = scglue.graph.compose_multigraph(g_rna_atac, g_rna_prot)
        for node in guidance.nodes:
            if not guidance.has_edge(node, node):
                guidance.add_edge(node, node, weight=1.0, sign=1, type="loop")
    else:
        scglue.data.get_gene_annotation(prot, gtf=args.gtf, gtf_by="gene_name")
        remove_duplicate_var_columns(prot)
        prot = filter_missing_coordinates(prot)
        guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac, prot)

    scglue.graph.check_graph(guidance, [rna, atac, prot])
    return rna, atac, prot, guidance


def configure_datasets(
    rna: ad.AnnData,
    atac: ad.AnnData,
    prot: ad.AnnData,
    args: argparse.Namespace,
) -> None:
    common = {
        "use_batch": args.batch_key,
        "use_obs_names": args.paired,
    }
    scglue.models.configure_dataset(
        rna,
        "NB",
        use_highly_variable=True,
        use_layer=args.rna_layer,
        use_rep="X_pca",
        **common,
    )
    scglue.models.configure_dataset(
        atac,
        "NB",
        use_highly_variable=True,
        use_layer=args.atac_layer,
        use_rep="X_lsi",
        **common,
    )
    scglue.models.configure_dataset(
        prot,
        args.prot_model,
        use_highly_variable=False,
        use_layer=args.prot_layer,
        use_rep="X_pca",
        **common,
    )


def train_glue(
    rna: ad.AnnData,
    atac: ad.AnnData,
    prot: ad.AnnData,
    guidance: nx.MultiDiGraph,
    args: argparse.Namespace,
):
    if args.model == "disentangled":
        model_cls = scglue.models.DisentangledSCGLUEModel
    elif args.model == "paired" or args.paired:
        model_cls = scglue.models.PairedSCGLUEModel
    else:
        model_cls = scglue.models.SCGLUEModel
    guidance_hvf = guidance.subgraph(
        chain(
            rna.var.query("highly_variable").index,
            atac.var.query("highly_variable").index,
            prot.var_names,
        )
    ).copy()
    init_kws = {
        "shared_batches": args.shared_batches,
        "random_seed": args.random_seed,
        "latent_dim": args.latent_dim,
    }
    compile_kws = {}
    if args.model == "disentangled":
        init_kws["shared_dim"] = args.shared_dim
        compile_kws.update(
            {
                "beta_shared": args.beta_shared,
                "beta_private": args.beta_private,
            }
        )
    return scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac, "prot": prot},
        guidance_hvf,
        model=model_cls,
        init_kws=init_kws,
        compile_kws=compile_kws,
        fit_kws={"directory": os.fspath(Path(args.output_dir) / "glue")},
    ), guidance_hvf


def save_embeddings(
    glue,
    rna: ad.AnnData,
    atac: ad.AnnData,
    prot: ad.AnnData,
    output_dir: Path,
    run_umap: bool,
) -> None:
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    prot.obsm["X_glue"] = glue.encode_data("prot", prot)

    for key, adata in (("rna", rna), ("atac", atac), ("prot", prot)):
        if "domain" not in adata.obs:
            adata.obs["domain"] = key

    rna_obs = rna.obs.copy()
    atac_obs = atac.obs.copy()
    prot_obs = prot.obs.copy()
    rna_obs.index = pd.Index([f"rna:{item}" for item in rna_obs.index])
    atac_obs.index = pd.Index([f"atac:{item}" for item in atac_obs.index])
    prot_obs.index = pd.Index([f"prot:{item}" for item in prot_obs.index])

    combined = ad.AnnData(
        obs=pd.concat([rna_obs, atac_obs, prot_obs], join="outer"),
        obsm={
            "X_glue": np.concatenate(
                [rna.obsm["X_glue"], atac.obsm["X_glue"], prot.obsm["X_glue"]]
            )
        },
    )
    if run_umap:
        sc.pp.neighbors(combined, n_pcs=50, use_rep="X_glue", metric="cosine")
        sc.tl.umap(combined)

    np.save(output_dir / "combined_glue.npy", combined.obsm["X_glue"])
    rna.write(output_dir / "rna_glue.h5ad", compression="gzip")
    atac.write(output_dir / "atac_glue.h5ad", compression="gzip")
    prot.write(output_dir / "prot_glue.h5ad", compression="gzip")
    combined.write(output_dir / "combined_glue.h5ad", compression="gzip")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.bedtools:
        scglue.config.BEDTOOLS_PATH = args.bedtools

    rna = deduplicate_vars(read_adata(args.rna))
    atac = deduplicate_vars(read_adata(args.atac))
    prot = deduplicate_vars(read_adata(args.prot))

    ensure_layer(rna, args.rna_layer)
    ensure_layer(atac, args.atac_layer)
    ensure_layer(prot, args.prot_layer)
    auto_mark_rna_hvg(rna)
    ensure_hvg(rna, "rna")
    ensure_hvg(atac, "atac")
    ensure_obs_names_unique({"rna": rna, "atac": atac, "prot": prot})

    prepare_representations(rna, atac, prot)
    rna, atac, prot, guidance = build_guidance_graph(rna, atac, prot, args)

    rna.write(output_dir / "rna_pp.h5ad", compression="gzip")
    atac.write(output_dir / "atac_pp.h5ad", compression="gzip")
    prot.write(output_dir / "prot_pp.h5ad", compression="gzip")
    nx.write_graphml(guidance, output_dir / "guidance.graphml.gz")

    configure_datasets(rna, atac, prot, args)
    glue, guidance_hvf = train_glue(rna, atac, prot, guidance, args)
    glue.save(output_dir / "glue.dill")
    nx.write_graphml(guidance_hvf, output_dir / "guidance_hvf.graphml.gz")

    save_embeddings(glue, rna, atac, prot, output_dir, args.umap)


if __name__ == "__main__":
    main()
