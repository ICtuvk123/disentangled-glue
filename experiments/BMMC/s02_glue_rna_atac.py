import argparse
import os
from itertools import chain
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from pandas.api.types import is_categorical_dtype

import scglue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SCGLUE on BMMC RNA/ATAC data (two-modality)."
    )
    parser.add_argument("--rna", required=True, help="Path to RNA h5ad")
    parser.add_argument("--atac", required=True, help="Path to ATAC h5ad")
    parser.add_argument("--gtf", required=True, help="Path to GTF annotation file")
    parser.add_argument(
        "--output-dir",
        default="s02_glue_rna_atac",
        help="Output directory for preprocessed data and model artifacts",
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
        "--model",
        default="disentangled",
        choices=["scglue", "paired", "disentangled"],
        help="SCGLUE variant to train",
    )
    parser.add_argument(
        "--prob-model",
        default="NB",
        choices=["NB", "Normal", "ZINormal", "ZINB", "Bernoulli"],
        help="Probability model for data decoders (use Normal for log-normalized data)",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Use obs_names to align matched cells",
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
        "--shared-dim",
        type=int,
        default=50,
        help="Shared latent dimensionality",
    )
    parser.add_argument(
        "--private-dim",
        type=int,
        default=20,
        help="Per-modality private latent dimensionality (disentangled model only)",
    )
    parser.add_argument(
        "--batch-embed-dim",
        type=int,
        default=8,
        help="Batch embedding dimensionality used by the disentangled decoder",
    )
    parser.add_argument(
        "--beta-shared",
        type=float,
        default=4.0,
        help="Shared KL weight for the disentangled model",
    )
    parser.add_argument(
        "--lam-iso",
        type=float,
        default=0.0,
        help="Isometric loss weight (disentangled model only)",
    )
    parser.add_argument(
        "--lam-align",
        type=float,
        default=0.05,
        help="Adversarial alignment weight",
    )

    parser.add_argument(
        "--beta-private",
        type=float,
        default=None,
        help="Private KL weight (broadcast to all modalities). "
             "Overridden by --beta-private-rna/atac if set.",
    )
    parser.add_argument(
        "--beta-private-rna",
        type=float,
        default=None,
        help="Private KL weight for the RNA modality (disentangled model only)",
    )
    parser.add_argument(
        "--beta-private-atac",
        type=float,
        default=None,
        help="Private KL weight for the ATAC modality (disentangled model only)",
    )
    parser.add_argument(
        "--align-support",
        action="store_true",
        help="Estimate unsupported-cell alignment weights after pretraining "
             "and apply them during fine-tuning",
    )
    parser.add_argument(
        "--align-support-k",
        type=int,
        default=15,
        help="Cross-modal neighbor count used for unsupported-cell support estimation",
    )
    parser.add_argument(
        "--align-support-strategy",
        default="soft",
        choices=["soft", "hard"],
        help="Whether to convert support scores into soft weights or hard masks",
    )
    parser.add_argument(
        "--align-support-min-weight",
        type=float,
        default=0.05,
        help="Minimum adversarial alignment weight under soft support weighting",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Compute neighbors and UMAP for the combined embedding",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run preprocessing and graph construction only, then exit",
    )
    parser.add_argument(
        "--preprocessed-dir",
        default=None,
        help="Load preprocessed h5ad and guidance graph from this directory, "
             "skipping data preprocessing and graph construction",
    )
    parser.add_argument(
        "--skip-modality-h5ad",
        action="store_true",
        help="Skip writing rna_glue.h5ad and atac_glue.h5ad (saves ~900MB per run)",
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


def auto_mark_atac_hvg(atac: ad.AnnData, n_top_features: int = 30000) -> None:
    if "highly_variable" in atac.var:
        return
    import scipy.sparse
    X = atac.X
    if scipy.sparse.issparse(X):
        mean_acc = np.asarray(X.mean(axis=0)).ravel()
    else:
        mean_acc = X.mean(axis=0)
    ranked = np.argsort(mean_acc)[::-1]
    mask = np.zeros(atac.n_vars, dtype=bool)
    mask[ranked[:n_top_features]] = True
    atac.var["highly_variable"] = mask


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


def prepare_representations(rna: ad.AnnData, atac: ad.AnnData) -> None:
    if "X_pca" not in rna.obsm:
        sc.tl.pca(rna, n_comps=100, svd_solver="auto")
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


def align_shared_batch_categories(adatas: dict[str, ad.AnnData], batch_key: str) -> None:
    """Coerce all modalities to share the same batch category vocabulary."""
    categories = pd.Index([])
    for adata in adatas.values():
        if batch_key not in adata.obs:
            raise ValueError(f"Missing batch key {batch_key!r} in one modality")
        series = adata.obs[batch_key]
        if is_categorical_dtype(series):
            cats = pd.Index(series.cat.categories)
        else:
            cats = pd.Index(series.dropna().astype(str).unique())
        categories = categories.union(cats)
    categories = categories.sort_values()
    for adata in adatas.values():
        values = adata.obs[batch_key].astype(str)
        values = values.where(~adata.obs[batch_key].isna(), other=pd.NA)
        adata.obs[batch_key] = pd.Categorical(values, categories=categories)


def build_guidance_graph(
    rna: ad.AnnData,
    atac: ad.AnnData,
    args: argparse.Namespace,
) -> tuple:
    scglue.data.get_gene_annotation(rna, gtf=args.gtf, gtf_by="gene_name")
    remove_duplicate_var_columns(rna)
    parse_peak_coordinates(atac)
    remove_duplicate_var_columns(atac)

    rna = filter_missing_coordinates(rna)
    atac = filter_missing_coordinates(atac)

    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    scglue.graph.check_graph(guidance, [rna, atac])
    return rna, atac, guidance


def configure_datasets(
    rna: ad.AnnData,
    atac: ad.AnnData,
    args: argparse.Namespace,
) -> None:
    if args.shared_batches and args.batch_key:
        align_shared_batch_categories({"rna": rna, "atac": atac}, args.batch_key)
    common = {
        "use_batch": args.batch_key,
        "use_obs_names": args.paired,
    }
    scglue.models.configure_dataset(
        rna,
        args.prob_model,
        use_highly_variable=True,
        use_layer=args.rna_layer,
        use_rep="X_pca",
        **common,
    )
    scglue.models.configure_dataset(
        atac,
        args.prob_model,
        use_highly_variable=True,
        use_layer=args.atac_layer,
        use_rep="X_lsi",
        **common,
    )


def train_glue(
    rna: ad.AnnData,
    atac: ad.AnnData,
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
        )
    ).copy()

    init_kws = {
        "shared_batches": args.shared_batches,
        "random_seed": args.random_seed,
        "latent_dim": args.shared_dim,
    }
    compile_kws = {}
    align_support_kws = None

    if args.model == "disentangled":
        init_kws.pop("latent_dim")
        init_kws["shared_dim"] = args.shared_dim
        init_kws["private_dim"] = args.private_dim
        init_kws["batch_embed_dim"] = args.batch_embed_dim
        fallback = args.beta_private if args.beta_private is not None else 1.0
        beta_private = {
            "rna":  args.beta_private_rna  if args.beta_private_rna  is not None else fallback,
            "atac": args.beta_private_atac if args.beta_private_atac is not None else fallback,
        }
        compile_kws.update(
            {
                "beta_shared":  args.beta_shared,
                "beta_private": beta_private,
                "lam_iso":      args.lam_iso,
                "lam_align":    args.lam_align,
            }
        )
    else:
        compile_kws["lam_align"] = args.lam_align

    if args.align_support:
        align_support_kws = {
            "n_neighbors": args.align_support_k,
            "strategy": args.align_support_strategy,
            "min_weight": args.align_support_min_weight,
        }

    return scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac},
        guidance_hvf,
        model=model_cls,
        init_kws=init_kws,
        compile_kws=compile_kws,
        fit_kws={"directory": os.fspath(Path(args.output_dir) / "glue")},
        align_support_kws=align_support_kws,
    ), guidance_hvf


def save_embeddings(
    glue,
    rna: ad.AnnData,
    atac: ad.AnnData,
    output_dir: Path,
    run_umap: bool,
    skip_modality_h5ad: bool = False,
) -> None:
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    for key, adata in (("rna", rna), ("atac", atac)):
        if "domain" not in adata.obs:
            adata.obs["domain"] = key

    rna_obs = rna.obs.copy()
    atac_obs = atac.obs.copy()
    rna_obs.index = pd.Index([f"rna:{item}" for item in rna_obs.index])
    atac_obs.index = pd.Index([f"atac:{item}" for item in atac_obs.index])

    combined = ad.AnnData(
        obs=pd.concat([rna_obs, atac_obs], join="outer"),
        obsm={
            "X_glue": np.concatenate(
                [rna.obsm["X_glue"], atac.obsm["X_glue"]]
            )
        },
    )
    if run_umap:
        n_pcs = min(50, combined.obsm["X_glue"].shape[1])
        sc.pp.neighbors(combined, n_pcs=n_pcs, use_rep="X_glue", metric="cosine")
        sc.tl.umap(combined)

    np.save(output_dir / "combined_glue.npy", combined.obsm["X_glue"])
    if not skip_modality_h5ad:
        rna.write(output_dir / "rna_glue.h5ad", compression="gzip")
        atac.write(output_dir / "atac_glue.h5ad", compression="gzip")
    combined.write(output_dir / "combined_glue.h5ad", compression="gzip")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.bedtools:
        scglue.config.BEDTOOLS_PATH = args.bedtools

    if args.preprocessed_dir:
        pp = Path(args.preprocessed_dir)
        rna = sc.read(pp / "rna_pp.h5ad")
        atac = sc.read(pp / "atac_pp.h5ad")
        guidance = nx.read_graphml(pp / "guidance.graphml.gz")
    else:
        rna = deduplicate_vars(read_adata(args.rna))
        atac = deduplicate_vars(read_adata(args.atac))

        ensure_layer(rna, args.rna_layer)
        ensure_layer(atac, args.atac_layer)
        auto_mark_rna_hvg(rna)
        auto_mark_atac_hvg(atac)
        ensure_hvg(rna, "rna")
        ensure_hvg(atac, "atac")
        ensure_obs_names_unique({"rna": rna, "atac": atac})

        prepare_representations(rna, atac)
        rna, atac, guidance = build_guidance_graph(rna, atac, args)

        rna.write(output_dir / "rna_pp.h5ad", compression="gzip")
        atac.write(output_dir / "atac_pp.h5ad", compression="gzip")
        nx.write_graphml(guidance, output_dir / "guidance.graphml.gz")

    if args.preprocess_only:
        return

    configure_datasets(rna, atac, args)
    glue, guidance_hvf = train_glue(rna, atac, guidance, args)
    glue.save(output_dir / "glue.dill")
    nx.write_graphml(guidance_hvf, output_dir / "guidance_hvf.graphml.gz")

    save_embeddings(glue, rna, atac, output_dir, args.umap, args.skip_modality_h5ad)


if __name__ == "__main__":
    main()
