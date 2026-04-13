#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import chain
from pathlib import Path


def _resolve_cli_gpu(default: str = "0") -> str:
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == "--gpu" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
    return os.environ.get("CUDA_VISIBLE_DEVICES", default)


os.environ["CUDA_VISIBLE_DEVICES"] = _resolve_cli_gpu()
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

DGLUE_ROOT = Path("/data1/users/zhutianci/proj/disentangled-glue")
if str(DGLUE_ROOT) not in sys.path:
    sys.path.insert(0, str(DGLUE_ROOT))

import scglue


DEFAULT_DATA_DIR = DGLUE_ROOT / "data" / "dataset" / "Yao"
DEFAULT_OUTPUT_DIR = DGLUE_ROOT / "runs" / "yao_disentangled"
DEFAULT_BEDTOOLS = "/data1/users/zhutianci/.conda/envs/scMRDR/bin/bedtools"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train disentangled SCGLUE on the Yao RNA/ATAC dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bedtools", default=DEFAULT_BEDTOOLS)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--lsi-n-components", type=int, default=100)
    parser.add_argument("--lsi-n-iter", type=int, default=15)
    parser.add_argument(
        "--atac-hvf-top-n",
        type=int,
        default=None,
        help="Recompute ATAC highly-variable features on ATAC_tfidf_hvg.h5ad and keep the top N.",
    )
    parser.add_argument("--shared-dim", type=int, default=48)
    parser.add_argument("--private-dim", type=int, default=16)
    parser.add_argument("--batch-embed-dim", type=int, default=8)
    parser.add_argument("--h-depth", type=int, default=2)
    parser.add_argument("--h-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--beta-shared", type=float, default=0.75)
    parser.add_argument("--beta-private-rna", type=float, default=0.1)
    parser.add_argument("--beta-private-atac", type=float, default=0.1)
    parser.add_argument("--lam-graph", type=float, default=0.02)
    parser.add_argument("--lam-align", type=float, default=0.03)
    parser.add_argument("--lam-iso", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--reduce-lr-patience", type=int, default=20)
    parser.add_argument("--reuse-preprocessed", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--umap", action="store_true")
    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.bedtools:
        bedtools = Path(args.bedtools)
        os.environ["PATH"] += os.pathsep + str(bedtools.parent)
        scglue.config.BEDTOOLS_PATH = os.fspath(bedtools)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(", ".join(str(path) for path in paths))


def select_atac_hvf_mask(
    data_dir: Path,
    top_n: int | None,
    atac_path: Path,
) -> pd.Series:
    atac_tfidf_path = data_dir / "ATAC_tfidf_hvg.h5ad"
    if top_n is None:
        atac_hvf_source = first_existing_path(atac_tfidf_path, atac_path)
        atac_hvf = ad.read_h5ad(atac_hvf_source, backed="r")
        if "highly_variable" not in atac_hvf.var:
            atac_hvf.file.close()
            raise ValueError(f"{atac_hvf_source} is missing var['highly_variable']")
        atac_hvg = atac_hvf.var["highly_variable"].copy()
        atac_hvf.file.close()
        return atac_hvg

    atac_tfidf = sc.read_h5ad(require_file(atac_tfidf_path))
    top_n = min(top_n, atac_tfidf.n_vars)
    sc.pp.highly_variable_genes(atac_tfidf, batch_key="batch", n_top_genes=top_n)
    return atac_tfidf.var["highly_variable"].copy()


def load_inputs(data_dir: Path, atac_hvf_top_n: int | None) -> tuple[ad.AnnData, ad.AnnData, pd.Series]:
    rna = sc.read_h5ad(require_file(data_dir / "RNA_counts_qc.h5ad"))
    atac_path = first_existing_path(data_dir / "ATAC_raw_qc.h5ad", data_dir / "ATAC_counts_qc.h5ad")
    atac = sc.read_h5ad(atac_path)
    atac_hvg = select_atac_hvf_mask(data_dir, atac_hvf_top_n, atac_path)

    rna = rna[:, ~rna.var_names.duplicated(keep="first")].copy()
    atac = atac[:, ~atac.var_names.duplicated(keep="first")].copy()

    if "highly_variable" not in rna.var:
        raise ValueError("RNA_counts_qc.h5ad is missing var['highly_variable']")
    if "batch" not in rna.obs or "batch" not in atac.obs:
        raise ValueError("Both modalities must contain obs['batch']")
    return rna, atac, atac_hvg


def sparse_safe_lsi(
    atac: ad.AnnData,
    n_components: int,
    n_iter: int,
    random_state: int,
) -> None:
    atac_use = atac[:, atac.var["highly_variable"].to_numpy()]
    X = scglue.num.tfidf(atac_use.X)
    X = normalize(X, norm="l1")
    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.log1p(X.data * 1e4)
    else:
        X = np.log1p(X * 1e4)

    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state,
    )
    X_us = svd.fit_transform(X)
    X_lsi = X_us / svd.singular_values_
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    row_std = X_lsi.std(axis=1, ddof=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    X_lsi /= row_std
    atac.obsm["X_lsi"] = X_lsi.astype(np.float32, copy=False)


def preprocess(args: argparse.Namespace) -> tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]:
    rna, atac, atac_hvg = load_inputs(args.data_dir, args.atac_hvf_top_n)

    atac.layers["counts"] = atac.X.copy()

    rna.obs["batch"] = rna.obs["batch"].astype("category")
    atac.obs["batch"] = atac.obs["batch"].astype("category")
    atac.var["highly_variable"] = atac_hvg.reindex(atac.var_names).fillna(False).to_numpy()
    atac.uns["atac_hvf_top_n"] = args.atac_hvf_top_n if args.atac_hvf_top_n is not None else int(atac.var["highly_variable"].sum())

    sc.tl.pca(rna, n_comps=args.lsi_n_components, use_highly_variable=True, svd_solver="auto")
    sparse_safe_lsi(
        atac,
        n_components=args.lsi_n_components,
        n_iter=args.lsi_n_iter,
        random_state=args.random_seed,
    )

    rna = rna[:, rna.var["highly_variable"].to_numpy()].copy()
    atac = atac[:, atac.var["highly_variable"].to_numpy()].copy()

    rna = rna[:, ~rna.var[["chrom", "chromStart", "chromEnd"]].isna().any(axis=1)].copy()
    atac = atac[:, ~atac.var[["chrom", "chromStart", "chromEnd"]].isna().any(axis=1)].copy()

    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    scglue.graph.check_graph(guidance, [rna, atac])
    return rna, atac, guidance


def configure_datasets(rna: ad.AnnData, atac: ad.AnnData) -> None:
    scglue.models.configure_dataset(
        rna,
        "Normal",
        use_highly_variable=True,
        use_rep="X_pca",
        use_batch="batch",
    )
    scglue.models.configure_dataset(
        atac,
        "NB",
        use_highly_variable=True,
        use_layer="counts",
        use_rep="X_lsi",
        use_batch="batch",
    )


def train_model(
    rna: ad.AnnData,
    atac: ad.AnnData,
    guidance: nx.MultiDiGraph,
    args: argparse.Namespace,
):
    guidance_hvf = guidance.subgraph(
        chain(
            rna.var.query("highly_variable").index,
            atac.var.query("highly_variable").index,
        )
    ).copy()

    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac},
        guidance_hvf,
        model=scglue.models.DisentangledSCGLUEModel,
        init_kws={
            "shared_dim": args.shared_dim,
            "private_dim": args.private_dim,
            "batch_embed_dim": args.batch_embed_dim,
            "h_depth": args.h_depth,
            "h_dim": args.h_dim,
            "dropout": args.dropout,
            "random_seed": args.random_seed,
        },
        compile_kws={
            "beta_shared": args.beta_shared,
            "beta_private": {
                "rna": args.beta_private_rna,
                "atac": args.beta_private_atac,
            },
            "lam_graph": args.lam_graph,
            "lam_align": args.lam_align,
            "lam_iso": args.lam_iso,
            "lr": args.lr,
        },
        fit_kws={
            "directory": os.fspath(args.output_dir / "glue"),
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "reduce_lr_patience": args.reduce_lr_patience,
        },
    )
    return glue, guidance_hvf


def build_combined(rna: ad.AnnData, atac: ad.AnnData) -> ad.AnnData:
    for key, adata in (("rna", rna), ("atac", atac)):
        if "domain" not in adata.obs:
            adata.obs["domain"] = key
        if "modality" not in adata.obs:
            adata.obs["modality"] = key

    rna_obs = rna.obs.copy()
    atac_obs = atac.obs.copy()
    rna_obs.index = pd.Index([f"rna:{item}" for item in rna_obs.index])
    atac_obs.index = pd.Index([f"atac:{item}" for item in atac_obs.index])

    return ad.AnnData(
        obs=pd.concat([rna_obs, atac_obs], join="outer"),
        obsm={
            "X_glue": np.concatenate([rna.obsm["X_glue"], atac.obsm["X_glue"]], axis=0),
            "X_glue_shared": np.concatenate(
                [rna.obsm["X_glue_shared"], atac.obsm["X_glue_shared"]], axis=0
            ),
        },
    )


def encode_and_save(
    glue,
    rna: ad.AnnData,
    atac: ad.AnnData,
    guidance: nx.MultiDiGraph,
    guidance_hvf: nx.MultiDiGraph,
    output_dir: Path,
    run_umap: bool,
) -> None:
    rna_shared, rna_private = glue.encode_data("rna", rna, return_private=True)
    atac_shared, atac_private = glue.encode_data("atac", atac, return_private=True)

    rna.obsm["X_glue"] = rna_shared
    atac.obsm["X_glue"] = atac_shared
    rna.obsm["X_glue_shared"] = rna_shared
    atac.obsm["X_glue_shared"] = atac_shared
    rna.obsm["X_glue_private"] = rna_private
    atac.obsm["X_glue_private"] = atac_private

    combined = build_combined(rna, atac)
    if run_umap:
        n_pcs = min(50, combined.obsm["X_glue"].shape[1])
        sc.pp.neighbors(combined, n_pcs=n_pcs, use_rep="X_glue", metric="cosine")
        sc.tl.umap(combined)

    output_dir.mkdir(parents=True, exist_ok=True)
    glue.save(output_dir / "glue.dill")
    rna.write(output_dir / "rna_glue.h5ad", compression="gzip")
    atac.write(output_dir / "atac_glue.h5ad", compression="gzip")
    combined.write(output_dir / "combined_glue.h5ad", compression="gzip")
    np.save(output_dir / "combined_glue.npy", combined.obsm["X_glue"])
    nx.write_graphml(guidance, output_dir / "guidance.graphml.gz")
    nx.write_graphml(guidance_hvf, output_dir / "guidance_hvf.graphml.gz")


def load_preprocessed(output_dir: Path) -> tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]:
    return (
        sc.read_h5ad(output_dir / "rna_pp.h5ad"),
        sc.read_h5ad(output_dir / "atac_pp.h5ad"),
        nx.read_graphml(output_dir / "guidance.graphml.gz"),
    )


def save_preprocessed(
    rna: ad.AnnData,
    atac: ad.AnnData,
    guidance: nx.MultiDiGraph,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rna.write(output_dir / "rna_pp.h5ad", compression="gzip")
    atac.write(output_dir / "atac_pp.h5ad", compression="gzip")
    nx.write_graphml(guidance, output_dir / "guidance.graphml.gz")
    meta = {
        "rna_hvg_n": int(rna.var["highly_variable"].sum()) if "highly_variable" in rna.var else None,
        "atac_hvf_n": int(atac.var["highly_variable"].sum()) if "highly_variable" in atac.var else None,
        "atac_hvf_top_n": atac.uns.get("atac_hvf_top_n"),
    }
    (output_dir / "preprocess_meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    args = parse_args()
    setup_environment(args)

    if args.reuse_preprocessed and (args.output_dir / "rna_pp.h5ad").exists():
        rna, atac, guidance = load_preprocessed(args.output_dir)
    else:
        rna, atac, guidance = preprocess(args)
        save_preprocessed(rna, atac, guidance, args.output_dir)

    if args.preprocess_only:
        return

    configure_datasets(rna, atac)
    glue, guidance_hvf = train_model(rna, atac, guidance, args)
    encode_and_save(
        glue,
        rna,
        atac,
        guidance,
        guidance_hvf,
        args.output_dir,
        args.umap,
    )


if __name__ == "__main__":
    main()
