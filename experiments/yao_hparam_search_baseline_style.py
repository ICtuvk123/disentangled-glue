#!/usr/bin/env python3

"""
Baseline-aligned hyperparameter search for DisentangledSCGLUE on Yao-2021.

This script is designed to make the disentangled search comparable to the
working non-disentangled SCGLUE pipeline:
1. Read raw Yao RNA/ATAC counts from data/dataset/Yao.
2. Preprocess with PCA on RNA and either:
   - raw_svd: TruncatedSVD directly on ATAC counts (baseline-aligned), or
   - tfidf: TF-IDF/log scaled LSI (current disentangled path, for comparison).
3. Search across:
   - NB/NB vs Normal/Normal vs mixed Normal/NB decoder presets
   - HVG/HVF only vs all features
   - disentangled model hyperparameters
4. Evaluate on feature_aligned.h5ad with the same metric pipeline used by the
   existing Yao search script.

The raw_svd branch intentionally avoids `.toarray()` to keep the baseline math
without forcing a dense ATAC matrix in memory.
"""

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
            value = argv[idx + 1].strip()
            if value:
                return value
        if arg.startswith("--gpu="):
            value = arg.split("=", 1)[1].strip()
            if value:
                return value
    for idx, arg in enumerate(argv):
        if arg == "--gpu-id" and idx + 1 < len(argv):
            value = argv[idx + 1].strip()
            if value:
                return value
        if arg.startswith("--gpu-id="):
            value = arg.split("=", 1)[1].strip()
            if value:
                return value
    env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return env_gpu or default


os.environ["CUDA_VISIBLE_DEVICES"] = _resolve_cli_gpu()
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
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
SCMRDR_ROOT = Path("/data1/users/zhutianci/proj/scMRDR")
if str(DGLUE_ROOT) not in sys.path:
    sys.path.insert(0, str(DGLUE_ROOT))
if str(SCMRDR_ROOT / "experiments" / "plots") not in sys.path:
    sys.path.insert(0, str(SCMRDR_ROOT / "experiments" / "plots"))

import scglue
from metrics import BatchCorrection2, Benchmarker2, BioConservation2, ModalityIntegration2
from scglue.data import estimate_balancing_weight
from scglue.models import configure_dataset
from scglue.models.scglue import DisentangledSCGLUEModel
from scglue.utils import config as scglue_config


DEFAULT_DATA_DIR = DGLUE_ROOT / "data" / "dataset" / "Yao"
DEFAULT_SEARCH_DIR = DGLUE_ROOT / "hparam_search_v4_yao_baseline_style"
DEFAULT_CACHE_DIR = DGLUE_ROOT / "runs" / "yao_baseline_style_cache"
DEFAULT_BEDTOOLS = "/data1/users/zhutianci/.conda/envs/scMRDR/bin/bedtools"

PRESET_SPACE = {
    "nb_hvg": {
        "use_highly_variable": True,
        "rna_prob_model": "NB",
        "rna_use_layer": "counts",
        "atac_prob_model": "NB",
        "atac_use_layer": "counts",
        "feature_space": "hvg",
    },
    "nb_all": {
        "use_highly_variable": False,
        "rna_prob_model": "NB",
        "rna_use_layer": "counts",
        "atac_prob_model": "NB",
        "atac_use_layer": "counts",
        "feature_space": "all",
    },
    "normal_hvg": {
        "use_highly_variable": True,
        "rna_prob_model": "Normal",
        "rna_use_layer": None,
        "atac_prob_model": "Normal",
        "atac_use_layer": None,
        "feature_space": "hvg",
    },
    "normal_all": {
        "use_highly_variable": False,
        "rna_prob_model": "Normal",
        "rna_use_layer": None,
        "atac_prob_model": "Normal",
        "atac_use_layer": None,
        "feature_space": "all",
    },
    "mixed_hvg": {
        "use_highly_variable": True,
        "rna_prob_model": "Normal",
        "rna_use_layer": None,
        "atac_prob_model": "NB",
        "atac_use_layer": "counts",
        "feature_space": "hvg",
    },
    "mixed_all": {
        "use_highly_variable": False,
        "rna_prob_model": "Normal",
        "rna_use_layer": None,
        "atac_prob_model": "NB",
        "atac_use_layer": "counts",
        "feature_space": "all",
    },
}

MODEL_SEARCH_SPACE = {
    "shared_dim": [32, 48, 64],
    "private_dim": [8, 16],
    "batch_embed_dim": [4, 8],
    "beta_shared": [0.5, 0.75, 1.0, 1.25],
    "lam_iso": [0.5, 1.0, 2.0],
    "lam_align": [0.02, 0.03, 0.05, 0.07],
    "beta_private_rna": [0.05, 0.1, 0.25],
    "beta_private_atac": [0.1, 0.25, 0.5],
    "dropout": [0.1, 0.2],
    "lr": [1e-3, 2e-3],
}

FIXED_KWS = {
    "lam_data": 1.0,
    "lam_graph": 0.02,
    "shared_batches": False,
    "h_depth": 2,
    "h_dim": 256,
}

FAST_PRETRAIN_KWS = {
    "max_epochs": 60,
    "patience": 12,
    "reduce_lr_patience": 6,
}

FAST_FINETUNE_KWS = {
    "max_epochs": 30,
    "patience": 8,
    "reduce_lr_patience": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline-aligned hyperparameter search for Yao disentangled GLUE."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--feature-aligned", type=Path, default=DEFAULT_DATA_DIR / "feature_aligned.h5ad")
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--bedtools", default=DEFAULT_BEDTOOLS)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--seed", type=int, default=0, help="Search sampling seed")
    parser.add_argument("--random-seed", type=int, default=0, help="Model/preprocess seed")
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-trials-per-preset", type=int, default=6)
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=sorted(PRESET_SPACE),
        default=["nb_hvg", "nb_all", "normal_hvg", "normal_all"],
    )
    parser.add_argument(
        "--lsi-methods",
        nargs="+",
        choices=["raw_svd", "tfidf"],
        default=["raw_svd"],
    )
    parser.add_argument("--lsi-n-components", type=int, default=100)
    parser.add_argument("--lsi-n-iter", type=int, default=15)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--reduce-lr-patience", type=int, default=20)
    parser.add_argument(
        "--quick-search",
        action="store_true",
        help="Use a shorter two-stage training budget for hyperparameter screening.",
    )
    parser.add_argument("--pretrain-max-epochs", type=int, default=None)
    parser.add_argument("--pretrain-patience", type=int, default=None)
    parser.add_argument("--pretrain-reduce-lr-patience", type=int, default=None)
    parser.add_argument("--finetune-max-epochs", type=int, default=None)
    parser.add_argument("--finetune-patience", type=int, default=None)
    parser.add_argument("--finetune-reduce-lr-patience", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=10)
    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.bedtools:
        bedtools = Path(args.bedtools)
        os.environ["PATH"] += os.pathsep + str(bedtools.parent)
        scglue.config.BEDTOOLS_PATH = os.fspath(bedtools)


def _stage_value(explicit: int | None, fallback: int) -> int:
    return explicit if explicit is not None else fallback


def resolve_training_schedule(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    if args.quick_search:
        pretrain_default = FAST_PRETRAIN_KWS
        finetune_default = FAST_FINETUNE_KWS
    else:
        pretrain_default = {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "reduce_lr_patience": args.reduce_lr_patience,
        }
        finetune_default = pretrain_default
    pretrain_kws = {
        "max_epochs": _stage_value(args.pretrain_max_epochs, pretrain_default["max_epochs"]),
        "patience": _stage_value(args.pretrain_patience, pretrain_default["patience"]),
        "reduce_lr_patience": _stage_value(
            args.pretrain_reduce_lr_patience, pretrain_default["reduce_lr_patience"]
        ),
    }
    finetune_kws = {
        "max_epochs": _stage_value(args.finetune_max_epochs, finetune_default["max_epochs"]),
        "patience": _stage_value(args.finetune_patience, finetune_default["patience"]),
        "reduce_lr_patience": _stage_value(
            args.finetune_reduce_lr_patience, finetune_default["reduce_lr_patience"]
        ),
    }
    return {"pretrain": pretrain_kws, "finetune": finetune_kws}


def format_training_schedule(training_schedule: dict[str, dict[str, int]]) -> str:
    pretrain = training_schedule["pretrain"]
    finetune = training_schedule["finetune"]
    return (
        "pretrain("
        f"max_epochs={pretrain['max_epochs']}, "
        f"patience={pretrain['patience']}, "
        f"reduce_lr_patience={pretrain['reduce_lr_patience']}"
        ") "
        "fine_tune("
        f"max_epochs={finetune['max_epochs']}, "
        f"patience={finetune['patience']}, "
        f"reduce_lr_patience={finetune['reduce_lr_patience']}"
        ")"
    )


def maybe_use_layer(layer_name: str | None) -> dict:
    return {"use_layer": layer_name} if layer_name else {}


def require_columns(adata: ad.AnnData, axis: str, columns: list[str], name: str) -> None:
    table = adata.obs if axis == "obs" else adata.var
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing required {axis} columns: {missing}")


def load_inputs(data_dir: Path) -> tuple[ad.AnnData, ad.AnnData]:
    rna = sc.read_h5ad(data_dir / "RNA_counts_qc.h5ad")
    atac = sc.read_h5ad(data_dir / "ATAC_counts_qc.h5ad")

    rna = rna[:, ~rna.var_names.duplicated(keep="first")].copy()
    atac = atac[:, ~atac.var_names.duplicated(keep="first")].copy()

    require_columns(rna, "obs", ["batch"], "RNA_counts_qc.h5ad")
    require_columns(atac, "obs", ["batch"], "ATAC_counts_qc.h5ad")
    require_columns(
        rna, "var", ["chrom", "chromStart", "chromEnd", "highly_variable"], "RNA_counts_qc.h5ad"
    )
    require_columns(
        atac, "var", ["chrom", "chromStart", "chromEnd", "highly_variable"], "ATAC_counts_qc.h5ad"
    )

    if "counts" not in rna.layers:
        rna.layers["counts"] = rna.X.copy()
    if "counts" not in atac.layers:
        atac.layers["counts"] = atac.X.copy()

    rna.obs["batch"] = rna.obs["batch"].astype("category")
    atac.obs["batch"] = atac.obs["batch"].astype("category")
    return rna, atac


def fit_atac_raw_svd(
    atac: ad.AnnData,
    n_components: int,
    n_iter: int,
    random_state: int,
) -> None:
    X = atac.X
    if sparse.issparse(X):
        X = X.astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state,
    )
    atac.obsm["X_lsi"] = svd.fit_transform(X).astype(np.float32, copy=False)


def fit_atac_tfidf_lsi(
    atac: ad.AnnData,
    n_components: int,
    n_iter: int,
    random_state: int,
) -> None:
    atac_hvg = atac[:, atac.var["highly_variable"].to_numpy()].copy()
    X = scglue.num.tfidf(atac_hvg.X)
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


def preprocess_cached(
    args: argparse.Namespace,
    feature_space: str,
    lsi_method: str,
) -> tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]:
    cache_name = (
        f"{feature_space}_{lsi_method}"
        f"_pc{args.lsi_n_components}"
        f"_it{args.lsi_n_iter}"
        f"_rs{args.random_seed}"
    )
    cache_dir = args.cache_dir / cache_name
    rna_path = cache_dir / "rna_pp.h5ad"
    atac_path = cache_dir / "atac_pp.h5ad"
    guidance_path = cache_dir / "guidance.graphml.gz"

    if rna_path.exists() and atac_path.exists() and guidance_path.exists():
        return (
            ad.read_h5ad(rna_path),
            ad.read_h5ad(atac_path),
            nx.read_graphml(guidance_path),
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    rna, atac = load_inputs(args.data_dir)

    sc.tl.pca(rna, n_comps=args.lsi_n_components, svd_solver="arpack")
    if lsi_method == "raw_svd":
        fit_atac_raw_svd(atac, args.lsi_n_components, args.lsi_n_iter, args.random_seed)
    elif lsi_method == "tfidf":
        fit_atac_tfidf_lsi(atac, args.lsi_n_components, args.lsi_n_iter, args.random_seed)
    else:
        raise ValueError(f"Unknown lsi_method: {lsi_method}")

    if feature_space == "hvg":
        rna = rna[:, rna.var["highly_variable"].to_numpy()].copy()
        atac = atac[:, atac.var["highly_variable"].to_numpy()].copy()
    elif feature_space != "all":
        raise ValueError(f"Unknown feature_space: {feature_space}")

    rna = rna[:, ~rna.var[["chrom", "chromStart", "chromEnd"]].isna().any(axis=1)].copy()
    atac = atac[:, ~atac.var[["chrom", "chromStart", "chromEnd"]].isna().any(axis=1)].copy()

    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    scglue.graph.check_graph(guidance, [rna, atac])

    rna.write(rna_path, compression="gzip")
    atac.write(atac_path, compression="gzip")
    nx.write_graphml(guidance, guidance_path)
    meta = {
        "feature_space": feature_space,
        "lsi_method": lsi_method,
        "rna_shape": list(rna.shape),
        "atac_shape": list(atac.shape),
        "rna_hvg_n": int(rna.var["highly_variable"].sum()),
        "atac_hvg_n": int(atac.var["highly_variable"].sum()),
    }
    (cache_dir / "preprocess_meta.json").write_text(json.dumps(meta, indent=2))
    return rna, atac, guidance


def canonicalize_modality(value) -> str:
    key = str(value).strip().lower()
    mapping = {
        "0": "RNA",
        "1": "ATAC",
        "rna": "RNA",
        "atac": "ATAC",
        "scrna-seq": "RNA",
        "scatac-seq": "ATAC",
    }
    return mapping.get(key, str(value))


def load_eval_adata(path: Path) -> ad.AnnData:
    adata_eval = sc.read_h5ad(path)
    if "modality" not in adata_eval.obs:
        if "domain" in adata_eval.obs:
            adata_eval.obs["modality"] = adata_eval.obs["domain"]
        else:
            raise ValueError("feature_aligned is missing both `modality` and `domain` columns")
    adata_eval.obs["modality"] = adata_eval.obs["modality"].map(canonicalize_modality).astype(str)
    return adata_eval


def sample_model_config(rng: np.random.Generator, trial_id: int) -> dict:
    config = {"trial_id": trial_id}
    for key, choices in MODEL_SEARCH_SPACE.items():
        config[key] = choices[rng.integers(len(choices))]
    return config


def build_trial_configs(args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    configs = []
    trial_id = 0
    for preset_name in args.presets:
        for lsi_method in args.lsi_methods:
            for trial_in_bucket in range(args.n_trials_per_preset):
                cfg = sample_model_config(rng, trial_id)
                cfg["preset"] = preset_name
                cfg["lsi_method"] = lsi_method
                cfg["trial_in_bucket"] = trial_in_bucket
                cfg["feature_space"] = PRESET_SPACE[preset_name]["feature_space"]
                configs.append(cfg)
                trial_id += 1
    return configs


def trial_name(cfg: dict) -> str:
    return (
        f"t{cfg['trial_id']:04d}"
        f"_{cfg['preset']}"
        f"_{cfg['lsi_method']}"
        f"_sd{cfg['shared_dim']}"
        f"_pd{cfg['private_dim']}"
        f"_be{cfg['batch_embed_dim']}"
        f"_bs{cfg['beta_shared']}"
        f"_li{cfg['lam_iso']}"
        f"_la{cfg['lam_align']}"
        f"_bpr{cfg['beta_private_rna']}"
        f"_bpa{cfg['beta_private_atac']}"
        f"_do{cfg['dropout']}"
        f"_lr{cfg['lr']}"
    )


def configure_modalities(rna: ad.AnnData, atac: ad.AnnData, preset_name: str) -> dict:
    preset = PRESET_SPACE[preset_name]
    configure_dataset(
        rna,
        preset["rna_prob_model"],
        use_highly_variable=preset["use_highly_variable"],
        use_rep="X_pca",
        use_batch="batch",
        **maybe_use_layer(preset["rna_use_layer"]),
    )
    configure_dataset(
        atac,
        preset["atac_prob_model"],
        use_highly_variable=preset["use_highly_variable"],
        use_rep="X_lsi",
        use_batch="batch",
        **maybe_use_layer(preset["atac_use_layer"]),
    )
    return preset


def fit_scglue_hpo(
    adatas: dict[str, ad.AnnData],
    graph: nx.MultiDiGraph,
    *,
    model: type[DisentangledSCGLUEModel],
    init_kws: dict,
    compile_kws: dict,
    training_schedule: dict[str, dict[str, int]],
    train_dir: Path,
) -> DisentangledSCGLUEModel:
    print("[INFO] fit_SCGLUE: Pretraining SCGLUE model...")
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False})
    pretrain = model(adatas, sorted(graph.nodes), **pretrain_init_kws)
    pretrain.compile(**compile_kws)
    pretrain.fit(
        adatas,
        graph,
        directory=str(train_dir / "pretrain"),
        align_burnin=np.inf,
        safe_burnin=False,
        **training_schedule["pretrain"],
    )
    pretrain.save(train_dir / "pretrain" / "pretrain.dill")

    print("[INFO] fit_SCGLUE: Estimating balancing weight...")
    tmp_key = f"X_{scglue_config.TMP_PREFIX}"
    for key, adata in adatas.items():
        adata.obsm[tmp_key] = pretrain.encode_data(key, adata)
    estimate_balancing_weight(
        *adatas.values(),
        use_rep=tmp_key,
        use_batch=None,
        key_added="balancing_weight",
    )
    for adata in adatas.values():
        adata.uns[scglue_config.ANNDATA_KEY]["use_dsc_weight"] = "balancing_weight"
        del adata.obsm[tmp_key]

    print("[INFO] fit_SCGLUE: Fine-tuning SCGLUE model...")
    finetune = model(adatas, sorted(graph.nodes), **init_kws)
    finetune.adopt_pretrained_model(pretrain)
    finetune.compile(**compile_kws)
    finetune.random_seed += 1
    finetune.fit(
        adatas,
        graph,
        directory=str(train_dir / "fine-tune"),
        **training_schedule["finetune"],
    )
    finetune.save(train_dir / "fine-tune" / "fine-tune.dill")
    return finetune


def collect_completed_results(search_dir: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(search_dir.glob("t*/metrics.json")):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists():
            continue
        with config_path.open() as handle:
            cfg = json.load(handle)
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        rows.append({**cfg, **metrics})
    return rows


def write_live_summary(search_dir: Path, announce: bool = False) -> None:
    rows = collect_completed_results(search_dir)
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "Total" in df.columns:
        cols_first = [
            "trial_id",
            "preset",
            "feature_space",
            "lsi_method",
            "Total",
            "Bio conservation",
            "Batch correction",
            "Modality integration",
            "shared_dim",
            "private_dim",
            "batch_embed_dim",
            "beta_shared",
            "lam_iso",
            "lam_align",
            "beta_private_rna",
            "beta_private_atac",
            "dropout",
            "lr",
        ]
        cols_first = [column for column in cols_first if column in df.columns]
        rest = [column for column in df.columns if column not in cols_first]
        df = df[cols_first + rest].sort_values("Total", ascending=False, na_position="last")
        summary_path = search_dir / "summary.tsv"
    else:
        summary_path = search_dir / "summary.partial.tsv"
    tmp_path = search_dir / f".{summary_path.name}.{os.getpid()}.tmp"
    df.to_csv(tmp_path, sep="\t", index=False)
    tmp_path.replace(summary_path)
    if announce:
        print(f"Updated live summary: {summary_path}")


def build_guidance_subgraph(
    rna: ad.AnnData,
    atac: ad.AnnData,
    guidance: nx.MultiDiGraph,
    use_highly_variable: bool,
) -> nx.MultiDiGraph:
    if use_highly_variable:
        nodes = chain(
            rna.var.query("highly_variable").index,
            atac.var.query("highly_variable").index,
        )
    else:
        nodes = chain(rna.var.index, atac.var.index)
    return guidance.subgraph(nodes).copy()


def run_trial(
    cfg: dict,
    rna: ad.AnnData,
    atac: ad.AnnData,
    guidance: nx.MultiDiGraph,
    adata_eval: ad.AnnData,
    search_dir: Path,
    args: argparse.Namespace,
    training_schedule: dict[str, dict[str, int]],
) -> dict:
    tag = trial_name(cfg)
    out_dir = search_dir / tag
    npy_path = out_dir / "embedding.npy"
    metrics_path = out_dir / "metrics.json"
    config_path = out_dir / "config.json"

    if npy_path.exists() and metrics_path.exists():
        with metrics_path.open() as handle:
            return json.load(handle)

    out_dir.mkdir(parents=True, exist_ok=True)
    preset = configure_modalities(rna, atac, cfg["preset"])
    guidance_use = build_guidance_subgraph(rna, atac, guidance, preset["use_highly_variable"])

    init_kws = {
        "shared_dim": cfg["shared_dim"],
        "private_dim": cfg["private_dim"],
        "batch_embed_dim": cfg["batch_embed_dim"],
        "shared_batches": FIXED_KWS["shared_batches"],
        "h_depth": FIXED_KWS["h_depth"],
        "h_dim": FIXED_KWS["h_dim"],
        "dropout": cfg["dropout"],
        "random_seed": args.random_seed,
    }
    compile_kws = {
        "lam_data": FIXED_KWS["lam_data"],
        "lam_graph": FIXED_KWS["lam_graph"],
        "lam_align": cfg["lam_align"],
        "beta_shared": cfg["beta_shared"],
        "beta_private": {
            "rna": cfg["beta_private_rna"],
            "atac": cfg["beta_private_atac"],
        },
        "lam_iso": cfg["lam_iso"],
        "lr": cfg["lr"],
    }

    (out_dir / "dataset_meta.json").write_text(
        json.dumps(
            {
                "preset": cfg["preset"],
                "lsi_method": cfg["lsi_method"],
                "feature_space": cfg["feature_space"],
                "rna_prob_model": preset["rna_prob_model"],
                "atac_prob_model": preset["atac_prob_model"],
                "use_highly_variable": preset["use_highly_variable"],
                "rna_use_layer": preset["rna_use_layer"],
                "atac_use_layer": preset["atac_use_layer"],
            },
            indent=2,
        )
    )
    config_path.write_text(json.dumps(cfg, indent=2))
    print(f"  training_schedule={format_training_schedule(training_schedule)}")

    try:
        glue = fit_scglue_hpo(
            {"rna": rna, "atac": atac},
            guidance_use,
            model=DisentangledSCGLUEModel,
            init_kws=init_kws,
            compile_kws=compile_kws,
            training_schedule=training_schedule,
            train_dir=out_dir / "glue",
        )
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return {}

    rna.obsm["X_glue_shared"], _ = glue.encode_data("rna", rna, return_private=True)
    atac.obsm["X_glue_shared"], _ = glue.encode_data("atac", atac, return_private=True)
    emb = np.vstack([rna.obsm["X_glue_shared"], atac.obsm["X_glue_shared"]])
    np.save(npy_path, emb)

    adata_eval.obsm[tag] = emb
    bm2 = Benchmarker2(
        adata_eval,
        batch_key="batch",
        label_key="cell_type",
        modality_key="modality",
        bio_conservation_metrics=BioConservation2(),
        batch_correction_metrics=BatchCorrection2(),
        modality_integration_metrics=ModalityIntegration2(),
        embedding_obsm_keys=[tag],
        n_jobs=args.n_jobs,
    )
    bm2.benchmark()
    df = bm2.get_results(min_max_scale=False)

    metrics = {}
    for column in df.columns:
        value = df[column].iloc[0]
        if pd.notna(value):
            metrics[column] = float(value)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.tsv", sep="\t", index=False)

    del adata_eval.obsm[tag]
    return metrics


def write_summary(results: list[dict], search_dir: Path) -> None:
    rows = [result for result in results if result]
    if not rows:
        print("\nNo successful trials yet.")
        return
    write_live_summary(search_dir, announce=False)
    summary_path = search_dir / "summary.tsv"
    if summary_path.exists():
        df = pd.read_csv(summary_path, sep="\t")
        cols_first = [
            "trial_id",
            "preset",
            "feature_space",
            "lsi_method",
            "Total",
            "Bio conservation",
            "Batch correction",
            "Modality integration",
            "shared_dim",
            "private_dim",
            "batch_embed_dim",
            "beta_shared",
            "lam_iso",
            "lam_align",
            "beta_private_rna",
            "beta_private_atac",
            "dropout",
            "lr",
        ]
        cols_first = [column for column in cols_first if column in df.columns]
        print(f"\nSummary written to {summary_path}")
        print(df[cols_first].head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    args = parse_args()
    setup_environment(args)
    args.search_dir.mkdir(parents=True, exist_ok=True)
    training_schedule = resolve_training_schedule(args)

    configs = build_trial_configs(args)
    my_configs = [cfg for cfg in configs if cfg["trial_id"] % args.n_gpus == args.gpu_id]
    print(
        f"Total configs: {len(configs)} | Worker GPU {args.gpu_id}: {len(my_configs)} | "
        f"search_dir={args.search_dir}"
    )
    print(f"Training schedule: {format_training_schedule(training_schedule)}")

    adata_eval = load_eval_adata(args.feature_aligned)

    dataset_cache: dict[tuple[str, str], tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]] = {}
    for cfg in my_configs:
        key = (cfg["feature_space"], cfg["lsi_method"])
        if key not in dataset_cache:
            print(f"Preparing cache for feature_space={key[0]} lsi_method={key[1]}")
            dataset_cache[key] = preprocess_cached(args, feature_space=key[0], lsi_method=key[1])
            rna, atac, _ = dataset_cache[key]
            expected_n_obs = rna.n_obs + atac.n_obs
            if adata_eval.n_obs != expected_n_obs:
                raise ValueError(
                    f"feature_aligned n_obs mismatch for {key}: "
                    f"{adata_eval.n_obs} vs {expected_n_obs}"
                )

    results = []
    for idx, cfg in enumerate(my_configs, start=1):
        key = (cfg["feature_space"], cfg["lsi_method"])
        rna, atac, guidance = dataset_cache[key]
        tag = trial_name(cfg)
        print(f"\n[{idx}/{len(my_configs)}] {tag}")
        metrics = run_trial(
            cfg,
            rna,
            atac,
            guidance,
            adata_eval,
            args.search_dir,
            args,
            training_schedule,
        )
        result = {**cfg, **metrics}
        results.append(result)
        if metrics:
            print(
                f"  Total={metrics.get('Total', float('nan')):.4f} "
                f"Bio={metrics.get('Bio conservation', float('nan')):.4f} "
                f"Batch={metrics.get('Batch correction', float('nan')):.4f} "
                f"Modality={metrics.get('Modality integration', float('nan')):.4f}"
            )
            write_live_summary(args.search_dir, announce=True)

    write_summary(results, args.search_dir)


if __name__ == "__main__":
    main()
