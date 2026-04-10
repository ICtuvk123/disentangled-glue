"""
Refined random hyperparameter search for DisentangledSCGLUEModel.

Supports dataset-specific profiles:
- Muto: original baseline behavior
- Yao: uses the Yao-specific preprocessed disentangled inputs and evaluation base
"""

import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import sys
import json
import argparse
from itertools import chain
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

DGLUE_ROOT = Path("/data1/users/zhutianci/proj/disentangled-glue")
SCMRDR_ROOT = Path("/data1/users/zhutianci/proj/scMRDR")
sys.path.insert(0, str(DGLUE_ROOT))
sys.path.insert(0, str(SCMRDR_ROOT / "experiments" / "plots"))

from scglue.models import configure_dataset, fit_SCGLUE
from scglue.models.scglue import DisentangledSCGLUEModel
from metrics import Benchmarker2, BioConservation2, BatchCorrection2, ModalityIntegration2


DATASET_PROFILES = {
    "muto": {
        "label": "Muto",
        "search_space": {
            "shared_dim": [24, 32, 48],
            "private_dim": [4, 8, 16],
            "batch_embed_dim": [4, 8],
            "beta_shared": [1.0, 1.25, 1.5],
            "lam_iso": [0.5, 1.0, 2.0],
            "lam_align": [0.02, 0.03, 0.05, 0.07, 0.10],
            "beta_private_rna": [0.5, 1.0],
            "beta_private_atac": [0.5, 1.0],
        },
        "fixed": {
            "lam_data": 1.0,
            "lam_graph": 0.02,
            "lr": 2e-3,
            "shared_batches": False,
            "h_depth": 2,
            "h_dim": 256,
            "dropout": 0.2,
        },
        "defaults": {
            "preprocessed_dir": DGLUE_ROOT,
            "feature_aligned": DGLUE_ROOT / "feature_aligned_trained.h5ad",
            "search_dir": DGLUE_ROOT / "hparam_search_v3",
            "rna_prob_model": "NB",
            "rna_use_layer": "counts",
            "atac_prob_model": "NB",
            "atac_use_layer": "counts",
        },
    },
    "yao": {
        "label": "Yao",
        "search_space": {
            "shared_dim": [32, 48, 64],
            "private_dim": [8, 16],
            "batch_embed_dim": [4, 8],
            "beta_shared": [0.5, 0.75, 1.0, 1.25],
            "lam_iso": [0.5, 1.0, 2.0],
            "lam_align": [0.02, 0.03, 0.05, 0.07],
            "beta_private_rna": [0.05, 0.1, 0.25],
            "beta_private_atac": [0.1, 0.25, 0.5],
        },
        "fixed": {
            "lam_data": 1.0,
            "lam_graph": 0.02,
            "lr": 2e-3,
            "shared_batches": False,
            "h_depth": 2,
            "h_dim": 256,
            "dropout": 0.2,
        },
        "defaults": {
            "preprocessed_dir": DGLUE_ROOT / "runs" / "yao_disentangled",
            "feature_aligned": DGLUE_ROOT / "data" / "dataset" / "Yao" / "feature_aligned.h5ad",
            "search_dir": DGLUE_ROOT / "hparam_search_v3_yao",
            "rna_prob_model": "Normal",
            "rna_use_layer": None,
            "atac_prob_model": "NB",
            "atac_use_layer": "counts",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_PROFILES), default="muto")
    parser.add_argument("--n-trials", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--feature-aligned", type=Path, default=None)
    parser.add_argument("--search-dir", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=10)
    return parser.parse_args()


def profile_for(args: argparse.Namespace) -> dict:
    profile = DATASET_PROFILES[args.dataset]
    defaults = profile["defaults"]
    return {
        "label": profile["label"],
        "search_space": profile["search_space"],
        "fixed": profile["fixed"],
        "preprocessed_dir": args.preprocessed_dir or defaults["preprocessed_dir"],
        "feature_aligned": args.feature_aligned or defaults["feature_aligned"],
        "search_dir": args.search_dir or defaults["search_dir"],
        "rna_prob_model": defaults["rna_prob_model"],
        "rna_use_layer": defaults["rna_use_layer"],
        "atac_prob_model": defaults["atac_prob_model"],
        "atac_use_layer": defaults["atac_use_layer"],
    }


def sample_config(rng: np.random.Generator, trial_id: int, search_space: dict) -> dict:
    cfg = {"trial_id": trial_id}
    for key, choices in search_space.items():
        cfg[key] = choices[rng.integers(len(choices))]
    return cfg


def trial_name(cfg: dict, fixed: dict) -> str:
    return (
        f"t{cfg['trial_id']:04d}"
        f"_sd{cfg['shared_dim']}"
        f"_pd{cfg['private_dim']}"
        f"_be{cfg['batch_embed_dim']}"
        f"_bs{cfg['beta_shared']}"
        f"_li{cfg['lam_iso']}"
        f"_la{cfg['lam_align']}"
        f"_bpr{cfg['beta_private_rna']}"
        f"_bpa{cfg['beta_private_atac']}"
        f"_sb{int(fixed['shared_batches'])}"
    )


def maybe_use_layer(layer_name: str | None) -> dict:
    return {"use_layer": layer_name} if layer_name else {}


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


def load_preprocessed(profile: dict) -> tuple[ad.AnnData, ad.AnnData, nx.MultiDiGraph]:
    pp_dir = Path(profile["preprocessed_dir"])
    rna_path = pp_dir / "rna_pp.h5ad"
    atac_path = pp_dir / "atac_pp.h5ad"
    guidance_path = pp_dir / "guidance.graphml.gz"
    if not rna_path.exists() or not atac_path.exists() or not guidance_path.exists():
        hint = ""
        if profile["label"] == "Yao":
            hint = (
                "\nPrepare them first with:\n"
                "python /data1/users/zhutianci/proj/disentangled-glue/experiments/"
                "yao_disentangled_glue.py --preprocess-only"
            )
        raise FileNotFoundError(
            "Missing preprocessed files. Expected "
            f"{rna_path}, {atac_path}, {guidance_path}{hint}"
        )
    return (
        ad.read_h5ad(rna_path),
        ad.read_h5ad(atac_path),
        nx.read_graphml(guidance_path),
    )


def configure_modalities(rna: ad.AnnData, atac: ad.AnnData, profile: dict) -> None:
    configure_dataset(
        rna,
        profile["rna_prob_model"],
        use_highly_variable=True,
        use_rep="X_pca",
        use_batch="batch",
        **maybe_use_layer(profile["rna_use_layer"]),
    )
    configure_dataset(
        atac,
        profile["atac_prob_model"],
        use_highly_variable=True,
        use_rep="X_lsi",
        use_batch="batch",
        **maybe_use_layer(profile["atac_use_layer"]),
    )


def load_eval_adata(path: Path) -> ad.AnnData:
    adata_eval = sc.read_h5ad(path)
    if "modality" not in adata_eval.obs:
        if "domain" in adata_eval.obs:
            adata_eval.obs["modality"] = adata_eval.obs["domain"]
        else:
            raise ValueError("feature_aligned is missing both `modality` and `domain` columns")
    adata_eval.obs["modality"] = adata_eval.obs["modality"].map(canonicalize_modality).astype(str)
    return adata_eval


def run_trial(
    cfg: dict,
    rna: ad.AnnData,
    atac: ad.AnnData,
    guidance_hvf,
    adata_eval: ad.AnnData,
    search_dir: Path,
    fixed: dict,
    n_jobs: int,
) -> dict:
    tag = trial_name(cfg, fixed)
    out_dir = search_dir / tag
    npy_path = out_dir / "embedding.npy"
    metrics_path = out_dir / "metrics.json"

    if npy_path.exists() and metrics_path.exists():
        print("  Already done, skipping.")
        with metrics_path.open() as f:
            return json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    compile_kws = {
        "lam_data": fixed["lam_data"],
        "lam_graph": fixed["lam_graph"],
        "lam_align": cfg["lam_align"],
        "beta_shared": cfg["beta_shared"],
        "lam_iso": cfg["lam_iso"],
        "beta_private": {
            "rna": cfg["beta_private_rna"],
            "atac": cfg["beta_private_atac"],
        },
        "lr": fixed["lr"],
    }

    init_kws = {
        "shared_dim": cfg["shared_dim"],
        "private_dim": cfg["private_dim"],
        "batch_embed_dim": cfg["batch_embed_dim"],
        "shared_batches": fixed["shared_batches"],
        "h_depth": fixed["h_depth"],
        "h_dim": fixed["h_dim"],
        "dropout": fixed["dropout"],
    }

    try:
        glue = fit_SCGLUE(
            {"rna": rna, "atac": atac},
            guidance_hvf,
            model=DisentangledSCGLUEModel,
            init_kws=init_kws,
            compile_kws=compile_kws,
            fit_kws={"directory": str(out_dir / "glue")},
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return {}

    rna.obsm["X_glue_shared"], _ = glue.encode_data("rna", rna, return_private=True)
    atac.obsm["X_glue_shared"], _ = glue.encode_data("atac", atac, return_private=True)
    emb = np.vstack([rna.obsm["X_glue_shared"], atac.obsm["X_glue_shared"]])
    np.save(str(npy_path), emb)

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
        n_jobs=n_jobs,
    )
    bm2.benchmark()
    df = bm2.get_results(min_max_scale=False)

    metrics = {}
    for col in df.columns:
        val = df[col].iloc[0]
        if pd.notna(val):
            metrics[col] = float(val)

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.tsv", sep="\t", index=False)

    del adata_eval.obsm[tag]
    return metrics


def write_summary(results: list[dict], search_dir: Path) -> None:
    rows = []
    for cfg, metrics in results:
        if metrics:
            rows.append({**cfg, **metrics})
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols_first = [
        "trial_id",
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
    ]
    cols_first = [c for c in cols_first if c in df.columns]
    rest = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + rest].sort_values("Total", ascending=False, na_position="last")
    summary_path = search_dir / "summary.tsv"
    df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary written to {summary_path}")
    print(df[cols_first].head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    profile = profile_for(args)
    search_dir = Path(profile["search_dir"])
    search_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {profile['label']} data...")
    rna, atac, guidance = load_preprocessed(profile)
    configure_modalities(rna, atac, profile)

    guidance_hvf = guidance.subgraph(
        chain(
            rna.var.query("highly_variable").index,
            atac.var.query("highly_variable").index,
        )
    ).copy()

    adata_eval = load_eval_adata(Path(profile["feature_aligned"]))
    if adata_eval.n_obs != rna.n_obs + atac.n_obs:
        raise ValueError(
            f"feature_aligned n_obs mismatch: {adata_eval.n_obs} vs {rna.n_obs + atac.n_obs}"
        )

    rng = np.random.default_rng(args.seed)
    configs = [sample_config(rng, i, profile["search_space"]) for i in range(args.n_trials)]
    my_configs = [cfg for cfg in configs if cfg["trial_id"] % args.n_gpus == args.gpu_id]
    print(
        f"Dataset: {args.dataset}  |  Total trials: {args.n_trials}  |  "
        f"This worker (GPU {args.gpu_id}): {len(my_configs)} trials"
    )
    print(f"Search dir: {search_dir}")

    results = []
    for i, cfg in enumerate(my_configs):
        tag = trial_name(cfg, profile["fixed"])
        print(f"\n[{i + 1}/{len(my_configs)}] {tag}")
        metrics = run_trial(
            cfg,
            rna,
            atac,
            guidance_hvf,
            adata_eval,
            search_dir,
            profile["fixed"],
            args.n_jobs,
        )
        results.append((cfg, metrics))
        if metrics:
            print(
                f"  Total={metrics.get('Total', float('nan')):.4f}  "
                f"Bio={metrics.get('Bio conservation', float('nan')):.4f}  "
                f"Batch={metrics.get('Batch correction', float('nan')):.4f}  "
                f"Modality={metrics.get('Modality integration', float('nan')):.4f}"
            )

    write_summary(results, search_dir)


if __name__ == "__main__":
    main()
