#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import chain
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "muto_best"

BEST_CFG = {
    "trial_id": 18,
    "shared_dim": 32,
    "private_dim": 8,
    "batch_embed_dim": 4,
    "beta_shared": 1.5,
    "lam_iso": 0.5,
    "lam_align": 0.05,
    "beta_private_rna": 0.5,
    "beta_private_atac": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the best Muto config from hparam_search_v3."
    )
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--preprocessed-dir", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--feature-aligned",
        type=Path,
        default=REPO_ROOT / "feature_aligned_trained.h5ad",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_muto_preprocessed(pp_dir: Path) -> tuple[Path, Path, Path]:
    candidates = [
        (
            pp_dir / "rna_pp.h5ad",
            pp_dir / "atac_pp.h5ad",
            pp_dir / "guidance.graphml.gz",
        ),
        (
            pp_dir / "rna-pp.h5ad",
            pp_dir / "atac-pp.h5ad",
            pp_dir / "guidance.graphml.gz",
        ),
    ]
    for rna_path, atac_path, guidance_path in candidates:
        if rna_path.exists() and atac_path.exists() and guidance_path.exists():
            return rna_path, atac_path, guidance_path
    raise FileNotFoundError(
        f"Missing Muto preprocessed inputs under {pp_dir}. "
        "Tried {rna_pp.h5ad,atac_pp.h5ad,guidance.graphml.gz} and "
        "{rna-pp.h5ad,atac-pp.h5ad,guidance.graphml.gz}."
    )


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

    import anndata as ad
    import hparam_search_v3 as hpo
    import networkx as nx

    profile_defaults = hpo.DATASET_PROFILES["muto"]
    profile = {
        "label": profile_defaults["label"],
        "search_space": profile_defaults["search_space"],
        "fixed": profile_defaults["fixed"],
        "preprocessed_dir": args.preprocessed_dir,
        "feature_aligned": args.feature_aligned,
        "search_dir": args.output_dir,
        "rna_prob_model": profile_defaults["defaults"]["rna_prob_model"],
        "rna_use_layer": profile_defaults["defaults"]["rna_use_layer"],
        "atac_prob_model": profile_defaults["defaults"]["atac_prob_model"],
        "atac_use_layer": profile_defaults["defaults"]["atac_use_layer"],
    }

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pp_dir = args.preprocessed_dir.resolve()
    rna_path, atac_path, guidance_path = resolve_muto_preprocessed(pp_dir)
    print(f"Loading Muto preprocessed inputs from {pp_dir}")
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)
    guidance = nx.read_graphml(guidance_path)
    hpo.configure_modalities(rna, atac, profile)

    guidance_hvf = guidance.subgraph(
        chain(
            rna.var.query("highly_variable").index,
            atac.var.query("highly_variable").index,
        )
    ).copy()

    adata_eval = hpo.load_eval_adata(Path(profile["feature_aligned"]))
    expected_n_obs = rna.n_obs + atac.n_obs
    if adata_eval.n_obs != expected_n_obs:
        raise ValueError(
            f"feature_aligned n_obs mismatch: {adata_eval.n_obs} vs {expected_n_obs}"
        )

    print("Training best Muto config:")
    print(json.dumps(BEST_CFG, indent=2))
    metrics = hpo.run_trial(
        BEST_CFG,
        rna,
        atac,
        guidance_hvf,
        adata_eval,
        output_root,
        profile["fixed"],
        args.n_jobs,
    )
    hpo.write_summary([(BEST_CFG, metrics)], output_root)

    run_dir = output_root / hpo.trial_name(BEST_CFG, profile["fixed"])
    (run_dir / "config.json").write_text(json.dumps(BEST_CFG, indent=2))
    (run_dir / "repro_info.json").write_text(
        json.dumps(
            {
                "dataset": "Muto",
                "source_search_dir": str(REPO_ROOT / "hparam_search_v3"),
                "preprocessed_dir": str(pp_dir),
                "resolved_rna": str(rna_path),
                "resolved_atac": str(atac_path),
                "resolved_guidance": str(guidance_path),
                "feature_aligned": str(args.feature_aligned.resolve()),
            },
            indent=2,
        )
    )

    print(f"\nRun directory: {run_dir}")
    if metrics:
        print(
            "Metrics: "
            f"Total={metrics.get('Total', float('nan')):.4f} "
            f"Bio={metrics.get('Bio conservation', float('nan')):.4f} "
            f"Batch={metrics.get('Batch correction', float('nan')):.4f} "
            f"Modality={metrics.get('Modality integration', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
