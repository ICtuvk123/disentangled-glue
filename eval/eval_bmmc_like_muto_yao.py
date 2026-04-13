#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BMMC_ROOT = REPO_ROOT / "experiments" / "BMMC"

_NP_ASARRAY = np.asarray


def _compat_asarray(a, dtype=None, order=None, *, like=None, copy=None):
    if copy is None:
        return _NP_ASARRAY(a, dtype=dtype, order=order, like=like)
    return np.array(a, dtype=dtype, order=order, like=like, copy=copy)


np.asarray = _compat_asarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a BMMC trimodal run with the same Benchmarker2-style metric extraction used by Muto/Yao."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-aligned",
        type=Path,
        default=BMMC_ROOT / "s01_preprocessing" / "feature_aligned_sampled.h5ad",
        help="Shared pre-integration baseline for PCR and metric computation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/benchmarker_eval",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Defaults to run dir name",
    )
    parser.add_argument("--cell-type-key", default="celltype")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--domain-key", default="domain")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--write-scaled", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "benchmarker_eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_dir.name

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(BMMC_ROOT))

    import scanpy as sc
    import s06_eval

    combined_path = run_dir / "combined_glue.h5ad"
    if not combined_path.exists():
        raise FileNotFoundError(f"Missing combined_glue.h5ad under {run_dir}")
    if not args.feature_aligned.exists():
        raise FileNotFoundError(f"Missing feature_aligned file: {args.feature_aligned}")

    print(f"Loading run embedding from {combined_path}")
    adata = sc.read_h5ad(combined_path)
    adata.obsm["X_embed"] = adata.obsm["X_glue"]

    print(f"Loading shared baseline from {args.feature_aligned}")
    aligned = s06_eval.load_feature_aligned(
        args.feature_aligned.resolve(),
        adata.obs_names,
        args.domain_key,
    )
    var = aligned.var.copy()
    if "highly_variable" not in var:
        var["highly_variable"] = True
    adata = sc.AnnData(
        X=aligned.X.copy(),
        obs=adata.obs.copy(),
        var=var,
        obsm=dict(adata.obsm),
    )

    bm = s06_eval.Benchmarker2(
        adata,
        batch_key=args.batch_key,
        label_key=args.cell_type_key,
        modality_key=args.domain_key,
        embedding_obsm_keys=["X_embed"],
        bio_conservation_metrics=s06_eval.BioConservation2(),
        batch_correction_metrics=s06_eval.BatchCorrection2(pcr_comparison_b=True),
        modality_integration_metrics=s06_eval.ModalityIntegration2(pcr_comparison_m=True),
        pre_integrated_embedding_obsm_key=None,
        n_jobs=args.n_jobs,
        progress_bar=True,
    )
    bm.benchmark()

    df_unscaled = bm.get_results(min_max_scale=False)
    df_unscaled.to_csv(output_dir / f"{tag}_unscaled.csv")

    if args.write_scaled:
        df_scaled = bm.get_results(min_max_scale=True)
        df_scaled.to_csv(output_dir / f"{tag}_scaled.csv")

    if "X_embed" not in df_unscaled.index:
        raise ValueError("Expected X_embed row in unscaled results")
    row = df_unscaled.loc["X_embed"]
    metrics = {}
    for key, value in row.items():
        if pd.notna(value):
            metrics[key] = float(value)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.tsv", sep="\t", index=False)
    (output_dir / "eval_info.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "feature_aligned": str(args.feature_aligned.resolve()),
                "cell_type_key": args.cell_type_key,
                "batch_key": args.batch_key,
                "domain_key": args.domain_key,
                "n_jobs": args.n_jobs,
            },
            indent=2,
        )
    )

    print(
        "Metrics: "
        f"Total={metrics.get('Total', float('nan')):.4f} "
        f"Bio={metrics.get('Bio conservation', float('nan')):.4f} "
        f"Batch={metrics.get('Batch correction', float('nan')):.4f} "
        f"Modality={metrics.get('Modality integration', float('nan')):.4f}"
    )
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
