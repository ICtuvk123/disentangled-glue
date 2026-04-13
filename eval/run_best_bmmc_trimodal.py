#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BMMC_ROOT = REPO_ROOT / "experiments" / "BMMC"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "bmmc_trimodal_best"
ORIGINAL_PREPROCESSED_DIR = BMMC_ROOT / "s11_long_search" / "preprocessed"

BEST_CFG = {
    "trial_id": 88,
    "mode": "baseline",
    "shared_dim": 48,
    "private_dim": 8,
    "beta_shared": 0.75,
    "lam_iso": 2.0,
    "lam_align": 0.3,
    "beta_private_rna": 0.25,
    "beta_private_atac": 1.0,
    "beta_private_prot": 1.0,
    "align_support_k": 15,
    "align_support_strategy": "hard",
    "align_support_min_weight": 0.1,
}

_NP_ASARRAY = np.asarray


def _compat_asarray(a, dtype=None, order=None, *, like=None, copy=None):
    if copy is None:
        return _NP_ASARRAY(a, dtype=dtype, order=order, like=like)
    return np.array(a, dtype=dtype, order=order, like=like, copy=copy)


np.asarray = _compat_asarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the best BMMC trimodal config from s11_long_search."
    )
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--rna",
        default=BMMC_ROOT / "s01_preprocessing" / "RNA_counts_qc_sampled.h5ad",
        type=Path,
    )
    parser.add_argument(
        "--atac",
        default=BMMC_ROOT / "s01_preprocessing" / "ATAC_counts_qc_sampled.h5ad",
        type=Path,
    )
    parser.add_argument(
        "--prot",
        default=BMMC_ROOT / "s01_preprocessing" / "protein_counts_qc_sampled.h5ad",
        type=Path,
    )
    parser.add_argument(
        "--gtf",
        default=BMMC_ROOT / "gencode.v38.chr_patch_hapl_scaff.annotation.gtf",
        type=Path,
    )
    parser.add_argument(
        "--protein-gene-map",
        default=BMMC_ROOT / "s01_preprocessing" / "protein_gene_map.tsv",
        type=Path,
    )
    parser.add_argument(
        "--feature-aligned",
        default=BMMC_ROOT / "s01_preprocessing" / "feature_aligned_sampled.h5ad",
        type=Path,
    )
    parser.add_argument(
        "--preprocessed-dir",
        default=ORIGINAL_PREPROCESSED_DIR,
        type=Path,
        help=(
            "Preprocessed directory to reuse. Defaults to the original "
            "s11_long_search/preprocessed used by the historical search."
        ),
    )
    parser.add_argument(
        "--bedtools",
        default="/data1/users/zhutianci/.conda/envs/scMRDR/bin/bedtools",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--cell-type-key", default="celltype")
    parser.add_argument("--domain-key", default="domain")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def extract_standard_metrics(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path, index_col=0)
    if "X_embed" not in df.index:
        raise ValueError(f"Expected X_embed row in {csv_path}")
    row = df.loc["X_embed"]
    metrics = {}
    for key, value in row.items():
        if pd.notna(value):
            metrics[key] = float(value)
    return metrics


def _needs_prot_mask_var_fix(prot_path: Path) -> bool:
    with h5py.File(prot_path, "r") as f:
        if "/uns/pca/params/mask_var" not in f:
            return False
        ds = f["/uns/pca/params/mask_var"]
        return ds.attrs.get("encoding-type") == "null"


def _make_original_preprocessed_compat(source_dir: Path, output_root: Path) -> Path:
    compat_dir = output_root / "preprocessed_from_s11_long_search_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)

    for name in ("rna_pp.h5ad", "atac_pp.h5ad", "guidance.graphml.gz"):
        src = source_dir / name
        dst = compat_dir / name
        if not dst.exists():
            dst.symlink_to(src)

    prot_src = source_dir / "prot_pp.h5ad"
    prot_dst = compat_dir / "prot_pp.h5ad"
    if not prot_dst.exists():
        shutil.copy2(prot_src, prot_dst)
        with h5py.File(prot_dst, "r+") as f:
            if "/uns/pca/params/mask_var" in f:
                del f["/uns/pca/params/mask_var"]

    (compat_dir / "compat_info.json").write_text(
        json.dumps(
            {
                "source_preprocessed_dir": str(source_dir),
                "patched_files": ["prot_pp.h5ad"],
                "patch": "Removed /uns/pca/params/mask_var with encoding-type=null for current anndata compatibility.",
            },
            indent=2,
        )
    )
    return compat_dir


def resolve_preprocessed_dir(requested_dir: Path, output_root: Path) -> Path:
    requested_dir = requested_dir.resolve()
    prot_path = requested_dir / "prot_pp.h5ad"
    if (
        requested_dir == ORIGINAL_PREPROCESSED_DIR.resolve()
        and prot_path.exists()
        and _needs_prot_mask_var_fix(prot_path)
    ):
        compat_dir = _make_original_preprocessed_compat(requested_dir, output_root)
        print(
            "Using compatibility copy of original s11 preprocessed inputs: "
            f"{compat_dir}"
        )
        return compat_dir
    return requested_dir


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(BMMC_ROOT))

    import s11_hparam_search as bmmc_search

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    preprocess_dir = resolve_preprocessed_dir(args.preprocessed_dir, output_root)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    if not (preprocess_dir / "guidance.graphml.gz").exists():
        if not bmmc_search.preprocess_once(args, preprocess_dir):
            raise RuntimeError("BMMC preprocessing failed")

    run_name = bmmc_search.trial_name(BEST_CFG)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    quick_metrics_path = run_dir / "metrics.json"
    combined_glue_path = run_dir / "combined_glue.h5ad"
    if quick_metrics_path.exists():
        quick_metrics = json.loads(quick_metrics_path.read_text())
    else:
        if not combined_glue_path.exists():
            ok = bmmc_search.train_one(BEST_CFG, args, run_dir, preprocess_dir)
            if not ok:
                raise RuntimeError("BMMC training failed")
        quick_metrics = bmmc_search.quick_eval_one(BEST_CFG, args, run_dir)
        bmmc_search.save_artifacts(run_dir, BEST_CFG, quick_metrics)

    standard_eval_dir = run_dir / "standard_eval"
    standard_eval_dir.mkdir(parents=True, exist_ok=True)
    tag = run_name
    standard_csv = standard_eval_dir / f"{tag}_unscaled.csv"
    if not (args.resume and standard_csv.exists()):
        cmd = [
            args.python_bin,
            str(BMMC_ROOT / "s09_eval_true_scmrdr.py"),
            "--run-dir",
            str(run_dir),
            "--feature-aligned",
            str(args.feature_aligned.resolve()),
            "--output-dir",
            str(standard_eval_dir),
            "--tag",
            tag,
            "--cell-type-key",
            args.cell_type_key,
            "--batch-key",
            args.batch_key,
            "--domain-key",
            args.domain_key,
            "--n-jobs",
            str(args.n_jobs),
        ]
        subprocess.check_call(cmd)

    standard_metrics = extract_standard_metrics(standard_csv)
    (run_dir / "standard_metrics.json").write_text(json.dumps(standard_metrics, indent=2))
    (run_dir / "repro_info.json").write_text(
        json.dumps(
            {
                "dataset": "BMMC",
                "source_search_dir": str(
                    BMMC_ROOT / "s11_long_search" / "20260329_204614"
                ),
                "feature_aligned": str(args.feature_aligned.resolve()),
                "preprocessed_dir": str(preprocess_dir),
                "requested_preprocessed_dir": str(args.preprocessed_dir.resolve()),
                "quick_eval_source": "experiments/BMMC/s11_hparam_search.py",
                "standard_eval_source": "experiments/BMMC/s09_eval_true_scmrdr.py",
            },
            indent=2,
        )
    )

    print(f"\nRun directory: {run_dir}")
    print(
        "Search-style metrics: "
        f"Total={quick_metrics.get('Total', float('nan')):.4f} "
        f"Bio={quick_metrics.get('Bio conservation', float('nan')):.4f} "
        f"Batch={quick_metrics.get('Batch correction', float('nan')):.4f} "
        f"Modality={quick_metrics.get('Modality integration', float('nan')):.4f}"
    )
    print(
        "Standard metrics: "
        f"Total={standard_metrics.get('Total', float('nan')):.4f} "
        f"Bio={standard_metrics.get('Bio conservation', float('nan')):.4f} "
        f"Batch={standard_metrics.get('Batch correction', float('nan')):.4f} "
        f"Modality={standard_metrics.get('Modality integration', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
