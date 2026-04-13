#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BMMC_ROOT = REPO_ROOT / "experiments" / "BMMC"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "bmmc_trimodal_manual"
ORIGINAL_PREPROCESSED_DIR = BMMC_ROOT / "s11_long_search" / "preprocessed"
DEFAULT_CFG = {
    "trial_id": 0,
    "mode": "baseline",
    "shared_dim": 50,
    "private_dim": 20,
    "beta_shared": 4.0,
    "lam_iso": 0.0,
    "lam_align": 0.05,
    "beta_private_rna": 1.0,
    "beta_private_atac": 1.0,
    "beta_private_prot": 1.0,
    "align_support_k": 15,
    "align_support_strategy": "soft",
    "align_support_min_weight": 0.05,
}

_NP_ASARRAY = np.asarray


def _compat_asarray(a, dtype=None, order=None, *, like=None, copy=None):
    if copy is None:
        return _NP_ASARRAY(a, dtype=dtype, order=order, like=like)
    return np.array(a, dtype=dtype, order=order, like=like, copy=copy)


np.asarray = _compat_asarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a single BMMC trimodal run from explicit "
            "hyperparameters using the same training and Benchmarker2-style "
            "evaluation pipeline as the s11 search."
        )
    )
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional hparams.json to load first; explicit CLI flags override it.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run directory name")
    parser.add_argument("--trial-id", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["baseline", "support"],
        default=None,
        help="Training mode. Support mode enables unsupported-cell alignment weights.",
    )
    parser.add_argument("--shared-dim", type=int, default=None)
    parser.add_argument("--private-dim", type=int, default=None)
    parser.add_argument("--beta-shared", type=float, default=None)
    parser.add_argument("--lam-iso", type=float, default=None)
    parser.add_argument("--lam-align", type=float, default=None)
    parser.add_argument("--beta-private-rna", type=float, default=None)
    parser.add_argument("--beta-private-atac", type=float, default=None)
    parser.add_argument("--beta-private-prot", type=float, default=None)
    parser.add_argument("--align-support-k", type=int, default=None)
    parser.add_argument(
        "--align-support-strategy",
        choices=["soft", "hard"],
        default=None,
    )
    parser.add_argument("--align-support-min-weight", type=float, default=None)
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
    parser.add_argument("--skip-quick-eval", action="store_true")
    parser.add_argument("--skip-standard-eval", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_cfg(args: argparse.Namespace) -> dict:
    cfg = DEFAULT_CFG.copy()
    if args.config_json:
        loaded = json.loads(args.config_json.read_text())
        cfg.update(loaded)
    overrides = {
        "trial_id": args.trial_id,
        "mode": args.mode,
        "shared_dim": args.shared_dim,
        "private_dim": args.private_dim,
        "beta_shared": args.beta_shared,
        "lam_iso": args.lam_iso,
        "lam_align": args.lam_align,
        "beta_private_rna": args.beta_private_rna,
        "beta_private_atac": args.beta_private_atac,
        "beta_private_prot": args.beta_private_prot,
        "align_support_k": args.align_support_k,
        "align_support_strategy": args.align_support_strategy,
        "align_support_min_weight": args.align_support_min_weight,
    }
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    required = [
        "trial_id",
        "mode",
        "shared_dim",
        "private_dim",
        "beta_shared",
        "lam_iso",
        "lam_align",
        "beta_private_rna",
        "beta_private_atac",
        "beta_private_prot",
        "align_support_k",
        "align_support_strategy",
        "align_support_min_weight",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing hyperparameter keys: {', '.join(missing)}")
    if cfg["mode"] not in {"baseline", "support"}:
        raise ValueError(f"Unsupported mode: {cfg['mode']}")
    return cfg


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
    cfg = load_cfg(args)

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

    run_name = args.run_name or bmmc_search.trial_name(cfg)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    quick_metrics = None
    metrics_path = run_dir / "metrics.json"
    combined_glue_path = run_dir / "combined_glue.h5ad"

    if not combined_glue_path.exists():
        ok = bmmc_search.train_one(cfg, args, run_dir, preprocess_dir)
        if not ok:
            raise RuntimeError("BMMC training failed")
    elif not args.resume:
        print(f"Training artifacts already exist, reusing: {combined_glue_path}")

    if not args.skip_quick_eval:
        if metrics_path.exists() and args.resume:
            quick_metrics = json.loads(metrics_path.read_text())
        else:
            quick_metrics = bmmc_search.quick_eval_one(cfg, args, run_dir)
            bmmc_search.save_artifacts(run_dir, cfg, quick_metrics)
    else:
        (run_dir / "hparams.json").write_text(json.dumps(cfg, indent=2))

    standard_metrics = None
    standard_eval_dir = run_dir / "standard_eval"
    standard_eval_dir.mkdir(parents=True, exist_ok=True)
    tag = run_name
    standard_csv = standard_eval_dir / f"{tag}_unscaled.csv"

    if not args.skip_standard_eval:
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
        (run_dir / "standard_metrics.json").write_text(
            json.dumps(standard_metrics, indent=2)
        )

    (run_dir / "repro_info.json").write_text(
        json.dumps(
            {
                "dataset": "BMMC",
                "config_json": str(args.config_json.resolve()) if args.config_json else None,
                "feature_aligned": str(args.feature_aligned.resolve()),
                "preprocessed_dir": str(preprocess_dir),
                "requested_preprocessed_dir": str(args.preprocessed_dir.resolve()),
                "quick_eval_source": (
                    None if args.skip_quick_eval else "experiments/BMMC/s11_hparam_search.py"
                ),
                "standard_eval_source": (
                    None if args.skip_standard_eval else "experiments/BMMC/s09_eval_true_scmrdr.py"
                ),
            },
            indent=2,
        )
    )

    print(f"\nRun directory: {run_dir}")
    print(f"Config: {json.dumps(cfg, ensure_ascii=True, sort_keys=True)}")
    if quick_metrics is not None:
        print(
            "Search-style metrics: "
            f"Total={quick_metrics.get('Total', float('nan')):.4f} "
            f"Bio={quick_metrics.get('Bio conservation', float('nan')):.4f} "
            f"Batch={quick_metrics.get('Batch correction', float('nan')):.4f} "
            f"Modality={quick_metrics.get('Modality integration', float('nan')):.4f}"
        )
    if standard_metrics is not None:
        print(
            "Benchmarker2 metrics: "
            f"Total={standard_metrics.get('Total', float('nan')):.4f} "
            f"Bio={standard_metrics.get('Bio conservation', float('nan')):.4f} "
            f"Batch={standard_metrics.get('Batch correction', float('nan')):.4f} "
            f"Modality={standard_metrics.get('Modality integration', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
