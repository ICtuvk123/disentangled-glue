#!/usr/bin/env python
"""
Strictly evaluate a BMMC run with the repo's original scMRDR-style workflow.

This wrapper does not reimplement the metrics. Instead, it exports RNA/ATAC
latents from a GLUE run and calls the exact scripts used by
`evaluation/workflow/rules/utils.smk`:

- `rna_unirep.py`
- `atac_unirep.py`
- `cell_integration.py`

Note
----
The original workflow is RNA/ATAC only. To stay strictly comparable, the main
score here is also RNA/ATAC only. Protein remains outside the strict scMRDR
benchmark and can be evaluated separately if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import pandas as pd
import scanpy as sc
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Run directory containing rna_glue.h5ad and atac_glue.h5ad")
    parser.add_argument("--rna", required=True,
                        help="Path to the raw/full RNA h5ad used for training")
    parser.add_argument("--atac", required=True,
                        help="Path to the raw/full ATAC h5ad used for training")
    parser.add_argument("--output-dir", default="s08_eval_scmrdr",
                        help="Directory to save strict benchmark artifacts")
    parser.add_argument("--tag", default=None,
                        help="Output tag (defaults to run dir name)")
    parser.add_argument("--cell-type-key", default="cell_type",
                        help="obs column for cell type labels")
    parser.add_argument("--domain-key", default="domain",
                        help="obs column for omics/domain labels")
    parser.add_argument("--python-bin", default=sys.executable,
                        help="Python executable used to run the workflow scripts")
    return parser.parse_args()


def ensure_rna_hvg(input_path: Path, output_path: Path) -> Path:
    adata = sc.read(input_path)
    if "highly_variable" not in adata.var:
        tmp = adata.copy()
        if "counts" in tmp.layers:
            tmp.X = tmp.layers["counts"].copy()
        sc.pp.normalize_total(tmp)
        sc.pp.log1p(tmp)
        hvg_kws = {"min_mean": 0.02, "max_mean": 4, "min_disp": 0.5}
        if "batch" in tmp.obs:
            hvg_kws["batch_key"] = "batch"
        sc.pp.highly_variable_genes(tmp, **hvg_kws)
        adata.var["highly_variable"] = tmp.var["highly_variable"].to_numpy()
    adata.write(output_path, compression="gzip")
    return output_path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_dir.name

    repo_root = Path(__file__).resolve().parents[2]
    workflow_dir = repo_root / "evaluation" / "workflow" / "scripts"
    rna_unirep_script = workflow_dir / "rna_unirep.py"
    atac_unirep_script = workflow_dir / "atac_unirep.py"
    cell_integration_script = workflow_dir / "cell_integration.py"

    rna_glue = sc.read(run_dir / "rna_glue.h5ad")
    atac_glue = sc.read(run_dir / "atac_glue.h5ad")

    rna_latent_csv = output_dir / f"{tag}_rna_latent.csv"
    atac_latent_csv = output_dir / f"{tag}_atac_latent.csv"
    pd.DataFrame(rna_glue.obsm["X_glue"], index=rna_glue.obs_names).to_csv(
        rna_latent_csv, header=False
    )
    pd.DataFrame(atac_glue.obsm["X_glue"], index=atac_glue.obs_names).to_csv(
        atac_latent_csv, header=False
    )

    rna_unirep_h5ad = output_dir / f"{tag}_rna_unirep.h5ad"
    atac_unirep_h5ad = output_dir / f"{tag}_atac_unirep.h5ad"
    cell_integration_yaml = output_dir / f"{tag}_cell_integration.yaml"
    metrics_json = output_dir / f"{tag}_metrics.json"
    metrics_tsv = output_dir / f"{tag}_metrics.tsv"
    rna_unirep_input = ensure_rna_hvg(Path(args.rna), output_dir / f"{tag}_rna_with_hvg.h5ad")

    subprocess.check_call([
        args.python_bin, os.fspath(rna_unirep_script),
        "-i", os.fspath(rna_unirep_input),
        "-o", os.fspath(rna_unirep_h5ad),
    ])
    subprocess.check_call([
        args.python_bin, os.fspath(atac_unirep_script),
        "-i", args.atac,
        "-o", os.fspath(atac_unirep_h5ad),
    ])
    subprocess.check_call([
        args.python_bin, os.fspath(cell_integration_script),
        "-d", os.fspath(rna_unirep_h5ad), os.fspath(atac_unirep_h5ad),
        "-l", os.fspath(rna_latent_csv), os.fspath(atac_latent_csv),
        "--cell-type", args.cell_type_key,
        "--domain", args.domain_key,
        "-o", os.fspath(cell_integration_yaml),
    ])

    with cell_integration_yaml.open("r", encoding="utf-8") as fh:
        metrics = yaml.load(fh, Loader=yaml.Loader)
    with metrics_json.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    pd.DataFrame([metrics]).to_csv(metrics_tsv, sep="\t", index=False, float_format="%.6f")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved strict scMRDR-style metrics to {cell_integration_yaml}")


if __name__ == "__main__":
    main()
