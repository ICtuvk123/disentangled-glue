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


def canonicalize_domain(value: object) -> str:
    key = str(value).strip().lower()
    mapping = {
        "0": "rna",
        "1": "atac",
        "gex": "rna",
        "rna": "rna",
        "scrna-seq": "rna",
        "sc rna-seq": "rna",
        "atac": "atac",
        "scatac-seq": "atac",
        "sc atac-seq": "atac",
    }
    return mapping.get(key, key)


def prefixed_obs_names(path: Path, domain: str) -> pd.Index:
    adata = sc.read_h5ad(path, backed="r")
    try:
        return pd.Index([f"{domain}:{obs_name}" for obs_name in adata.obs_names])
    finally:
        adata.file.close()


def load_modality_latents(
    run_dir: Path, rna_path: Path, atac_path: Path, domain_key: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rna_glue_path = run_dir / "rna_glue.h5ad"
    atac_glue_path = run_dir / "atac_glue.h5ad"
    if rna_glue_path.exists() and atac_glue_path.exists():
        rna_glue = sc.read(rna_glue_path)
        atac_glue = sc.read(atac_glue_path)
        return (
            pd.DataFrame(rna_glue.obsm["X_glue"], index=rna_glue.obs_names),
            pd.DataFrame(atac_glue.obsm["X_glue"], index=atac_glue.obs_names),
        )

    combined_path = run_dir / "combined_glue.h5ad"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Missing {rna_glue_path.name}/{atac_glue_path.name} and {combined_path.name} under {run_dir}"
        )

    combined = sc.read(combined_path)
    if "X_glue" not in combined.obsm:
        raise KeyError(f"Missing X_glue in {combined_path}")
    if domain_key not in combined.obs:
        raise KeyError(f"Missing obs[{domain_key!r}] in {combined_path}")

    obs_index = pd.Index(combined.obs_names)
    def extract(domain: str, source_path: Path) -> pd.DataFrame:
        desired = prefixed_obs_names(source_path, domain)
        missing = desired.difference(obs_index)
        if not missing.empty:
            raise ValueError(
                f"{combined_path.name} is missing {domain} cells required by {source_path.name}; "
                f"first missing entries: {missing[:5].tolist()}"
            )
        subset = combined[desired].copy()
        subset_domains = subset.obs[domain_key].map(canonicalize_domain).astype(str)
        if not (subset_domains == domain).all():
            bad = subset.obs_names[subset_domains != domain][:5].tolist()
            raise ValueError(
                f"{combined_path.name} contains cells with non-{domain} domain labels in the {domain} slice: {bad}"
            )
        return pd.DataFrame(subset.obsm["X_glue"], index=subset.obs_names.str.split(":", n=1).str[-1])

    return extract("rna", rna_path), extract("atac", atac_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Run directory containing rna_glue.h5ad/atac_glue.h5ad or combined_glue.h5ad")
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


def ensure_domain_obs(
    input_path: Path, output_path: Path, domain_key: str, domain_value: str
) -> Path:
    adata = sc.read(input_path)
    if domain_key not in adata.obs:
        adata.obs[domain_key] = domain_value
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

    rna_latent_csv = output_dir / f"{tag}_rna_latent.csv"
    atac_latent_csv = output_dir / f"{tag}_atac_latent.csv"
    rna_latent_df, atac_latent_df = load_modality_latents(
        run_dir, Path(args.rna), Path(args.atac), args.domain_key
    )
    rna_latent_df.to_csv(rna_latent_csv, header=False)
    atac_latent_df.to_csv(atac_latent_csv, header=False)

    rna_unirep_h5ad = output_dir / f"{tag}_rna_unirep.h5ad"
    atac_unirep_h5ad = output_dir / f"{tag}_atac_unirep.h5ad"
    cell_integration_yaml = output_dir / f"{tag}_cell_integration.yaml"
    metrics_json = output_dir / f"{tag}_metrics.json"
    metrics_tsv = output_dir / f"{tag}_metrics.tsv"
    rna_unirep_input = ensure_rna_hvg(Path(args.rna), output_dir / f"{tag}_rna_with_hvg.h5ad")
    rna_unirep_input = ensure_domain_obs(
        rna_unirep_input, output_dir / f"{tag}_rna_with_hvg_domain.h5ad", args.domain_key, "rna"
    )
    atac_unirep_input = ensure_domain_obs(
        Path(args.atac), output_dir / f"{tag}_atac_with_domain.h5ad", args.domain_key, "atac"
    )

    subprocess.check_call([
        args.python_bin, os.fspath(rna_unirep_script),
        "-i", os.fspath(rna_unirep_input),
        "-o", os.fspath(rna_unirep_h5ad),
    ])
    subprocess.check_call([
        args.python_bin, os.fspath(atac_unirep_script),
        "-i", os.fspath(atac_unirep_input),
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
