#!/usr/bin/env python
"""
Train/evaluate full-data BMMC models from the preserved sampled-model shortlist.

Default behavior:
- read candidate definitions from preserved_models_sampled_20260318/manifest.tsv
- build full preprocessing if missing
- build shared graph preprocessing if missing
- retrain each shortlisted model on the full dataset with the saved hyperparameters
- evaluate each full run with the true scMRDR-aligned metric stack
- summarize aggregate scores in a TSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import networkx as nx


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=os.fspath(script_dir / "preserved_models_sampled_20260318" / "manifest.tsv"),
        help="TSV manifest describing preserved candidate models",
    )
    parser.add_argument(
        "--slots",
        type=int,
        nargs="*",
        default=None,
        help="Optional manifest slot numbers to run; default runs all rows",
    )
    parser.add_argument(
        "--prep-dir",
        default=os.fspath(script_dir / "s01_preprocessing_full"),
        help="Directory for full preprocessing outputs",
    )
    parser.add_argument(
        "--pp-dir",
        default=os.fspath(script_dir / "s10_preserved_full_runs" / "preprocessed"),
        help="Directory for shared graph preprocessing outputs",
    )
    parser.add_argument(
        "--run-root",
        default=os.fspath(script_dir / "s10_preserved_full_runs" / "models"),
        help="Directory for full trained model runs",
    )
    parser.add_argument(
        "--eval-root",
        default=os.fspath(script_dir / "s10_preserved_full_runs" / "eval_true_scmrdr"),
        help="Directory for true scMRDR-aligned evaluation outputs",
    )
    parser.add_argument(
        "--summary-tsv",
        default=os.fspath(script_dir / "s10_preserved_full_runs" / "summary.tsv"),
        help="Summary TSV path",
    )
    parser.add_argument(
        "--multiome",
        default=os.fspath(script_dir / "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"),
        help="Path to processed BMMC multiome h5ad",
    )
    parser.add_argument(
        "--cite",
        default=os.fspath(script_dir / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"),
        help="Path to processed BMMC CITE-seq h5ad",
    )
    parser.add_argument(
        "--gtf",
        default=os.fspath(script_dir / "gencode.v38.chr_patch_hapl_scaff.annotation.gtf"),
        help="Path to the BMMC GTF annotation",
    )
    parser.add_argument(
        "--hgnc",
        default=os.fspath(script_dir / "hgnc_complete_set.txt"),
        help="Path to the HGNC mapping table",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch the pipeline scripts",
    )
    parser.add_argument(
        "--batch-key",
        default="batch",
        help="Batch column for training/evaluation",
    )
    parser.add_argument(
        "--cell-type-key",
        default="celltype",
        help="Cell type column for true scMRDR evaluation",
    )
    parser.add_argument(
        "--domain-key",
        default="domain",
        help="Modality/domain column for true scMRDR evaluation",
    )
    parser.add_argument(
        "--align-support-k",
        type=int,
        default=15,
        help="Support weighting K for support-mode full retraining",
    )
    parser.add_argument(
        "--align-support-strategy",
        choices=["soft", "hard"],
        default="soft",
        help="Support weighting strategy for support-mode full retraining",
    )
    parser.add_argument(
        "--align-support-min-weight",
        type=float,
        default=0.05,
        help="Minimum weight for soft support weighting",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Parallel jobs for true scMRDR evaluation",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train full models; skip evaluation and summary",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; only run evaluation/summary on existing full runs",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return rows


def selected_rows(rows: list[dict[str, str]], slots: list[int] | None) -> list[dict[str, str]]:
    if not slots:
        return rows
    wanted = {str(slot) for slot in slots}
    return [row for row in rows if row["slot"] in wanted]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def shared_preprocessing_valid(pp_dir: Path) -> bool:
    required = [
        pp_dir / "rna_pp.h5ad",
        pp_dir / "atac_pp.h5ad",
        pp_dir / "prot_pp.h5ad",
        pp_dir / "guidance.graphml.gz",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        nx.read_graphml(pp_dir / "guidance.graphml.gz")
    except Exception as exc:
        print(f"Detected invalid shared preprocessing under {pp_dir}: {exc}")
        return False
    return True


def ensure_full_preprocessing(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    prep_dir = Path(args.prep_dir).resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)

    rna = prep_dir / "RNA_counts_qc.h5ad"
    atac = prep_dir / "ATAC_counts_qc.h5ad"
    prot = prep_dir / "protein_counts_qc.h5ad"
    feature_aligned = prep_dir / "feature_aligned.h5ad"
    protein_gene_map = prep_dir / "protein_gene_map.tsv"

    required = [rna, atac, prot, feature_aligned, protein_gene_map]
    if not all(path.exists() for path in required):
        run([
            args.python_bin,
            os.fspath(Path(__file__).resolve().parent / "s01_preprocessing.py"),
            "--multiome", os.fspath(Path(args.multiome).resolve()),
            "--cite", os.fspath(Path(args.cite).resolve()),
            "--gtf", os.fspath(Path(args.gtf).resolve()),
            "--hgnc", os.fspath(Path(args.hgnc).resolve()),
            "--output-dir", os.fspath(prep_dir),
        ])

    return rna, atac, prot, feature_aligned, protein_gene_map


def ensure_shared_preprocessing(
    args: argparse.Namespace,
    rna: Path,
    atac: Path,
    prot: Path,
    protein_gene_map: Path,
) -> Path:
    pp_dir = Path(args.pp_dir).resolve()
    pp_dir.mkdir(parents=True, exist_ok=True)
    if shared_preprocessing_valid(pp_dir):
        return pp_dir
    for path in (
        pp_dir / "rna_pp.h5ad",
        pp_dir / "atac_pp.h5ad",
        pp_dir / "prot_pp.h5ad",
        pp_dir / "guidance.graphml.gz",
    ):
        if path.exists():
            path.unlink()

    cmd = [
        args.python_bin,
        os.fspath(Path(__file__).resolve().parent / "s02_glue.py"),
        "--model", "disentangled",
        "--rna", os.fspath(rna),
        "--atac", os.fspath(atac),
        "--prot", os.fspath(prot),
        "--gtf", os.fspath(Path(args.gtf).resolve()),
        "--protein-gene-map", os.fspath(protein_gene_map),
        "--batch-key", args.batch_key,
        "--output-dir", os.fspath(pp_dir),
        "--preprocess-only",
    ]
    run(cmd)
    return pp_dir


def load_hparams(source_dir: Path) -> dict[str, object]:
    hparams_path = source_dir / "hparams.json"
    if not hparams_path.exists():
        return {}
    with hparams_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_tag(row: dict[str, str]) -> str:
    return f"{row['link_name']}_{row['label']}_full"


def train_candidate(
    args: argparse.Namespace,
    row: dict[str, str],
    run_dir: Path,
    pp_dir: Path,
    rna: Path,
    atac: Path,
    prot: Path,
    protein_gene_map: Path,
) -> None:
    if (run_dir / "combined_glue.h5ad").exists():
        print(f"Skipping training for {row['label']} because {run_dir} already exists")
        return

    source_dir = (Path(__file__).resolve().parent / row["source_dir"]).resolve()
    hparams = load_hparams(source_dir)
    mode = (hparams.get("mode") or row.get("mode") or "").strip().lower()
    is_scglue = row.get("mode", "").strip().lower() == "scglue"

    cmd = [
        args.python_bin,
        os.fspath(Path(__file__).resolve().parent / "s02_glue.py"),
        "--rna", os.fspath(rna),
        "--atac", os.fspath(atac),
        "--prot", os.fspath(prot),
        "--gtf", os.fspath(Path(args.gtf).resolve()),
        "--protein-gene-map", os.fspath(protein_gene_map),
        "--batch-key", args.batch_key,
        "--output-dir", os.fspath(run_dir),
        "--preprocessed-dir", os.fspath(pp_dir),
    ]

    if is_scglue:
        cmd.extend(["--model", "scglue"])
    else:
        cmd.extend(["--model", "disentangled"])
        for key, arg_name in (
            ("shared_dim", "--shared-dim"),
            ("private_dim", "--private-dim"),
            ("beta_shared", "--beta-shared"),
            ("lam_iso", "--lam-iso"),
            ("lam_align", "--lam-align"),
            ("beta_private_rna", "--beta-private-rna"),
            ("beta_private_atac", "--beta-private-atac"),
            ("beta_private_prot", "--beta-private-prot"),
        ):
            if key in hparams and hparams[key] not in (None, ""):
                cmd.extend([arg_name, str(hparams[key])])
        if mode == "support":
            cmd.extend([
                "--align-support",
                "--align-support-k", str(args.align_support_k),
                "--align-support-strategy", args.align_support_strategy,
                "--align-support-min-weight", str(args.align_support_min_weight),
            ])

    run(cmd)


def evaluate_candidate(
    args: argparse.Namespace,
    row: dict[str, str],
    run_dir: Path,
    eval_root: Path,
    feature_aligned: Path,
) -> Path:
    tag = build_tag(row)
    out_dir = eval_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    unscaled = out_dir / f"{tag}_unscaled.csv"
    if unscaled.exists():
        print(f"Skipping evaluation for {row['label']} because {unscaled} already exists")
        return unscaled

    run([
        args.python_bin,
        os.fspath(Path(__file__).resolve().parent / "s09_eval_true_scmrdr.py"),
        "--run-dir", os.fspath(run_dir),
        "--feature-aligned", os.fspath(feature_aligned),
        "--output-dir", os.fspath(out_dir),
        "--tag", tag,
        "--cell-type-key", args.cell_type_key,
        "--batch-key", args.batch_key,
        "--domain-key", args.domain_key,
        "--n-jobs", str(args.n_jobs),
        "--python-bin", args.python_bin,
    ])
    return unscaled


def parse_unscaled_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        first = next(reader)
    return first


def write_summary(summary_path: Path, rows: list[dict[str, str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slot",
        "label",
        "link_name",
        "source_dir",
        "mode",
        "full_run_dir",
        "eval_unscaled_csv",
        "Batch correction",
        "Bio conservation",
        "Modality integration",
        "Total",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.train_only and args.eval_only:
        raise ValueError("`--train-only` and `--eval-only` cannot be used together.")

    manifest = Path(args.manifest).resolve()
    rows = selected_rows(read_manifest(manifest), args.slots)
    if not rows:
        raise ValueError("No manifest rows selected.")

    rna, atac, prot, feature_aligned, protein_gene_map = ensure_full_preprocessing(args)
    pp_dir = ensure_shared_preprocessing(args, rna, atac, prot, protein_gene_map)

    run_root = Path(args.run_root).resolve()
    eval_root = Path(args.eval_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str]] = []
    for row in rows:
        tag = build_tag(row)
        run_dir = run_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        if not args.eval_only:
            train_candidate(args, row, run_dir, pp_dir, rna, atac, prot, protein_gene_map)

        if args.train_only:
            continue

        unscaled = evaluate_candidate(args, row, run_dir, eval_root, feature_aligned)
        metrics = parse_unscaled_csv(unscaled)
        summary_rows.append({
            "slot": row["slot"],
            "label": row["label"],
            "link_name": row["link_name"],
            "source_dir": row["source_dir"],
            "mode": row["mode"],
            "full_run_dir": os.fspath(run_dir),
            "eval_unscaled_csv": os.fspath(unscaled),
            "Batch correction": metrics["Batch correction"],
            "Bio conservation": metrics["Bio conservation"],
            "Modality integration": metrics["Modality integration"],
            "Total": metrics["Total"],
        })

    if not args.train_only:
        write_summary(Path(args.summary_tsv).resolve(), summary_rows)
        print(f"Saved summary to {Path(args.summary_tsv).resolve()}")


if __name__ == "__main__":
    main()
