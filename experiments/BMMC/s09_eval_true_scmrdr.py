#!/usr/bin/env python
"""
Evaluate a BMMC run with the same metric stack used in
`scMRDR/experiments/plots/metrics.py`.

This is a thin wrapper around `s06_eval.py` with the settings required to stay
aligned with the "true" scMRDR benchmark:

- use the shared `feature_aligned(.h5ad)` matrix as the pre-integration baseline
- enable PCR metrics
- use the BMMC label/batch defaults from the original scMRDR scripts

For full-data evaluation, point `--feature-aligned` at the full preprocessing
output, e.g. `s01_preprocessing_full/feature_aligned.h5ad`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing combined_glue.h5ad for the run to evaluate",
    )
    parser.add_argument(
        "--feature-aligned",
        default=os.fspath(script_dir / "s01_preprocessing_full" / "feature_aligned.h5ad"),
        help="Path to feature_aligned.h5ad used as the shared pre-integration baseline",
    )
    parser.add_argument(
        "--output-dir",
        default="s09_eval_true_scmrdr",
        help="Directory to save CSV/PDF outputs",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Filename tag for outputs (defaults to run dir name)",
    )
    parser.add_argument(
        "--cell-type-key",
        default="celltype",
        help="obs column for BMMC cell type labels",
    )
    parser.add_argument(
        "--batch-key",
        default="batch",
        help="obs column for BMMC batch labels",
    )
    parser.add_argument(
        "--domain-key",
        default="domain",
        help="obs column for integrated modality labels",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Parallel jobs for neighbor computation",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch s06_eval.py",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of saving only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    feature_aligned = Path(args.feature_aligned).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (run_dir / "combined_glue.h5ad").exists():
        raise FileNotFoundError(f"Missing combined_glue.h5ad under {run_dir}")
    if not feature_aligned.exists():
        raise FileNotFoundError(
            "Missing feature_aligned.h5ad required for true scMRDR-style evaluation: "
            f"{feature_aligned}"
        )

    script_dir = Path(__file__).resolve().parent
    s06_eval = script_dir / "s06_eval.py"
    cmd = [
        args.python_bin,
        os.fspath(s06_eval),
        "--run-dir", os.fspath(run_dir),
        "--feature-aligned", os.fspath(feature_aligned),
        "--enable-pcr",
        "--output-dir", os.fspath(output_dir),
        "--cell-type-key", args.cell_type_key,
        "--batch-key", args.batch_key,
        "--domain-key", args.domain_key,
        "--n-jobs", str(args.n_jobs),
    ]
    if args.tag:
        cmd.extend(["--tag", args.tag])
    if not args.show:
        cmd.append("--no-show")

    print("Running true scMRDR-aligned evaluation:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
