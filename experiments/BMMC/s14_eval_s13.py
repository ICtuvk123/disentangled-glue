#!/usr/bin/env python
"""
Batch-evaluate all completed s13 RNA+ATAC search trials with full scib-metrics (PCR included).

Finds all trial directories under --search-dir that have combined_glue.h5ad,
skips those already evaluated (unless --force), and runs s06_eval.py in parallel
using a process pool.

Usage:
    python s14_eval_s13.py \
        --search-dir s13_rna_atac_search \
        --feature-aligned s01_preprocessing/feature_aligned_sampled.h5ad \
        --output-dir s14_eval_s13 \
        --n-workers 4

After completion, prints a ranked summary table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
S06_EVAL   = SCRIPT_DIR / "s06_eval.py"
PYTHON_BIN = sys.executable


def find_trials(search_dir: Path) -> list[Path]:
    """Return trial directories that have combined_glue.h5ad."""
    return sorted(
        p for p in search_dir.iterdir()
        if p.is_dir() and (p / "combined_glue.h5ad").exists()
    )


def already_done(trial_dir: Path, output_dir: Path) -> bool:
    tag = trial_dir.name
    return (output_dir / f"{tag}_unscaled.csv").exists()


def run_one(trial_dir: Path, output_dir: Path, feature_aligned: Path,
            n_jobs: int) -> tuple[str, bool, str]:
    tag = trial_dir.name
    cmd = [
        PYTHON_BIN, str(S06_EVAL),
        "--run-dir",        str(trial_dir),
        "--output-dir",     str(output_dir),
        "--tag",            tag,
        "--feature-aligned", str(feature_aligned),
        "--enable-pcr",
        "--domain-key",     "domain",
        "--cell-type-key",  "celltype",
        "--batch-key",      "batch",
        "--n-jobs",         str(n_jobs),
        "--no-show",
    ]
    log_path = output_dir / f"{tag}.log"
    try:
        with log_path.open("w") as fh:
            result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
        ok = result.returncode == 0
        return tag, ok, str(log_path)
    except subprocess.TimeoutExpired:
        return tag, False, f"TIMEOUT after 1800s — see {log_path}"
    except Exception as e:
        return tag, False, str(e)


def collect_results(output_dir: Path, search_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in output_dir.glob("*_unscaled.csv"):
        tag = csv_path.stem.replace("_unscaled", "")
        df = pd.read_csv(csv_path, index_col=0)
        # df rows = metrics, col = embedding key ("X_embed")
        # Aggregate scores are in rows with "Aggregate score" label or similar
        # scib-metrics output: index = metric name, columns = embeddings
        scores = df.get("X_embed", df.iloc[:, 0])
        row = {"trial": tag}
        for idx, val in scores.items():
            row[idx] = val
        # Also load hparams
        hp_path = search_dir / tag / "hparams.json"
        if hp_path.exists():
            with hp_path.open() as f:
                row.update(json.load(f))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results to summarize.")
        return

    score_cols = [c for c in df.columns if c in (
        "Bio conservation", "Batch correction", "Modality integration", "Total"
    )]
    if not score_cols:
        # Try aggregate score rows from scib output transposed differently
        print("Available columns:", df.columns.tolist()[:20])
        return

    hparam_cols = ["mode", "shared_dim", "private_dim", "lam_align", "beta_shared",
                   "lam_iso", "beta_private_rna", "beta_private_atac"]
    hparam_cols = [c for c in hparam_cols if c in df.columns]
    show_cols = ["trial"] + score_cols + hparam_cols

    df_show = df[show_cols].copy()
    if "Total" in df_show:
        df_show = df_show.sort_values("Total", ascending=False)

    print(f"\n{'='*80}")
    print(f"s13 full-metrics summary  (n={len(df_show)})")
    print(f"{'='*80}")
    print(df_show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if "mode" in df_show.columns and "Total" in df_show.columns:
        print("\n=== Mean ± Std by mode ===")
        for mode, grp in df_show.groupby("mode"):
            print(f"\n{mode} (n={len(grp)}):")
            for c in score_cols:
                print(f"  {c}: {grp[c].mean():.4f} ± {grp[c].std():.4f}  [max={grp[c].max():.4f}]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search-dir",
                   default=str(SCRIPT_DIR / "s13_rna_atac_search"),
                   help="Directory containing s13 trial subdirectories")
    p.add_argument("--feature-aligned",
                   default=str(SCRIPT_DIR / "s01_preprocessing" / "feature_aligned_sampled.h5ad"),
                   help="Path to feature_aligned.h5ad for PCR baseline")
    p.add_argument("--output-dir",
                   default=str(SCRIPT_DIR / "s14_eval_s13"),
                   help="Directory to write per-trial CSVs and logs")
    p.add_argument("--n-workers", type=int, default=4,
                   help="Number of parallel evaluation workers")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="scib-metrics n_jobs per trial (keep low to avoid memory pressure)")
    p.add_argument("--force", action="store_true",
                   help="Re-evaluate even if output CSVs already exist")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip evaluation, just print summary of existing results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    search_dir     = Path(args.search_dir)
    feature_aligned = Path(args.feature_aligned)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not feature_aligned.exists():
        print(f"ERROR: feature_aligned not found: {feature_aligned}", file=sys.stderr)
        sys.exit(1)

    trials = find_trials(search_dir)
    print(f"Found {len(trials)} trials with combined_glue.h5ad in {search_dir}")

    if args.summary_only:
        df = collect_results(output_dir, search_dir)
        print_summary(df)
        return

    to_run = [t for t in trials if args.force or not already_done(t, output_dir)]
    skipped = len(trials) - len(to_run)
    print(f"  {skipped} already evaluated (skipping), {len(to_run)} to run")

    if to_run:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {
                pool.submit(run_one, t, output_dir, feature_aligned, args.n_jobs): t
                for t in to_run
            }
            done = 0
            for fut in as_completed(futures):
                tag, ok, info = fut.result()
                done += 1
                status = "OK" if ok else "FAIL"
                print(f"[{done}/{len(to_run)}] {status}  {tag}  ({info})")

    df = collect_results(output_dir, search_dir)
    print_summary(df)

    summary_path = output_dir / "summary.csv"
    if not df.empty:
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
