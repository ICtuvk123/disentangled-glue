r"""
Compute unimodal representation of ATAC data
(for computing the neighbor conservation metric)
"""

import argparse
import faulthandler
import os
import pathlib
import resource
import sys
import time
import traceback

import anndata
import scglue


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Compute unimodal representation of ATAC data"
    )
    parser.add_argument(
        "-i", "--input", dest="input", type=pathlib.Path, required=True,
        help="Path to input file (.h5ad)"
    )
    parser.add_argument(
        "-d", "--dim", dest="dim", type=int, default=50,
        help="Dimensionality of the representation"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=pathlib.Path, required=True,
        help="Path to output file (.h5ad)"
    )
    return parser.parse_args()


def rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def describe_matrix(matrix: object) -> str:
    shape = getattr(matrix, "shape", None)
    dtype = getattr(matrix, "dtype", None)
    nnz = getattr(matrix, "nnz", None)
    parts = [f"type={type(matrix).__name__}"]
    if shape is not None:
        parts.append(f"shape={tuple(shape)}")
    if dtype is not None:
        parts.append(f"dtype={dtype}")
    if nnz is not None:
        parts.append(f"nnz={nnz}")
    return ", ".join(parts)


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(300, repeat=True)
    t0 = time.time()
    try:
        print(f"[0/3] PID={os.getpid()} rss_mb={rss_mb():.1f}", flush=True)
        print(f"[1/3] Reading data from {args.input} ...", flush=True)
        adata = anndata.read_h5ad(args.input)
        print(
            f"[1/3] Read complete: n_obs={adata.n_obs}, n_vars={adata.n_vars}, "
            f"rss_mb={rss_mb():.1f}",
            flush=True,
        )
        print(f"[1/3] Matrix: {describe_matrix(adata.X)}", flush=True)

        print(
            f"[2/3] Computing LSI (dim={args.dim}, use_highly_variable=False, n_iter=15) ...",
            flush=True,
        )
        lsi_start = time.time()
        scglue.data.lsi(adata, n_components=args.dim, use_highly_variable=False, n_iter=15)
        print(
            f"[2/3] LSI complete in {time.time() - lsi_start:.1f}s, "
            f"X_lsi_shape={adata.obsm['X_lsi'].shape}, rss_mb={rss_mb():.1f}",
            flush=True,
        )

        print(f"[3/3] Saving results to {args.output} ...", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        anndata.AnnData(
            X=adata.obsm["X_lsi"], obs=adata.obs
        ).write(args.output, compression="gzip")
        print(
            f"[3/3] Done in {time.time() - t0:.1f}s, output={args.output}",
            flush=True,
        )
    except Exception:
        print(
            f"[ERROR] atac_unirep failed after {time.time() - t0:.1f}s, rss_mb={rss_mb():.1f}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main(parse_args())
