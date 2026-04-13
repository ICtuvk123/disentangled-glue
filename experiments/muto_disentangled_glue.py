"""
Disentangled-GLUE training on the Muto (mop) dataset.

RNA_counts_qc.h5ad / ATAC_counts_qc.h5ad already contain:
  - chrom / chromStart / chromEnd
  - highly_variable
  - counts layer

So Part 1 only needs to run PCA / LSI and build the guidance graph.
Part 2 trains DisentangledSCGLUEModel.
"""

import os
import sys

# Ensure a valid GPU is visible before any CUDA-touching import
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from itertools import chain
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import scanpy as sc
from matplotlib import rcParams

# ── local disentangled-glue checkout ──────────────────────────────────────────
DGLUE_ROOT = Path("/data1/users/zhutianci/proj/disentangled-glue")
sys.path.insert(0, str(DGLUE_ROOT))

import scglue
from scglue.models import configure_dataset, fit_SCGLUE
from scglue.models.scglue import DisentangledSCGLUEModel

scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/data1/users/zhutianci/proj/disentangled-glue")
OUT_DIR  = DATA_DIR
BEDTOOLS = "/data1/users/zhutianci/.conda/envs/scMRDR/bin/bedtools"

# ==============================================================================
# PART 1 – PCA / LSI + guidance graph
# ==============================================================================

def preprocess():
    print("=== Part 1: PCA / LSI + guidance graph ===")

    os.environ["PATH"] += os.pathsep + str(Path(BEDTOOLS).parent)
    scglue.config.BEDTOOLS_PATH = BEDTOOLS

    rna  = sc.read(str(DATA_DIR / "RNA_counts_qc.h5ad"))
    atac = sc.read(str(DATA_DIR / "ATAC_counts_qc.h5ad"))

    # deduplicate
    rna  = rna[:,  ~rna.var_names.duplicated(keep="first")]
    atac = atac[:, ~atac.var_names.duplicated(keep="first")]

    # dimensionality reduction
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")
    scglue.data.lsi(atac, n_components=100, n_iter=15)

    # restrict to HVF
    rna  = rna[:,  rna.var["highly_variable"]]
    atac = atac[:, atac.var["highly_variable"]]

    # drop features without chrom annotation (safety check)
    rna  = rna[:,  ~rna.var["chrom"].isnull()]
    atac = atac[:, ~atac.var["chrom"].isnull()]

    # build guidance graph
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    scglue.graph.check_graph(guidance, [rna, atac])

    # save
    rna.write( str(OUT_DIR / "rna-pp.h5ad"),  compression="gzip")
    atac.write(str(OUT_DIR / "atac-pp.h5ad"), compression="gzip")
    nx.write_graphml(guidance, str(OUT_DIR / "guidance.graphml.gz"))
    print("Done.")


# ==============================================================================
# PART 2 – Disentangled-GLUE training
# ==============================================================================

def train():
    print("=== Part 2: Disentangled-GLUE training ===")

    rna      = ad.read_h5ad(str(OUT_DIR / "rna-pp.h5ad"))
    atac     = ad.read_h5ad(str(OUT_DIR / "atac-pp.h5ad"))
    guidance = nx.read_graphml(str(OUT_DIR / "guidance.graphml.gz"))

    configure_dataset(
        rna, "NB",
        use_highly_variable=True,
        use_layer="counts",
        use_rep="X_pca",
        use_batch="batch",
    )
    configure_dataset(
        atac, "NB",
        use_highly_variable=True,
        use_layer="counts",
        use_rep="X_lsi",
        use_batch="batch",
    )

    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index,
    )).copy()

    glue = fit_SCGLUE(
        {"rna": rna, "atac": atac},
        guidance_hvf,
        model=DisentangledSCGLUEModel,
        init_kws={
            "shared_dim":  48,
            "private_dim": 16,
            "h_depth":      2,
            "h_dim":      256,
            "dropout":    0.2,
        },
        compile_kws={
            "lam_data":     1.0,
            "beta_shared":  0.75,   # 降低，防止 shared KL collapse
            "beta_private": 0.1,
            "lam_graph":    0.02,
            "lam_align":    0.03,   # 提高，强迫 shared 空间对齐
            "lam_iso":      1.0,
            "lr":           2e-3,
        },
        fit_kws={"directory": str(OUT_DIR / "glue_disentangled")},
    )

    glue.save(str(OUT_DIR / "glue_disentangled.dill"))

    # shared embedding
    rna.obsm["X_glue"]  = glue.encode_data("rna",  rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    # shared + private embeddings
    rna.obsm["X_glue_shared"],  rna.obsm["X_glue_private"]  = \
        glue.encode_data("rna",  rna,  return_private=True)
    atac.obsm["X_glue_shared"], atac.obsm["X_glue_private"] = \
        glue.encode_data("atac", atac, return_private=True)

    combined = ad.concat([rna, atac])
    np.save(str(OUT_DIR / "glue_disentangled.npy"), combined.obsm["X_glue"])

    rna.write( str(OUT_DIR / "rna-disentangled.h5ad"),  compression="gzip")
    atac.write(str(OUT_DIR / "atac-disentangled.h5ad"), compression="gzip")

    print(f"Done. Shared embedding: {combined.obsm['X_glue'].shape}")


# ==============================================================================

if __name__ == "__main__":
    pp_done = (
        (OUT_DIR / "rna-pp.h5ad").exists()
        and (OUT_DIR / "atac-pp.h5ad").exists()
        and (OUT_DIR / "guidance.graphml.gz").exists()
    )
    if not pp_done:
        preprocess()
    else:
        print("rna-pp / atac-pp / guidance already exist, skipping Part 1.")

    train()
