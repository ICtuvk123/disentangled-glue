"""
Evaluate disentangled-GLUE on the Muto dataset, consistent with how other
methods are evaluated in metrics.py (same feature_aligned_trained.h5ad base).
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
from pathlib import Path
import numpy as np
import scanpy as sc

DGLUE_ROOT  = Path("/data1/users/zhutianci/proj/disentangled-glue")
SCMRDR_ROOT = Path("/data1/users/zhutianci/proj/scMRDR")
sys.path.insert(0, str(SCMRDR_ROOT / "experiments" / "plots"))

from metrics import Benchmarker2, BioConservation2, BatchCorrection2, ModalityIntegration2

OUT_DIR = str(DGLUE_ROOT)

# ── load the shared base adata (same as all other methods) ────────────────────
adata = sc.read_h5ad(str(DGLUE_ROOT / "feature_aligned_trained.h5ad"))
adata.obs["modality"] = adata.obs["modality"].astype(str)
adata.obs.loc[adata.obs["modality"] == "0", "modality"] = "RNA"
adata.obs.loc[adata.obs["modality"] == "1", "modality"] = "ATAC"

# ── load embeddings ───────────────────────────────────────────────────────────
adata.obsm["Ours"] = adata.obsm["latent_shared"].copy()

glue_dis = np.load(str(DGLUE_ROOT / "glue_disentangled.npy"))
adata.obsm["DisentangledGLUE"] = glue_dis.copy()

# ── benchmark ─────────────────────────────────────────────────────────────────
bm2 = Benchmarker2(
    adata,
    batch_key="batch",
    label_key="cell_type",
    modality_key="modality",
    bio_conservation_metrics=BioConservation2(),
    batch_correction_metrics=BatchCorrection2(),
    modality_integration_metrics=ModalityIntegration2(),
    # pre_integrated_embedding_obsm_key not set → auto PCA on adata.X
    embedding_obsm_keys=["Ours", "DisentangledGLUE"],
    n_jobs=20,
)
bm2.benchmark()

df = bm2.get_results(min_max_scale=False)
df.to_csv(os.path.join(OUT_DIR, "unscaled_metrics_muto_dis.csv"))
df = bm2.get_results(min_max_scale=True)
df.to_csv(os.path.join(OUT_DIR, "scaled_metrics_muto_dis.csv"))
bm2.plot_results_table(tag="muto_dis",        min_max_scale=False, save_dir=OUT_DIR)
bm2.plot_results_table(tag="muto_dis_scaled",  min_max_scale=True,  save_dir=OUT_DIR)

print(bm2.get_results(min_max_scale=False))
