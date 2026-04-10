"""
Evaluate all completed hparam_search trials using the same Benchmarker2
pipeline as metrics.py.

Embeddings in hparam_search/<tag>/embedding.npy are stacked as
[rna (19985), atac (24205)] which matches the cell order in
feature_aligned_trained.h5ad (modality 0=RNA, 1=ATAC).
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
from pathlib import Path
import numpy as np
import scanpy as sc
import pandas as pd

DGLUE_ROOT  = Path("/data1/users/zhutianci/proj/disentangled-glue")
SCMRDR_ROOT = Path("/data1/users/zhutianci/proj/scMRDR")
sys.path.insert(0, str(SCMRDR_ROOT / "experiments" / "plots"))

from metrics import Benchmarker2, BioConservation2, BatchCorrection2, ModalityIntegration2

SEARCH_DIR = DGLUE_ROOT / "hparam_search"
OUT_DIR    = SEARCH_DIR

# ── load base adata ───────────────────────────────────────────────────────────
adata = sc.read_h5ad(str(DGLUE_ROOT / "feature_aligned_trained.h5ad"))
adata.obs["modality"] = adata.obs["modality"].astype(str)
adata.obs.loc[adata.obs["modality"] == "0", "modality"] = "RNA"
adata.obs.loc[adata.obs["modality"] == "1", "modality"] = "ATAC"

# ── collect completed trials ──────────────────────────────────────────────────
embedding_keys = []
for trial_dir in sorted(SEARCH_DIR.iterdir()):
    npy = trial_dir / "embedding.npy"
    if not npy.exists():
        continue
    tag = trial_dir.name
    emb = np.load(str(npy))
    adata.obsm[tag] = emb
    embedding_keys.append(tag)
    print(f"Loaded: {tag}")

print(f"\nTotal trials to evaluate: {len(embedding_keys)}")

# ── benchmark ─────────────────────────────────────────────────────────────────
bm2 = Benchmarker2(
    adata,
    batch_key="batch",
    label_key="cell_type",
    modality_key="modality",
    bio_conservation_metrics=BioConservation2(),
    batch_correction_metrics=BatchCorrection2(),
    modality_integration_metrics=ModalityIntegration2(),
    embedding_obsm_keys=embedding_keys,
    n_jobs=20,
)
bm2.benchmark()

df_unscaled = bm2.get_results(min_max_scale=False)
df_scaled   = bm2.get_results(min_max_scale=True)

df_unscaled.to_csv(str(OUT_DIR / "hparam_eval_unscaled.csv"))
df_scaled.to_csv(str(OUT_DIR / "hparam_eval_scaled.csv"))

print("\n=== Unscaled Results ===")
print(df_unscaled.to_string())
print("\n=== Scaled Results ===")
print(df_scaled.to_string())
