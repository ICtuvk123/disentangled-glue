"""Evaluate integration metrics for the standard GLUE baseline."""
import sys
import numpy as np
import scanpy as sc
sys.path.insert(0, "/data1/users/zhutianci/proj/disentangled-glue")

from scglue.metrics import (
    avg_silhouette_width,
    avg_silhouette_width_batch,
    graph_connectivity,
    mean_average_precision,
    normalized_mutual_info,
    seurat_alignment_score,
)

COMBINED = "s02_compare/scglue/combined_glue.h5ad"
CELL_TYPE_KEY = "cell_type"
DOMAIN_KEY    = "domain"
BATCH_KEY     = None   # set to e.g. "batch" if applicable

print(f"Loading {COMBINED} ...")
adata = sc.read(COMBINED)
x  = adata.obsm["X_glue"]
ct = adata.obs[CELL_TYPE_KEY].to_numpy().astype(str)
domain = adata.obs[DOMAIN_KEY].to_numpy().astype(str)

print("Computing metrics ...")
results = {
    "NMI":          normalized_mutual_info(x, ct),
    "ARI_MAP":       mean_average_precision(x, ct),
    "ASW_celltype":  avg_silhouette_width(x, ct),
    "Graph_conn":    graph_connectivity(x, ct),
    "Seurat_domain": seurat_alignment_score(x, domain),
}

if BATCH_KEY and BATCH_KEY in adata.obs:
    batch = adata.obs[BATCH_KEY].to_numpy().astype(str)
    results["ASW_batch"]    = avg_silhouette_width_batch(x, batch, ct)
    results["Seurat_batch"] = seurat_alignment_score(x, batch)

bio   = np.mean([results["NMI"], results["ARI_MAP"],
                 results["ASW_celltype"], results["Graph_conn"]])
integ = results.get("Seurat_batch", results["Seurat_domain"])
results["Bio_avg"]     = bio
results["Integration"] = integ
results["Overall"]     = 0.6 * bio + 0.4 * integ

print("\n=== Standard GLUE baseline metrics ===")
for k, v in results.items():
    print(f"  {k:<20s} {v:.4f}")
