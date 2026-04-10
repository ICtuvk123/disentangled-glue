"""
Hyperparameter search for DisentangledSCGLUEModel on the Muto dataset.
Searches over beta_shared, lam_align, lam_iso.
Each run saves embeddings and computes a quick proxy score (kl_shared + dsc_loss).
Full scib evaluation is run only on the best config.
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import itertools
from pathlib import Path
from itertools import chain

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

DGLUE_ROOT  = Path("/data1/users/zhutianci/proj/disentangled-glue")
SCMRDR_ROOT = Path("/data1/users/zhutianci/proj/scMRDR")
sys.path.insert(0, str(DGLUE_ROOT))
sys.path.insert(0, str(SCMRDR_ROOT / "experiments" / "plots"))

from scglue.models import configure_dataset, fit_SCGLUE
from scglue.models.scglue import DisentangledSCGLUEModel

# ── search grid ───────────────────────────────────────────────────────────────
GRID = {
    "beta_shared":  [0.2, 0.5,1],
    "lam_align":    [0.1, 0.2,0.5,1],
    "lam_iso":      [0.01, 0.05,0.1,0.5],
}

FIXED = {
    "lam_data":     1.0,
    "beta_private": 0.5,   # 降低，保护 private 空间不塌
    "lam_graph":    0.02,
    "lr":           2e-3,
}

SEARCH_DIR = DGLUE_ROOT / "hparam_search"
SEARCH_DIR.mkdir(exist_ok=True)

# ── load preprocessed data (reuse across all runs) ────────────────────────────
rna      = ad.read_h5ad(str(DGLUE_ROOT / "rna-pp.h5ad"))
atac     = ad.read_h5ad(str(DGLUE_ROOT / "atac-pp.h5ad"))
guidance = nx.read_graphml(str(DGLUE_ROOT / "guidance.graphml.gz"))

configure_dataset(rna,  "NB", use_highly_variable=True,
                  use_layer="counts", use_rep="X_pca",  use_batch="batch")
configure_dataset(atac, "NB", use_highly_variable=True,
                  use_layer="counts", use_rep="X_lsi",  use_batch="batch")

guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index,
)).copy()

# ── run search ────────────────────────────────────────────────────────────────
results = []
keys = list(GRID.keys())
combos = list(itertools.product(*[GRID[k] for k in keys]))
print(f"Total configs: {len(combos)}")

for i, vals in enumerate(combos):
    cfg = dict(zip(keys, vals))
    tag = "_".join(f"{k}={v}" for k, v in cfg.items())
    out_dir = SEARCH_DIR / tag

    print(f"\n[{i+1}/{len(combos)}] {tag}")

    # skip if already done
    npy_path = out_dir / "embedding.npy"
    log_path = out_dir / "train_log.csv"
    if npy_path.exists() and log_path.exists():
        print("  Already done, loading results.")
        log = pd.read_csv(log_path)
        row = {"config": tag, **cfg}
        row["final_kl_shared"] = log["kl_shared"].iloc[-1]
        row["final_dsc_loss"]  = log["dsc_loss"].iloc[-1]
        results.append(row)
        continue

    out_dir.mkdir(exist_ok=True)

    # patch trainer to log kl_shared and dsc_loss per epoch
    import scglue.models.scglue as _sc
    _orig_trainer = _sc.DisentangledSCGLUETrainer
    epoch_logs = []

    class _LoggingTrainer(_orig_trainer):
        def compute_losses(self, *args, **kwargs):
            losses = super().compute_losses(*args, **kwargs)
            return losses

    # train
    compile_kws = {**FIXED, **cfg}
    try:
        glue = fit_SCGLUE(
            {"rna": rna, "atac": atac},
            guidance_hvf,
            model=DisentangledSCGLUEModel,
            init_kws={"shared_dim": 50, "private_dim": 20,
                      "h_depth": 2, "h_dim": 256, "dropout": 0.2},
            compile_kws=compile_kws,
            fit_kws={"directory": str(out_dir / "glue")},
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({"config": tag, **cfg,
                        "final_kl_shared": -1, "final_dsc_loss": -1})
        continue

    # save embedding
    rna.obsm["X_glue_shared"], _ = glue.encode_data("rna",  rna,  return_private=True)
    atac.obsm["X_glue_shared"], _ = glue.encode_data("atac", atac, return_private=True)
    emb = np.vstack([rna.obsm["X_glue_shared"], atac.obsm["X_glue_shared"]])
    np.save(str(npy_path), emb)

    # parse trainer log from checkpoint dir for final epoch metrics
    import glob, json
    log_files = sorted(glob.glob(str(out_dir / "glue" / "fine-tune" / "*.log")))
    kl_shared_final, dsc_final = np.nan, np.nan
    if log_files:
        try:
            with open(log_files[-1]) as f:
                last = json.loads(f.readlines()[-1])
            kl_shared_final = (last.get("x_rna_kl_shared", 0)
                               + last.get("x_atac_kl_shared", 0)) / 2
            dsc_final = last.get("dsc_loss", np.nan)
        except Exception:
            pass

    pd.DataFrame([{"kl_shared": kl_shared_final, "dsc_loss": dsc_final}]
                 ).to_csv(log_path, index=False)

    row = {"config": tag, **cfg,
           "final_kl_shared": kl_shared_final,
           "final_dsc_loss":  dsc_final}
    results.append(row)
    print(f"  kl_shared={kl_shared_final:.4f}  dsc_loss={dsc_final:.4f}")

# ── summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(str(SEARCH_DIR / "summary.csv"), index=False)
print("\n=== Search Summary ===")
print(df.to_string(index=False))

# best config: kl_shared as high as possible AND dsc_loss closest to ln(2)≈0.693
df_valid = df[df["final_kl_shared"] > 0]
if not df_valid.empty:
    df_valid = df_valid.copy()
    df_valid["score"] = (
        df_valid["final_kl_shared"] / df_valid["final_kl_shared"].max()
        + 1 - (df_valid["final_dsc_loss"] - 0.693).abs() / 0.693
    )
    best = df_valid.loc[df_valid["score"].idxmax()]
    print(f"\nBest config: {best['config']}")
    print(best)
