"""
Evaluate a trained SCGLUE run with the full scib-metrics suite (Benchmarker2).

Usage
-----
    # Evaluate best run (default: PCR disabled for tri-modal BMMC)
    python s06_eval.py --run-dir s06_sweep/run_023 --output-dir s06_eval

    # Enable PCR with a proper shared pre-integration baseline
    python s06_eval.py --run-dir s06_sweep/run_023 --feature-aligned s01_preprocessing/feature_aligned.h5ad --enable-pcr --output-dir s06_eval
"""

import argparse
import os
import re
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

Kwargs = dict[str, Any]
MetricType = bool | Kwargs

_LABELS = "labels"
_BATCH = "batch"
_MODALITY = "modality"
_X_PRE = "X_pre"
_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"
_METRIC_NAME = "Metric Name"

metric_name_cleaner2 = {
    "silhouette_label": "Silhouette label",
    "silhouette_batch_b": "Silhouette batch",
    "silhouette_batch_m": "Silhouette modality",
    "isolated_labels": "Isolated labels",
    "nmi_ari_cluster_labels_leiden_nmi": "Leiden NMI",
    "nmi_ari_cluster_labels_leiden_ari": "Leiden ARI",
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "clisi_knn": "cLISI",
    "ilisi_knn_b": "iLISI",
    "ilisi_knn_m": "iLISI",
    "kbet_per_label_b": "KBET",
    "kbet_per_label_m": "KBET",
    "graph_connectivity": "Graph connectivity",
    "pcr_comparison_b": "PCR comparison",
    "pcr_comparison_m": "PCR comparison",
}

# ──────────────────────────────────────────────────────────────────────
# Metric specifications
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BioConservation2:
    isolated_labels: MetricType = True
    nmi_ari_cluster_labels_leiden: MetricType = False
    nmi_ari_cluster_labels_kmeans: MetricType = True
    silhouette_label: MetricType = True
    clisi_knn: MetricType = True


@dataclass(frozen=True)
class BatchCorrection2:
    silhouette_batch_b: MetricType = True
    ilisi_knn_b: MetricType = True
    kbet_per_label_b: MetricType = True
    pcr_comparison_b: MetricType = False   # requires X_pre; enabled via --preprocessed-dir


@dataclass(frozen=True)
class ModalityIntegration2:
    silhouette_batch_m: MetricType = True
    ilisi_knn_m: MetricType = True
    kbet_per_label_m: MetricType = True
    graph_connectivity: MetricType = True
    pcr_comparison_m: MetricType = False   # requires X_pre; enabled via --preprocessed-dir


# ──────────────────────────────────────────────────────────────────────
# AnnData API dispatching
# ──────────────────────────────────────────────────────────────────────

class MetricAnnDataAPI2(Enum):
    isolated_labels              = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_MODALITY])
    nmi_ari_cluster_labels_leiden = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    nmi_ari_cluster_labels_kmeans = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    silhouette_label             = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    clisi_knn                    = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_LABELS])
    silhouette_batch_b           = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    pcr_comparison_b             = lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_BATCH], categorical=True)
    ilisi_knn_b                  = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_BATCH])
    kbet_per_label_b             = lambda ad, fn: fn(ad.uns["50_neighbor_res"], ad.obs[_BATCH], ad.obs[_LABELS])
    graph_connectivity           = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    silhouette_batch_m           = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_MODALITY])
    pcr_comparison_m             = lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_MODALITY], categorical=True)
    ilisi_knn_m                  = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_MODALITY])
    kbet_per_label_m             = lambda ad, fn: fn(ad.uns["50_neighbor_res"], ad.obs[_MODALITY], ad.obs[_LABELS])


# ──────────────────────────────────────────────────────────────────────
# Benchmarker2
# ──────────────────────────────────────────────────────────────────────

class Benchmarker2:
    def __init__(
        self,
        adata: AnnData,
        batch_key: str,
        label_key: str,
        modality_key: str,
        embedding_obsm_keys: list[str],
        bio_conservation_metrics: BioConservation2 | None,
        batch_correction_metrics: BatchCorrection2 | None,
        modality_integration_metrics: ModalityIntegration2 | None,
        pre_integrated_embedding_obsm_key: str | None = None,
        n_jobs: int = 1,
        progress_bar: bool = True,
    ):
        self._adata = adata
        self._embedding_obsm_keys = embedding_obsm_keys
        self._pre_integrated_embedding_obsm_key = pre_integrated_embedding_obsm_key
        self._bio_conservation_metrics = bio_conservation_metrics
        self._batch_correction_metrics = batch_correction_metrics
        self._modality_integration_metrics = modality_integration_metrics
        self._results = pd.DataFrame(columns=list(self._embedding_obsm_keys) + [_METRIC_TYPE])
        self._emb_adatas = {}
        self._neighbor_values = (15, 50, 90)
        self._prepared = False
        self._benchmarked = False
        self._batch_key = batch_key
        self._modality_key = modality_key
        self._label_key = label_key
        self._n_jobs = n_jobs
        self._progress_bar = progress_bar
        if self._bio_conservation_metrics is None and self._batch_correction_metrics is None:
            raise ValueError("Either batch or bio metrics must be defined.")
        self._metric_collection_dict = {}
        if self._bio_conservation_metrics is not None:
            self._metric_collection_dict["Bio conservation"] = self._bio_conservation_metrics
        if self._batch_correction_metrics is not None:
            self._metric_collection_dict["Batch correction"] = self._batch_correction_metrics
        if self._modality_integration_metrics is not None:
            self._metric_collection_dict["Modality integration"] = self._modality_integration_metrics

    def prepare(self, neighbor_computer: Callable[[np.ndarray, int], NeighborsResults] | None = None) -> None:
        if self._pre_integrated_embedding_obsm_key is None:
            # Some BMMC combined outputs only store latent embeddings in `obsm`
            # and have no feature matrix to run PCA on.
            if self._adata.n_vars >= 2 and self._adata.X is not None:
                n_comps = min(50, self._adata.n_obs, self._adata.n_vars) - 1
                if n_comps >= 1:
                    sc.tl.pca(self._adata, n_comps=n_comps, use_highly_variable=False)
                    self._pre_integrated_embedding_obsm_key = "X_pca"
        for emb_key in self._embedding_obsm_keys:
            self._emb_adatas[emb_key] = AnnData(self._adata.obsm[emb_key], obs=self._adata.obs)
            self._emb_adatas[emb_key].obs[_BATCH] = np.asarray(self._adata.obs[self._batch_key].values)
            self._emb_adatas[emb_key].obs[_MODALITY] = np.asarray(self._adata.obs[self._modality_key].values)
            self._emb_adatas[emb_key].obs[_LABELS] = np.asarray(self._adata.obs[self._label_key].values)
            if self._pre_integrated_embedding_obsm_key in self._adata.obsm:
                self._emb_adatas[emb_key].obsm[_X_PRE] = self._adata.obsm[self._pre_integrated_embedding_obsm_key]
        progress = self._emb_adatas.values()
        if self._progress_bar:
            progress = tqdm(progress, desc="Computing neighbors")
        for ad in progress:
            if neighbor_computer is not None:
                neigh_result = neighbor_computer(ad.X, max(self._neighbor_values))
            else:
                neigh_result = pynndescent(
                    ad.X, n_neighbors=max(self._neighbor_values), random_state=0, n_jobs=self._n_jobs
                )
            for n in self._neighbor_values:
                ad.uns[f"{n}_neighbor_res"] = neigh_result.subset_neighbors(n=n)
        self._prepared = True

    def benchmark(self) -> None:
        if self._benchmarked:
            warnings.warn("Benchmark already run; overwriting previous results.", UserWarning)
        if not self._prepared:
            self.prepare()
        num_metrics = sum(
            sum(v is not False for v in asdict(mc).values())
            for mc in self._metric_collection_dict.values()
        )
        progress_embs = self._emb_adatas.items()
        if self._progress_bar:
            progress_embs = tqdm(self._emb_adatas.items(), desc="Embeddings", position=0, colour="green")
        for emb_key, ad in progress_embs:
            pbar = tqdm(total=num_metrics, desc="Metrics", position=1, leave=False, colour="blue") if self._progress_bar else None
            for metric_type, metric_collection in self._metric_collection_dict.items():
                for metric_name, use_metric_or_kwargs in asdict(metric_collection).items():
                    if use_metric_or_kwargs:
                        if pbar is not None:
                            pbar.set_postfix_str(f"{metric_type}: {metric_name}")
                        metric_fn = getattr(scib_metrics, re.sub(r"(_b|_m)$", "", metric_name))
                        if isinstance(use_metric_or_kwargs, dict):
                            metric_fn = partial(metric_fn, **use_metric_or_kwargs)
                        metric_value = getattr(MetricAnnDataAPI2, metric_name)(ad, metric_fn)
                        if isinstance(metric_value, dict):
                            for k, v in metric_value.items():
                                self._results.loc[f"{metric_type}_{metric_name}_{k}", emb_key] = v
                                self._results.loc[f"{metric_type}_{metric_name}_{k}", _METRIC_TYPE] = metric_type
                                self._results.loc[f"{metric_type}_{metric_name}_{k}", _METRIC_NAME] = f"{metric_name}_{k}"
                        else:
                            self._results.loc[f"{metric_type}_{metric_name}", emb_key] = metric_value
                            self._results.loc[f"{metric_type}_{metric_name}", _METRIC_TYPE] = metric_type
                            self._results.loc[f"{metric_type}_{metric_name}", _METRIC_NAME] = metric_name
                        if pbar is not None:
                            pbar.update(1)
        self._benchmarked = True

    def get_results(self, min_max_scale: bool = True) -> pd.DataFrame:
        df = self._results.transpose()
        df.index.name = "Embedding"
        df = df.loc[~df.index.isin([_METRIC_TYPE, _METRIC_NAME])]
        values = MinMaxScaler().fit_transform(df) if min_max_scale else df.to_numpy()
        df = pd.DataFrame(values, columns=self._results[_METRIC_NAME].values, index=df.index)
        df = df.transpose()
        df[_METRIC_TYPE] = self._results[_METRIC_TYPE].values
        per_class_score = df.groupby(_METRIC_TYPE).mean().transpose()
        if (self._modality_integration_metrics is not None
                and self._batch_correction_metrics is not None
                and self._bio_conservation_metrics is not None):
            per_class_score["Total"] = (
                0.3 * per_class_score["Batch correction"]
                + 0.3 * per_class_score["Modality integration"]
                + 0.4 * per_class_score["Bio conservation"]
            )
        elif (self._modality_integration_metrics is not None
              and self._bio_conservation_metrics is not None
              and self._batch_correction_metrics is None):
            per_class_score["Total"] = (
                0.4 * per_class_score["Modality integration"]
                + 0.6 * per_class_score["Bio conservation"]
            )
        df[_METRIC_NAME] = self._results[_METRIC_NAME].values
        df = pd.concat([df.transpose(), per_class_score], axis=1)
        df.loc[_METRIC_TYPE, per_class_score.columns] = _AGGREGATE_SCORE
        df.loc[_METRIC_NAME, per_class_score.columns] = per_class_score.columns
        return df

    def plot_results_table(self, tag: str, min_max_scale: bool = True,
                           show: bool = True, save_dir: str | None = None) -> Table:
        num_embeds = len(self._embedding_obsm_keys)
        cmap_fn = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)
        df = self.get_results(min_max_scale=min_max_scale)
        plot_df = df.drop([_METRIC_TYPE, _METRIC_NAME], axis=0)
        if self._batch_correction_metrics is not None and self._bio_conservation_metrics is not None:
            sort_col = "Total"
        elif self._modality_integration_metrics is not None:
            sort_col = "Modality integration"
        elif self._batch_correction_metrics is not None:
            sort_col = "Batch correction"
        else:
            sort_col = "Bio conservation"
        plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
        plot_df["Method"] = plot_df.index
        score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
        other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
        column_definitions = [
            ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
        ]
        column_definitions += [
            ColumnDefinition(
                col,
                title=metric_name_cleaner2[df.loc[_METRIC_NAME, col]].replace(" ", "\n", 1),
                width=1,
                textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.25}},
                cmap=cmap_fn(plot_df[col]),
                group=df.loc[_METRIC_TYPE, col],
                formatter="{:.2f}",
            )
            for i, col in enumerate(other_cols)
        ]
        column_definitions += [
            ColumnDefinition(
                col,
                width=1,
                title=df.loc[_METRIC_NAME, col].replace(" ", "\n", 1),
                plot_fn=bar,
                plot_kw={
                    "cmap": mpl.cm.YlGnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                },
                group=df.loc[_METRIC_TYPE, col],
                border="left" if i == 0 else None,
            )
            for i, col in enumerate(score_cols)
        ]
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
            tab = Table(
                plot_df,
                cell_kw={"linewidth": 0, "edgecolor": "k"},
                column_definitions=column_definitions,
                ax=ax,
                row_dividers=True,
                footer_divider=True,
                textprops={"fontsize": 10, "ha": "center"},
                row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
                column_border_kw={"linewidth": 1, "linestyle": "-"},
                index_col="Method",
            ).autoset_fontcolors(colnames=plot_df.columns)
        if show:
            plt.show()
        if save_dir is not None:
            fig.savefig(
                os.path.join(save_dir, tag + ".pdf"),
                facecolor=ax.get_facecolor(),
                dpi=300,
                format="pdf",
                bbox_inches="tight",
            )
        return tab


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="Path to a sweep run directory containing combined_glue.h5ad")
    p.add_argument("--output-dir", default="s06_eval",
                   help="Directory to save CSV results and PDF plots")
    p.add_argument("--tag", default=None,
                   help="Filename tag for output files (defaults to run dir name)")
    p.add_argument("--cell-type-key", default="celltype",
                   help="obs column for cell type labels")
    p.add_argument("--batch-key", default="batch",
                   help="obs column for batch labels")
    p.add_argument("--domain-key", default="domain",
                   help="obs column for modality labels (rna/atac/prot)")
    p.add_argument("--feature-aligned", default=None,
                   help="Path to feature_aligned(.h5ad) from s01_preprocessing. "
                        "Required for a valid PCR baseline on tri-modal BMMC.")
    p.add_argument("--preprocessed-dir", default=None,
                   help="Deprecated fallback path to preprocessed h5ads. "
                        "Kept for backward compatibility; use --feature-aligned for PCR.")
    p.add_argument("--enable-pcr", action="store_true",
                   help="Enable PCR comparison metrics. For tri-modal BMMC this should "
                        "be used together with --feature-aligned.")
    p.add_argument("--n-jobs", type=int, default=8,
                   help="Parallel jobs for neighbor computation")
    p.add_argument("--no-show", action="store_true",
                   help="Do not display plots interactively")
    return p.parse_args()


def canonicalize_modality(value: Any) -> str:
    key = str(value).strip().lower()
    mapping = {
        "0": "rna",
        "1": "atac",
        "2": "prot",
        "gex": "rna",
        "rna": "rna",
        "atac": "atac",
        "adt": "prot",
        "protein": "prot",
        "prot": "prot",
    }
    return mapping.get(key, key)


def load_feature_aligned(
    path: Path, obs_names: pd.Index, domain_key: str = "modality"
) -> AnnData:
    """Load and reorder feature_aligned.h5ad to match combined_glue obs_names."""
    aligned = sc.read(path)
    if domain_key in aligned.obs:
        modality_col = domain_key
    elif "modality" in aligned.obs:
        modality_col = "modality"
    else:
        raise ValueError(
            "feature_aligned.h5ad must contain a modality/domain column. "
            f"Tried `{domain_key}` and `modality`."
        )

    prefixed_index = pd.Index(
        [
            f"{canonicalize_modality(mod)}:{obs_name}"
            for obs_name, mod in zip(aligned.obs_names, aligned.obs[modality_col])
        ]
    )
    aligned.obs_names = prefixed_index
    if not aligned.obs_names.is_unique:
        raise ValueError("feature_aligned.h5ad produces non-unique prefixed obs_names")

    missing = obs_names.difference(aligned.obs_names)
    if not missing.empty:
        raise ValueError(
            "feature_aligned.h5ad is missing cells required by combined_glue.h5ad; "
            f"first missing entries: {missing[:5].tolist()}"
        )
    return aligned[obs_names].copy()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_dir.name

    print(f"Loading {run_dir / 'combined_glue.h5ad'} ...")
    adata = sc.read(run_dir / "combined_glue.h5ad")
    adata.obsm["X_embed"] = adata.obsm["X_glue"]

    enable_pcr = args.enable_pcr and args.feature_aligned is not None
    if args.enable_pcr and args.feature_aligned is None:
        raise ValueError("`--enable-pcr` now requires `--feature-aligned`!")
    if args.preprocessed_dir is not None:
        print("Ignoring deprecated `--preprocessed-dir`; use `--feature-aligned` for PCR.")
    if enable_pcr:
        print("Loading shared feature-aligned baseline for PCR ...")
        aligned = load_feature_aligned(Path(args.feature_aligned), adata.obs_names, args.domain_key)
        var = aligned.var.copy()
        if "highly_variable" not in var:
            var["highly_variable"] = True
        adata = sc.AnnData(
            X=aligned.X.copy(),
            obs=adata.obs.copy(),
            var=var,
            obsm=dict(adata.obsm),
        )

    bm = Benchmarker2(
        adata,
        batch_key=args.batch_key,
        label_key=args.cell_type_key,
        modality_key=args.domain_key,
        embedding_obsm_keys=["X_embed"],
        bio_conservation_metrics=BioConservation2(),
        batch_correction_metrics=BatchCorrection2(pcr_comparison_b=enable_pcr),
        modality_integration_metrics=ModalityIntegration2(pcr_comparison_m=enable_pcr),
        pre_integrated_embedding_obsm_key=None,
        n_jobs=args.n_jobs,
        progress_bar=True,
    )
    bm.benchmark()

    print("\n=== Raw results ===")
    print(bm._results.transpose().to_string())

    df_raw = bm.get_results(min_max_scale=False)
    df_raw.to_csv(output_dir / f"{tag}_unscaled.csv")
    df_scaled = bm.get_results(min_max_scale=True)
    df_scaled.to_csv(output_dir / f"{tag}_scaled.csv")

    print(f"\nSaved CSVs to {output_dir}/")

    bm.plot_results_table(tag=tag, min_max_scale=False,
                          show=not args.no_show, save_dir=str(output_dir))
    bm.plot_results_table(tag=f"{tag}_scaled", min_max_scale=True,
                          show=not args.no_show, save_dir=str(output_dir))
    print(f"Saved PDFs to {output_dir}/")


if __name__ == "__main__":
    main()
