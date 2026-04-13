# Results Summary

## Yao-2021

- Dataset: `Yao-2021`
- Task: DisentangledSCGLUE RNA+ATAC integration, baseline-aligned hyperparameter search with raw RNA/ATAC counts, PCA + raw SVD preprocessing, and evaluation on `feature_aligned.h5ad`.
- Source: `/data1/users/zhutianci/proj/disentangled-glue/hparam_search_v4_yao_baseline_style_fast`

1. `trial_id=6`
Hyperparameters: `trial_id=6`, `preset=nb_all`, `feature_space=all`, `lsi_method=raw_svd`, `shared_dim=64`, `private_dim=16`, `batch_embed_dim=8`, `beta_shared=0.7500`, `lam_iso=2.0000`, `lam_align=0.0200`, `beta_private_rna=0.1000`, `beta_private_atac=0.5000`, `dropout=0.2000`, `lr=0.0020`
Metrics: `Total=0.6215`, `Bio conservation=0.5753`, `Batch correction=0.5929`, `Modality integration=0.7118`, `isolated_labels=0.5515`, `nmi_ari_cluster_labels_kmeans_nmi=0.5120`, `nmi_ari_cluster_labels_kmeans_ari=0.2904`, `silhouette_label=0.5225`, `clisi_knn=1.0000`, `silhouette_batch_b=0.9109`, `ilisi_knn_b=0.2277`, `kbet_per_label_b=0.2527`, `pcr_comparison_b=0.9803`, `silhouette_batch_m=0.9152`, `ilisi_knn_m=0.4669`, `kbet_per_label_m=0.2305`, `graph_connectivity=0.9507`, `pcr_comparison_m=0.9957`

2. `trial_id=4`
Hyperparameters: `trial_id=4`, `preset=nb_all`, `feature_space=all`, `lsi_method=raw_svd`, `shared_dim=48`, `private_dim=8`, `batch_embed_dim=4`, `beta_shared=0.5000`, `lam_iso=0.5000`, `lam_align=0.0500`, `beta_private_rna=0.1000`, `beta_private_atac=0.2500`, `dropout=0.1000`, `lr=0.0020`
Metrics: `Total=0.6157`, `Bio conservation=0.5695`, `Batch correction=0.5865`, `Modality integration=0.7064`, `isolated_labels=0.5443`, `nmi_ari_cluster_labels_kmeans_nmi=0.5106`, `nmi_ari_cluster_labels_kmeans_ari=0.2800`, `silhouette_label=0.5128`, `clisi_knn=0.9997`, `silhouette_batch_b=0.9072`, `ilisi_knn_b=0.2304`, `kbet_per_label_b=0.2314`, `pcr_comparison_b=0.9772`, `silhouette_batch_m=0.8432`, `ilisi_knn_m=0.5666`, `kbet_per_label_m=0.2072`, `graph_connectivity=0.9179`, `pcr_comparison_m=0.9972`

3. `trial_id=7`
Hyperparameters: `trial_id=7`, `preset=nb_all`, `feature_space=all`, `lsi_method=raw_svd`, `shared_dim=48`, `private_dim=8`, `batch_embed_dim=4`, `beta_shared=0.7500`, `lam_iso=2.0000`, `lam_align=0.0700`, `beta_private_rna=0.0500`, `beta_private_atac=0.5000`, `dropout=0.2000`, `lr=0.0010`
Metrics: `Total=0.6126`, `Bio conservation=0.6101`, `Batch correction=0.5831`, `Modality integration=0.6454`, `isolated_labels=0.5326`, `nmi_ari_cluster_labels_kmeans_nmi=0.5886`, `nmi_ari_cluster_labels_kmeans_ari=0.3999`, `silhouette_label=0.5296`, `clisi_knn=0.9998`, `silhouette_batch_b=0.9096`, `ilisi_knn_b=0.2271`, `kbet_per_label_b=0.2175`, `pcr_comparison_b=0.9781`, `silhouette_batch_m=0.6905`, `ilisi_knn_m=0.5665`, `kbet_per_label_m=0.1672`, `graph_connectivity=0.8047`, `pcr_comparison_m=0.9979`

4. `trial_id=5`
Hyperparameters: `trial_id=5`, `preset=nb_all`, `feature_space=all`, `lsi_method=raw_svd`, `shared_dim=64`, `private_dim=8`, `batch_embed_dim=4`, `beta_shared=1.2500`, `lam_iso=2.0000`, `lam_align=0.0700`, `beta_private_rna=0.1000`, `beta_private_atac=0.5000`, `dropout=0.2000`, `lr=0.0020`
Metrics: `Total=0.5757`, `Bio conservation=0.5571`, `Batch correction=0.5627`, `Modality integration=0.6135`, `isolated_labels=0.5248`, `nmi_ari_cluster_labels_kmeans_nmi=0.4496`, `nmi_ari_cluster_labels_kmeans_ari=0.2870`, `silhouette_label=0.5241`, `clisi_knn=1.0000`, `silhouette_batch_b=0.9203`, `ilisi_knn_b=0.1939`, `kbet_per_label_b=0.1600`, `pcr_comparison_b=0.9766`, `silhouette_batch_m=0.7890`, `ilisi_knn_m=0.3115`, `kbet_per_label_m=0.1390`, `graph_connectivity=0.8317`, `pcr_comparison_m=0.9964`

5. `trial_id=2`
Hyperparameters: `trial_id=2`, `preset=nb_hvg`, `feature_space=hvg`, `lsi_method=raw_svd`, `shared_dim=32`, `private_dim=16`, `batch_embed_dim=8`, `beta_shared=0.5000`, `lam_iso=1.0000`, `lam_align=0.0700`, `beta_private_rna=0.1000`, `beta_private_atac=0.1000`, `dropout=0.2000`, `lr=0.0020`
Metrics: `Total=0.5752`, `Bio conservation=0.6662`, `Batch correction=0.5409`, `Modality integration=0.4881`, `isolated_labels=0.5509`, `nmi_ari_cluster_labels_kmeans_nmi=0.6851`, `nmi_ari_cluster_labels_kmeans_ari=0.5216`, `silhouette_label=0.5738`, `clisi_knn=1.0000`, `silhouette_batch_b=0.9023`, `ilisi_knn_b=0.1916`, `kbet_per_label_b=0.0884`, `pcr_comparison_b=0.9813`, `silhouette_batch_m=0.5845`, `ilisi_knn_m=0.0810`, `kbet_per_label_m=0.0282`, `graph_connectivity=0.7491`, `pcr_comparison_m=0.9978`

## Muto

- Dataset: `Muto`
- Task: DisentangledSCGLUE RNA+ATAC integration random hyperparameter search on the Muto dataset, evaluated on `feature_aligned_trained.h5ad`.
- Source: `/data1/users/zhutianci/proj/disentangled-glue/hparam_search_v3`

1. `trial_id=16`
Hyperparameters: `trial_id=16`, `shared_dim=48`, `private_dim=4`, `batch_embed_dim=8`, `beta_shared=1.5000`, `lam_iso=2.0000`, `lam_align=0.0200`, `beta_private_rna=1.0000`, `beta_private_atac=1.0000`
Metrics: `Total=0.6492`, `Bio conservation=0.7485`, `Batch correction=0.4091`, `Modality integration=0.7567`, `isolated_labels=0.6613`, `nmi_ari_cluster_labels_kmeans_nmi=0.7806`, `nmi_ari_cluster_labels_kmeans_ari=0.6290`, `silhouette_label=0.6717`, `clisi_knn=1.0000`, `silhouette_batch_b=0.8980`, `ilisi_knn_b=0.4221`, `kbet_per_label_b=0.3163`, `pcr_comparison_b=0.0000`, `silhouette_batch_m=0.8299`, `ilisi_knn_m=0.6594`, `kbet_per_label_m=0.3838`, `graph_connectivity=0.9229`, `pcr_comparison_m=0.9877`

2. `trial_id=34`
Hyperparameters: `trial_id=34`, `shared_dim=32`, `private_dim=16`, `batch_embed_dim=8`, `beta_shared=1.2500`, `lam_iso=2.0000`, `lam_align=0.1000`, `beta_private_rna=1.0000`, `beta_private_atac=0.5000`
Metrics: `Total=0.6456`, `Bio conservation=0.7337`, `Batch correction=0.4125`, `Modality integration=0.7612`, `isolated_labels=0.6847`, `nmi_ari_cluster_labels_kmeans_nmi=0.7441`, `nmi_ari_cluster_labels_kmeans_ari=0.5819`, `silhouette_label=0.6578`, `clisi_knn=1.0000`, `silhouette_batch_b=0.9112`, `ilisi_knn_b=0.4114`, `kbet_per_label_b=0.3273`, `pcr_comparison_b=0.0000`, `silhouette_batch_m=0.8700`, `ilisi_knn_m=0.5933`, `kbet_per_label_m=0.4017`, `graph_connectivity=0.9511`, `pcr_comparison_m=0.9900`

3. `trial_id=22`
Hyperparameters: `trial_id=22`, `shared_dim=24`, `private_dim=4`, `batch_embed_dim=4`, `beta_shared=1.5000`, `lam_iso=2.0000`, `lam_align=0.1000`, `beta_private_rna=1.0000`, `beta_private_atac=1.0000`
Metrics: `Total=0.6435`, `Bio conservation=0.7442`, `Batch correction=0.4033`, `Modality integration=0.7495`, `isolated_labels=0.6709`, `nmi_ari_cluster_labels_kmeans_nmi=0.7605`, `nmi_ari_cluster_labels_kmeans_ari=0.6115`, `silhouette_label=0.6780`, `clisi_knn=1.0000`, `silhouette_batch_b=0.9049`, `ilisi_knn_b=0.4094`, `kbet_per_label_b=0.2988`, `pcr_comparison_b=0.0000`, `silhouette_batch_m=0.8543`, `ilisi_knn_m=0.5974`, `kbet_per_label_m=0.3960`, `graph_connectivity=0.9092`, `pcr_comparison_m=0.9905`

4. `trial_id=28`
Hyperparameters: `trial_id=28`, `shared_dim=48`, `private_dim=16`, `batch_embed_dim=4`, `beta_shared=1.0000`, `lam_iso=0.5000`, `lam_align=0.0700`, `beta_private_rna=0.5000`, `beta_private_atac=1.0000`
Metrics: `Total=0.6409`, `Bio conservation=0.7185`, `Batch correction=0.4077`, `Modality integration=0.7707`, `isolated_labels=0.6595`, `nmi_ari_cluster_labels_kmeans_nmi=0.7421`, `nmi_ari_cluster_labels_kmeans_ari=0.5607`, `silhouette_label=0.6301`, `clisi_knn=1.0000`, `silhouette_batch_b=0.9129`, `ilisi_knn_b=0.4043`, `kbet_per_label_b=0.3136`, `pcr_comparison_b=0.0000`, `silhouette_batch_m=0.8787`, `ilisi_knn_m=0.6278`, `kbet_per_label_m=0.4204`, `graph_connectivity=0.9357`, `pcr_comparison_m=0.9910`

5. `trial_id=31`
Hyperparameters: `trial_id=31`, `shared_dim=48`, `private_dim=4`, `batch_embed_dim=4`, `beta_shared=1.0000`, `lam_iso=0.5000`, `lam_align=0.1000`, `beta_private_rna=0.5000`, `beta_private_atac=0.5000`
Metrics: `Total=0.6391`, `Bio conservation=0.7312`, `Batch correction=0.4038`, `Modality integration=0.7515`, `isolated_labels=0.6635`, `nmi_ari_cluster_labels_kmeans_nmi=0.7544`, `nmi_ari_cluster_labels_kmeans_ari=0.5909`, `silhouette_label=0.6471`, `clisi_knn=1.0000`, `silhouette_batch_b=0.9209`, `ilisi_knn_b=0.3924`, `kbet_per_label_b=0.3018`, `pcr_comparison_b=0.0000`, `silhouette_batch_m=0.8792`, `ilisi_knn_m=0.5450`, `kbet_per_label_m=0.3938`, `graph_connectivity=0.9510`, `pcr_comparison_m=0.9885`

## BMMC

- Dataset: `BMMC`
- Task: DisentangledSCGLUE RNA+ATAC+protein integration unified long random search on BMMC, jointly searching `baseline` vs `support` mode and support-specific hyperparameters.
- Source: `/data1/users/zhutianci/proj/disentangled-glue/experiments/BMMC/s11_long_search/20260329_204614`

1. `baseline_t0088_sd48_pd8_bs0.75_li2.0_la0.3_bpr0.25_bpa1.0_bpp1.0`
Hyperparameters: `trial_id=88`, `mode=baseline`, `shared_dim=48`, `private_dim=8`, `beta_shared=0.7500`, `lam_iso=2.0000`, `lam_align=0.3000`, `beta_private_rna=0.2500`, `beta_private_atac=1.0000`, `beta_private_prot=1.0000`, `align_support_k=15`, `align_support_strategy=hard`, `align_support_min_weight=0.1000`
Metrics: `Total=0.6336`, `Bio conservation=0.8013`, `Batch correction=0.5667`, `Modality integration=0.5327`, `ARI_MAP=0.8649`, `ASW_batch=0.8781`, `ASW_celltype=0.6132`, `Graph_conn=0.9440`, `NMI=0.7831`, `Seurat_batch=0.9270`, `Seurat_domain=0.9104`, `iLISI_batch=0.2596`, `iLISI_modality=0.4214`, `kBET_batch=0.2022`, `kBET_modality=0.2662`

2. `baseline_t0075_sd24_pd4_bs1.5_li1.0_la0.3_bpr0.5_bpa0.25_bpp0.5`
Hyperparameters: `trial_id=75`, `mode=baseline`, `shared_dim=24`, `private_dim=4`, `beta_shared=1.5000`, `lam_iso=1.0000`, `lam_align=0.3000`, `beta_private_rna=0.5000`, `beta_private_atac=0.2500`, `beta_private_prot=0.5000`, `align_support_k=10`, `align_support_strategy=hard`, `align_support_min_weight=0.0100`
Metrics: `Total=0.6186`, `Bio conservation=0.7635`, `Batch correction=0.5477`, `Modality integration=0.5446`, `ARI_MAP=0.8194`, `ASW_batch=0.8227`, `ASW_celltype=0.6142`, `Graph_conn=0.9219`, `NMI=0.6983`, `Seurat_batch=0.9233`, `Seurat_domain=0.9096`, `iLISI_batch=0.2697`, `iLISI_modality=0.5062`, `kBET_batch=0.1751`, `kBET_modality=0.2180`

3. `baseline_t0074_sd24_pd4_bs1.25_li1.0_la0.5_bpr1.0_bpa0.5_bpp0.5`
Hyperparameters: `trial_id=74`, `mode=baseline`, `shared_dim=24`, `private_dim=4`, `beta_shared=1.2500`, `lam_iso=1.0000`, `lam_align=0.5000`, `beta_private_rna=1.0000`, `beta_private_atac=0.5000`, `beta_private_prot=0.5000`, `align_support_k=10`, `align_support_strategy=soft`, `align_support_min_weight=0.1000`
Metrics: `Total=0.6171`, `Bio conservation=0.7712`, `Batch correction=0.5602`, `Modality integration=0.5200`, `ARI_MAP=0.8219`, `ASW_batch=0.8523`, `ASW_celltype=0.6056`, `Graph_conn=0.9122`, `NMI=0.7453`, `Seurat_batch=0.9257`, `Seurat_domain=0.9022`, `iLISI_batch=0.2673`, `iLISI_modality=0.4663`, `kBET_batch=0.1954`, `kBET_modality=0.1915`

4. `baseline_t0039_sd48_pd16_bs1.5_li1.0_la0.1_bpr0.5_bpa1.0_bpp1.0`
Hyperparameters: `trial_id=39`, `mode=baseline`, `shared_dim=48`, `private_dim=16`, `beta_shared=1.5000`, `lam_iso=1.0000`, `lam_align=0.1000`, `beta_private_rna=0.5000`, `beta_private_atac=1.0000`, `beta_private_prot=1.0000`, `align_support_k=10`, `align_support_strategy=hard`, `align_support_min_weight=0.0100`
Metrics: `Total=0.6152`, `Bio conservation=0.7844`, `Batch correction=0.5456`, `Modality integration=0.5156`, `ARI_MAP=0.8387`, `ASW_batch=0.8491`, `ASW_celltype=0.6164`, `Graph_conn=0.9261`, `NMI=0.7566`, `Seurat_batch=0.9179`, `Seurat_domain=0.8821`, `iLISI_batch=0.2513`, `iLISI_modality=0.4531`, `kBET_batch=0.1639`, `kBET_modality=0.2117`

5. `baseline_t0055_sd48_pd4_bs0.75_li1.0_la0.3_bpr0.5_bpa1.0_bpp1.0`
Hyperparameters: `trial_id=55`, `mode=baseline`, `shared_dim=48`, `private_dim=4`, `beta_shared=0.7500`, `lam_iso=1.0000`, `lam_align=0.3000`, `beta_private_rna=0.5000`, `beta_private_atac=1.0000`, `beta_private_prot=1.0000`, `align_support_k=10`, `align_support_strategy=hard`, `align_support_min_weight=0.1000`
Metrics: `Total=0.6118`, `Bio conservation=0.7795`, `Batch correction=0.5504`, `Modality integration=0.5055`, `ARI_MAP=0.8514`, `ASW_batch=0.8564`, `ASW_celltype=0.6016`, `Graph_conn=0.9181`, `NMI=0.7469`, `Seurat_batch=0.9160`, `Seurat_domain=0.8703`, `iLISI_batch=0.2542`, `iLISI_modality=0.4271`, `kBET_batch=0.1749`, `kBET_modality=0.2191`
