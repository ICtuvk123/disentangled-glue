# Top 5 Hyperparameter Tables

Ranking is by `Total` in descending order.

## `hparam_search_v3`

Dataset: `Muto`  
Source: `/data1/users/zhutianci/proj/disentangled-glue/hparam_search_v3`  
Ranking source: per-trial `metrics.json` (`summary.tsv` in this folder is stale and incomplete)

| Rank | Dataset | Hyperparameter Combo | Bio | Batch | Integration | Total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Muto | `t0018_sd32_pd8_be4_bs1.5_li0.5_la0.05_bpr0.5_bpa0.5_sb0` | 0.7983 | 0.4099 | 0.7803 | 0.6764 |
| 2 | Muto | `t0012_sd24_pd4_be4_bs1.5_li1.0_la0.07_bpr0.5_bpa0.5_sb0` | 0.7392 | 0.4192 | 0.7848 | 0.6569 |
| 3 | Muto | `t0002_sd32_pd8_be8_bs1.5_li0.5_la0.1_bpr1.0_bpa0.5_sb0` | 0.7436 | 0.4205 | 0.7724 | 0.6553 |
| 4 | Muto | `t0027_sd48_pd4_be8_bs1.5_li0.5_la0.07_bpr1.0_bpa0.5_sb0` | 0.7638 | 0.4124 | 0.7514 | 0.6546 |
| 5 | Muto | `t0021_sd24_pd8_be4_bs1.5_li2.0_la0.07_bpr1.0_bpa1.0_sb0` | 0.7373 | 0.4164 | 0.7814 | 0.6543 |

## `hparam_search_v4_yao_baseline_style_fast`

Dataset: `Yao-2021`  
Source: `/data1/users/zhutianci/proj/disentangled-glue/hparam_search_v4_yao_baseline_style_fast`

| Rank | Dataset | Hyperparameter Combo | Bio | Batch | Integration | Total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Yao-2021 | `t6_nb_all_all_raw_svd_sd64_pd16_be8_bs0.75_li2.0_la0.02_bpr0.1_bpa0.5_do0.2_lr0.002` | 0.5753 | 0.5929 | 0.7118 | 0.6215 |
| 2 | Yao-2021 | `t4_nb_all_all_raw_svd_sd48_pd8_be4_bs0.5_li0.5_la0.05_bpr0.1_bpa0.25_do0.1_lr0.002` | 0.5695 | 0.5865 | 0.7064 | 0.6157 |
| 3 | Yao-2021 | `t7_nb_all_all_raw_svd_sd48_pd8_be4_bs0.75_li2.0_la0.07_bpr0.05_bpa0.5_do0.2_lr0.001` | 0.6101 | 0.5831 | 0.6454 | 0.6126 |
| 4 | Yao-2021 | `t5_nb_all_all_raw_svd_sd64_pd8_be4_bs1.25_li2.0_la0.07_bpr0.1_bpa0.5_do0.2_lr0.002` | 0.5571 | 0.5627 | 0.6135 | 0.5757 |
| 5 | Yao-2021 | `t2_nb_hvg_hvg_raw_svd_sd32_pd16_be8_bs0.5_li1.0_la0.07_bpr0.1_bpa0.1_do0.2_lr0.002` | 0.6662 | 0.5409 | 0.4881 | 0.5752 |

## `experiments/BMMC/s11_long_search/20260329_204614`

Dataset: `BMMC`  
Source: `/data1/users/zhutianci/proj/disentangled-glue/experiments/BMMC/s11_long_search/20260329_204614`

| Rank | Dataset | Hyperparameter Combo | Bio | Batch | Integration | Total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | BMMC | `baseline_t0088_sd48_pd8_bs0.75_li2.0_la0.3_bpr0.25_bpa1.0_bpp1.0` | 0.8013 | 0.5667 | 0.5327 | 0.6336 |
| 2 | BMMC | `baseline_t0075_sd24_pd4_bs1.5_li1.0_la0.3_bpr0.5_bpa0.25_bpp0.5` | 0.7635 | 0.5477 | 0.5446 | 0.6186 |
| 3 | BMMC | `baseline_t0074_sd24_pd4_bs1.25_li1.0_la0.5_bpr1.0_bpa0.5_bpp0.5` | 0.7712 | 0.5602 | 0.5200 | 0.6171 |
| 4 | BMMC | `baseline_t0039_sd48_pd16_bs1.5_li1.0_la0.1_bpr0.5_bpa1.0_bpp1.0` | 0.7844 | 0.5456 | 0.5156 | 0.6152 |
| 5 | BMMC | `baseline_t0055_sd48_pd4_bs0.75_li1.0_la0.3_bpr0.5_bpa1.0_bpp1.0` | 0.7795 | 0.5504 | 0.5055 | 0.6118 |
