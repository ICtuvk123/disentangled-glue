# Disentangled SCGLUE 训练说明

本文档说明如何使用当前仓库中的 `DisentangledSCGLUEModel` 训练一个带共享/特异潜变量分解的 GLUE 模型。

该实现的核心改动如下：

- 每个模态的编码器输出两部分潜变量：
  - `z_shared`：跨模态共享表示
  - `z_private`：模态特异表示
- 图编码器只学习与 `z_shared` 对齐的顶点表示 `v`
- 数据重构时：
  - `z_shared` 与图顶点表示 `v` 共同参与重构
  - `z_private` 通过私有投影分支参与模态内重构
- 对抗判别器只使用 `z_shared`
- 分类器只使用 `z_shared`
- KL 项拆分为：
  - `beta_shared * KL(z_shared || prior)`
  - `beta_private * KL(z_private || prior)`


## 1. 代码入口

当前实现的主要入口如下：

- 模型类：`scglue.models.DisentangledSCGLUEModel`
- 训练快捷入口：`scglue.models.fit_SCGLUE(..., model=scglue.models.DisentangledSCGLUEModel, ...)`

对应源码位置：

- `scglue/models/scglue.py`
- `scglue/models/sc.py`


## 2. 训练前准备

### 2.1 配置 AnnData

每个模态都必须先调用 `scglue.models.configure_dataset`。

示例：

```python
import scglue

scglue.models.configure_dataset(
    rna,
    prob_model="NB",
    use_highly_variable=True,
    use_layer="counts",
    use_rep="X_pca",
    use_batch="batch",
    use_cell_type="cell_type"
)

scglue.models.configure_dataset(
    atac,
    prob_model="Bernoulli",
    use_highly_variable=True,
    use_rep="X_lsi",
    use_batch="batch",
    use_cell_type="cell_type"
)
```

注意：

- `prob_model="NB"` 在解耦模型中会自动升级为 `ZINB`
- `shared_dim` 必须满足 `0 < shared_dim < latent_dim`
- 图中的顶点必须覆盖所有模态的特征名


### 2.2 准备 guidance graph

训练前需要一个 `networkx.Graph`，其节点名必须与各模态特征名一致。

常见做法：

- RNA-ATAC：graph 节点为 gene + peak
- 边表示正调控或负调控关系
- GLUE 会使用图编码器将顶点编码到共享潜空间


## 3. 推荐训练方式

推荐直接使用 `fit_SCGLUE`，因为它已经封装了：

- 预训练
- balancing weight 估计
- 微调

示例：

```python
import scglue

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac},
    graph,
    model=scglue.models.DisentangledSCGLUEModel,
    init_kws={
        "latent_dim": 50,
        "shared_dim": 30,
        "h_depth": 2,
        "h_dim": 256,
        "dropout": 0.2,
        "shared_batches": False,
        "random_seed": 0,
    },
    compile_kws={
        "lam_data": 1.0,
        "beta_shared": 4.0,
        "beta_private": 1.0,
        "lam_graph": 0.02,
        "lam_align": 0.05,
        "lam_sup": 0.02,
        "dsc_steps": 1,
        "normalize_u": False,
        "lr": 2e-3,
    },
    fit_kws={
        "directory": "runs/disentangled_scglue",
        "max_epochs": 200,
        "patience": 40,
        "reduce_lr_patience": 20,
    },
    align_support_kws={
        "n_neighbors": 15,
        "strategy": "soft",
        "min_weight": 0.05,
    },
)
```

其中 `align_support_kws` 会在预训练后的共享空间里估计每个细胞的跨模态支持度：

- `unsupported_score`：细胞是否被其它模态支持的分数
- `unsupported_align_weight`：对抗对齐部分使用的软权重
- 如果同时启用 balancing，最终判别器权重为两者乘积后的 `combined_dsc_weight`

推荐先使用 `strategy="soft"`，而不是直接把疑似 unsupported 细胞从 GAN 对齐中完全移除。


## 4. 手动训练方式

如果你想显式控制训练流程，也可以手动分三步：

```python
import scglue

model = scglue.models.DisentangledSCGLUEModel(
    {"rna": rna, "atac": atac},
    vertices=sorted(graph.nodes),
    latent_dim=50,
    shared_dim=30,
    h_depth=2,
    h_dim=256,
    dropout=0.2,
    shared_batches=False,
    random_seed=0,
)

model.compile(
    lam_data=1.0,
    beta_shared=4.0,
    beta_private=1.0,
    lam_graph=0.02,
    lam_align=0.05,
    lam_sup=0.02,
    dsc_steps=1,
    normalize_u=False,
    lr=2e-3,
)

model.fit(
    {"rna": rna, "atac": atac},
    graph,
    max_epochs=200,
    patience=40,
    reduce_lr_patience=20,
    directory="runs/disentangled_scglue_manual",
)
```


## 5. 训练输出与嵌入提取

### 5.1 提取共享/特异 embedding

```python
z_rna_shared, z_rna_private = glue.encode_data(
    "rna", rna, return_private=True
)

z_atac_shared, z_atac_private = glue.encode_data(
    "atac", atac, return_private=True
)
```

返回结果：

- `shared`：用于跨模态对齐、聚类、可视化、GAN 判别
- `private`：保留模态特异信息，不参与跨模态对齐


### 5.2 提取图顶点 embedding

```python
v = glue.encode_graph(graph)
```

这里的 `v` 位于共享潜空间，维度等于 `shared_dim`。


### 5.3 跨模态解码

```python
rna_to_atac = glue.decode_data(
    source_key="rna",
    target_key="atac",
    adata=rna,
    graph=graph,
)
```

当前实现中：

- 跨模态解码只使用源模态的 `z_shared`
- `z_private` 在跨模态预测时被置零
- 这是有意设计，因为 `z_private` 被视为不可迁移的模态专属信息


## 6. 关键超参说明

### 6.1 `latent_dim` 与 `shared_dim`

- `latent_dim = shared_dim + private_dim`
- `shared_dim` 决定跨模态公共空间容量
- `private_dim = latent_dim - shared_dim`

经验建议：

- RNA-ATAC 二模态任务可先试：
  - `latent_dim=50`
  - `shared_dim=30`
- 如果发现模态特异信息丢失，可以适当增大 `private_dim`
- 如果发现跨模态对齐不足，可以适当增大 `shared_dim`


### 6.2 `beta_shared`

控制共享空间的约束强度。

- 更大：共享表示更规整，更利于对齐
- 过大：可能损伤重构能力

建议起点：

- `beta_shared = 2.0 ~ 6.0`


### 6.3 `beta_private`

控制特异空间的约束强度。

- 更大：特异空间更紧凑
- 过小：`z_private` 可能过度吸收信息
- 过大：模态特异信息表达不足

建议起点：

- `beta_private = 0.5 ~ 2.0`


### 6.4 `lam_align`

控制对抗对齐强度。

- 这里只对 `z_shared` 做判别对齐
- 如果模态混合差，可以适当增大
- 如果出现过对齐导致生物学差异被抹平，可以减小

建议起点：

- `lam_align = 0.02 ~ 0.1`


### 6.5 `lam_graph`

控制 guidance graph 对训练的约束强度。

- 更大：特征侧先验更强
- 过大：可能过度依赖图结构

建议起点：

- `lam_graph = 0.01 ~ 0.05`


## 7. 建议的调参顺序

建议按以下顺序调参：

1. 先固定 `latent_dim=50, shared_dim=30`
2. 先调 `beta_shared`
3. 再调 `beta_private`
4. 再调 `lam_align`
5. 最后调 `lam_graph`

如果你的目标是“更强对齐”：

- 增大 `shared_dim`
- 增大 `beta_shared`
- 适当增大 `lam_align`

如果你的目标是“保留更多模态特异信息”：

- 增大 `private_dim`
- 适当减小 `beta_private`
- 避免 `lam_align` 过强


## 8. 常见问题

### 8.1 `shared_dim` 报错

如果出现：

```python
ValueError: `shared_dim` must be greater than 0 and smaller than `latent_dim`!
```

说明你设置了非法维度，需要保证：

```python
0 < shared_dim < latent_dim
```


### 8.2 图节点不覆盖特征

如果图中缺少某个模态特征，会在模型初始化时报错。需要确保：

- 所有 `configure_dataset` 后保留的特征
- 都存在于 `graph.nodes`


### 8.3 想只看共享 embedding 做下游分析

直接使用：

```python
z = glue.encode_data("rna", rna)
```

默认只返回 `z_shared`。


## 9. 推荐实验记录

建议至少记录以下内容：

- `latent_dim`
- `shared_dim`
- `beta_shared`
- `beta_private`
- `lam_align`
- `lam_graph`
- 是否使用 `normalize_u`
- 对齐指标
- 生物学保真指标
- 重构损失

如果你要做消融，推荐最少比较以下几组：

- 原始 SCGLUE
- 解耦模型，`beta_private=0`
- 解耦模型，GAN 只判别 `z_shared`
- 解耦模型，图重构只使用 `z_shared`
- 完整解耦模型


## 10. 一句话总结

当前版本的训练逻辑可以概括为：

- 用 `z_shared` 学跨模态公共结构
- 用 `z_private` 保留模态特异信息
- 用图先验约束共享空间
- 用 GAN 只对齐共享空间

如果你的目标是减少过对齐、同时保留模态特异结构，这个版本就是为这个目的设计的。
