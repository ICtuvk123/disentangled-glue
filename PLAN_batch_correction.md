# Plan: Decoder-Conditioned Batch Correction

## 背景

### 当前机制

目前 disentangled SCGLUE 有两种 batch correction 机制：

1. **`du`（modality discriminator）内嵌 batch**：`shared_batches=True` 时，batch one-hot 拼接到
   embedding 输入 `du`，做 batch-aware modality alignment。
2. **`db`（batch discriminator）对抗训练**：独立判别器预测 batch，通过对抗训练让 z 去除 batch
   信息，权重 `lam_batch`。

s15 消融实验结果表明 `db` 不仅没有帮助，反而损害了 Bio conservation 和 Modality integration（
Total: lb=0 0.59 > lb=0.05 0.57 > lb=0.10 0.54 > lb=0.20 0.50），说明对抗式 batch correction
在这个设置下过度约束了 embedding。

此外，disentangled decoder 已有 batch-indexed 参数（`scale_lin[b]`, `bias[b]`），`xbch` 也
已经传入 decoder，但仅做简单的线性偏移（lookup-table 式），无法捕捉 batch 与 z 之间的非线性交互。

### scMRDR 的方式

scMRDR 将 batch one-hot 编码**拼接到 latent z** 上再输入 decoder：

```
[z_shared ‖ z_private ‖ b_one_hot] → Decoder → 重建数据
```

decoder 通过重建损失自然学会"从 z 中分离出 batch 效应"，无需对抗训练，训练更稳定。

---

## 目标

将 decoder conditioning 增强为 scMRDR 式的 z-concat 方式，同时移除不稳定的 `db` 对抗判别器，
使 batch correction 完全依赖 decoder 的重建路径。

---

## 现状分析

### 关键文件和位置

| 文件 | 位置 | 说明 |
|------|------|------|
| `scglue/models/sc.py` | 线 687–890 | Disentangled decoder 定义 |
| `scglue/models/scglue.py` | 线 1523–1540 | `DisentangledSCGLUE` net 构造 |
| `scglue/models/scglue.py` | 线 1895–1899 | decoder 实例化（传入 `n_batches`、`private_dim`） |
| `scglue/models/scglue.py` | 线 1708–1719 | 训练时 decoder 调用（已传入 `xbch[k]`） |

### decoder 当前调用方式（已传入 xbch）

```python
# scglue.py 线 1708-1719
x_nll = {
    k: -net.u2x[k](
        z_shared_samp[k],          # (N, shared_dim)
        z_private_samp[k],         # (N, private_dim)
        vsamp[getattr(net, f"{k}_idx")],
        xbch[k],                   # batch index，已传入
        l[k],
    ).log_prob(x[k]).nanmean()
    for k in net.keys
}
```

---

## 实施方案

### Step 1：给 decoder 增加 batch embedding 层

**文件**：`scglue/models/sc.py`

在所有 `Disentangled*DataDecoder` 的 `__init__` 中，增加一个可学习的 batch embedding：

```python
# 新增参数
self.batch_embed = torch.nn.Embedding(n_batches, batch_embed_dim)
# 同时在 Linear 投影层输入维度加上 batch_embed_dim
self.fc = torch.nn.Linear(shared_dim + private_dim_proj + batch_embed_dim, out_features)
```

`batch_embed_dim` 建议默认 `min(n_batches, 8)`，作为新的超参数。

在 `forward` 中：

```python
def forward(self, z_shared, z_private, v, b, l):
    b_emb = self.batch_embed(b)                          # (N, batch_embed_dim)
    h = torch.cat([z_shared, self.private_proj(z_private), b_emb], dim=1)  # concat
    # 用 h 替换原来只用 z_shared 的部分
    ...
```

保留原有的 `scale_lin[b]`, `bias[b]` 参数（兼容现有 checkpoint），但在 forward 中
额外拼接 batch embedding 提供更强的非线性 batch 信号。

**涉及的 decoder 类**（共 9 个，均在 `_DISENTANGLED_DECODER_MAP` 中）：
- `DisentangledNormalDataDecoder`
- `DisentangledZINDataDecoder`
- `DisentangledZILNDataDecoder`
- `DisentangledNBDataDecoder`
- `DisentangledNBMixtureDataDecoder`
- `DisentangledZINBDataDecoder`
- `DisentangledBetaDataDecoder`
- `DisentangledBetaBinomialDataDecoder`
- `DisentangledBernoulliDataDecoder`

### Step 2：更新 decoder 实例化

**文件**：`scglue/models/scglue.py`，线 1895–1899

```python
u2x[k] = _DISENTANGLED_DECODER_MAP[prob_model](
    len(data_config["features"]),
    private_dim=private_dim,
    n_batches=max(data_config["batches"].size, 1),
    batch_embed_dim=batch_embed_dim,   # 新增参数
)
```

`batch_embed_dim` 从 `DisentangledSCGLUEModel.__init__` 的参数传入（默认 8）。

### Step 3：移除 `db` 对抗判别器

**文件**：`scglue/models/scglue.py`

- `DisentangledSCGLUE.__init__`：移除 `db` 参数（或保留但默认 `None`，向后兼容）
- `DisentangledSCGLUETrainer.compute_losses`：移除 `batch_dsc_loss` 相关计算
- `DisentangledSCGLUEModel.__init__`：移除 `db` 构造，移除 `lam_batch` 参数
- 训练脚本 `s02_glue_rna_atac.py`：移除 `--lam-batch` 参数（或保留但忽略）

### Step 4：（可选）Encoder 也接收 batch 信息

类似 scMRDR 的 `encoder_covariates=True`，让 encoder 在编码时也知道 batch：

```python
# x2u encoder forward 中
h = torch.cat([x, b_one_hot], dim=1)
```

这一步可以作为后续消融，先不做。

---

## 超参数变更

| 参数 | 状态 | 说明 |
|------|------|------|
| `lam_batch` | 移除 | 对抗 batch loss 权重，不再需要 |
| `batch_embed_dim` | 新增 | decoder batch embedding 维度，默认 8 |

---

## 实验设计（改完后）

复用 s15 的设定，做消融对比：

| Run | 说明 |
|-----|------|
| baseline (lb=0) | 当前最好结果，无 db |
| decoder_concat | 新方案：batch embedding 拼接到 decoder |
| decoder_concat + encoder_batch | 进一步：encoder 也接收 batch |

---

## 风险与注意事项

1. **n_batches=1 时**：`batch_embed_dim` 无意义，需要 skip 或固定为零向量
2. **维度变化**：decoder 输入维度增加，与旧 checkpoint 不兼容，需要重新训练
3. **9 个 decoder 类**：需逐一修改，注意不同 decoder 的 forward 逻辑差异（NB vs Normal vs Beta 等）
4. **`private_proj` 输出维度**：各 decoder 中 private 部分的投影可能不一致，需统一处理
