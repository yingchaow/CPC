# jaccard_cmp_knn_softlabel_multi_negative_center 损失函数与组件共识

本文档对应当前方法配置：

`configs/methods/jaccard_cmp_knn_softlabel_multi_negative_center.yaml`

主要源码位置：

- `train.py`
- `models/encoders.py`
- `losses/composite.py`
- `losses/pairwise.py`
- `losses/center.py`
- `losses/cmp.py`
- `losses/regularization.py`
- `training/selection.py`

## 1. 当前方法的总体结构

模型是双塔跨模态哈希结构：

- 图像输入 `x_i` 经过 image encoder 得到哈希表示 `u_i`
- 文本输入 `x_t` 经过 text encoder 得到哈希表示 `v_i`
- encoder 末端使用 `tanh`，并在当前配置中继续做 L2 normalization
- 当前开启 classification head，因此模型还输出图像/文本分类 logits：`a_i`, `b_i`

记：

- batch size 为 `N`
- hash bits 为 `B`
- 类别数为 `C`
- 图像哈希矩阵 `U in R^{N x B}`
- 文本哈希矩阵 `V in R^{N x B}`
- 多标签矩阵 `Y in {0,1}^{N x C}`
- 图像-文本相似度矩阵 `S = U V^T`
- 对角线正配对相似度 `d_i = S_{ii}`

当前配置的核心开关：

```yaml
pairwise:       enabled, mode=jaccard_contrast, weight=0.9
center:         enabled, update=learnable, dual_center=enabled, weight=0.5
cmp:            enabled, weight=0.1
classification: enabled, weight=1.0
quantization:   enabled, weight=2.5
balance:        disabled
ema_consistency: disabled
small_loss:      disabled
ema_teacher:     disabled
knn_soft_label:  enabled after warmup_epochs=5
```

因此当前训练总损失为：

```math
L =
0.9 L_{pair}
+ 0.5 L_{center}
+ 0.1 L_{cmp}
+ 1.0 L_{cls}
+ 2.5 L_{quant}
```

其中 `L_balance`、`L_ema` 在当前方法中不参与训练。

## 2. Pairwise jaccard_contrast 损失

对应源码：`losses/pairwise.py`

当前配置：

```yaml
mode: jaccard_contrast
similarity: dice
similarity_type: hard_jaccard
margin: 0.10
shift: 0.8
temperature: 1.0
hard_negative.enabled: false
```

虽然参数名为 `hard_jaccard`，但当前 `similarity: dice`，所以监督目标实际是 hard label 的 Dice 相似度：

```math
T_{ij} =
\frac{2 |Y_i \cap Y_j|}{|Y_i| + |Y_j|}
```

图文相似度：

```math
S_{ij} = u_i^T v_j
```

对每个图像查询行和文本查询列，分别构造 margin cost：

```math
C^{t}_{ij} =
\begin{cases}
S_{ij}, & S_{ij} \ge d_i - m \\
S_{ij} - s, & S_{ij} < d_i - m
\end{cases}
```

```math
C^{i}_{ij} =
\begin{cases}
S_{ij}, & S_{ij} \ge d_j - m \\
S_{ij} - s, & S_{ij} < d_j - m
\end{cases}
```

其中：

- `m = 0.10`
- `s = 0.8`
- `d_i = S_{ii}`

负样本权重：

```math
W^{-}_{ij} = 1 - T_{ij}
```

当前未启用 pairwise hard negative，因此没有额外 hard negative 放大项。

图像到文本、文本到图像的 pair 项：

```math
L^{t}_{i} =
\frac{1}{N} \sum_j
\tau \exp\left(\frac{C^{t}_{ij}}{\tau} W^{-}_{ij}\right)
```

```math
L^{i}_{j} =
\frac{1}{N} \sum_i
\tau \exp\left(\frac{C^{i}_{ij}}{\tau} W^{-}_{ij}\right)
```

正相关非对角样本的 separation 项：

```math
L^{sep}_i =
\frac{1}{N} \sum_j
1[T_{ij} > 0, i \ne j] \exp(s - S_{ij})
```

最终 per-sample pairwise 损失：

```math
L_{pair,i}
=
L^t_i + L^i_i + L^{sep}_i - d_i
```

batch 损失：

```math
L_{pair} = \frac{1}{N} \sum_i L_{pair,i}
```

注意：当前 method 虽然启用了 kNN soft label，但由于 `similarity_type: hard_jaccard`，pairwise target 和 pairwise weight 仍然只使用原始 hard label，不直接使用 kNN soft target。

## 3. Prototype Center / Multi-Negative Center 损失

对应源码：`losses/center.py`

当前配置：

```yaml
center.enabled: true
center.update: learnable
center.temperature: 0.1
center.rgce_r: 0.7
center.dual_center.enabled: true
center.dual_center.warmup_epochs: 5
center.dual_center.top_k: 2
center.dual_center.negative_centers: 3
center.dual_center.hard_weight: 0.03
center.dual_center.separation_weight: 0.03
center.dual_center.diversity_weight: 0.01
center.dual_center.hash_quantization_weight: 0.01
```

### 3.1 类原型 softmax 置信度

每个类别有一个可学习正中心 `p_c`。归一化后，图像哈希对所有类别中心的 softmax 概率：

```math
q^u_{ic}
=
\frac{\exp(\bar{u}_i^T \bar{p}_c / T)}
{\sum_k \exp(\bar{u}_i^T \bar{p}_k / T)}
```

文本同理：

```math
q^v_{ic}
=
\frac{\exp(\bar{v}_i^T \bar{p}_c / T)}
{\sum_k \exp(\bar{v}_i^T \bar{p}_k / T)}
```

对多标签样本，取其正类平均置信度：

```math
c^u_i =
\frac{\sum_c Y_{ic} q^u_{ic}}
{\max(1, \sum_c Y_{ic})}
```

```math
c^v_i =
\frac{\sum_c Y_{ic} q^v_{ic}}
{\max(1, \sum_c Y_{ic})}
```

### 3.2 Robust generalized cross entropy

源码中的 `_robust_loss` 为：

```math
R(c)
=
(1-r)\frac{1-c^r}{r}
+ r(1-c)
```

其中 `r = 0.7`。

基础 center loss：

```math
L^{base}_{center,i}
=
\frac{R(c^u_i) + R(c^v_i)}{2}
```

### 3.3 Dual / multi-negative center

当前启用 dual center，并且每个类别有 `M=3` 个 hard negative centers，记为 `n_{c,m}`。

该项在 epoch >= 5 后启用。对一个哈希向量 `h_i`，计算：

```math
A_{ic} = \bar{h}_i^T \bar{p}_c
```

```math
H_{ic} = \max_m \bar{h}_i^T \bar{n}_{c,m}
```

正类平均中心分数：

```math
A^+_i =
\frac{\sum_c Y_{ic} A_{ic}}
{\max(1, \sum_c Y_{ic})}
```

对负类中心构造 push-away：

```math
P_{ic}
=
1[Y_{ic}=0] \cdot
\max(0, A_{ic} - A^+_i + m_c)
```

其中 `m_c = 0.2`。当前 `top_k=2`，因此每个样本只保留最大的 2 个负类 push-away 项。

若 kNN soft label 已产生 soft target，则 center reliability 为：

```math
\rho_i =
\frac{\sum_c Y_{ic} \hat{Y}_{ic}}
{\max(1, \sum_c Y_{ic})}
```

当前 `dual_center.reliability_enabled: true`，因此：

```math
P_{ic} \leftarrow \rho_i P_{ic}
```

hard negative attraction：

```math
L^{hard}_i
=
\frac{\sum_c stopgrad(P_{ic}) (1 - H_{ic})}
{\max(\epsilon, \sum_c stopgrad(P_{ic}))}
```

positive-center repulsion：

```math
L^{repel}_i
=
\frac{\sum_c P_{ic}}
{\max(1, \sum_c 1[Y_{ic}=0])}
```

单模态 dual-center 项：

```math
L^{dual}_i(h)
=
0.03 L^{hard}_i
+ 0.03 L^{repel}_i
```

图像和文本取平均：

```math
L^{dual}_{center,i}
=
\frac{L^{dual}_i(u) + L^{dual}_i(v)}{2}
+ L^{neg-reg}
```

### 3.4 Negative center regularization

当每类有多个负中心时，源码中额外加入两个正则。

负中心多样性：

```math
L^{div}
=
0.01 \cdot
mean_{c,m \ne n}
\left(
\bar{n}_{c,m}^T \bar{n}_{c,n}
\right)^2
```

负中心量化约束：

```math
L^{neg-q}
=
0.01 \cdot
mean_{c,m,b}
\left(
|n_{c,m,b}| - B^{-1/2}
\right)^2
```

```math
L^{neg-reg}=L^{div}+L^{neg-q}
```

最终 center loss：

```math
L_{center}
=
\frac{1}{N}
\sum_i
\left(
L^{base}_{center,i}
+ 1[epoch \ge 5] L^{dual}_{center,i}
\right)
```

## 4. CMP margin 损失

对应源码：`losses/cmp.py`

当前配置：

```yaml
cmp.enabled: true
cmp.margin: 0.3
cmp.weight: 0.1
```

相似度矩阵：

```math
S = U V^T
```

正配对：

```math
d_i = S_{ii}
```

图像到文本方向：

```math
L^{i2t}_{ij}
=
1[i \ne j]\max(0, S_{ij} - d_i + m)
```

文本到图像方向：

```math
L^{t2i}_{ij}
=
1[i \ne j]\max(0, S_{ij} - d_j + m)
```

per-sample：

```math
L_{cmp,i}
=
\frac{1}{2}
\left(
\frac{1}{N-1}\sum_{j \ne i} L^{i2t}_{ij}
+
\frac{1}{N-1}\sum_{j \ne i} L^{t2i}_{ji}
\right)
```

batch 损失：

```math
L_{cmp} = \frac{1}{N}\sum_i L_{cmp,i}
```

## 5. kNN soft label 与分类损失

对应源码：`training/selection.py`、`losses/composite.py`

当前配置：

```yaml
classification.enabled: true
knn_classification_weight.enabled: true
knn_classification_weight.warmup_epochs: 5
knn_classification_weight.k: 20
knn_classification_weight.gamma: 0.5
knn_classification_weight.soft_label_enabled: true
```

### 5.1 kNN soft target

epoch < 5 时：

```math
w_i = 1,\quad \hat{Y}_i = Y_i
```

epoch >= 5 时，先收集全训练集的当前哈希表示。跨模态邻居相似度为：

```math
R_{ij}
=
\frac{
\bar{u}_i^T \bar{u}_j
+
\bar{v}_i^T \bar{v}_j
}{2}
```

取 top-k 邻居，负相似度截断为 0 并归一化：

```math
\alpha_{ij}
=
\frac{\max(0, R_{ij})}
{\sum_{j \in N_k(i)} \max(0, R_{ij}) + \epsilon}
```

邻居标签聚合：

```math
G_i = \sum_{j \in N_k(i)} \alpha_{ij} Y_j
```

一致性权重：

```math
w_i
=
\gamma + (1-\gamma) cosine(Y_i, G_i)
```

其中 `gamma=0.5`，所以 `w_i in [0.5,1]`。

soft target：

```math
\hat{Y}_i
=
w_i Y_i + (1-w_i)G_i
```

### 5.2 分类 BCE

图像分类 logits 为 `a_i`，文本分类 logits 为 `b_i`。

单样本分类损失：

```math
L_{cls,i}
=
\frac{w_i}{2}
\left(
BCEWithLogits(a_i, \hat{Y}_i)
+
BCEWithLogits(b_i, \hat{Y}_i)
\right)
```

batch 损失：

```math
L_{cls} = \frac{1}{N}\sum_i L_{cls,i}
```

这个 soft target 还用于 center reliability：

```math
\rho_i =
\frac{\sum_c Y_{ic}\hat{Y}_{ic}}
{\max(1,\sum_c Y_{ic})}
```

但在当前 `pairwise.similarity_type: hard_jaccard` 下，它不改变 pairwise target。

## 6. Quantization 损失

对应源码：`losses/regularization.py`

当前配置：

```yaml
quantization.enabled: true
quantization.weight: 2.5
```

因为哈希向量经过 L2 normalization，理想二值哈希每一维幅值接近 `B^{-1/2}`。

```math
L_{quant}
=
\frac{1}{2}
\left[
mean(|U| - B^{-1/2})^2
+
mean(|V| - B^{-1/2})^2
\right]
```

## 7. 项目中已有但当前方法未启用的损失/组件

### 7.1 Balance loss

源码：`losses/regularization.py`

```math
L_{balance}
=
\frac{1}{2}
\left[
mean_b(\frac{1}{N}\sum_i U_{ib})^2
+
mean_b(\frac{1}{N}\sum_i V_{ib})^2
\right]
```

作用：鼓励每个 hash bit 在 batch 内均衡使用，避免所有样本塌缩到同一符号。

当前方法未启用。

### 7.2 EMA consistency loss

源码：`losses/regularization.py`、`models/ema.py`

```math
L_{ema}
=
\frac{1}{2}
\left[
MSE(U_s, U_t) + MSE(V_s, V_t)
\right]
```

只对 unselected 样本计算。当前方法未启用 EMA teacher 和 EMA consistency。

### 7.3 Small-loss selection

源码：`training/selection.py`

记每个样本的 selection score 为：

```math
score_i = w_{pair} L_{pair,i} + w_{center} L_{center,i}
```

按 score 从小到大选择 remember rate 对应比例的样本参与监督损失。

当前方法 `small_loss.enabled: false`，所以所有样本均被选中。

### 7.4 Semantic multi-center

源码：`losses/center.py`

项目中存在 semantic multi-center 分支，每类可有多个正中心，并支持正中心 intra regularization 和 label graph pull。当前方法启用的是 dual/multi-negative center，因此 semantic multi-center 未启用，且源码中不允许它和 dual center 同时开启。

### 7.5 Relational KD

源码：`losses/relational.py`

项目里实现了基于 fuzzy Jaccard teacher relation 的 relational KD：

```math
R^T_{ij}=2\cdot fuzzyJaccard(\hat{Y}_i,\hat{Y}_j)-1
```

再用 Huber 或 MSE 约束图文相似度矩阵接近 teacher relation。当前 composite loss 没有接入该项，因此当前方法不使用它。

## 8. 当前方法的组件共识

这个 method 的设计共识可以概括为：

1. 图像和文本必须落到同一个归一化 hash 空间，检索依赖 `u_i^T v_j` 的跨模态相似度。
2. 多标签监督不是简单的正负二值关系；当前 pairwise 用 Dice label similarity 表达样本对之间的语义重叠程度。
3. 噪声标签主要通过 kNN soft label 和 center reliability 缓解，而不是通过 small-loss 删除样本。
4. learnable class prototypes 提供类别级锚点；multi-negative centers 为难负类提供额外吸引/分离结构。
5. CMP margin 直接保护 batch 内正确图文配对，使正配对相似度高于错配样本。
6. Classification head 让 hash 表示保留类别可判别性；kNN soft target 让分类监督随表示空间动态修正。
7. Quantization loss 把连续 hash 表示推向可二值化的幅值，降低训练表示和最终二值 hash 之间的落差。

整体上，当前方法不是单一损失驱动，而是由“样本对排序 + 类中心结构 + 配对 margin + 软标签分类 + 量化约束”共同塑形。

