# PUNCH2 与 LoRA-DR-Suite 基线复现规范

更新时间：2026-08-23

## 1. 本阶段的目的

本阶段不是直接改进模型，而是先得到可信、可审计的基线。只有基线的数据、划分、输出定义和评价代码与我们的实验一致，后续声称“改进带来 AUC 提升”才成立。

所有模型最终都只输出一个长度为 `L` 的向量：

```text
p[i] = 第 i 个氨基酸残基属于经典固有无序（IDR）的概率，0 <= p[i] <= 1
```

不训练或融合 SoftDis，不设置 NOX/PDB 双头，不使用 CAID2/CAID3 标签进行训练、验证、阈值选择、模型选择或校准。

## 2. 两条复现轨道

### 轨道 A：原始模型复现

目的：验证作者公开模型和论文结果是否能在我们的环境中正常运行。

- 使用作者公开的权重、预处理和推理逻辑。
- 保留作者自己的训练数据和输出含义。
- 单独报告“论文定义版本”和“当前仓库原样版本”。
- 这条轨道不能用于证明某个网络结构在我们的训练数据上更优，因为训练数据并不相同。

### 轨道 B：公平重训

目的：在相同数据、相同同源划分、相同标签掩码和相同评价代码下比较架构能力。

- 所有基线和改进模型使用完全相同的五个同源簇 fold。
- 每个 fold 只用 4 个 fold 训练、1 个 fold 验证。
- 未知残基从损失和指标中排除。
- 记录每折最优验证 ROC-AUC、OOF 概率和测试运行时间。
- 主要指标为全部已知残基合并后的 micro ROC-AUC。
- 辅助指标为 PR-AUC、蛋白级 macro ROC-AUC、MCC、F1、参数量、显存和推理速度。
- 选择模型时只看内部验证结果；CAID2/CAID3 只在模型冻结后运行一次正式测试。

## 3. PUNCH2 原始模型规范

### 3.1 论文定义

PUNCH2 的最终模型由三类逐残基特征分别送入 CNN，再对多个 fold 模型的概率做算术平均：

| 分支 | 输入维度 | fold 数 | 论文最终网络 |
|---|---:|---:|---|
| One-hot | 21 | 3 | CNN_L12_narrow |
| ProtTrans | 1024 | 5 | CNN_L12_narrow |
| MSA-Transformer | 768 | 5 | CNN_L12_narrow |

因此论文中的 PUNCH2 应为 `3 + 5 + 5 = 13` 个模型；PUNCH2-light 应为 `3 + 5 = 8` 个模型。

论文训练数据是 `PDB_missing: clstr100 + 3 × DisProt_FD`。PDB 缺失残基作为无序正例，实验可见残基作为有序负例，并增加完全无序的 DisProt 序列。论文说明训练集中过滤了与 CAID2 Disorder_PDB 序列 identity 大于 30% 的序列。

### 3.2 官方源码确认的 CNN_L12_narrow

当前公开源码中的 12 层卷积通道与卷积核为：

```text
input -> 50(k=3) -> 30(k=3) -> 30(k=3)
      -> 20(k=7) -> 20(k=7)
      -> 20(k=15) -> 20(k=15)
      -> 20(k=7) -> 20(k=7)
      -> 10(k=3) -> 10(k=3) -> 1(k=1) -> sigmoid
```

所有卷积 stride 为 1，并用 padding 保持长度不变。公开参数文件给出的学习率为 `1e-4`、推理 dropout 为 `0`。PUNCH2 阈值为 `0.39`，PUNCH2-light 阈值为 `0.35`；阈值不影响 ROC-AUC，只用于二值标签、MCC 和 F1。

### 3.3 已发现的论文—代码差异

当前 PUNCH2 `main.py` 加载：

```text
3 × One-hot/CNN_L12
5 × ProtTrans/CNN_L3
5 × ProtTrans/CNN_L12
5 × MSA-Transformer/CNN_L11
= 18 个模型
```

当前 PUNCH2-light `main.py` 加载：

```text
3 × One-hot/CNN_L12
5 × ProtTrans/CNN_L3
5 × ProtTrans/CNN_L12
= 13 个模型
```

这与论文文字中的 13/8 模型以及“最终使用 CNN_L12_narrow”的描述不完全一致。正式复现必须同时输出：

1. `PUNCH2-paper`：严格按论文 13 模型定义；
2. `PUNCH2-repo`：严格按当前仓库 18 模型入口运行；
3. `PUNCH2-light-paper`：严格按论文 8 模型定义；
4. `PUNCH2-light-repo`：严格按当前仓库 13 模型入口运行。

在没有核实权重文件名称和每个权重对应的训练架构前，不把其中任何一个称为“复现成功”。

### 3.4 长序列行为

公开入口在序列长度大于 1022 时跳过全部 MSA-Transformer 模型，再对剩余模型重新求平均。这意味着 PUNCH2 对短序列和长序列实际使用不同的集成器。

正式报告至少拆分：

- 全部序列；
- `L <= 1022`；
- `L > 1022`。

并记录每条序列实际参与平均的子模型数，防止将“长序列少一个分支”误认为模型自然性能变化。

## 4. PUNCH2 公平重训规范

公平重训不是复制作者的 PDB_missing 数据，而是在我们的五折数据上复现其核心思想。

### 4.1 主比较版本

为避免 MSA 数据库搜索引入额外数据源和巨大计算量，主公平基线使用：

```text
One-hot -> CNN_L12_narrow -> p_onehot
冻结的 ProtTrans 表征 -> CNN_L12_narrow -> p_plm
最终概率 = 两个分支概率的等权平均
```

每个 fold 各训练一个 One-hot 模型和一个 ProtTrans 模型。这样保留 PUNCH2 的“互补表示 + 深窄 CNN + 概率集成”思想，同时遵守我们的训练集和划分。

### 4.2 补充比较版本

- `PUNCH2-CNN-onehot`：只保留 One-hot 分支，判断 CNN 本身贡献。
- `PUNCH2-CNN-PLM`：只保留 PLM 特征分支，判断表示贡献。
- `PUNCH2-CNN-ensemble`：两个分支等权平均，判断集成贡献。

不在 CAID 上学习集成权重。若以后尝试可学习权重，只能在每个训练 fold 的内部验证数据上拟合，并作为独立消融项报告。

## 5. LoRA-DR-Suite 原始模型规范

LoRA-DR-Suite 是一组分别训练的二分类模型，不是一个 ID/SoftDis 双输出头。我们只使用 intrinsic-disorder 检查点；SoftDis 检查点与本课题标签定义不同，完全排除。

### 5.1 推理输出

推荐 intrinsic 模型 `CQSB/esm2_650M-LoRA-ID` 使用 ESM2-650M 主干和逐残基二分类头。对每个非特殊 token 的两个 logits 做 softmax，取类别 1：

```text
p_idr[i] = softmax(logits[i])[1]
```

必须验证输出长度恰好等于原始氨基酸序列长度，CLS/EOS/PAD 等特殊 token 全部排除。

### 5.2 数据泄漏限制

公开 `esm2_650M-LoRA-ID` 使用 DisProt7、CAID1 和 CAID2 的并集训练。因此：

- 可以把它作为作者“as-released”模型在 CAID3 上外部比较；
- 不能把它在 CAID2 上的分数当成独立泛化结果；
- 不能用它作为我们的公平重训模型；
- CAID2 独立比较应使用只在 DisProt7 上训练的 `esm2_650M-LoRA-DisProt7`，前提是对应公开权重可获得并锁定版本。

### 5.3 论文训练搜索空间

公开论文描述的训练设置为：

- 冻结 PLM 基础权重，在每个多头注意力线性变换中加入 LoRA；
- LoRA 目标为 `{Q,K,V}` 或 `{Q,V}`；
- rank 为 `{8,16,32}`；
- dropout 为 `{0.1,0.2,0.3}`；
- warmup ratio 为 `{0.1,0.2}`；
- weight decay 为 `{0.1,0.01,0.001}`；
- 最大学习率在 `[1e-5, 1e-3]` 上对数均匀搜索；
- AdamW、cosine decay、最多 10 epoch、验证 ROC-AUC patience 2；
- 40 次超参数试验，按验证 ROC-AUC 选择；
- intrinsic 完整数据的 effective batch size 为 16；
- 训练和验证输入最大长度为 1024，论文实现对更长序列采用截断。

原始复现先加载作者权重做确定性推理，不重新做 40 次 HPO；只有在验证作者权重之后才进行公平重训。

## 6. LoRA-DR 公平重训规范

### 6.1 固定架构

```text
ESM2-650M（冻结）
  + attention Q/V LoRA
  + dropout
  + Linear(hidden_size -> 2)
  -> 每残基 softmax(class=1)
```

这一路只产生一个经典 IDR 概率，不引入 SoftDis 模型或标签。

### 6.2 第一轮配置

为控制计算预算，第一轮不是重复 40 次 HPO，而是使用一个预注册配置：

- LoRA target：Q/V；
- rank：16；
- dropout：0.2；
- optimizer：AdamW；
- scheduler：cosine with 0.1 warmup；
- 最大 10 epoch；
- early stopping：验证 micro ROC-AUC，patience 2；
- mixed precision：优先 BF16，硬件不支持时 FP16；
- gradient accumulation：使 effective batch size 达到 16；
- loss：带类别权重的逐残基交叉熵；
- unknown mask：未知残基完全不参与 loss；
- seed：至少 3 个，先用 seed 42 完成管线验收，再补 2 个 seed。

这一配置用于验证管线和取得公平基线，不声称是最优配置。后续小规模搜索只能使用内部 fold，不接触 CAID2/3 标签。

### 6.3 长序列修正

公平重训和我们的最终模型都不直接丢弃第 1024 位后的残基。统一使用重叠窗口：

- 窗口有效残基长度：1022；
- stride：511；
- 每个窗口独立输出 logits；
- 重叠区在 logit 空间按中心权重合并；
- 合并后再做 softmax；
- 第一和最后一个残基必须恰好各得到一个最终概率。

需要做“论文截断”和“重叠窗口”消融，验证提升是否主要来自长序列覆盖。

## 7. 统一验收测试

每个基线在进入五折训练前必须通过以下小样本测试：

1. 单条短序列：输出长度等于输入长度；
2. 一批不同长度序列：padding 不产生预测残基；
3. 1022、1023、1500 长度边界测试；
4. 全 unknown 标签蛋白：loss 不为 NaN，且不进入指标；
5. 固定 seed 重跑：输出误差在规定容差内；
6. OOF 完整性：每个已知残基只预测一次；
7. 泄漏检查：训练索引与 CAID2/3 蛋白 ID、序列哈希、同源簇均无交叉；
8. 评价一致性：同一预测文件用统一 evaluator 计算，禁止采用模型仓库自带的不同实现作为主结果。

## 8. 本阶段完成判据

只有同时满足以下条件，才宣布基线复现阶段完成：

- 第三方源码、Docker 镜像或 Hugging Face 权重锁定到不可变 revision；
- 保存环境版本、命令、输入 FASTA 哈希和原始输出；
- 论文版与仓库版 PUNCH2 差异有实测结果；
- LoRA-DR 输出 token 对齐测试全部通过；
- 小样本推理成功；
- 公平重训五折 OOF 文件完整并通过统一 evaluator；
- CAID2/3 标签仍未被训练或选择代码访问。

## 9. 下一执行顺序

1. 锁定 PUNCH2、PUNCH2-light、LoRA-DR-Suite 和 Hugging Face 权重 revision；
2. 下载到 `tools/third_party/`，保持第三方源码只读；
3. 编写统一 FASTA 输入和 `id, position, probability` 输出适配器；
4. 用 3 条合成短序列和 1 条 1500 aa 序列做形状测试；
5. 运行作者公开权重，生成原始模型 smoke-test 报告；
6. 实现并训练 PUNCH2 公平版本；
7. 实现并训练 LoRA-DR 公平版本；
8. 汇总五折 OOF 后，才进入 MUSE-IDR 改进模型阶段。

## 10. 官方来源

- PUNCH2 论文：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0319208
- PUNCH2 源码：https://github.com/deemeng/punch2
- PUNCH2-light 源码：https://github.com/deemeng/punch2_light
- LoRA-DR-Suite 论文：https://academic.oup.com/bioinformatics/article/41/Supplement_1/i439/8199360
- LoRA-DR-Suite 源码：https://gitlab.lcqb.upmc.fr/lombardi/LoRA-DR-suite
- LoRA-DR-Suite 权重集合：https://huggingface.co/collections/CQSB/lora-dr-suite
- ESM2-650M intrinsic checkpoint：https://huggingface.co/CQSB/esm2_650M-LoRA-ID
