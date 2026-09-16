# A6：A2局部多尺度 + 门控长程膨胀上下文

A6只检验一个新假设：A2的最大局部卷积核为31，可能不足以识别跨越更长
片段的无序模式。在完全保留A2局部多尺度结构的基础上，A6串联6个轻量级
膨胀深度可分离卷积残差块。

```text
冻结ESM2逐残基表示(1280)
  -> A2局部多尺度模块(kernel 3/7/15/31)
  -> 门控膨胀残差栈(dilation 1/2/4/8/16/32)
  -> 单个逐残基classic IDR logit
```

膨胀栈的感受野为127个残基；连同A2的最大局部分支，最大输入感受野约为
157个残基。每个残差块的门初始约为`sigmoid(-2)=0.119`，使模型从接近
A2的状态开始学习，降低新增长程模块破坏已有局部特征的风险。

模型仍只有一个通用IDR输出头。未知标签`-1`不进入损失或指标，CAID1/2/3
均不用于训练或调参。

## 开发门控

先只运行固定的开发折`fold 4 / seed 17`：

```bash
python experiments/ablations/a6_long_range_dilated_context/train_a6.py \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --cache-dir data/cache/a1_esm2_650m \
  --config experiments/ablations/a6_long_range_dilated_context/config.yaml \
  --fold 4 \
  --seed 17
```

主门槛是A6的fold-4 Micro ROC-AUC必须高于同协议A2的`0.967695179`。
通过后才执行五折、三个随机种子；未通过则停止A6，不用CAID结果决定结构。
