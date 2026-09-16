# A7：A2局部多尺度 + 门控双向序列上下文

A6表明，把固定卷积感受野从31扩大到157没有提高开发折Micro ROC-AUC。
A7因此检验不同的长程假设：不预先规定长程窗口，而是在A2局部特征之后
使用单层双向GRU，自适应地汇总每条蛋白前后两个方向的序列信息。

```text
冻结ESM2逐残基表示(1280)
  -> A2局部多尺度模块(kernel 3/7/15/31)
  -> 单层BiGRU(每个方向128维)
  -> 小初值门控残差融合
  -> 单个逐残基classic IDR logit
```

模型使用`pack_padded_sequence`，所以双向GRU只读取每条蛋白的真实残基，
不会把batch padding当作序列。序列门初始值约为`sigmoid(-2)=0.119`，
模型从接近A2的状态开始学习，并可以对不同残基选择不同程度的全序列信息。

模型仍只有一个通用IDR输出头；未知标签`-1`不进入损失或指标；
CAID1/2/3均不用于训练或调参。

## 开发门控

```bash
python experiments/ablations/a7_bidirectional_sequence_context/train_a7.py \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --cache-dir data/cache/a1_esm2_650m \
  --config experiments/ablations/a7_bidirectional_sequence_context/config.yaml \
  --fold 4 \
  --seed 17
```

主门槛仍为A2同协议开发折的`Micro ROC-AUC > 0.967695179`。只有通过
该门槛才运行完整五折、三个随机种子；否则把A7作为阴性消融停止扩大。
