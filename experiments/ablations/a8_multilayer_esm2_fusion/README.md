# A8：冻结ESM2最后四层可学习融合 + A2

A1–A7只使用ESM2第33层（最后一层）。A8检验一个新的表示假设：ESM2较早的
高层可能保留对无序预测有用的局部、结构和组成信息，因此缓存第30、31、32、
33层，并学习四个全局softmax权重。

```text
冻结ESM2 layer 30/31/32/33
  -> 可学习softmax标量融合（4个参数）
  -> 原样A2多尺度模块(kernel 3/7/15/31)
  -> 单个逐残基classic IDR logit
```

初始权重为`[0.1, 0.1, 0.1, 0.7]`，所以模型从偏向最后一层、接近A2的
状态开始，但可以学习增加其他层的贡献。A8总可训练参数仅比A2增加4个。

多层缓存使用与A1完全相同的1022残基窗口、511步长和中心加权合并。缓存输入
必须是无标签的`a1_sequences.jsonl`；每个文件保存`[L,4,1280]`的float16
数组，预计总大小为5,871,831,040字节（约5.87 GB十进制）。生成后必须运行
独立验证脚本，训练器拒绝未验证、哈希改变、层次错误或含标签审计失败的缓存。

CAID1/2/3均不用于训练或调参，模型始终只有一个通用IDR输出头。

## 生成并验证缓存

```bash
python experiments/ablations/a8_multilayer_esm2_fusion/cache_multilayer.py \
  --manifest data/processed/classic_idr_2018/a1_sequences.jsonl \
  --output-dir data/cache/a8_esm2_last4

python experiments/ablations/a8_multilayer_esm2_fusion/verify_multilayer_cache.py \
  --manifest data/processed/classic_idr_2018/a1_sequences.jsonl \
  --cache-dir data/cache/a8_esm2_last4
```

## 固定开发折

```bash
python experiments/ablations/a8_multilayer_esm2_fusion/train_a8.py \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --cache-dir data/cache/a8_esm2_last4 \
  --config experiments/ablations/a8_multilayer_esm2_fusion/config.yaml \
  --fold 4 \
  --seed 17
```

主门槛仍为A2同协议开发折`Micro ROC-AUC > 0.967695179`。通过后才运行
完整五折seed 17；完整OOF超过A2的`0.959449461`后才运行另外两个种子。
