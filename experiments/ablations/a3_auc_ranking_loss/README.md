# A3：A2 + 采样式 AUC 排序损失

A3 保持 A2 的缓存、数据划分、模型结构、优化器和早停规则不变。唯一主要变化是
训练目标从加权 BCE 改为：

```text
weighted_BCE + 0.2 * sampled_pairwise_AUC_loss
```

排序项让已知标签中的无序残基 logit 高于有序残基 logit。每个 protein batch 最多
采样 8192 个正负残基对，避免构造完整笛卡尔积。未知标签 `-1` 和 padding 在两种损失
中都被排除。模型仍然只输出一个经典 IDR 概率。

本地冒烟测试：

```bash
python experiments/ablations/a3_auc_ranking_loss/smoke_test_a3.py
python experiments/ablations/a3_auc_ranking_loss/smoke_test_a3_end_to_end.py
```

4090 开发折：

```bash
python -u experiments/ablations/a3_auc_ranking_loss/train_a3.py \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --cache-dir data/cache/a1_esm2_650m \
  --config experiments/ablations/a3_auc_ranking_loss/config.yaml \
  --fold 4 --seed 17
```

开发折只用于筛选。通过后才运行固定种子 `[17, 29, 43]` 的完整五折。
