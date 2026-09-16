# A1缓存训练入口

正式缓存训练使用`train_cached_a1.py`和`cached_config.yaml`。它读取已经
独立验证的冻结ESM2-650M逐残基表征，训练期间不会再次运行650M主干。

第一次只运行预先指定的第五折（编号4）和种子17：

```bash
python experiments/ablations/a1_frozen_esm2/train_cached_a1.py \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --cache-dir data/cache/a1_esm2_650m \
  --config experiments/ablations/a1_frozen_esm2/cached_config.yaml \
  --fold 4 \
  --seed 17
```

确认输出完整后，去掉`--fold 4`运行完整五折。正式重复种子固定为
17、29、43，不允许根据CAID2/3测试结果挑选种子。

`train_a1.py`和`run.py`保留为在线运行ESM主干的参考实现，不作为当前
缓存实验入口。缓存模型只有一个`1280 -> 1`经典IDR输出头。

输出按种子和fold隔离：

```text
outputs/ablations/a1_frozen_esm2/
  seed_17/
    fold_4/best_head.pt
    fold_4/metrics.json
    fold_4/predictions.csv
    oof_metrics.json
    oof_predictions.csv
```
