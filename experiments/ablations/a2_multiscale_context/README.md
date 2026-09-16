# A2：冻结ESM2缓存 + 多尺度上下文单输出头

A2与A1使用完全相同的1133条训练蛋白、同源隔离五折、冻结ESM2
revision、标签掩码和OOF评测器。唯一主要变化是把`1280 -> 1`线性头
替换为轻量多尺度局部上下文模块。

## 结构

```text
ESM2逐残基表示(1280)
  -> LayerNorm + 256维投影
  -> kernel 3/7/15/31并行深度可分离卷积
  -> 门控残差融合
  -> 单个逐残基IDR logit
```

模型没有SoftDis、NOX或PDB输出头。未知标签`-1`不进入损失或指标。

## 第一次开发运行

```bash
python experiments/ablations/a2_multiscale_context/train_a2.py \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --cache-dir data/cache/a1_esm2_650m \
  --config experiments/ablations/a2_multiscale_context/config.yaml \
  --fold 4 \
  --seed 17
```

结果写入`outputs/ablations/a2_multiscale_context/seed_17/`。只有开发fold
运行稳定且内部AUC具有竞争力后，才执行五折和三随机种子正式实验。
