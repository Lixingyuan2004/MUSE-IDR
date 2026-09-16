# 统一残基级评估协议

## 1. 为什么单独建立评估器

LoRA-DR-suite、PUNCH2-light和MUSE-IDR必须把逐残基预测交给同一套代码。模型代码不能各自实现AUC，否则掩码、并列分数或蛋白质汇总方式的差异会造成不可比较的结果。

评估器当前只在人工数据和未来的内部验证集上运行。CAID2/3参考答案继续封存，直到模型、检查点和集成方式全部冻结。

## 2. 输入格式

参考JSONL每行一条蛋白质：

```json
{"protein_id":"P1","sequence":"ACDE","labels":[0,1,-1,1]}
```

预测JSONL只包含同一个通用IDR输出：

```json
{"protein_id":"P1","scores":[0.1,0.9,0.4,0.8]}
```

评估器要求两侧ID集合完全一致，拒绝重复ID、缺失蛋白、额外蛋白、长度不一致、NaN/Infinity和不在`[0,1]`内的分数。`-1`标签被屏蔽，只有`0/1`进入指标。

## 3. 首要指标

首要指标是完整精度的residue micro ROC-AUC：先汇总所有蛋白质的已知标签残基，再计算正残基分数高于负残基分数的概率。相同分数按0.5次正确排序处理。数据只有一个类别时直接失败，不返回误导性的0或NaN。

## 4. 辅助指标

- 梯形积分PR-AUC和stepwise average precision分别报告，不能混称；
- protein macro ROC-AUC只平均同时含正负标签的蛋白质，并报告排除数量；
- F1、MCC和balanced accuracy使用预先声明的阈值，判定规则为`score >= threshold`；
- 阈值指标只用于解释，不参与以ROC-AUC为目标的模型选择。

## 5. 不确定性

正式结果以蛋白质为单位进行百分位bootstrap，而不是随机抽取残基。这样同一蛋白内部强相关的相邻残基不会被错误当成独立样本。默认正式配置为1000次、95%区间和固定随机种子。

## 6. 运行方式

```powershell
$env:PYTHONPATH = "src"
python scripts\evaluate_residue_predictions.py `
  --reference internal_reference.jsonl `
  --predictions model_predictions.jsonl `
  --output metrics.json `
  --threshold 0.5 `
  --bootstrap-replicates 1000
```

内部模型选择保存完整精度结果。最终论文另用官方CAID程序复核并报告其展示精度，但不根据两者差异反向修改模型。
