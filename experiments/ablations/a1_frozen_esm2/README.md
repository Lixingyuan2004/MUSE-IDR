# A1：冻结 ESM2-650M + 单输出头

这是 MUSE-IDR 消融实验的第一个深度学习基线。模型只输出每个残基的一个经典 IDR 概率，不包含 SoftDis、NOX 或 PDB 输出头。

## 代码位置

- `model.py`：冻结 ESM2-650M 和单个逐残基线性头。
- `data.py`：五折清单读取、未知标签掩码和重叠窗口。
- `run.py`：训练、early stopping、五折 OOF 预测和统一指标。
- `config.yaml`：实验超参数。

## 输入格式

输入是 JSONL，每行一个蛋白：

```json
{"id":"protein_1","sequence":"MSTN...","labels":[0,0,1,-1],"fold":"A"}
```

- `labels` 长度必须和序列相同；
- `1` 为经典固有无序，`0` 为有序，`-1` 为未知；
- `fold` 使用已经固定的同源聚类五折，不能重新随机划分；
- CAID2/CAID3 数据不得放入该训练清单。

## 运行方式

先只运行一个开发 fold：

```powershell
python experiments/ablations/a1_frozen_esm2/run.py `
  --manifest <五折训练清单.jsonl> `
  --fold E
```

通过形状和显存测试后运行全部五折：

```powershell
python experiments/ablations/a1_frozen_esm2/run.py `
  --manifest <五折训练清单.jsonl>
```

结果写入：

```text
outputs/ablations/a1_frozen_esm2/
  fold_A/best_head.pt
  fold_A/metrics.json
  fold_A/predictions.csv
  ...
  oof_metrics.json
  oof_predictions.csv
```

## 设计约束

- ESM2 主干参数全部冻结并保持 eval 模式；
- 只有 dropout 和一个 `hidden_size -> 1` 线性头参与训练；
- 未知残基不进入损失或指标；
- 超过 1022 aa 的序列按 1022 窗口、511 stride 切分；
- 重叠窗口在 logit 空间中心加权合并；
- 每个已知残基在 OOF 文件中只出现一次；
- 主要选择指标为验证集 residue-level micro ROC-AUC。

## 当前状态

代码和配置已经建立，但尚未宣称运行通过。第一次执行前需要：

1. 将现有 CAID-clean 五折数据转换为上述 JSONL 清单；
2. 锁定 `transformers`、PyTorch 和 ESM2 权重版本；
3. 完成短序列、1022/1023 边界和 1500 aa 长序列 smoke test；
4. 在 RTX 4090/5090 上记录显存与运行时间。
