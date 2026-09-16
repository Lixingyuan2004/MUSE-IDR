# 实验记录规范

## 实验编号

正式实验ID格式：

```text
YYYYMMDD_model_data_split_seed
```

示例：

```text
20260820_esm2lora_disprot_v1_fold0_s17
```

## 每次实验必须保存

```text
outputs/<experiment_id>/
├── config.resolved.yaml
├── environment.txt
├── git_state.txt
├── metrics.json
├── predictions.parquet
├── train.log
└── checkpoints/
```

生成文件不提交Git；用于复现实验的配置、数据manifest和最终汇总提交Git。

## 实验阶段

1. `smoke`：几十条人工或小样本数据，只验证代码能够运行。
2. `baseline`：复现LoRA与PUNCH2-light的内部对照。
3. `ablation`：每次只改变一个主要因素。
4. `candidate`：完整5折、固定随机种子的候选模型。
5. `frozen-final`：配置和检查点规则冻结后的最终模型。
6. `caid2-3-test`：只允许使用`frozen-final`在CAID2/3进行推理和报告。

## 公平比较

- 同一轮对照实验必须使用相同训练/验证划分和相同标签掩码。
- 不同模型必须报告参数量、可训练参数量、训练时间、峰值显存和推理速度。
- AUC比较必须同时给出折间均值、标准差和逐折结果。
- 候选模型差异很小时，使用同一批序列上的配对bootstrap置信区间。
