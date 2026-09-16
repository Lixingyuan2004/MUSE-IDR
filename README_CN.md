# MUSE-IDR

[![DOI](https://zenodo.org/badge/1372462628.svg)](https://doi.org/10.5281/zenodo.22784470)

MUSE-IDR 是一个残基级经典内在无序预测模型。它以冻结的 ESM-2 650M
为表征骨干，结合多尺度序列上下文、ESM-2 最后四层可学习融合以及跨架构
锁定集成，为输入序列中的每个残基输出一个无序概率。

本仓库是面向 Bioinformatics 投稿的复现材料。仓库整理不会重新训练、
调参、校准或改变已经锁定的 `A10-locked` 模型。

## 模型组成

- 15 个 A2 成员：使用 ESM-2 第 33 层和门控多尺度上下文预测头；
- 15 个 A8 成员：融合第 30、31、32、33 层后使用同一多尺度预测头；
- 每个家族包含 3 个随机种子和 5 个固定交叉验证折；
- 30 个成员采用固定等权 logit 平均，再经过 sigmoid 得到概率；
- ESM-2 在训练中冻结，CAID2/CAID3 标签不参与训练、选模或集成加权。

## 快速验证

```bash
python -m pip install -r requirements-paper.txt
python scripts/audit_repository_layout.py
python scripts/verify_release.py
python inference/predict.py \
  --locked-root models/weights/a10_locked \
  --device cpu \
  --verify-only
```

## 示例推理

```bash
python inference/predict.py \
  --fasta examples/example.fasta \
  --locked-root models/weights/a10_locked \
  --cache-dir cache/muse_idr \
  --output predictions.tsv.gz \
  --report prediction_report.json \
  --device cuda \
  --precision bf16
```

## 目录入口

- `models/`：模型接口和 30 个锁定权重；
- `training/`：统一训练入口；
- `inference/`：统一推理入口；
- `evaluation/`：统一评估入口；
- `experiments/ablations/`：A1–A10；
- `experiments/analysis/`：B1–B7；
- `experiments/supplementary/`：S1 与 S2；
- `results/`：论文和补充材料所需的精简锁定结果；
- `data/manifests/`：可公开的数据来源、划分与泄漏审计清单。

训练数据和外部参考数据可以存在于本机目录，但默认不由 Git 上传。公开
仓库不会重新分发 CAID 标签、第三方模型权重或第三方逐残基预测。项目自有
代码采用 Apache-2.0 许可证。本次投稿对应的固定版本已归档于 Zenodo，版本
DOI 为 [`10.5281/zenodo.22784471`](https://doi.org/10.5281/zenodo.22784471)；
项目总 DOI [`10.5281/zenodo.22784470`](https://doi.org/10.5281/zenodo.22784470)
始终指向最新归档版本。
