# S2 数据来源与序列候选审计

审计已完成；候选序列尚不是最终测试集。

训练集核验：1133 条蛋白，202662 个已知标签残基；正标签版本 2018_11，PDB 发布截止 2018-11-30。
训练数据构建使用过 2026_06 的序列/accession 和 2026 年 SIFTS 映射，不能描述为所有资源均冻结于 2018 年。

| 本地快照 | 条目数 |
| --- | ---: |
| 2023_06 | 2901 |
| 2025_06 | 3504 |
| 2026_06 | 3388 |

条目数减少不能解读为负的新增量；候选池通过元数据和序列逐条比较。

| 时间区间（至 2026_06） | 条目 released 字段在区间内 | 去旧条目、训练/CAID 重叠及重复后 | 同源过滤后 |
| --- | ---: | ---: | ---: |
| 晚于 2023_06 | 765 | 163 | 128 |
| 晚于 2025_06 | 153 | 143 | 114 |

时间解释：这些快照在模型锁定前已经下载。本次可支持回顾性版本划分的可行性分析，不能称为锁模后前瞻性验证。released/date 字段不等于首次实验发表日期。

同源规则：identity > 0.30 且双向 coverage >= 0.80；此规则不排除短片段/结构域同源，也不证明未进入 ESM-2 的预训练语料。

本次未提取 DisProt 区域标签、未读取 CAID 参考标签、未生成预测。已知标签计数只用于核验现有训练集。

## 后续必须完成

- Check entry history and migration/obsolete/ambiguous evidence before final eligibility
- Map direct positive evidence without treating unannotated residues as negatives
- Obtain reliable observed-structure negative labels with sequence-coordinate checks
- Define evaluation metrics, baseline cohort and test lock before predictions
- Audit baseline training overlap; frozen ESM2 pretraining novelty is not established
- Long-fragment/domain homology and test-set internal clustering need a separate sensitivity audit

版本与修订依据：[DisProt 发布记录](https://disprot.org/release-notes)；[条目历史示例](https://www.disprot.org/history/DP00609)。
