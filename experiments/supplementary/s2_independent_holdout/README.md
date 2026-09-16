# S2：独立外部测试的数据准备

当前阶段只完成来源、元数据与序列筛查，尚未建立最终测试标签或运行预测。

## 已完成的来源审计（2026-09-06）

- 核验 8 个固定输入的 SHA256，包括训练 manifest、训练来源文件、三个较新 DisProt 快照、历史快照及 CAID 无标签哨兵/序列。
- 将实际训练 manifest 与保留 accession 和证据信息的来源文件逐条对齐：1,133 条蛋白，105,036 个正标签、97,626 个负标签、370,759 个 unknown。
- 仅读取较新 DisProt 的元数据和序列；不利用 regions、consensus 或 disorder_content 生成筛选规则。
- 对当前快照中 released 字段处于两个时间区间的条目，排除旧快照中出现的编号、accession/isoform 主体及完全相同的序列；进一步排除训练集、CAID2/3 重叠和候选池内重复。
- 排除顶层元数据显式标记的 obsolete/ambiguous、序列异常和 polyprotein。区域级证据及状态审核仍待完成。
- 使用 MMseqs2 18-8cc5c，identity 严格大于 0.30 且双向覆盖率均不小于 0.80，查询实际训练集和 CAID2/3 无标签序列。原始比对、执行命令、工具哈希与日志均保留。

| 候选时间区间（结束于 2026_06） | released 字段在区间内 | 直接排除和去重后 | 同源筛查后 | 候选序列总残基数 |
| --- | ---: | ---: | ---: | ---: |
| 晚于 2023_06 | 765 | 163 | 128 | 84,334 |
| 晚于 2025_06 | 153 | 143 | 114 | 67,973 |

114 条是 128 条中的子集，不能作为两个相互独立测试集相加。候选数量还会受证据质量、负标签覆盖及条目历史审核影响。

`candidate_cohorts.excluded_records` 记录同源筛查之前的排除数；最终可用序列候选数见 `after_homology_sequences`。单条记录可以同时命中多个排除原因，因此原因计数不能相加当作删除蛋白数。

## 时间含义与待完成工作

训练正标签采用 DisProt 2018_11；结构负标签对应的 PDB 条目限制为 2018-11-30 前发布。但构建过程使用过 DisProt 2026_06 的序列/accession 和 2026 年 SIFTS 映射，因此不能表述为所有数据资源均冻结于 2018 年。

三个较新快照在 2026-08-21 已用于标识符映射。故本实验是回顾性的版本划分候选审计，不能称为锁模后才收集的前瞻性验证。`released` 与 `date` 字段含义不同：例如 DP03208 在当前快照中 released 为 2026_06，而 date 为 2021 年。仅根据这些字段不能认定实验首次发表于 2026 年。

下一阶段应完成：

1. 对候选的条目/区域版本、原始文献、obsolete/ambiguous 状态和直接实验证据进行审核；区分新记录、旧证据的新收录及旧条目修订。
2. 按预先确定的经典 IDR 证据规则构建正标签；通过可靠实验结构与一致序列坐标获取负标签，未知或冲突位置保持 -1。只有无序正标签时不能计算有意义的二分类 ROC-AUC。
3. 汇总正负残基和双类蛋白数量，再固定评价指标、基线来源、模型版本及最终测试清单。不能根据模型分数或效果选择候选。
4. 本次双向高覆盖筛查不保证排除了短结构域同源，仍需局部相似性敏感性分析及候选内部同源簇检查；也不证明候选从未出现于 ESM-2 预训练语料。

## 注释与结构证据审计（2026-09-06）

已对 128 条序列候选读取区域级证据，但尚未读取任何模型预测。新版
DisProt 结构状态包含 319 条 disorder、23 条 order、2 条 molten globule
和 1 条 pre-molten globule 注释。严格限制为直接实验 ECO、有效坐标、
非 obsolete、已记录 curator validation 且无构建体改变后，得到 41 条
蛋白、3,789 个暂定正残基。

预测推断、X-ray/EM 缺失坐标和不明确的组合证据未自动转成正标签。
需要人工检查的逐区域记录见
`outputs/supplementary/s2_independent_holdout/evidence_audit_20260906/annotation_review_queue.tsv`。

候选的 1,031 个 PDB 条目已经通过 PDBe 官方 API 补齐 summary 和
experiment 元数据并固定 SHA256。采用 PDB 发布日期不晚于 2026-06-30、
X-ray 分辨率不差于 3.0 Å、EM 不差于 4.0 Å、NMR 不要求分辨率的规则，
873 个 PDB 条目通过。精确 accession+sequence 映射及等长 SIFTS 坐标过滤后，
临时参考包含 3,789 个正残基、34,505 个结构负残基和 46,040 个 unknown；
29 条蛋白同时具有正负标签。该参考仍不得用于模型选择或正式结果声明。

固定 JSON 中候选结构状态均带 `unpublished: true`，但官网公开条目
DP04433 同时展示了这些已验证注释。因此该字段不能简单解释为“当前未公开”，
仍需结合条目历史和官方说明完成逐区域审核。

## 局部同源敏感性审计（2026-09-06）

在原全局同源排除之后，又以 MMseqs2 18-8cc5c 进行更宽松的局部搜索：
最低 20% identity、双向最低 10% coverage、E-value 不大于 1e-3，并启用
低复杂度屏蔽和组成偏倚校正。21/128 条候选与训练集或 CAID2/3 至少命中
一项局部敏感性规则，候选内部有 23/128 条至少命中一项相似性规则；候选与
训练/CAID2/3 的原严格全局规则命中仍为 0。敏感性命中只进入复核队列，
不会根据模型分数决定去留。

发布及历史说明：[DisProt release notes](https://disprot.org/release-notes)、[条目版本历史示例](https://www.disprot.org/history/DP00609)。

## 复现（在项目根目录运行）

```powershell
& .venv\Scripts\python.exe experiments\supplementary\s2_independent_holdout\test_audit_sources.py
& .venv\Scripts\python.exe -u experiments\supplementary\s2_independent_holdout\audit_sources.py --output-dir outputs\supplementary\s2_independent_holdout\source_audit_recheck --mmseqs tools\mmseqs2-windows-18-8cc5c\mmseqs\bin\mmseqs.exe
& .venv\Scripts\python.exe experiments\supplementary\s2_independent_holdout\test_audit_evidence.py
```

输出目录必须不存在，以保留之前的审计。省略 `--mmseqs` 时只做直接重叠检查，报告会把同源筛查标记为未运行。

本次结果目录：`outputs/supplementary/s2_independent_holdout/source_audit_20260906/`。
主要文件：`s2_source_audit.md`、`s2_source_audit.json`、两个时间区间的 `*_audit.jsonl` 和 `*_sequence_candidates.fasta`、`forbidden_homology_hits.json`、`s2_audit_lock.sha256`。

其中 FASTA 不含标签，仅供下一阶段数据准备；不能直接当作已确认的有序/无序测试参考文件。

新增的主要审计目录为：

- `outputs/supplementary/s2_independent_holdout/evidence_audit_20260906/`
- `outputs/supplementary/s2_independent_holdout/qualification_audit_20260906/`
- `outputs/supplementary/s2_independent_holdout/local_homology_audit_20260906/`

`qualification_audit_20260906/provisional_reference.jsonl` 含暂定标签，明确不是
正式测试集。完成逐区域人工证据审核、同源簇处理和最终清单冻结之前，不得在其上
运行 A10 或任何基线预测。

## 预测盲法预冻结队列（2026-09-06）

在没有读取任何 MUSE-IDR 或基线预测的前提下，将临时参考、外部局部同源标记和
候选内部相似性图合并。固定规则首先排除没有临时正/负标签的候选，再排除命中任一
预设外部局部相似性规则的有标签候选；对剩余候选的内部相似连通分量，仅按已知标签
残基数最多、DisProt ID 最小的顺序保留一个代表，不使用模型分数或测试性能决定去留。

- 128 条输入候选中，112 条至少含一个临时正或负标签。
- 21 条候选命中外部局部相似性规则；其中 3 条同时没有临时标签，因此有标签队列
  实际因外部相似性排除 18 条。
- 内部相似性去冗余进一步排除 7 条，得到 87 条预测盲法预冻结候选。
- 预冻结队列含 56,394 个残基：2,999 个暂定正标签、27,729 个暂定负标签和
  25,666 个 unknown；33 条蛋白含正标签，76 条含负标签，22 条同时含两类标签。
- 33 条含正标签的保留候选必须逐区域人工复核；当前 `ready_for_prediction` 仍为
  `false`，尚未运行 A10 或任何比较模型。

本次预冻结目录为
`outputs/supplementary/s2_independent_holdout/prefreeze_20260906/`。其中
`candidate_decisions.tsv` 记录每条候选的保留/排除原因，`prefreeze_lock.sha256`
固定了选择脚本、测试脚本、临时参考、无标签 FASTA、决策表和报告。该目录仍是
人工注释复核前的中间审计材料，不是最终测试集。

## 暂定正区段人工复核包（2026-09-06）

`outputs/supplementary/s2_independent_holdout/manual_review_packet_20260906/`
汇总了 33 条保留蛋白中的 66 个暂定正证据区段，共涉及 36 篇不同文献。每行包含
区段坐标、ECO 方法、引文、DisProt/文献链接、curator/validator、版本信息和原始
statement；所有内容均在模型预测之前生成并锁定。

`manual_review_form_TEMPLATE.tsv` 是不可原地修改的空白模板。实际审核时应复制为
`manual_review_form_COMPLETED.tsv`，逐行核对实验是否直接支持经典内在无序、整段
坐标是否得到证据支持、实验构建体能否映射到固定参考序列，以及公开注释/历史能否
解释快照中的 `unpublished` 字段。只允许 `accept`、`reject` 或 `uncertain`；任何
未解决问题均记为 `uncertain` 并从最终正标签排除。66 行全部审核、双人复核并通过
机器校验后，才能重建最终参考、固定最终 SHA256 锁并开始模型推理。

## 人工复核批次 01：DP03747（2026-09-06）

首批证据预审覆盖 DP03747 的 3 个暂定正区段，并保存到
`outputs/supplementary/s2_independent_holdout/manual_review_batches/batch01_DP03747/`。
在不读取任何模型分数的前提下，文献和官方补充材料支持暂定接受 CBM20 插入/连接区
66–175 与 218–254；DSP 活动位点区 388–413 的高 HDX 主要反映构象变化和溶剂可及性，
不能单独证明本征无序，因此建议拒绝。该批次只是带来源哈希的审阅建议，仍需具名人工
复核者独立核查并签字，不能直接写入最终参考，也不能据此启动模型预测。

Xingyuan Li 已于 2026-09-06 确认该批次的上述 3 个决定。确认内容、证据哈希和
审阅者信息保存在 `manual_review_batches/batch01_DP03747/confirmed_decisions.json`，
并已合并到新的 `manual_review_progress_20260906_batch01/manual_review_form_IN_PROGRESS.tsv`。
当前 66 个区段中完成 3 个、剩余 63 个；状态仍为 `manual_review_in_progress`，
`ready_for_prediction=false`。

## 数据库注释版最终冻结（2026-09-06）

项目负责人随后决定暂停逐篇文献复核，并采用统一的数据库注释政策：所有正标签均来自
DisProt 2026_06 公共快照中通过既定自动严格条件的实验性结构状态注释；负标签仍来自
质量过滤并经 SIFTS 精确映射的 PDB 实验观察残基。DP03747 的三行人工复核仅作为质量
抽查记录，不应用于标签修改，从而避免只人工修改部分蛋白造成不一致。

最终冻结目录为
`outputs/supplementary/s2_independent_holdout/final_database_curated_20260906/`，包含
87 条蛋白、56,394 个残基，其中正标签 2,999、负标签 27,729、unknown 25,666；
22 条蛋白同时包含正负两类。`s2_sequences.fasta` 不含标签，`s2_reference.jsonl` 与
`s2_labels.tsv.gz` 保存固定参考，`s2_final_lock.sha256` 固定输入政策、筛选产物、冻结
脚本、测试和所有最终输出。全部锁和 31 项 S2 测试通过后，状态变为
`ready_for_prediction=true`。

论文中必须将 S2 表述为“严格过滤的 DisProt 数据库注释版补充时间外验证集”，不能表述
为逐篇原始文献重新整理的人工金标准。CAID2 和 CAID3 仍是主要外部基准。

## A10 盲法推理上传包（2026-09-07）

`release/s2_a10_inference_upload_20260907.tar.gz` 仅包含 87 条序列的无标签 FASTA、
与 `v1.0.0-paper` 哈希一致的 A10 推理源文件，以及预测前后校验脚本；不包含
`s2_reference.jsonl`、`s2_labels.tsv.gz`、人工复核材料或任何 S2 参考答案。包内
清单 17 项全部通过 SHA256 复核，压缩包中共有 18 个文件，标签侧文件数为 0。

AutoDL 端复用已经存在并受 `a10_lock_manifest.json` 保护的
`models/weights/a10_locked`。正式预测前先运行 `--verify-only`，确认 30 个成员
权重哈希全部匹配；正式预测完成后，驱动器检查每个蛋白的残基位置、氨基酸和概率范围，
再生成独立预测锁。只有锁定的预测文件下载回本地之后，才允许读取 S2 标签进行评估。

## A10 本地标签侧评估（2026-09-07）

预测结果包下载后必须先与服务器端 SHA256
`10aa2b49d8aa7243aaef8d88ab1314f59415a9f398feda89315c951e57a4d7e6`
核对。`evaluate_s2_a10.py` 在语义标签访问前依次验证结果包、预测锁和最终参考锁，随后
严格按蛋白、位置和氨基酸对齐预测与标签，并排除 unknown（`-1`）标签。点估计复用
B1/B6 的全精度指标实现，95% 置信区间采用整条蛋白聚类 bootstrap。固定阈值为 0.5；
Fmax 只用于描述，不允许据此重新调整 A10。S2 仍是补充时间外验证，不能代替 CAID2/3，
在相同 S2 参考上完成基线预测之前也不能提出 S2 最优声明。

正式评估使用 2,000 次整蛋白 bootstrap（种子 20260907），处理 87 条蛋白中的
30,728 个已知标签残基。A10 的全精度 ROC-AUC 为 0.977827（95% CI
[0.964305, 0.989485]），AUPRC 为 0.847950（95% CI [0.654202, 0.939162]），
固定阈值 0.5 的 F1 和 MCC 分别为 0.747267 和 0.732873；22 条同时含有两类标签
蛋白的 Macro ROC-AUC 为 0.895952（95% CI [0.828303, 0.953254]）。正式输出位于
`outputs/supplementary/s2_independent_holdout/evaluation_20260907/`，结果锁自身 SHA256
为 `29986c46a1eb2b0f63d0f655805973a7c51c840db46b93d6f019bf0a53d834fd`。

## 两个正式基线的同参考比较（2026-09-07）

在未运行任何 S2 基线预测之前，正式比较范围固定为 PUNCH2-Light Released-13 与
LoRA-DR-Suite ESM2-650M DisProt7。该协议形成于 A10 的 S2 结果已经计算之后，因此
不能称为结果揭盲前预注册；但基线身份继承自此前 B4 复现计划，并在基线接触 S2
之前固定。Paper-8、完整 PUNCH2 和其他文献模型暂不运行，除非后续审稿意见要求。

正式主指标为同一已知残基上的全精度 ROC-AUC，两个 A10-vs-baseline 差值采用整条
蛋白配对 bootstrap（2,000 次，种子 20260907）并进行 Holm 校正；AUPRC 是关键辅助
指标。阈值相关指标仅作描述，因为 PUNCH2-Light 的官方阈值为 0.35，而另外两个
模型通常使用 0.5，不能用不同阈值下的 F1/MCC 建立优越性结论。协议文件为
`s2_baseline_comparison_protocol_20260907.json`，SHA256 为
`a231176522a0bd1bd5a0566c74d9006ab9ea6a65100d58b371b617d3533413e2`。

`run_s2_baseline_predictions.py` 只接受无标签 FASTA，没有 reference/labels 参数；
它固定模型版本、权重哈希、推理精度和长序列窗口策略，并逐残基检查输出对齐。
`build_s2_baseline_inference_upload.py` 生成的 AutoDL 包不含 S2 标签、A10 分数、S2
评估结果或第三方权重。PUNCH2-Light 使用 AutoDL 已有的官方固定提交与 13 个权重，
LoRA 适配器从固定 Hugging Face revision 获取并在加载后核对 SHA256。
