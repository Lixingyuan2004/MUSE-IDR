# 无泄漏数据协议

## 1. 目标

构建两个彼此隔离的数据域：

```text
训练域：CAID1历史候选 + DisProt实验无序正样本 + PDB实验结构负样本
测试域：CAID2/3参考文件
```

训练代码只接收处理后的训练文件和不含标签的CAID2/3哨兵manifest。CAID2/3真实标签保存在`data/caid_holdout`，该目录不提交Git，也不能作为训练、验证、早停或配置选择输入。CAID1若进入训练，必须从有版本和证据记录的历史源构建到训练数据域，不能让训练器遍历整个holdout目录。

## 2. CAID哨兵

哨兵manifest只包含：

- CAID轮次和参考类型；
- DisProt ID及可获得的UniProt accession；
- 序列长度；
- 规范化序列SHA-256；
- 原始文件URL、字节数和SHA-256。

正式训练过滤使用`caid23_test_sentinel.json`，只保护CAID2/3；三轮总清单继续保存用于来源审计。manifest故意不包含CAID的`0/1/-`标签。这样训练数据构建器可以排除测试目标，但不能读取测试答案。

### CAID1

原始CAID1数据目录已经迁移。根据CAID官方Git历史中的生成说明，目标集采用DisProt `2018_11`结构状态条目，排除`2016_10`已有条目和名称中含`polyprotein`的记录。当前官方历史API快照重建出652个训练候选目标。它们必须统一转换为经典IDR的`1/0/-1`标签，并先对CAID2/3完成三层排除。

### CAID2和CAID3

从CAID官方结果页下载Disorder-NOX和Disorder-PDB参考。两种参考只是同一个通用预测输出上的不同标签掩码；隔离清单使用两者目标并集。

## 3. 训练数据构建顺序

### 3.1 冻结原始快照

每个原始文件记录正式版本、下载时间、URL、许可、字节数和SHA-256。原始文件只读，不原地修改。

计划数据源：

- DisProt正式发布快照，用于实验支持的经典无序残基；
- wwPDB PDBx/mmCIF和SIFTS observed mapping，用于结构覆盖负样本；
- UniProt accession/isoform映射，用于标识符去泄漏；
- CAID1历史快照，可作为训练候选；
- CAID2/3，仅用于生成排除哨兵和最终测试。

### 3.2 正样本

保留直接实验支持的经典无序注释。SoftDis、预测结果、AlphaFold低pLDDT、高B-factor和仅由X-ray缺失推断的区域不能单独成为正式正标签。

当前正标签固定来自DisProt `2018_11`历史快照。显式证据白名单、原始文件哈希、处理计数和CAID2/3直接泄漏删除记录见`configs/data/historical_disprot_2018.json`及`data/manifests/training/historical_disprot_2018_positive.json`。更新版DisProt只可用于标识符补映射，不能向本实验系列补充训练标签。

### 3.3 负样本

负标签需要实验PDB坐标覆盖，并通过SIFTS映射回UniProt残基。未覆盖、映射冲突、结构与DisProt证据冲突的残基一律标记为`-1`。

当前实现仅使用DisProt `2018_11`已引用且发布日期不晚于2018-11-30的PDB条目。X-ray要求分辨率不大于3.0 Å，cryo-EM不大于4.0 Å，NMR不要求分辨率。SIFTS accession和候选序列必须完全匹配；Disorder、Molten globule、Pre-molten globule及其边界两侧5个残基不允许变成负标签。完整快照与结果见`configs/data/pdb_negative_2018.json`和`data/manifests/training/classic_idr_2018_pdb_merged.json`。

### 3.4 三层CAID2/3排除

1. 标识符层：删除相同DisProt ID、UniProt accession及isoform主体。
2. 精确序列层：删除规范化序列SHA-256相同的记录。
3. 同源层：候选训练序列与CAID序列一致性严格大于30%，且双方覆盖度均至少80%时删除。

同源搜索必须保存原始TSV、软件版本、数据库哈希和命令。没有完成同源审计的数据不能进入正式交叉验证。

当前使用本机Windows版MMseqs2 `18-8cc5c`生成以下显式字段：

```text
query,target,fident,alnlen,qlen,tlen,qcov,tcov
```

其中`fident`、`qcov`和`tcov`都必须是`0～1`小数。`alnlen`包含插入/缺失产生的缺口列，不能直接代替双向覆盖率；因此正式MMseqs审计必须显式输出`qcov`和`tcov`。这也避免把30%误写成30或只检查单向覆盖度。正式模式若不提供同源TSV会直接失败；`--allow-exact-only`只能用于冒烟测试。

同源搜索的测试目标FASTA由构建脚本写入`data/caid_holdout/derived/caid23/targets.fasta`。该文件不含标签、不提交Git，上传训练服务器时只需要它和训练候选FASTA，不需要上传CAID2/3答案。

正式CAID2/3哨兵中的目标即使无法补齐UniProt accession也不会被放行：序列SHA-256和同源层仍然强制覆盖它们，manifest同时保留原始DP编号以便后续补映射。

### 3.5 内部五折

完成CAID排除后，再对剩余训练数据做同源聚类。整个同源簇只能进入同一折，不能把同一蛋白、isoform或近同源序列拆到训练折和验证折两侧。

内部全对全搜索使用与CAID排除相同的严格身份和双向覆盖规则。每个通过规则的非自身比对构成一条无向边；同源簇定义为该无向图的连通分量。采用连通分量而不是只围绕一个代表序列分组，可以保证即使`A～B、B～C、A不直接命中C`，A、B、C仍不会跨折。

整簇五折分配固定使用seed 17处理并列情况，同时平衡每折的正标签残基、负标签残基、蛋白数和总残基数。模型初始化seed 17、29、43复用完全相同的折划分，不能为不同模型或随机种子重新生成更有利的验证折。

当前1,133条蛋白形成984个连通簇；190条严格无向同源边跨折数量为0。固定划分和审计结果记录在`data/manifests/training/classic_idr_2018_homology_5fold.json`。

## 4. 基线与候选模型分离

PUNCH2复现允许在独立的`baseline-punch2`数据命名空间中使用论文原始PDB-missing标签，因为这属于复现实验。它不能静默混入MUSE-IDR正式训练数据。

所有模型必须在相同的无泄漏内部验证划分上另做公平比较；同时报告“严格标签版本”和“历史基线原始标签版本”的差异。

## 5. 失败即停止原则

出现以下任一情况，数据构建必须失败而不能跳过：

- 下载结果是HTML而不是预期FASTA/JSON；
- 序列长度与标签长度不同；
- 文件哈希或预期记录数不符；
- 出现重复ID但序列不同；
- CAID2/3精确匹配或标识符匹配仍留在训练集；
- 同源审计文件缺失；
- 折间存在同源簇交叉。
