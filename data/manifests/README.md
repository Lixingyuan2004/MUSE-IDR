# 数据manifest

每个数据源需要一个manifest，至少记录：

- 数据源名称和正式版本；
- 官方下载地址和下载日期；
- 原始文件名、字节数和SHA-256；
- 数据许可；
- 标签证据筛选规则；
- 处理脚本的Git提交；
- 生成文件及SHA-256。

CAID清单也保存在此处，但不保存CAID真实标签。`caid_all_rounds_sentinel.json`记录三轮来源，`caid23_test_sentinel.json`是训练审计实际使用的外部测试保护清单。

`training/`保存可再生成的训练数据manifest。训练manifest必须明确写出是否已经具备训练资格；处于`blocked`状态的数据不能交给训练脚本。
