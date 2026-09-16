# A1 冻结 ESM2 表征缓存

缓存输入必须是只包含 `id`、`sequence`、`fold` 的 JSONL。`cache_embeddings.py` 如果发现 `labels`、`label` 或 `scores` 字段会立即报错。

先生成无标签序列清单：

```powershell
.\.venv\Scripts\python.exe experiments\ablations\a1_frozen_esm2\build_sequence_manifest.py `
  --source data\processed\classic_idr_2018\a1_frozen_esm2_manifest.jsonl `
  --output data\processed\classic_idr_2018\a1_sequences.jsonl `
  --report experiments\ablations\a1_frozen_esm2\sequence_manifest_report.json
```

在 4090/5090 上提取全部表征：

```powershell
.\.venv\Scripts\python.exe experiments\ablations\a1_frozen_esm2\cache_embeddings.py `
  --manifest data\processed\classic_idr_2018\a1_sequences.jsonl `
  --output-dir data\cache\a1_esm2_650m
```

缓存采用锁定的 ESM2 revision、1022 残基窗口、511 stride、重叠区中心加权和 float16 存储。再次运行会校验元数据和形状后复用已完成文件。
