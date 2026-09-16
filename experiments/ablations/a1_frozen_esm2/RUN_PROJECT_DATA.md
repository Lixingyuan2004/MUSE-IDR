# 使用冻结项目数据运行 A1

项目实际 fold 编号是 `0,1,2,3,4`。预先指定的开发验证 fold 为 `4`。

## 已完成

```powershell
.\.venv\Scripts\python.exe scripts\build_a1_manifest.py
.\.venv\Scripts\python.exe experiments\ablations\a1_frozen_esm2\smoke_test_model.py
.\.venv\Scripts\python.exe experiments\ablations\a1_frozen_esm2\train_a1.py `
  --manifest data\processed\classic_idr_2018\a1_frozen_esm2_smoke_manifest.jsonl `
  --config experiments\ablations\a1_frozen_esm2\smoke_train_config.yaml `
  --fold 4
```

## 4090/5090 上的正式开发 fold

```powershell
.\.venv\Scripts\python.exe experiments\ablations\a1_frozen_esm2\train_a1.py `
  --manifest data\processed\classic_idr_2018\a1_frozen_esm2_manifest.jsonl `
  --config experiments\ablations\a1_frozen_esm2\config.yaml `
  --fold 4
```

该命令只使用 fold 0–3 训练、fold 4 验证。它不会读取 CAID2/CAID3 测试标签。

开发 fold 验收后，去掉 `--fold 4` 才会运行完整五折 OOF。正式运行前优先实现冻结 ESM 表征缓存，避免每个 epoch 重复计算同一主干表示。
