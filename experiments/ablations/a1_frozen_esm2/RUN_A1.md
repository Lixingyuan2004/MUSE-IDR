# A1 当前运行入口

请使用 `train_a1.py` 作为 A1 的命令行入口：

```powershell
python experiments/ablations/a1_frozen_esm2/train_a1.py `
  --manifest <五折训练清单.jsonl> `
  --fold E
```

第一次只运行预先指定的开发 fold E。完成 token 对齐、显存和输出完整性检查后，再去掉 `--fold E` 运行全部五折。

`run.py` 当前保存训练、预测和评价函数；不要直接把它作为命令行入口。工作区沙箱恢复后会清理其中遗留的旧入口代码。
