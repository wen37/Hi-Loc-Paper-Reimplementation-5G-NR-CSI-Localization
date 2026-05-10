# 训练层消融实验

本目录用于分析“多基站模型为什么一开始变差，以及哪些训练稳定化策略真正起作用”。

## 实验分组

1. `no_stabilization`：无稳定化  
   - 无输入归一化
   - 无 Dropout
   - `lr=1e-3`
   - 无 `weight_decay`
   - 无 `grad_clip`
   - 直接复用已有 `result1`

2. `norm_dropout_only`：仅归一化 + Dropout  
   - 有输入归一化
   - 有 Dropout
   - `lr=1e-3`
   - 无 `weight_decay`
   - 无 `grad_clip`

3. `optimizer_stabilization_only`：仅优化层稳定化  
   - 无输入归一化
   - 无 Dropout
   - `lr=3e-4`
   - `weight_decay=1e-4`
   - `grad_clip=1.0`

4. `full_stabilization`：完整稳定化  
   - 有输入归一化
   - 有 Dropout
   - `lr=3e-4`
   - `weight_decay=1e-4`
   - `grad_clip=1.0`
   - 直接复用已有 `result2`

## 运行方式

在项目根目录运行：

```bash
python training_ablation\run_training_ablation.py
```

如需强制重建数据集缓存：

```bash
python training_ablation\run_training_ablation.py --force-reload
```

## 输出内容

运行后将在 `training_ablation/results/` 下生成：

- `training_ablation_summary.csv`
- `training_ablation_summary.md`
- `training_ablation_rmse.png`
- `training_ablation_mde.png`

以及每个方案下的：

- `train/loss_curve.png`
- `test/metrics.json`
- `test/error_histogram.png`
- `test/error_cdf.png`
- `test/prediction_scatter.png`
- `test/predictions.csv`
