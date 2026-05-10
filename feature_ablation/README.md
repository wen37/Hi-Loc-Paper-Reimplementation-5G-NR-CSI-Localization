# 特征层消融实验

本文件夹用于做统一设置下的特征层消融，对比以下三组方案：

1. `downsample_only`：仅降采样特征
2. `stats_only`：仅统计特征
3. `downsample_plus_stats`：降采样 + 统计特征（直接复用当前已完成的稳定多基站基线结果）

## 设计目标

- 验证频域降采样特征是否是主要信息来源
- 验证 8 个统计特征是否单独具备定位能力
- 验证“降采样 + 统计特征”的组合是否优于单一特征类型

## 运行方式

在项目根目录执行：

```bash
python feature_ablation\run_feature_ablation.py
```

如需强制重建缓存：

```bash
python feature_ablation\run_feature_ablation.py --force-reload
```

## 结果输出

脚本运行后会在 `feature_ablation/results/` 下生成：

- `downsample_only/`
- `stats_only/`
- `downsample_plus_stats/`
- `feature_ablation_summary.csv`
- `feature_ablation_summary.md`
- `feature_ablation_rmse.png`
- `feature_ablation_mde.png`

其中每个方案目录下会包含：

- `train/loss_curve.png`
- `test/metrics.json`
- `test/error_histogram.png`
- `test/error_cdf.png`
- `test/prediction_scatter.png`
- `test/predictions.csv`
