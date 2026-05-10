# Hi-Loc 5G NR CSI Indoor Localization Reproduction

本项目是对论文 **Hi-Loc: Hybrid Indoor Localization via Enhanced 5G NR CSI** 的工程化复现与扩展实验，围绕 5G NR CSI / CFR 室内定位任务，完成了从原始 `.mat` 数据读取、特征增强、序列构建、模型训练与测试，到多基站扩展和消融实验分析的完整流程。论文链接：https://ieeexplore.ieee.org/document/9855525

当前工程重点复现并验证了以下主线：

- 原始 CFR 读取与缓存
- 特征增强：频域降采样 + 统计特征拼接
- 时序样本构建：滑动窗口序列
- 定位模型：`CNN -> BiLSTM -> FC Regression`
- 可选模块：`Feature Attention`
- 多基站特征拼接
- 固定测试集与 manifest 导出
- 可视化与误差分析
- 特征层消融、训练策略层消融

## 1. Project Overview

论文原始思路可概括为：

```text
Raw CFR
-> Feature Enhancement
-> Data Construction
-> CNN
-> Feature Attention
-> BiLSTM
-> Sample Attention
-> FC Regression
-> (x, y)
```

本项目当前已经稳定实现的工程版本为：

```text
Raw CFR
-> Feature Engineering
-> Sequence Builder
-> CNN
-> Optional Feature Attention
-> BiLSTM
-> FC Regression
-> (x, y)
```

说明：

- 已复现论文主干中的 `CNN-BiLSTM` 建模思路
- 已实现 `Feature Attention` 的可选版本
- 当前更适合描述为“核心思想复现 + 工程化扩展”
- 论文流程中的 `Sample Attention` 尚未单独实现为独立模块

## 2. Main Features

- 支持 MATLAB `v7.3` 和 legacy `.mat` 双格式自动读取
- 支持复数 CFR 数据解析
- 支持单基站与多基站特征拼接
- 支持 `InF_DH`、`InF_DL` 多场景与多同步误差混合训练
- 支持首次处理后缓存为 `.npy`，避免重复读取大体积 `.mat`
- 支持 CPU / GPU 自动选择，GPU 不可用时自动回退到 CPU
- 支持固定随机种子和固定数据划分
- 支持导出训练集、验证集、测试集 manifest，便于结果追溯
- 支持输出训练损失曲线、误差直方图、CDF、预测散点图

## 3. Project Structure

```text
HLS/
├── config/
│   └── config.yaml
├── datasets/
│   ├── feature_engineering.py
│   ├── loader.py
│   └── sequence_builder.py
├── models/
│   ├── baseline_models.py
│   ├── bilstm_block.py
│   ├── cnn_block.py
│   └── feature_attention.py
├── trainers/
│   └── trainer.py
├── utils/
│   ├── checkpoint.py
│   ├── runtime.py
│   ├── seed.py
│   └── visualization.py
├── feature_ablation/
│   ├── dataset_ablation.py
│   ├── feature_variants.py
│   └── run_feature_ablation.py
├── training_ablation/
│   ├── ablation_model.py
│   └── run_training_ablation.py
├── train.py
├── test.py
├── infer.py
└── README.md
```

## 4. Dataset Format

数据链接：http://www.pmldatanet.com.cn/dataapp/5G-A_positioning_dataset_for_AI
项目默认数据组织形式如下：

```text
simulated_dataset/
├── InF_DH/
│   ├── sync_err_0/
│   │   ├── CFR1.mat ~ CFR18.mat
│   │   └── UE_pos.mat
│   ├── sync_err_10/
│   └── sync_err_50/
└── InF_DL/
    ├── sync_err_0/
    ├── sync_err_10/
    └── sync_err_50/
```

其中：

- `CFR*.mat` 为各基站的 CFR 数据
- `UE_pos.mat` 为用户位置标签
- 当前工程标签使用二维坐标 `(x, y)`

默认配置中的数据路径为：

```yaml
data:
  raw_path: "G:/signal/simulated_dataset"
```

请根据你的本地路径修改 `config/config.yaml`。

## 5. Feature Engineering

特征工程位于 `datasets/feature_engineering.py`，当前采用：

- 对 CFR 最后一维子载波进行处理
- 先将多维 CFR 聚合为一维复数子载波向量
- 每隔 8 个子载波取 1 个，得到 408 维降采样特征
- 计算 8 个统计特征：
  - mean
  - std
  - max
  - min
  - skew
  - kurtosis
  - energy
  - entropy
- 拼接后得到每基站 `416` 维输入特征

多基站模式下：

- `18` 个基站
- 每基站 `416` 维
- 总输入维度 `18 x 416 = 7488`

## 6. Model Architecture

当前主模型定义在 `models/baseline_models.py`，结构如下：

```text
Input (B, T, F)
-> LayerNorm
-> Dropout
-> CNNBlock
-> Optional FeatureAttention
-> BiLSTM
-> Dropout
-> FC
-> (x, y)
```

其中：

- `CNNBlock`：2 层 `Conv1d + BatchNorm + ReLU`
- `BiLSTMBlock`：双向 LSTM，默认 2 层
- `FeatureAttention`：两层 MLP + `Sigmoid` 的特征重加权
- `FC`：全连接回归头，输出二维坐标

这意味着：

- 已实现论文中的 `CNN-BiLSTM` 主干
- 已加入工程化稳定策略：`LayerNorm`、`Dropout`
- 已支持可选 `Feature Attention`
- 尚未完全逐层复刻论文中完整的 `Sample Attention` 结构

## 7. Environment Setup

建议使用 Python `3.10+` 或兼容版本。

安装依赖：

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 包含：

- `torch`
- `numpy`
- `scipy`
- `pyyaml`
- `tqdm`
- `matplotlib`
- `scikit-learn`

## 8. Configuration

核心配置文件为 `config/config.yaml`，主要字段包括：

- 数据路径与缓存路径
- 场景列表 `scenes`
- 同步误差列表 `sync_errors`
- 基站编号 `base_station_ids`
- 序列长度 `sequence_length`
- 降采样率 `downsample_rate`
- 模型超参数
- 训练超参数

当前默认设置示例：

```yaml
data:
  raw_path: "G:/signal/simulated_dataset"
  scenes: ["InF_DH", "InF_DL"]
  sync_errors: [0, 10, 50]
  base_station_ids: [1, 2, 3, ..., 18]
  sequence_length: 5
  step_size: 2
  downsample_rate: 8
  feature_dim_per_bs: 416

model:
  use_feature_attention: true
  dropout_rate: 0.3

train:
  batch_size: 32
  lr: 0.0003
  weight_decay: 0.0001
  grad_clip: 1.0
  epochs: 50
  device: "auto"
```

## 9. How To Run

### 9.1 Train

```bash
python train.py
```

流程包括：

- 加载配置
- 自动选择设备
- 构建混合数据集
- 固定随机种子并划分 train / val / test
- 导出划分清单
- 开始训练并保存最优模型

### 9.2 Test

```bash
python test.py
```

测试阶段会输出：

- `metrics.json`
- `predictions.csv`
- `error_histogram.png`
- `error_cdf.png`
- `prediction_scatter.png`

### 9.3 Inference

```bash
python infer.py --scene InF_DH --sync_err 0 --split test --index 0
```

推理脚本会输出：

- 目标样本所属场景
- 真实坐标
- 预测坐标
- 坐标绝对误差
- 欧氏距离误差

## 10. Outputs

默认输出目录：

```text
checkpoints/
outputs/
```

其中常见文件包括：

- `checkpoints/best_model.pt`
- `outputs/train/loss_history.json`
- `outputs/train/loss_curve.png`
- `outputs/test/metrics.json`
- `outputs/test/predictions.csv`
- `outputs/test/error_histogram.png`
- `outputs/test/error_cdf.png`
- `outputs/test/prediction_scatter.png`
- `outputs/splits/train_manifest.csv`
- `outputs/splits/val_manifest.csv`
- `outputs/splits/test_manifest.csv`

`manifest` 文件用于固定并追溯测试集，便于后续实验横向比较。

## 11. Experimental Results

### 11.1 Multi-BS Progressive Results

项目中已经完成了多组阶段性实验：

- `result1-直接多基站，但是效果比单基站差/`
- `result2-多基站，加了归一化正则化，结果好了很多。没加注意力机制/`
- `result3-多基站，加了注意力，还行/`

这些结果体现出一个重要结论：

- 多基站信息本身具有潜力
- 但简单拼接不会自动提升性能
- 训练稳定化策略对多基站模型非常关键

### 11.2 Feature Ablation

特征层消融结果表明：

- `stats_only` 最弱
- `downsample_only` 已经具备较强定位能力
- `downsample_plus_stats` 在整体表现和长尾误差控制上更优

这与原论文的趋势一致，即：

- 融合特征优于单一特征

### 11.3 Training Ablation

训练策略层消融表明：

- 仅依靠更小学习率、`weight_decay`、`grad_clip` 并不能显著提升效果
- 输入归一化与 Dropout 对稳定训练最关键
- 完整稳定化整体更稳，但不一定在当前设置下绝对最优

## 12. Reproduction Summary

从当前工程结果来看：

- 已较完整复现论文主干思路
- 已验证增强特征与融合特征的有效性
- 已完成多基站扩展与训练稳定化分析
- 当前结果趋势与论文“融合特征效果更好”的结论基本一致

因此，本项目可视为：

- 对 Hi-Loc 核心方法链路的工程化复现
- 并在此基础上完成了多基站与消融实验扩展

## 13. Known Limitations

- 当前大部分训练在 CPU 环境完成，训练效率有限
- 使用的是仿真数据，和真实商用 5G 信号仍有差异
- 复杂 NLOS、动态遮挡、多环境迁移等问题尚未充分验证
- 论文中的完整结构并未逐模块完全复刻

## 14. Future Work

后续可以继续从以下方向扩展：

- 更系统地搜索超参数
- 引入更完整的注意力结构
- 增加跨场景、跨同步误差的泛化验证
- 尝试仿真到真实数据迁移
- 融合其他模态数据，如 `PDR`、IMU、UWB、WiFi 等
