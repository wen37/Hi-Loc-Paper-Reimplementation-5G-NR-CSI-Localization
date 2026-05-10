# Hi-Loc-Paper-Reimplementation-5G-NR-CSI-Localization
基于 5G NR CSI 的 Hi-Loc 混合室内定位算法复现，采用 CNN-BiLSTM + 特征注意力机制，完成多基站数据预处理、序列构建、模型训练、消融实验与定位误差分析。


## 环境配置
### 依赖项
推荐使用Python 3.7+版本，核心依赖库如下：
```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.5
pandas>=1.3.0
scikit-learn>=0.24.2
```


## 数据准备
1. 数据下载链接：http://www.pmldatanet.com.cn/dataapp/5G-A_positioning_dataset_for_AI
2. 本项目用的是仿真数据InF-DL、InF-DH的0ns、10ns、50ns的数据。
3. 缓存数据：`/data/cache`目录下的缓存文件无需手动下载，运行`train.py`时会自动生成。
4. 消融实验相关：`feature_ablation`和`training_ablation`目录下`data`文件夹的实验数据、以及`test/checkpoints`下的`.pt`模型文件因体积过大未上传，且消融实验结果不影响核心功能运行，无需额外准备。

## 快速开始
### 模型训练
运行`train.py`启动训练流程，数据路径在config文件中修改，可根据需求调整命令行参数：


### 模型测试/推理
训练完成后，运行`test.py`进行模型评估或推理：
```bash
python test.py 
```

## 项目结构
```
├── data/                        # 数据目录
│   └── cache/                   # 缓存目录（train.py自动生成，将数据变成.npy文件，方便快速读取）
├── feature_ablation/            # 特征消融实验目录
│   ├── dataset_ablation.py
│   ├── feature_variants.py                
│   └── run_feature_ablation.py  # 运行此代码进行消融实验                    
├── training_ablation/           # 训练过程消融实验目录（数据未上传）
│   ├── ablation_model.py
│   └── run_training_ablation.py # 运行此代码进行消融实验  
├── train.py                     # 核心训练脚本
├── test.py                      # 核心测试/推理脚本
├── utils.py                     # 工具函数/类脚本
├── requirements.txt             # 依赖清单
└── README.md                    # 项目说明文档
```

## 实验说明
1. 核心实验：可通过`train.py`和`test.py`复现论文中核心实验结果，需确保数据集配置正确。
2. 消融实验：`feature_ablation`和`training_ablation`目录下为消融实验相关代码，但因实验数据、模型权重文件体积过大未上传，且消融实验结果不影响核心功能验证，无需复现。
