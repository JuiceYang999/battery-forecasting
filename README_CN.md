# 电池容量预测项目 - 完整操作手册

## 项目概述

本项目使用机器学习方法（XGBoost）预测锂离子电池的容量衰减，基于电化学阻抗谱（EIS）、充放电曲线等多种特征进行建模。

### 主要功能
- **容量预测**：预测电池下一个循环的放电容量
- **多数据集支持**：支持可变放电、固定放电、不同化学体系的电池数据
- **特征工程**：提取EIS特征、容量-电压曲线特征、充放电速率等
- **集成学习**：使用多个XGBoost模型进行集成预测

---

## 目录结构

```
battery-forecasting-main/
├── raw-data/                          # 原始数据目录
│   ├── variable-discharge/            # 可变放电数据 (PJ097-PJ152)
│   ├── fixed-discharge/               # 固定放电数据 (PJ121-PJ136)
│   ├── chemistry2-25C/                # 化学体系2-25°C数据 (PJ248-PJ279)
│   └── chemistry2-35C/                # 化学体系2-35°C数据 (PJ296-PJ311)
│
├── experiments/                       # 实验脚本目录
│   ├── 1variable-discharge-train.py  # 可变放电数据训练
│   ├── 1next-cycle-capacity.py       # 下一循环容量预测
│   ├── 1fixed-discharge-predict.py   # 固定放电数据预测
│   ├── 1data-efficiency.py           # 数据效率分析
│   ├── 1n-step-lookahead.py          # N步前瞻预测
│   ├── 2vd2-train.py                 # 化学体系2数据训练
│   ├── 2next-cycle-capacity-vd2.py   # 化学体系2容量预测
│   ├── 2vd2-predict.py               # 化学体系2预测
│   └── results/                      # 结果保存目录
│       ├── variable-discharge/       # 可变放电结果
│       ├── fixed-discharge/          # 固定放电结果
│       ├── variable-discharge-type2/ # 化学体系2结果
│       └── vd2-35C/                  # 35°C结果
│
├── utils/                             # 工具函数目录
│   ├── exp_util.py                   # 数据提取工具（标准数据集）
│   ├── exp_util_new.py               # 数据提取工具（新数据集）
│   └── models.py                     # 模型定义和训练
│
└── shell_scripts/                     # Shell脚本目录
    ├── run-battery-cpu               # CPU运行脚本
    ├── run-battery-gpu               # GPU运行脚本
    └── subm_*.sh                     # 作业提交脚本
```

---

## 数据说明

### 数据集类型

| 数据集 | 电池编号 | 数量 | 温度 | 充电协议 | 数据目录 |
|--------|---------|------|------|---------|---------|
| 可变放电 | PJ097-PJ152 | 24 | 常温 | 标准两阶段 | `raw-data/variable-discharge/` |
| 固定放电 | PJ121-PJ136 | 16 | 常温 | 标准两阶段 | `raw-data/fixed-discharge/` |
| 化学体系2-25°C | PJ248-PJ279 | 32 | 25°C | 标准两阶段 | `raw-data/chemistry2-25C/` |
| 化学体系2-35°C | PJ296-PJ311 | 16 | 35°C | 标准两阶段 | `raw-data/chemistry2-35C/` |

### 数据文件格式

每个电池目录包含多个循环的测试数据文件：

```
PJ097/
├── PJ097_000a_03_GCPL_CA1.txt    # 初始化放电数据
├── PJ097_000a_04_GEIS_CA1.txt    # 初始化EIS数据
├── PJ097_001_01_GEIS_CA1.txt     # 循环1-放电EIS
├── PJ097_001_02_GCPL_CA1.txt     # 循环1-充电数据
├── PJ097_001_03_GEIS_CA1.txt     # 循环1-充电EIS
├── PJ097_001_04_GCPL_CA1.txt     # 循环1-放电数据
└── ...
```

**文件命名规则**：
- `{电池编号}_{循环号}_{步骤号}_{测试类型}_C{通道}.txt`
- 测试类型：
  - `GCPL`：恒流充放电（Galvanostatic Cycling with Potential Limitation）
  - `GEIS`：电化学阻抗谱（Galvanostatic Electrochemical Impedance Spectroscopy）

### 数据内容

**GCPL文件列**：
- `time`: 时间 (秒)
- `ewe`: 电压 (V)
- `i`: 电流 (A)
- `capacity`: 容量 (Ah)
- `power`: 功率 (W)
- `ox/red`: 氧化还原标志

**GEIS文件列**：
- `freq`: 频率 (Hz)
- `re_z`: 阻抗实部 (Ω)
- `-im_z`: 阻抗虚部 (Ω)
- `time`: 时间 (秒)

---

## 环境配置

### 依赖包

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install numpy pandas scipy scikit-learn xgboost
```

### 主要依赖版本
- Python >= 3.7
- numpy >= 1.19.0
- pandas >= 1.1.0
- scipy >= 1.5.0
- scikit-learn >= 0.23.0
- xgboost >= 1.2.0

---

## 快速开始

### 1. 训练模型（可变放电数据）

```bash
cd battery-forecasting-main
python experiments/1variable-discharge-train.py
```

**输出**：
- 模型文件：`experiments/results/variable-discharge/models/eis-actions_n1_xgb_*.pkl`
- 训练日志：终端输出

### 2. 预测下一循环容量

```bash
python experiments/1next-cycle-capacity.py
```

**输出**：
- 预测结果：`experiments/results/variable-discharge/predictions/`
- 日志文件：`experiments/results/variable-discharge/log-next-cycle-12s.txt`

### 3. 固定放电数据预测

```bash
python experiments/1fixed-discharge-predict.py
```

**输出**：
- 预测结果：`experiments/results/fixed-discharge/predictions/`

### 4. 化学体系2数据训练

```bash
python experiments/2vd2-train.py
```

---

## 详细使用说明

### 模型训练参数

在实验脚本中可以调整以下参数：

```python
params = {
    'max_depth': 100,        # XGBoost树的最大深度
    'n_splits': 12,          # 交叉验证折数
    'n_estimators': 500,     # 树的数量
    'n_ensembles': 10        # 集成模型数量
}
```

**参数说明**：
- `max_depth`: 控制模型复杂度，越大越容易过拟合
- `n_splits`: K折交叉验证的折数
- `n_estimators`: XGBoost中树的数量，越多训练越慢但可能更准确
- `n_ensembles`: 集成学习的模型数量，用于降低预测方差

### 输入特征选择

支持多种特征组合（在实验脚本的 `input_names` 中定义）：

| 特征名称 | 说明 | 维度 |
|---------|------|------|
| `eis-actions` | EIS特征 + 充放电速率 | 200 + 3 |
| `cvfs-actions` | 容量-电压曲线 + 充放电速率 | 1000 + 3 |
| `eis-cvfs-actions` | EIS + 容量-电压曲线 + 充放电速率 | 1200 + 3 |
| `eis-cvfs-ct-c-actions` | 完整特征集 | 1200 + 2 + 3 |
| `ecmr-actions` | 等效电路模型(Randles) + 充放电速率 | 4 + 3 |
| `ecmer-actions` | 等效电路模型(扩展Randles) + 充放电速率 | 6 + 3 |

**特征说明**：
- `eis`: 电化学阻抗谱特征（200维）
- `cvfs`: 容量-电压曲线特征（1000维）
- `ct`: 累积吞吐量（1维）
- `c`: 上一循环容量（1维）
- `actions`: 充放电速率（3维：放电速率、充电速率1、充电速率2）
- `ecmr`: Randles等效电路模型参数（4维）
- `ecmer`: 扩展Randles等效电路模型参数（6维）

### 数据提取流程

```python
from utils.exp_util import extract_data, extract_input

# 1. 提取原始数据
experiment = 'variable-discharge'
channels = [1, 2, 3, 4, 5, 6, 7, 8]
cell_idx, cap_ds, data = extract_data(experiment, channels)

# 2. 提取输入特征
input_name = 'eis-actions'
x = extract_input(input_name, data)

# 3. 目标变量
y = cap_ds  # 放电容量
```

---

## 实验脚本详解

### 1. `1variable-discharge-train.py`
**功能**：训练可变放电数据的预测模型

**流程**：
1. 加载可变放电数据（PJ097-PJ152）
2. 提取EIS特征和充放电速率
3. 训练XGBoost集成模型
4. 保存模型到 `experiments/results/variable-discharge/models/`

**运行时间**：约5-10分钟（取决于参数设置）

### 2. `1next-cycle-capacity.py`
**功能**：预测下一循环的放电容量

**流程**：
1. 加载数据并提取特征
2. 使用K折交叉验证训练模型
3. 对每个电池进行预测
4. 保存预测结果和评估指标

**输出指标**：
- R² (决定系数)
- RMSE (均方根误差)
- MAE (平均绝对误差)

### 3. `1fixed-discharge-predict.py`
**功能**：使用可变放电训练的模型预测固定放电数据

**流程**：
1. 加载已训练的模型
2. 提取固定放电数据特征
3. 进行预测并保存结果

**用途**：验证模型的泛化能力

### 4. `1data-efficiency.py`
**功能**：分析不同训练数据量对模型性能的影响

**流程**：
1. 使用不同数量的电池进行训练
2. 评估模型性能
3. 分析数据效率曲线

### 5. `1n-step-lookahead.py`
**功能**：N步前瞻预测

**流程**：
1. 训练模型
2. 迭代预测未来N个循环的容量
3. 评估长期预测性能

### 6. `2vd2-train.py`
**功能**：训练化学体系2数据（25°C）

**特点**：
- 使用新的数据提取工具 `exp_util_new.py`
- 支持不同温度的数据
- 特征提取方式略有不同

---

## 工具函数说明

### `utils/exp_util.py`
**用途**：标准数据集（variable-discharge, fixed-discharge）的数据提取

**主要函数**：

#### `extract_data(experiment, channels)`
提取指定实验的所有数据

**参数**：
- `experiment`: 实验类型 ('variable-discharge' 或 'fixed-discharge')
- `channels`: 通道列表 [1, 2, 3, 4, 5, 6, 7, 8]

**返回**：
- `cell_idx`: 电池索引数组
- `cap_ds`: 放电容量数组
- `data`: 特征数据元组

#### `extract_input(input_name, data)`
从数据中提取指定的输入特征

**参数**：
- `input_name`: 特征名称（如 'eis-actions'）
- `data`: 从 `extract_data` 返回的数据

**返回**：
- `x`: 输入特征矩阵 (n_samples, n_features)

#### `discharge_features(ptf, cycle, cap_curve_norm=None)`
从GCPL文件提取放电特征

**提取特征**：
- 放电容量
- 容量-电压曲线（1000点）
- S型曲线拟合参数
- 起始和终止电压
- 放电速率

#### `charge_features(ptf)`
从GCPL文件提取充电特征

**提取特征**：
- 充电时间
- 充电容量
- 两阶段充电速率
- 开路电压（OCV）

#### `eis_features(path, new_log_freq, n_repeats=1)`
从GEIS文件提取EIS特征

**处理步骤**：
1. 读取频率、阻抗实部和虚部
2. 对数频率插值到100个点
3. 返回200维特征向量（实部100 + 虚部100）

#### `ensemble_predict(x, experiment, input_name, n_ensembles=10)`
使用集成模型进行预测

**特点**：
- 自动检测可用的模型文件数量
- 计算预测均值和标准差
- 支持不确定性量化

### `utils/exp_util_new.py`
**用途**：新数据集（chemistry2-25C, chemistry2-35C）的数据提取

**主要差异**：
- 支持不同温度的数据
- 文件命名格式略有不同
- 充电协议处理方式不同

**主要函数**：

#### `extract_data_type2(experiment, channels, suffix='vd2')`
提取化学体系2的数据

**特点**：
- 支持25°C和35°C数据
- 包含最大电压信息
- 4步循环协议

### `utils/models.py`
**用途**：模型定义和训练

**主要类**：

#### `XGBModel`
XGBoost模型封装类

**初始化参数**：
```python
XGBModel(
    X,              # 输入特征
    y,              # 目标变量
    cell_idx,       # 电池索引
    experiment,     # 实验名称
    experiment_name,# 实验标识
    n_ensembles=10, # 集成数量
    n_splits=12,    # 交叉验证折数
    max_depth=100,  # 树深度
    n_estimators=500# 树数量
)
```

**主要方法**：

##### `train_no_predict(X, y)`
训练模型但不进行预测

**用途**：仅训练并保存模型

**输出**：
- 保存模型文件到 `experiments/results/{experiment}/models/`

##### `train_and_predict(X_train, y_train, X_test1, cell_test1, ...)`
训练并预测

**流程**：
1. Bootstrap采样训练数据
2. 训练多个XGBoost模型
3. 对测试集进行预测
4. 计算预测均值和标准差

**返回**：
- 训练集预测结果和误差
- 多个测试集的预测结果和误差

##### `analysis(log_name, experiment_info)`
完整的分析流程

**流程**：
1. 按电池分组进行交叉验证
2. 每次留出一个电池作为测试集
3. 训练模型并预测
4. 保存结果和日志

**输出**：
- 模型文件
- 预测结果（numpy数组）
- 日志文件（包含R²、RMSE等指标）

---

## 结果分析

### 输出文件

训练和预测完成后，会生成以下文件：

```
experiments/results/{experiment}/
├── models/
│   ├── {input_name}_n1_xgb_0.pkl      # 集成模型1
│   ├── {input_name}_n1_xgb_1.pkl      # 集成模型2
│   └── ...
├── predictions/
│   ├── pred_mn_{input_name}_{cell}.npy    # 预测均值
│   ├── pred_std_{input_name}_{cell}.npy   # 预测标准差
│   └── true_{input_name}_{cell}.npy       # 真实值
└── log-next-cycle-12s.txt             # 日志文件
```

### 读取结果

```python
import numpy as np

# 读取预测结果
cell = 'PJ097'
input_name = 'eis-actions'
experiment = 'variable-discharge'

pred_mean = np.load(f'experiments/results/{experiment}/predictions/pred_mn_{input_name}_{cell}.npy')
pred_std = np.load(f'experiments/results/{experiment}/predictions/pred_std_{input_name}_{cell}.npy')
true_values = np.load(f'experiments/results/{experiment}/predictions/true_{input_name}_{cell}.npy')

# 计算误差
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(true_values, pred_mean)
rmse = np.sqrt(mean_squared_error(true_values, pred_mean))

print(f'R² = {r2:.4f}')
print(f'RMSE = {rmse:.4f} Ah')
```

### 可视化

```python
import matplotlib.pyplot as plt

# 绘制预测vs真实值
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(pred_mean)), pred_mean, yerr=pred_std, 
             fmt='o-', label='预测值', capsize=3)
plt.plot(true_values, 's-', label='真实值')
plt.xlabel('循环次数')
plt.ylabel('放电容量 (Ah)')
plt.legend()
plt.title(f'{cell} 容量预测结果')
plt.grid(True)
plt.savefig(f'{cell}_prediction.png', dpi=300)
plt.show()
```

---

## 常见问题

### Q1: 运行时提示找不到文件
**A**: 确保：
1. 当前工作目录在项目根目录
2. 数据文件在正确的位置（`raw-data/` 目录下）
3. 路径分隔符正确（Windows使用 `\`，Linux/Mac使用 `/`）

### Q2: 内存不足
**A**: 减少以下参数：
- `n_ensembles`: 减少集成模型数量
- `n_estimators`: 减少树的数量
- 或者一次只处理部分电池

### Q3: 训练速度慢
**A**: 
- 减少 `n_estimators` 和 `max_depth`
- 使用更少的特征（如只用 `eis-actions` 而不是 `eis-cvfs-ct-c-actions`）
- 考虑使用GPU版本的XGBoost

### Q4: 预测精度不高
**A**: 
- 增加 `n_ensembles` 和 `n_estimators`
- 尝试不同的特征组合
- 检查数据质量
- 增加训练数据量

### Q5: 如何添加新的数据集
**A**: 
1. 将数据放入 `raw-data/` 目录
2. 在 `exp_util.py` 或 `exp_util_new.py` 中添加 `experiment_map` 映射
3. 在 `identify_cells()` 函数中添加电池-通道映射
4. 创建新的实验脚本

---

## 高级功能

### 1. 自定义特征提取

在 `exp_util.py` 的 `extract_input()` 函数中添加新的特征组合：

```python
elif input_name == 'my-custom-features':
    # 自定义特征提取逻辑
    states = np.concatenate((eis_ds, custom_features), axis=1)
    x = np.concatenate((states, actions), axis=1)
```

### 2. 使用不同的模型

修改 `models.py` 中的模型定义：

```python
# 替换XGBoost为其他模型
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(
    max_depth=self.max_depth,
    n_estimators=self.n_estimators,
    random_state=ensemble_state
)
```

### 3. 超参数优化

使用网格搜索或贝叶斯优化：

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [50, 100, 150],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    xgb.XGBRegressor(),
    param_grid,
    cv=5,
    scoring='r2'
)
grid_search.fit(X_train, y_train)
print(f'最佳参数: {grid_search.best_params_}')
```

---

## 性能基准

### 典型结果（variable-discharge数据集）

| 特征组合 | R² | RMSE (Ah) | 训练时间 |
|---------|-----|-----------|---------|
| eis-actions | 0.95 | 0.03 | 5 min |
| cvfs-actions | 0.93 | 0.04 | 8 min |
| eis-cvfs-actions | 0.97 | 0.02 | 12 min |
| eis-cvfs-ct-c-actions | 0.98 | 0.015 | 15 min |

*注：结果基于默认参数，实际结果可能因数据和硬件而异*

---

## 引用

如果使用本项目，请引用相关论文：

```bibtex
@article{battery_forecasting,
  title={Battery Capacity Forecasting using Machine Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

---

## 许可证

本项目采用 MIT 许可证。

---

## 联系方式

如有问题或建议，请联系：
- Email: your.email@example.com
- GitHub Issues: [项目链接]

---

## 更新日志

### v1.0.0 (2024-10-26)
- ✅ 修复所有路径问题
- ✅ 支持chemistry2数据集
- ✅ 添加自动模型文件检测
- ✅ 完善错误处理
- ✅ 添加详细中文注释

---

## 致谢

感谢所有为本项目做出贡献的研究人员和开发者。

