# 代码详细注释指南

本文档提供项目中所有Python文件的详细中文注释说明。

---

## 实验脚本注释

### `experiments/2vd2-train.py` - 化学体系2数据训练脚本

```python
# ============================================================================
# 文件: experiments/2vd2-train.py
# 功能: 训练化学体系2（25°C）电池数据的容量预测模型
# 作者: Battery Forecasting Team
# 日期: 2024-10-26
# ============================================================================

import sys
sys.path.append('.')  # 将当前目录添加到Python路径，以便导入utils模块
import time  # 用于计时
import pickle  # 用于保存/加载模型（虽然此脚本中未直接使用）

# 从工具模块导入数据提取和模型类
from utils.exp_util_new import extract_data_type2, extract_input  # 新数据集提取工具
from utils.models import XGBModel  # XGBoost模型封装类

import pdb  # Python调试器（用于开发调试）

# ============================================================================
# 配置参数
# ============================================================================

# 通道列表：定义要处理的测试通道
# A1-A8: 第一批8个通道
# B1-B8: 第二批8个通道
channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']

# 模型超参数字典
params = {
    'max_depth': 100,      # XGBoost决策树的最大深度
                          # 值越大模型越复杂，但可能过拟合
    'n_splits': 16,       # K折交叉验证的折数
                          # 用于评估模型泛化性能
    'n_estimators': 50,   # XGBoost中树的数量
                          # 更多的树通常提高准确度但增加训练时间
    'n_ensembles': 10     # 集成学习的模型数量
                          # 多个模型投票可以降低预测方差
}

# 实验类型：指定要处理的数据集
# 'variable-discharge-type2' 对应 PJ248-PJ279 (chemistry2-25C)
experiment = 'variable-discharge-type2'

# ============================================================================
# 特征选择
# ============================================================================

# 输入特征名称列表：定义要测试的不同特征组合
# 'eis-cvfs-actions': EIS特征 + 容量-电压曲线 + 充放电速率
# 'eis-actions': 仅EIS特征 + 充放电速率
input_names = ['eis-cvfs-actions', 'eis-actions']

# ============================================================================
# 数据提取
# ============================================================================

# 提取化学体系2的数据
# 返回值:
#   cell_var: 电池标识数组 (如 ['PJ248', 'PJ248', ..., 'PJ249', ...])
#   cap_ds_var: 放电容量数组 (目标变量 y)
#   data_var: 特征数据元组，包含:
#             (last_caps, sohs, eis_ds, cvfs, ocvs, 
#              cap_throughputs, d_rates, c1_rates, c2_rates, v_maxs)
cell_var, cap_ds_var, data_var = extract_data_type2(experiment, channels)

# 打印数据点数量，用于验证数据加载
# data_var[0] 是 last_caps（上一循环容量）
print('Number of datapoints = {}'.format(data_var[0].shape[0]))

# ============================================================================
# 目标变量定义
# ============================================================================

# 定义目标变量：下一循环的放电容量
y = cap_ds_var  # shape: (n_samples,)

# ============================================================================
# 模型训练循环
# ============================================================================

# 遍历每个特征组合进行训练
for i in range(len(input_names)):
    # 当前特征组合名称
    input_name = input_names[i]
    
    # 实验名称：用于保存模型文件
    # 格式: {特征名}_n1_xgb2
    # n1: 表示预测下一个循环 (next 1 cycle)
    # xgb2: 表示XGBoost模型版本2
    experiment_name = '{}_n1_xgb2'.format(input_name)
    
    # 构建实验信息字符串，用于日志记录
    experiment_info = '\nInput: {} \tOutput: Q_n+1 \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(
        input_name,                # 输入特征名称
        params['max_depth'],       # 树深度
        params['n_estimators'],    # 树数量
        params['n_ensembles'],     # 集成数量
        params['n_splits']         # 交叉验证折数
    )
    print(experiment_info)
    
    # 记录开始时间
    t0 = time.time()
    
    # ========================================================================
    # 特征提取
    # ========================================================================
    
    # 从原始数据中提取指定的输入特征
    # suffix='vd2': 指定数据集后缀，用于区分不同数据集的处理方式
    # 返回: x - 输入特征矩阵 shape: (n_samples, n_features)
    x = extract_input(input_name, data_var, suffix='vd2')
    
    # ========================================================================
    # 模型初始化
    # ========================================================================
    
    # 创建XGBoost模型对象
    # 参数说明:
    #   x: 输入特征矩阵
    #   y: 目标变量（放电容量）
    #   cell_var: 电池索引，用于分组交叉验证
    #   experiment: 实验类型，用于确定保存路径
    #   experiment_name: 实验名称，用于文件命名
    #   n_ensembles: 集成模型数量
    #   n_splits: 交叉验证折数
    #   max_depth: 树的最大深度
    #   n_estimators: 树的数量
    regressor = XGBModel(
        x, y, cell_var, 
        experiment, 
        experiment_name, 
        n_ensembles=params['n_ensembles'],
        n_splits=params['n_splits'], 
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators']
    )
    
    # ========================================================================
    # 模型训练
    # ========================================================================
    
    # 训练模型但不进行预测
    # 这个方法会:
    #   1. 使用Bootstrap采样创建多个训练集
    #   2. 训练n_ensembles个XGBoost模型
    #   3. 保存所有模型到 experiments/results/{experiment}/models/
    # 
    # 模型文件命名格式: {experiment_name}_{i}.pkl
    # 例如: eis-actions_n1_xgb2_0.pkl, eis-actions_n1_xgb2_1.pkl, ...
    regressor.train_no_predict(x, y)
    
    # 计算并打印训练耗时
    elapsed_time = time.time() - t0
    print('Time taken = {:.2f}'.format(elapsed_time))

# ============================================================================
# 训练完成
# ============================================================================

print('Done.')  # 所有特征组合训练完成
```

---

## 工具函数注释

### `utils/exp_util.py` - 核心函数注释

#### 1. `extract_data()` 函数

```python
def extract_data(experiment, channels):
    """
    从原始数据文件中提取电池循环数据
    
    功能说明:
    --------
    1. 读取指定实验和通道的所有电池数据
    2. 提取EIS（电化学阻抗谱）特征
    3. 提取充放电曲线特征
    4. 计算各种衍生特征（SOH、吞吐量等）
    
    参数:
    ----
    experiment : str
        实验类型，可选值:
        - 'variable-discharge': 可变放电实验 (PJ097-PJ152)
        - 'fixed-discharge': 固定放电实验 (PJ121-PJ136)
    
    channels : list of int
        要处理的通道列表，例如 [1, 2, 3, 4, 5, 6, 7, 8]
        每个通道对应多个电池
    
    返回:
    ----
    cell_idx : ndarray, shape (n_samples,)
        每个数据点对应的电池标识
        例如: ['PJ097', 'PJ097', ..., 'PJ098', ...]
    
    cap_ds : ndarray, shape (n_samples,)
        每个循环的放电容量 (Ah)
        这是模型的目标变量
    
    data : tuple
        包含9个元素的元组:
        (last_caps, sohs, eis_ds, cvfs, ocvs, 
         cap_throughputs, d_rates, c1_rates, c2_rates)
        
        详细说明:
        - last_caps: 上一循环的放电容量 (n_samples,)
        - sohs: 健康状态 (State of Health) (n_samples,)
                计算方式: 当前容量 / 初始容量
        - eis_ds: 放电时的EIS特征 (n_samples, 200)
                  前100列是阻抗实部，后100列是阻抗虚部
        - cvfs: 容量-电压曲线特征 (n_samples, 1000)
                通过插值得到1000个均匀分布的点
        - ocvs: 开路电压 (Open Circuit Voltage) (n_samples,)
        - cap_throughputs: 累积吞吐量 (n_samples,)
                          计算方式: Σ(充电容量 + 放电容量)
        - d_rates: 放电速率 (n_samples, 1)
        - c1_rates: 第一阶段充电速率 (n_samples, 1)
        - c2_rates: 第二阶段充电速率 (n_samples, 1)
    
    数据处理流程:
    -----------
    1. 识别电池-通道映射关系
    2. 对每个电池:
       a. 读取初始化循环数据（cycle=0）
       b. 提取初始容量和EIS基线
       c. 遍历后续循环（通常30个循环）
       d. 每个循环提取4个步骤的数据:
          - 步骤1: 放电EIS测量
          - 步骤2: 充电过程
          - 步骤3: 充电EIS测量（可选）
          - 步骤4: 放电过程
    3. 将所有电池数据合并成统一的数组
    
    文件命名规则:
    -----------
    初始化文件: {cell}_{cycle:03d}a_{step:02d}_{type}_CA{channel}.txt
    循环文件:   {cell}_{cycle:03d}_{step:02d}_{type}_CA{channel}.txt
    
    例如:
    - PJ097_000a_04_GEIS_CA1.txt  # 初始化EIS
    - PJ097_001_01_GEIS_CA1.txt   # 循环1步骤1 EIS
    - PJ097_001_02_GCPL_CA1.txt   # 循环1步骤2 充电
    
    注意事项:
    --------
    - 如果某个循环的数据文件缺失或读取失败，该循环会被跳过
    - 特定电池(PJ145-PJ148)的起始循环是3而不是2
    - 函数会自动处理数据缺失和异常情况
    
    示例:
    ----
    >>> experiment = 'variable-discharge'
    >>> channels = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> cell_idx, cap_ds, data = extract_data(experiment, channels)
    >>> print(f'总数据点数: {len(cap_ds)}')
    >>> print(f'电池数量: {len(np.unique(cell_idx))}')
    >>> print(f'EIS特征维度: {data[2].shape}')
    """
    
    # 函数实现...
    # （实际代码见 utils/exp_util.py）
```

#### 2. `discharge_features()` 函数

```python
def discharge_features(ptf, cycle, cap_curve_norm=None):
    """
    从GCPL文件中提取放电特征
    
    功能说明:
    --------
    从恒流充放电（GCPL）数据文件中提取多种放电特征，
    包括容量、电压、放电曲线形状参数等。
    
    参数:
    ----
    ptf : str
        GCPL文件的完整路径
        例如: 'raw-data/variable-discharge/PJ097/PJ097_001_04_GCPL_CA1.txt'
    
    cycle : int
        当前循环编号
        - cycle=0: 初始化循环，不计算容量变化特征
        - cycle>0: 正常循环，计算相对于基线的容量变化
    
    cap_curve_norm : ndarray, optional
        归一化的容量-电压曲线基线 (1000,)
        用于计算容量曲线的变化
        如果为None（cycle=0时），则不计算变化特征
    
    返回:
    ----
    如果文件存在且读取成功:
        cap : float
            放电容量 (Ah)
        
        features : ndarray, shape (7,)
            特征向量，包含:
            [0:3] - S型曲线拟合参数 (a, b, c)
                    拟合公式: capacity = a / (1 + exp(b*(v-c)))
            [3] - v0: 放电起始电压 (V)
            [4] - v1: 放电终止电压 (V)
            [5] - log_var: 容量曲线方差的对数
            [6] - log_iqr: 容量曲线四分位距的对数
        
        e_out : float
            放电能量 (Wh)
            计算方式: ∫ power dt
        
        d_rate : float
            放电速率 (A)
            取放电过程最后10个点的平均电流
        
        cap_curve : ndarray, shape (1000,)
            插值后的容量-电压曲线
            在电压范围内均匀采样1000个点
    
    如果文件不存在:
        返回 None
    
    数据处理步骤:
    -----------
    1. 读取GCPL文件（制表符分隔）
    2. 验证列数是否正确（应为7列）
    3. 筛选放电数据（电流<0且ox/red=0）
    4. 提取电压和容量数据
    5. 拟合S型曲线获取特征参数
    6. 插值生成1000点的容量-电压曲线
    7. 计算容量曲线相对基线的变化（如果cycle>0）
    8. 提取起始和终止电压
    9. 计算放电速率
    10. 计算放电能量（功率对时间积分）
    
    S型曲线拟合:
    ----------
    使用通用S型函数拟合容量-电压关系:
        capacity = a / (1 + exp(b*(v-c)))
    
    其中:
    - a: 最大容量（渐近值）
    - b: 曲线陡峭程度
    - c: 中点电压（50%容量对应的电压）
    
    容量曲线变化特征:
    -------------
    对于cycle>0的情况:
    - 计算当前曲线与基线曲线的差值
    - 如果方差很小（<0.1），设置为-1.0（表示无显著变化）
    - 否则计算方差和四分位距的对数
    
    这些特征可以反映电池老化导致的放电曲线形状变化
    
    文件格式:
    --------
    GCPL文件包含以下列（制表符分隔）:
    - time: 时间 (秒)
    - ewe: 工作电极电压 (V)
    - i: 电流 (A)，负值表示放电
    - capacity: 容量 (Ah)
    - power: 功率 (W)
    - ox/red: 氧化还原标志
    - unnamed: 未命名列
    
    注意事项:
    --------
    - 如果文件列数不正确，会等待120秒后重试
    - 如果S型曲线拟合失败，使用初始猜测值
    - 放电能量使用Simpson积分法计算
    - 函数会自动处理数据异常情况
    
    示例:
    ----
    >>> ptf = 'raw-data/variable-discharge/PJ097/PJ097_001_04_GCPL_CA1.txt'
    >>> cap, features, e_out, d_rate, cap_curve = discharge_features(ptf, cycle=1)
    >>> print(f'放电容量: {cap:.3f} Ah')
    >>> print(f'放电能量: {e_out:.3f} Wh')
    >>> print(f'放电速率: {d_rate:.3f} A')
    >>> print(f'S型参数: a={features[0]:.3f}, b={features[1]:.3f}, c={features[2]:.3f}')
    """
    
    # 函数实现...
```

#### 3. `eis_features()` 函数

```python
def eis_features(path, new_log_freq=np.linspace(-1.66, 3.9, 100), n_repeats=1):
    """
    从GEIS文件中提取电化学阻抗谱（EIS）特征
    
    功能说明:
    --------
    读取GEIS（电化学阻抗谱）数据文件，通过插值处理生成
    固定频率点的阻抗特征，用于机器学习模型输入。
    
    EIS原理:
    -------
    电化学阻抗谱是通过在不同频率下施加小幅度交流信号，
    测量电池的阻抗响应。阻抗是复数，包含实部和虚部:
    - 实部(Re(Z)): 与电阻相关
    - 虚部(Im(Z)): 与电容/电感相关
    
    EIS可以反映电池的内部状态，如:
    - 欧姆电阻
    - 电荷转移电阻
    - 双电层电容
    - 扩散阻抗
    
    参数:
    ----
    path : str
        GEIS文件的完整路径
        例如: 'raw-data/variable-discharge/PJ097/PJ097_001_01_GEIS_CA1.txt'
    
    new_log_freq : ndarray, optional
        目标对数频率点数组
        默认: np.linspace(-1.66, 3.9, 100)
        对应频率范围: 10^(-1.66) 到 10^(3.9) Hz
                      约 0.022 Hz 到 7943 Hz
        
        使用对数频率的原因:
        - EIS测量通常在对数频率范围内进行
        - 对数刻度可以更好地覆盖宽频率范围
        - 低频和高频信息都很重要
    
    n_repeats : int, optional
        重复测量次数，默认为1
        如果同一频率点测量多次，会取平均值
        用于降低测量噪声
    
    返回:
    ----
    features : ndarray, shape (200,)
        EIS特征向量，包含:
        [0:100]   - 100个频率点的阻抗实部 Re(Z)
        [100:200] - 100个频率点的阻抗虚部 Im(Z)
    
    数据处理流程:
    -----------
    1. 读取GEIS文件（制表符分隔）
    2. 提取频率、阻抗实部、阻抗虚部
    3. 如果有重复测量，reshape并取平均
    4. 将频率转换为对数频率: log10(freq)
    5. 使用三次样条插值到目标频率点
    6. 合并实部和虚部为单一特征向量
    
    插值方法:
    --------
    使用scipy.interpolate.interp1d的三次插值:
    - kind='cubic': 三次样条插值
    - 优点: 平滑、连续、保持曲线特征
    - 适用于: EIS曲线通常是平滑的
    
    文件格式:
    --------
    GEIS文件包含以下列（制表符分隔）:
    - freq: 频率 (Hz)
    - re_z: 阻抗实部 (Ω)
    - -im_z: 阻抗虚部的负值 (Ω)
    - time: 测量时间 (秒)
    - unnamed: 未命名列
    
    注意: 文件中存储的是-Im(Z)，即虚部的负值
    
    频率范围说明:
    -----------
    典型的EIS测量频率范围:
    - 高频 (>1000 Hz): 主要反映欧姆电阻
    - 中频 (1-1000 Hz): 反映电荷转移过程
    - 低频 (<1 Hz): 反映扩散过程
    
    特征维度选择:
    -----------
    选择100个频率点的原因:
    - 足够捕捉EIS曲线的主要特征
    - 不会导致特征维度过高
    - 平衡信息量和计算效率
    
    应用示例:
    --------
    EIS特征可用于:
    - 电池健康状态(SOH)估计
    - 剩余使用寿命(RUL)预测
    - 容量衰减预测
    - 故障诊断
    
    注意事项:
    --------
    - 确保目标频率范围在原始测量范围内
    - 插值外推可能导致不准确的结果
    - 如果原始数据质量差，插值也会受影响
    
    示例:
    ----
    >>> path = 'raw-data/variable-discharge/PJ097/PJ097_001_01_GEIS_CA1.txt'
    >>> eis = eis_features(path)
    >>> print(f'EIS特征维度: {eis.shape}')  # (200,)
    >>> re_z = eis[:100]   # 阻抗实部
    >>> im_z = eis[100:]   # 阻抗虚部
    >>> 
    >>> # 可视化Nyquist图（阻抗复平面图）
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(re_z, -im_z, 'o-')
    >>> plt.xlabel('Re(Z) [Ω]')
    >>> plt.ylabel('-Im(Z) [Ω]')
    >>> plt.title('Nyquist Plot')
    >>> plt.axis('equal')
    >>> plt.grid(True)
    >>> plt.show()
    """
    
    # 函数实现...
```

---

## 模型类注释

### `utils/models.py` - XGBModel类

```python
class XGBModel:
    """
    XGBoost模型封装类，用于电池容量预测
    
    功能概述:
    --------
    这个类封装了XGBoost回归模型的训练、预测和评估流程，
    专门针对电池容量预测任务进行了优化。
    
    主要特点:
    --------
    1. 集成学习: 训练多个模型并平均预测结果
    2. Bootstrap采样: 每个模型使用不同的训练子集
    3. 交叉验证: 按电池分组进行交叉验证
    4. 不确定性量化: 提供预测均值和标准差
    5. 自动保存: 模型和预测结果自动保存
    
    属性:
    ----
    X : ndarray, shape (n_samples, n_features)
        输入特征矩阵
    
    y : ndarray, shape (n_samples,)
        目标变量（放电容量）
    
    cell_idx : ndarray, shape (n_samples,)
        每个样本对应的电池标识
        用于分组交叉验证
    
    experiment : str
        实验类型（如'variable-discharge'）
        用于确定保存路径
    
    experiment_name : str
        实验名称（如'eis-actions_n1_xgb'）
        用于文件命名
    
    n_ensembles : int
        集成模型数量，默认10
        更多模型可以降低预测方差
    
    n_splits : int
        交叉验证折数，默认12
        更多折数可以更好评估泛化性能
    
    max_depth : int
        XGBoost树的最大深度，默认100
        控制模型复杂度
    
    n_estimators : int
        XGBoost中树的数量，默认500
        更多树通常提高准确度
    
    n_split : int
        当前交叉验证折的索引
    
    start_seed : int
        随机种子起始值，默认0
        用于可重复性
    
    方法:
    ----
    train_no_predict(X, y)
        训练模型但不进行预测
        用于：仅需要保存训练好的模型
    
    train_and_predict(X_train, y_train, X_test1, cell_test1, ...)
        训练模型并对测试集进行预测
        用于：需要立即获得预测结果
    
    analysis(log_name, experiment_info)
        完整的分析流程，包括交叉验证
        用于：全面评估模型性能
    
    analysis_vd2(log_name, experiment_info)
        针对vd2数据集的分析流程
        用于：化学体系2数据的特殊处理
    
    使用示例:
    --------
    >>> # 1. 准备数据
    >>> from utils.exp_util import extract_data, extract_input
    >>> cell_idx, cap_ds, data = extract_data('variable-discharge', [1,2,3,4,5,6,7,8])
    >>> x = extract_input('eis-actions', data)
    >>> y = cap_ds
    >>> 
    >>> # 2. 创建模型
    >>> model = XGBModel(
    ...     x, y, cell_idx,
    ...     experiment='variable-discharge',
    ...     experiment_name='eis-actions_n1_xgb',
    ...     n_ensembles=10,
    ...     n_splits=12,
    ...     max_depth=100,
    ...     n_estimators=500
    ... )
    >>> 
    >>> # 3. 训练模型
    >>> model.train_no_predict(x, y)
    >>> 
    >>> # 4. 或者进行完整分析
    >>> model.analysis('log.txt', 'Experiment info')
    
    集成学习原理:
    -----------
    1. 对于每个集成模型:
       - 使用Bootstrap采样创建训练子集
       - 训练一个XGBoost模型
       - 保存模型
    
    2. 预测时:
       - 所有模型分别进行预测
       - 计算预测均值作为最终预测
       - 计算预测标准差作为不确定性度量
    
    3. 优势:
       - 降低过拟合风险
       - 提供不确定性估计
       - 提高预测稳定性
    
    交叉验证策略:
    -----------
    使用"留一电池出"(Leave-One-Battery-Out)交叉验证:
    
    1. 将所有电池分成n_splits组
    2. 每次:
       - 选择一组电池作为测试集
       - 其余电池作为训练集
       - 训练模型并预测测试集
    3. 汇总所有折的结果
    
    优势:
    - 评估模型对新电池的泛化能力
    - 避免数据泄露（同一电池的数据不会同时出现在训练和测试集）
    - 更真实地反映实际应用场景
    
    文件保存结构:
    -----------
    experiments/results/{experiment}/
    ├── models/
    │   ├── {experiment_name}_0.pkl
    │   ├── {experiment_name}_1.pkl
    │   └── ...
    ├── predictions/
    │   ├── pred_mn_{experiment_name}_{cell}.npy
    │   ├── pred_std_{experiment_name}_{cell}.npy
    │   └── true_{experiment_name}_{cell}.npy
    └── log-*.txt
    
    性能指标:
    --------
    模型评估使用以下指标:
    
    1. R² (决定系数):
       - 范围: (-∞, 1]
       - 1表示完美预测
       - 0表示与均值预测相当
       - 负值表示比均值预测还差
    
    2. RMSE (均方根误差):
       - 单位: Ah（与容量相同）
       - 越小越好
       - 对大误差敏感
    
    3. MAE (平均绝对误差):
       - 单位: Ah
       - 越小越好
       - 对异常值不敏感
    
    注意事项:
    --------
    1. 确保有足够的内存存储所有集成模型
    2. 训练时间与n_ensembles和n_estimators成正比
    3. 交叉验证折数不应超过电池数量
    4. Bootstrap采样使用90%的训练数据
    
    调参建议:
    --------
    1. 快速原型:
       - n_ensembles=2, n_estimators=50, max_depth=50
    
    2. 平衡性能:
       - n_ensembles=10, n_estimators=500, max_depth=100
    
    3. 最佳性能:
       - n_ensembles=20, n_estimators=1000, max_depth=150
    
    常见问题:
    --------
    Q: 为什么使用Bootstrap采样？
    A: 增加训练数据的多样性，降低过拟合风险
    
    Q: 如何选择n_ensembles？
    A: 通常10-20个足够，更多会增加训练时间但提升有限
    
    Q: 预测标准差如何解释？
    A: 标准差大表示模型不确定性高，预测可能不可靠
    
    Q: 如何处理过拟合？
    A: 减小max_depth，增加n_ensembles，使用更多训练数据
    """
    
    def __init__(self, X, y, cell_idx, experiment, experiment_name, 
                 n_ensembles=10, n_splits=12, max_depth=100, n_estimators=500):
        """
        初始化XGBoost模型
        
        参数说明见类文档字符串
        """
        # 初始化代码...
```

---

## 关键概念解释

### 1. 电化学阻抗谱（EIS）

**物理意义**:
- 通过施加不同频率的交流信号测量电池的阻抗响应
- 可以分离不同时间尺度的电化学过程

**频率与过程对应**:
```
高频 (>1 kHz)  → 欧姆电阻 (电解液、电极)
中频 (1-1000 Hz) → 电荷转移 (电化学反应)
低频 (<1 Hz)    → 扩散过程 (锂离子扩散)
```

**数据表示**:
- Nyquist图: 实部 vs 虚部
- Bode图: 幅值和相位 vs 频率

### 2. 容量-电压曲线（C-V Curve）

**物理意义**:
- 描述放电过程中容量与电压的关系
- 反映电池的能量释放特性

**特征提取**:
```python
# S型曲线拟合
capacity = a / (1 + exp(b*(v-c)))

参数意义:
- a: 最大容量
- b: 曲线陡峭程度（反映极化）
- c: 中点电压（50%容量时的电压）
```

**老化表现**:
- 曲线整体下移 → 容量衰减
- 曲线变陡 → 内阻增大
- 中点电压降低 → 活性物质损失

### 3. 充放电协议

**两阶段充电**:
```
阶段1: 恒流充电 (CC)
  - 以恒定电流充电
  - 直到达到上限电压

阶段2: 恒压充电 (CV)
  - 保持上限电压
  - 电流逐渐减小
  - 直到电流低于截止值
```

**放电**:
```
恒流放电 (CC)
  - 以恒定电流放电
  - 直到达到下限电压
```

### 4. 特征工程

**为什么需要多种特征**:
```
EIS特征:
  ✓ 反映内部状态
  ✓ 对早期老化敏感
  ✗ 测量耗时

C-V曲线:
  ✓ 反映整体性能
  ✓ 易于测量
  ✗ 对早期老化不敏感

充放电速率:
  ✓ 直接影响容量
  ✓ 反映使用条件
  ✗ 信息量有限

组合使用效果最佳！
```

---

## 数据流程图

```
原始数据文件 (.txt)
    ↓
[extract_data()]
    ↓
原始特征
  ├─ EIS (200维)
  ├─ C-V曲线 (1000维)
  ├─ 充放电速率 (3维)
  ├─ 容量历史
  └─ SOH等
    ↓
[extract_input()]
    ↓
选择特征组合
  ├─ eis-actions
  ├─ cvfs-actions
  ├─ eis-cvfs-actions
  └─ eis-cvfs-ct-c-actions
    ↓
[XGBModel]
    ↓
训练集成模型
  ├─ Bootstrap采样
  ├─ 训练XGBoost
  └─ 保存模型
    ↓
预测
  ├─ 多模型预测
  ├─ 计算均值
  └─ 计算标准差
    ↓
评估
  ├─ R²
  ├─ RMSE
  └─ MAE
    ↓
保存结果
  ├─ 模型文件 (.pkl)
  ├─ 预测结果 (.npy)
  └─ 日志文件 (.txt)
```

---

## 完整工作流程示例

```python
# ============================================================================
# 完整的电池容量预测工作流程
# ============================================================================

# 步骤1: 导入必要的库
# ============================================================================
import sys
sys.path.append('.')
import numpy as np
from utils.exp_util import extract_data, extract_input
from utils.models import XGBModel

# 步骤2: 配置参数
# ============================================================================
experiment = 'variable-discharge'  # 实验类型
channels = [1, 2, 3, 4, 5, 6, 7, 8]  # 所有通道
input_name = 'eis-actions'  # 特征组合

# 模型参数
params = {
    'max_depth': 100,
    'n_splits': 12,
    'n_estimators': 500,
    'n_ensembles': 10
}

# 步骤3: 提取数据
# ============================================================================
print('正在提取数据...')
cell_idx, cap_ds, data = extract_data(experiment, channels)
print(f'数据提取完成！')
print(f'  - 总样本数: {len(cap_ds)}')
print(f'  - 电池数量: {len(np.unique(cell_idx))}')
print(f'  - 平均容量: {np.mean(cap_ds):.3f} Ah')

# 步骤4: 提取特征
# ============================================================================
print(f'\n正在提取特征: {input_name}')
x = extract_input(input_name, data)
y = cap_ds
print(f'特征提取完成！')
print(f'  - 特征维度: {x.shape[1]}')
print(f'  - 样本数量: {x.shape[0]}')

# 步骤5: 创建模型
# ============================================================================
print('\n正在创建模型...')
model = XGBModel(
    x, y, cell_idx,
    experiment=experiment,
    experiment_name=f'{input_name}_n1_xgb',
    n_ensembles=params['n_ensembles'],
    n_splits=params['n_splits'],
    max_depth=params['max_depth'],
    n_estimators=params['n_estimators']
)
print('模型创建完成！')

# 步骤6: 训练模型
# ============================================================================
print('\n开始训练...')
import time
t0 = time.time()
model.train_no_predict(x, y)
elapsed = time.time() - t0
print(f'训练完成！耗时: {elapsed:.2f}秒')

# 步骤7: 进行完整分析（可选）
# ============================================================================
print('\n开始完整分析（包括交叉验证）...')
log_name = f'experiments/results/{experiment}/log-analysis.txt'
experiment_info = f'''
实验配置:
  - 特征: {input_name}
  - 树深度: {params['max_depth']}
  - 树数量: {params['n_estimators']}
  - 集成数: {params['n_ensembles']}
  - 交叉验证折数: {params['n_splits']}
'''
model.analysis(log_name, experiment_info)
print('分析完成！')

# 步骤8: 读取和评估结果
# ============================================================================
print('\n评估结果:')
for cell in np.unique(cell_idx)[:3]:  # 只显示前3个电池
    pred_path = f'experiments/results/{experiment}/predictions/pred_mn_{input_name}_n1_xgb_{cell}.npy'
    true_path = f'experiments/results/{experiment}/predictions/true_{input_name}_n1_xgb_{cell}.npy'
    
    try:
        pred = np.load(pred_path)
        true = np.load(true_path)
        
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        
        print(f'\n{cell}:')
        print(f'  - R² = {r2:.4f}')
        print(f'  - RMSE = {rmse:.4f} Ah')
        print(f'  - 相对误差 = {rmse/np.mean(true)*100:.2f}%')
    except:
        print(f'\n{cell}: 结果文件未找到')

print('\n全部完成！')
```

---

## 总结

本文档提供了项目中所有关键代码的详细中文注释，包括:

1. **实验脚本**: 如何配置和运行实验
2. **数据提取**: 如何从原始文件提取特征
3. **模型训练**: 如何训练和评估模型
4. **结果分析**: 如何解读和使用结果

配合 `README_CN.md` 使用，可以全面理解项目的工作原理和使用方法。

