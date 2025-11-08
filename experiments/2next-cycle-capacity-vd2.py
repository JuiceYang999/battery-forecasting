# ============================================================================
# 文件名: 2next-cycle-capacity-vd2.py
# 功能: 化学体系2数据（25°C）的交叉验证评估
# 数据集: PJ248-PJ279 (32个电池, chemistry2-25C)
# 方法: 随机分割交叉验证
# 说明: Variable discharge data type2 (将数据分成4批，分布不同)
# ============================================================================

import sys
sys.path.append('.')  # 将当前目录添加到Python路径
import time  # 用于计算训练耗时
import pickle  # 序列化模块（未使用但保留）
import numpy as np  # 数值计算库

# 从新数据集工具模块导入数据提取和模型类
from utils.exp_util_new import extract_data_type2, extract_input
from utils.models import XGBModel

import pdb  # Python调试器（用于开发调试）

# ============================================================================
# 配置参数
# ============================================================================

# 指定实验类型（化学体系2-25°C）
experiment = 'variable-discharge-type2'

# 定义要处理的测试通道列表
# 使用字母数字组合命名：A1-A8, B1-B8
# 总共16个通道，对应32个电池
channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',]

# 模型超参数配置字典
params = {
    'max_depth': 100,       # XGBoost决策树的最大深度
    'n_splits': 16,         # K折交叉验证的折数（比标准数据集多，因为电池更多）
    'n_estimators': 50,     # XGBoost中树的数量（比标准数据集少，用于快速训练）
    'n_ensembles': 10       # 集成学习的模型数量
}

# 定义日志文件路径
# 文件名包含'random-split-32'表示32个电池的随机分割
log_name = 'experiments/results/{}/log-next-cycle-random-split-32.txt'.format(experiment)

# ============================================================================
# 特征选择
# ============================================================================

# 定义要测试的输入特征组合列表
# 包含多种特征组合，从简单到复杂
input_names = [
    'actions',                      # 仅充放电速率
    'cvfs-actions',                 # 容量-电压曲线 + 速率
    'eis-actions',                  # EIS + 速率
    'ecmer-actions',                # 扩展Randles模型 + 速率
    'ecmr-actions',                 # Randles模型 + 速率
    'ecmer-cvfs-actions',           # 扩展Randles + C-V曲线 + 速率
    'ecmr-cvfs-actions',            # Randles + C-V曲线 + 速率
    'eis-cvfs-actions',             # EIS + C-V曲线 + 速率
    'c-actions',                    # 上一容量 + 速率
    'ecmr-cvfs-ct-c-actions',       # Randles + C-V + 吞吐量 + 上一容量 + 速率
    'ecmer-cvfs-ct-c-actions',      # 扩展Randles + C-V + 吞吐量 + 上一容量 + 速率
    'eis-cvfs-ct-c-actions'         # EIS + C-V + 吞吐量 + 上一容量 + 速率（完整特征）
]

# ============================================================================
# 数据提取
# ============================================================================

# 从原始数据文件中提取化学体系2的数据
# 使用extract_data_type2函数，专门处理chemistry2数据格式
# 返回值:
#   cell_var: 电池标识数组
#   cap_ds_var: 放电容量数组（目标变量）
#   data_var: 特征数据元组（包含10个元素，比标准数据集多v_maxs）
cell_var, cap_ds_var, data_var = extract_data_type2(experiment, channels)

# 注释掉的代码：可选的数据加载方式（从预保存的numpy文件加载）
# cell_var = np.load('cell_vd2.npy')
# cap_ds_var = np.load('cap_ds_vd2.npy')
# data_var = None

# data_var元组包含以下10个元素:
# (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates, v_maxs)
# 注意: 比标准数据集多了v_maxs（最大电压）

# 注释掉的代码：打印数据点数量
# print('Number of datapoints = {}'.format(data_var[0].shape[0]))

# ============================================================================
# 定义目标变量
# ============================================================================

# 目标变量: 下一循环的放电容量 (Ah)
y = cap_ds_var

# ============================================================================
# 模型训练和评估循环
# ============================================================================

# 遍历每个特征组合，分别训练和评估
for i in range(len(input_names)):
    # 获取当前特征组合名称
    input_name = input_names[i]
    
    # 构建实验名称
    # 格式: {特征名}_n1_xgb2
    # n1表示预测下一个循环(next 1 cycle)
    # xgb2表示XGBoost模型版本2（用于区分不同数据集的模型）
    experiment_name = '{}_n1_xgb2'.format(input_name)
    
    # 构建实验信息字符串，用于日志记录和控制台输出
    experiment_info = '\nInput: {} \tOutput: Q_n+1 \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(
        input_name,                # 输入特征名称
        params['max_depth'],       # 树的最大深度
        params['n_estimators'],    # 树的数量
        params['n_ensembles'],     # 集成模型数量
        params['n_splits']         # 交叉验证折数
    )
    print(experiment_info)
    
    # 记录开始时间
    t0 = time.time()

    # ------------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------------
    
    # 从data_var中提取指定的输入特征
    # suffix='vd2': 指定数据集后缀，用于区分不同数据集的处理方式
    #               vd2表示variable-discharge-type2
    # 返回: x - 输入特征矩阵，shape (n_samples, n_features)
    x = extract_input(input_name, data_var, suffix='vd2')

    # ------------------------------------------------------------------------
    # 模型初始化
    # ------------------------------------------------------------------------
    
    # 创建XGBoost模型对象
    # 参数说明:
    #   x: 输入特征矩阵
    #   y: 目标变量（放电容量）
    #   cell_var: 电池索引，用于分组交叉验证
    #   experiment: 实验类型，决定模型和结果保存路径
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
    
    # ------------------------------------------------------------------------
    # 完整分析流程
    # ------------------------------------------------------------------------
    
    # 执行完整的分析流程，包括:
    # 1. 按电池分组进行交叉验证（随机分割）
    #    - 每次留出一组电池作为测试集
    #    - 其余电池作为训练集
    # 2. 对每个折:
    #    - 训练n_ensembles个集成模型
    #    - 对测试电池进行预测
    #    - 计算评估指标（R²、RMSE、MAE）
    # 3. 保存结果:
    #    - 模型文件: experiments/results/{experiment}/models/
    #    - 预测结果: experiments/results/{experiment}/predictions/
    #    - 日志文件: log_name指定的文件
    # 
    # 参数说明:
    #   log_name: 日志文件路径
    #   experiment_info: 实验信息字符串，会写入日志文件
    regressor.analysis(log_name, experiment_info)
    
    # 计算并打印训练耗时
    elapsed_time = time.time() - t0
    print('Time taken = {:.2f}'.format(elapsed_time))

# ============================================================================
# 完成
# ============================================================================

print('Done.')  # 所有特征组合的训练和评估完成

# ============================================================================
# 输出文件说明
# ============================================================================
# 
# 1. 模型文件:
#    experiments/results/variable-discharge-type2/models/
#    ├── {experiment_name}_0.pkl
#    ├── {experiment_name}_1.pkl
#    └── ... (每个特征组合n_ensembles个模型)
# 
# 2. 预测结果:
#    experiments/results/variable-discharge-type2/predictions/
#    ├── pred_mn_{experiment_name}_{cell}.npy    # 预测均值
#    ├── pred_std_{experiment_name}_{cell}.npy   # 预测标准差
#    └── true_{experiment_name}_{cell}.npy       # 真实值
# 
# 3. 日志文件:
#    experiments/results/variable-discharge-type2/log-next-cycle-random-split-32.txt
#    包含每个电池和每个特征组合的性能指标:
#    - R² (决定系数)
#    - RMSE (均方根误差)
#    - MAE (平均绝对误差)
#    - 相对误差百分比
# 
# 4. 与标准数据集的区别:
#    - 电池数量更多（32个 vs 24个）
#    - 交叉验证折数更多（16折 vs 12折）
#    - 树数量更少（50 vs 500，用于快速训练）
#    - 包含最大电压特征（v_maxs）
