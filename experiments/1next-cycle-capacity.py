# ============================================================================
# 文件名: 1next-cycle-capacity.py
# 功能: 使用交叉验证评估模型性能，预测下一循环容量
# 方法: 留一电池出(Leave-One-Battery-Out)交叉验证
# ============================================================================

import sys
sys.path.append('.')  # 将当前目录添加到Python路径
import time  # 用于计算训练耗时
import pickle  # 序列化模块（未使用但保留）

# 从工具模块导入数据提取和模型类
from utils.exp_util import extract_data, extract_input
from utils.models import XGBModel

import pdb  # Python调试器（用于开发调试）

# ============================================================================
# 配置参数
# ============================================================================

# 定义要处理的测试通道列表
channels = [1, 2, 3, 4, 5, 6, 7, 8]

# 模型超参数配置字典
params = {
    'max_depth': 100,       # XGBoost决策树的最大深度
    'n_splits': 12,         # K折交叉验证的折数
    'n_estimators': 500,    # XGBoost中树的数量
    'n_ensembles': 10       # 集成学习的模型数量
}

# 指定实验类型
experiment = 'variable-discharge'

# 定义日志文件路径
# 日志文件将记录每个电池的预测性能指标（R²、RMSE、MAE等）
log_name = 'experiments/results/{}/log-next-cycle-12s.txt'.format(experiment)

# ============================================================================
# 特征选择
# ============================================================================

# 定义要测试的输入特征组合列表
# 注释掉的是其他可选的特征组合，可以根据需要启用
# input_names = ['eis-actions', 'ecmer-actions', 'ecmr-actions', 'cvfs-actions', 
#                'eis-cvfs-actions', 'ecmer-cvfs-actions', 'ecmr-cvfs-actions']

# 当前使用的特征组合（完整特征集）:
# - 'ecmr-cvfs-ct-c-actions': Randles等效电路模型 + C-V曲线 + 累积吞吐量 + 上一容量 + 速率
# - 'ecmer-cvfs-ct-c-actions': 扩展Randles模型 + C-V曲线 + 累积吞吐量 + 上一容量 + 速率
# - 'eis-cvfs-ct-c-actions': EIS + C-V曲线 + 累积吞吐量 + 上一容量 + 速率
input_names = ['ecmr-cvfs-ct-c-actions', 'ecmer-cvfs-ct-c-actions', 'eis-cvfs-ct-c-actions']

# ============================================================================
# 数据提取
# ============================================================================

# 从原始数据文件中提取可变放电数据集
# 返回值:
#   cell_var: 电池标识数组，用于分组交叉验证
#   cap_ds_var: 放电容量数组（目标变量）
#   data_var: 特征数据元组，包含9个元素
cell_var, cap_ds_var, data_var = extract_data(experiment, channels)

# data_var元组包含以下9个元素:
# (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates)
# - last_caps: 上一循环的放电容量
# - sohs: 健康状态(State of Health)
# - eis_ds: 放电时的EIS特征
# - cvfs: 容量-电压曲线特征
# - ocvs: 开路电压
# - cap_throughputs: 累积吞吐量
# - d_rates: 放电速率
# - c1_rates: 第一阶段充电速率
# - c2_rates: 第二阶段充电速率

# 打印数据集信息
print('Number of datapoints = {}'.format(data_var[0].shape[0]))

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
    
    # 构建实验名称，用于保存模型和结果文件
    # 格式: {特征名}_n1_xgb
    # n1表示预测下一个循环(next 1 cycle)
    experiment_name = '{}_n1_xgb'.format(input_name)
    
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
    # 根据input_name的不同，会提取不同的特征组合
    # 返回: x - 输入特征矩阵，shape (n_samples, n_features)
    x = extract_input(input_name, data_var)

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
    # 1. 按电池分组进行交叉验证
    #    - 每次留出一个电池作为测试集
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
#    experiments/results/variable-discharge/models/
#    ├── {experiment_name}_0.pkl
#    ├── {experiment_name}_1.pkl
#    └── ... (每个特征组合n_ensembles个模型)
# 
# 2. 预测结果:
#    experiments/results/variable-discharge/predictions/
#    ├── pred_mn_{experiment_name}_{cell}.npy    # 预测均值
#    ├── pred_std_{experiment_name}_{cell}.npy   # 预测标准差
#    └── true_{experiment_name}_{cell}.npy       # 真实值
# 
# 3. 日志文件:
#    experiments/results/variable-discharge/log-next-cycle-12s.txt
#    包含每个电池的性能指标:
#    - R² (决定系数)
#    - RMSE (均方根误差)
#    - MAE (平均绝对误差)
#    - 相对误差百分比
