# ============================================================================
# 文件名: 1variable-discharge-train.py
# 功能: 训练可变放电数据集的电池容量预测模型
# 数据集: PJ097-PJ152 (24个电池)
# 输出: 训练好的XGBoost集成模型
# ============================================================================

import sys
sys.path.append('.')  # 将项目根目录添加到Python路径，以便导入utils模块

import time  # 用于计算训练耗时
import pickle  # 用于序列化模型（虽然此脚本未直接使用，但保留以备用）

# 导入数据提取和模型训练工具
from utils.exp_util import extract_data, extract_input  # 数据提取函数
from utils.models import XGBModel  # XGBoost模型封装类

import pdb  # Python调试器（用于开发时调试）

# ============================================================================
# 配置参数
# ============================================================================

# 定义要处理的测试通道列表（每个通道对应多个电池）
channels = [1, 2, 3, 4, 5, 6, 7, 8]

# 模型超参数配置字典
params = {
    'max_depth': 100,       # XGBoost决策树的最大深度
                           # 控制模型复杂度，值越大模型越复杂，但可能过拟合
    'n_splits': 12,        # K折交叉验证的折数
                           # 用于评估模型的泛化性能
    'n_estimators': 500,   # XGBoost中树的数量
                           # 更多的树通常提高准确度，但增加训练时间
    'n_ensembles': 10      # 集成学习的模型数量
                           # 多个模型的预测结果取平均，可降低预测方差
}

# 指定实验类型（对应raw-data目录下的子目录名）
experiment = 'variable-discharge'

# ============================================================================
# 特征选择
# ============================================================================

# 定义要测试的输入特征组合列表
# 'eis-actions': EIS特征(200维) + 充放电速率(3维) = 203维
# 可选的其他特征组合:
#   'cvfs-actions': 容量-电压曲线(1000维) + 充放电速率(3维)
#   'eis-cvfs-actions': EIS + C-V曲线 + 充放电速率
#   'eis-cvfs-ct-c-actions': EIS + C-V曲线 + 累积吞吐量 + 上一容量 + 充放电速率
input_names = ['eis-actions', ]

# ============================================================================
# 数据提取
# ============================================================================

# 从原始数据文件中提取可变放电数据集
# 返回值:
#   cell_var: 电池标识数组，shape (n_samples,)
#   cap_ds_var: 放电容量数组（目标变量），shape (n_samples,)
#   data_var: 特征数据元组，包含9个元素:
#             (last_caps, sohs, eis_ds, cvfs, ocvs, 
#              cap_throughputs, d_rates, c1_rates, c2_rates)
cell_var, cap_ds_var, data_var = extract_data(experiment, channels)

# 打印数据集信息
# data_var[0]是last_caps（上一循环容量），其shape[0]即为总样本数
print('Number of datapoints = {}'.format(data_var[0].shape[0]))

# ============================================================================
# 定义目标变量
# ============================================================================

# 目标变量: 下一循环的放电容量 (Ah)
# 这是我们要预测的值
y = cap_ds_var

# ============================================================================
# 模型训练循环
# ============================================================================

# 遍历每个特征组合，分别训练模型
for i in range(len(input_names)):
    # 获取当前特征组合名称
    input_name = input_names[i]
    
    # 构建实验名称，用于保存模型文件
    # 格式: {特征名}_n1_xgb
    # n1表示预测下一个循环(next 1 cycle)
    experiment_name = '{}_n1_xgb'.format(input_name)
    
    # 构建实验信息字符串，用于日志记录
    experiment_info = '\nInput: {} \tOutput: Q_n+1 \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(
        input_name,                # 输入特征名称
        params['max_depth'],       # 树的最大深度
        params['n_estimators'],    # 树的数量
        params['n_ensembles'],     # 集成模型数量
        params['n_splits']         # 交叉验证折数
    )
    print(experiment_info)
    
    # 记录训练开始时间
    t0 = time.time()

    # ------------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------------
    
    # 从data_var中提取指定的输入特征
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
    #   experiment: 实验类型，决定模型保存路径
    #   experiment_name: 实验名称，用于模型文件命名
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
    # 模型训练
    # ------------------------------------------------------------------------
    
    # 训练模型但不进行预测
    # 这个方法会:
    #   1. 使用Bootstrap采样创建n_ensembles个训练子集
    #   2. 对每个子集训练一个XGBoost模型
    #   3. 将所有模型保存到 experiments/results/{experiment}/models/
    # 
    # 模型文件命名格式: {experiment_name}_{i}.pkl
    # 例如: eis-actions_n1_xgb_0.pkl, eis-actions_n1_xgb_1.pkl, ...
    regressor.train_no_predict(x, y)
    
    # 计算并打印训练耗时
    print('Time taken = {:.2f}'.format(time.time() - t0))

# ============================================================================
# 训练完成
# ============================================================================

print('Done.')  # 所有特征组合的模型训练完成
