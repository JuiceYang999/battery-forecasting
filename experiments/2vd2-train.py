# ============================================================================
# 文件名: 2vd2-train.py
# 功能: 训练化学体系2（25°C）电池数据的容量预测模型
# 数据集: PJ248-PJ279 (32个电池，chemistry2-25C)
# 输出: 训练好的XGBoost集成模型
# 与1variable-discharge-train.py的区别:
#   - 使用不同的数据集（chemistry2而非variable-discharge）
#   - 使用exp_util_new.py而非exp_util.py
#   - 支持不同温度条件的数据
# ============================================================================

import sys
sys.path.append('.')  # 将项目根目录添加到Python路径，以便导入utils模块

import time  # 用于计算训练耗时
import pickle  # 用于序列化模型（虽然此脚本未直接使用，但保留以备用）

# 导入新数据集的提取工具和模型类
from utils.exp_util_new import extract_data_type2, extract_input  # 新数据集提取函数
from utils.models import XGBModel  # XGBoost模型封装类

import pdb  # Python调试器（用于开发时调试）

# ============================================================================
# 配置参数
# ============================================================================

# 初始通道定义（后面会被覆盖，保留以示原始设计）
channels = [1, 2, 3, 4, 5, 6, 7, 8]

# 模型超参数配置字典
params = {
    'max_depth': 100,       # XGBoost决策树的最大深度
                           # 控制模型复杂度，值越大模型越复杂
    'n_splits': 16,        # K折交叉验证的折数（比标准数据集多，因为电池数更多）
                           # 用于评估模型的泛化性能
    'n_estimators': 50,    # XGBoost中树的数量（比标准数据集少，用于快速训练）
                           # 可根据需要调整
    'n_ensembles': 10      # 集成学习的模型数量
                           # 多个模型的预测结果取平均，可降低预测方差
}

# 指定实验类型（对应chemistry2-25C数据）
experiment = 'variable-discharge-type2'

# 重新定义通道列表（使用字母数字组合命名）
# A1-A8: 第一批8个通道
# B1-B8: 第二批8个通道
# 总共16个通道，对应32个电池（每个通道可能对应多个电池）
channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',]

# ============================================================================
# 特征选择
# ============================================================================

# 定义要测试的输入特征组合列表
# 'eis-cvfs-actions': EIS特征(200维) + 容量-电压曲线(1000维) + 充放电速率(3维) = 1203维
# 'eis-actions': EIS特征(200维) + 充放电速率(3维) = 203维
# 
# 注意: 这里测试两种特征组合，以比较不同特征对预测性能的影响
input_names = ['eis-cvfs-actions', 'eis-actions']

# ============================================================================
# 数据提取
# ============================================================================

# 从原始数据文件中提取化学体系2的数据
# 使用extract_data_type2函数，专门处理chemistry2数据格式
# 返回值:
#   cell_var: 电池标识数组，shape (n_samples,)
#             例如: ['PJ248', 'PJ248', ..., 'PJ249', ...]
#   cap_ds_var: 放电容量数组（目标变量），shape (n_samples,)
#   data_var: 特征数据元组，包含10个元素:
#             (last_caps, sohs, eis_ds, cvfs, ocvs, 
#              cap_throughputs, d_rates, c1_rates, c2_rates, v_maxs)
#             注意: 比标准数据集多了v_maxs（最大电压）
cell_var, cap_ds_var, data_var = extract_data_type2(experiment, channels)

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
    # 格式: {特征名}_n1_xgb2
    # n1表示预测下一个循环(next 1 cycle)
    # xgb2表示XGBoost模型版本2（用于区分不同数据集的模型）
    experiment_name = '{}_n1_xgb2'.format(input_name)
    
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
    #      Bootstrap采样: 从原始数据中有放回地随机抽样，每次抽取90%的数据
    #   2. 对每个子集训练一个XGBoost模型
    #      XGBoost: 梯度提升决策树，通过迭代方式构建多个决策树
    #   3. 将所有模型保存到 experiments/results/{experiment}/models/
    # 
    # 模型文件命名格式: {experiment_name}_{i}.pkl
    # 例如: eis-actions_n1_xgb2_0.pkl, eis-actions_n1_xgb2_1.pkl, ...
    # 
    # 集成学习的优势:
    #   - 降低过拟合风险（不同模型可能在不同数据上过拟合）
    #   - 提供不确定性估计（通过预测方差）
    #   - 提高预测稳定性（平均多个模型的预测）
    regressor.train_no_predict(x, y)
    
    # 计算并打印训练耗时
    elapsed_time = time.time() - t0
    print('Time taken = {:.2f}'.format(elapsed_time))

# ============================================================================
# 训练完成
# ============================================================================

print('Done.')  # 所有特征组合的模型训练完成

# ============================================================================
# 后续步骤（需要运行其他脚本）:
# ============================================================================
# 1. 运行 2next-cycle-capacity-vd2.py 进行交叉验证和性能评估
# 2. 运行 2vd2-predict.py 对测试集进行预测
# 3. 分析预测结果，查看 experiments/results/{experiment}/predictions/
# 4. 查看日志文件，了解模型性能指标（R²、RMSE、MAE等）
