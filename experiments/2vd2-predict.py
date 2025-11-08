# ============================================================================
# 文件名: 2vd2-predict.py
# 功能: 使用25°C训练的模型预测35°C数据（跨温度预测）
# 训练集: chemistry2-25C (PJ248-PJ279)
# 测试集: chemistry2-35C (PJ296-PJ311)
# 目的: 测试模型的跨温度泛化能力
# ============================================================================

import sys
sys.path.append('.')  # 将当前目录添加到Python路径
import time  # 时间模块（未使用但保留）
import pickle  # 序列化模块（未使用但保留）

import numpy as np  # 数值计算库，用于数组操作和保存结果

# 从工具模块导入数据提取和集成预测函数
from utils.exp_util_new import extract_data_type2, extract_input  # 新数据集提取工具
from utils.exp_util import ensemble_predict  # 集成预测函数（从标准工具模块导入）

import pdb  # Python调试器（用于开发调试）

# ============================================================================
# 配置参数
# ============================================================================

# 定义要处理的测试通道列表
# 使用字母数字组合命名：A1-A8, B1-B8
channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',]

# 指定使用的特征组合
# 'eis-actions': EIS特征(200维) + 充放电速率(3维)
input_name = 'eis-actions'

# ============================================================================
# 实验配置
# ============================================================================

# 测试集实验类型：35°C数据
# 对应 raw-data/chemistry2-35C/ 目录（PJ296-PJ311）
exp_test = 'vd2-35C'

# 训练集实验类型：标准可变放电数据
# 模型是用 raw-data/variable-discharge/ 数据训练的
# 注意: 这里使用的是标准数据集训练的模型，而非chemistry2-25C训练的模型
# 这是为了测试跨数据集和跨温度的泛化能力
exp_train = 'variable-discharge'

# ============================================================================
# 数据提取
# ============================================================================

# 提取35°C测试数据
# suffix='vd2': 指定数据集后缀，用于区分不同数据集的处理方式
# 返回值:
#   cells: 电池标识数组
#   cap_ds: 放电容量数组（真实值，用于后续对比）
#   data: 特征数据元组（包含10个元素）
cells, cap_ds, data = extract_data_type2(exp_test, channels, suffix='vd2')

# ============================================================================
# 特征提取
# ============================================================================

# 从35°C数据中提取与训练时相同的特征
# 必须使用与训练模型时相同的特征组合，否则预测会失败
# suffix='vd2': 指定数据集后缀
x = extract_input(input_name, data, suffix='vd2')

# ============================================================================
# 集成预测
# ============================================================================

# 使用集成模型进行预测
# 参数说明:
#   x: 35°C数据的特征矩阵
#   exp_train: 模型来源实验（'variable-discharge'）
#              模型是用标准可变放电数据训练的
#   input_name: 特征名称，用于定位模型文件
# 
# 函数会自动:
#   1. 加载 experiments/results/variable-discharge/models/ 中的所有模型
#   2. 每个模型分别进行预测
#   3. 计算所有模型预测的均值作为最终预测
#   4. 计算所有模型预测的标准差作为不确定性度量
# 
# 返回值:
#   pred: 预测均值，shape (n_samples,)
#   pred_err: 预测标准差，shape (n_samples,)
# 
# 注意: 这是一个跨温度预测实验
#       训练数据: 常温可变放电
#       测试数据: 35°C化学体系2
pred, pred_err = ensemble_predict(x, exp_train, input_name)

# ============================================================================
# 保存预测结果
# ============================================================================

# 定义结果保存目录
dts = 'experiments/results/{}'.format(exp_test)

# 保存预测均值
# 文件名格式: vd1_pred_mn_{特征名}.npy
# vd1表示使用variable-discharge（vd1）训练的模型
np.save('{}/predictions/vd1_pred_mn_{}.npy'.format(dts, input_name), pred)

# 保存预测标准差
# 文件名格式: vd1_pred_std_{特征名}.npy
# 用途: 不确定性分析，标准差大表示预测不确定性高
np.save('{}/predictions/vd1_pred_std_{}.npy'.format(dts, input_name), pred_err)

# 保存真实值
# 文件名格式: vd1_true_{特征名}.npy
# 用途: 与预测值对比，计算评估指标（R²、RMSE等）
np.save('{}/predictions/vd1_true_{}.npy'.format(dts, input_name), cap_ds)

# ============================================================================
# 完成
# ============================================================================
# 预测结果已保存到 experiments/results/vd2-35C/predictions/
# 
# 可以使用以下代码读取和分析结果:
#
# import numpy as np
# from sklearn.metrics import r2_score, mean_squared_error
# 
# pred = np.load('experiments/results/vd2-35C/predictions/vd1_pred_mn_eis-actions.npy')
# true = np.load('experiments/results/vd2-35C/predictions/vd1_true_eis-actions.npy')
# 
# r2 = r2_score(true, pred)
# rmse = np.sqrt(mean_squared_error(true, pred))
# print(f'跨温度预测性能:')
# print(f'R² = {r2:.4f}')
# print(f'RMSE = {rmse:.4f} Ah')
# 
# 预期结果:
# - R²可能略低于同温度预测（因为温度差异）
# - 但如果R²仍然较高（>0.85），说明模型具有良好的温度泛化能力
# 
# 应用价值:
# - 验证模型在不同温度条件下的适用性
# - 为实际应用中的温度变化提供性能参考
# - 评估模型的鲁棒性
