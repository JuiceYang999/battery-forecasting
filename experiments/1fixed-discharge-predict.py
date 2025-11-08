# ============================================================================
# 文件名: 1fixed-discharge-predict.py
# 功能: 使用可变放电数据训练的模型预测固定放电数据
# 目的: 测试模型的跨数据集泛化能力
# ============================================================================

import sys
sys.path.append('.')  # 将当前目录添加到Python路径
import time  # 时间模块（未使用但保留）
import pickle  # 序列化模块（未使用但保留）
import os  # 操作系统接口模块（未使用但保留）

import numpy as np  # 数值计算库，用于数组操作和保存结果

# 从工具模块导入数据提取和集成预测函数
from utils.exp_util import extract_data, extract_input, ensemble_predict

import pdb  # Python调试器（用于开发调试）

# ============================================================================
# 配置参数
# ============================================================================

# 定义要处理的测试通道列表
channels = [1, 2, 3, 4, 5, 6, 7, 8]

# 指定使用的特征组合
# 'eis-actions': EIS特征(200维) + 充放电速率(3维)
input_name = 'eis-actions'

# ============================================================================
# 数据提取
# ============================================================================

# 提取固定放电数据集
# 'fixed-discharge': 对应 raw-data/fixed-discharge/ 目录
# 返回值:
#   cell_fixed: 电池标识数组
#   cap_ds_fixed: 放电容量数组（真实值，用于后续对比）
#   data_fixed: 特征数据元组
cell_fixed, cap_ds_fixed, data_fixed = extract_data('fixed-discharge', channels)

# ============================================================================
# 特征提取
# ============================================================================

# 从固定放电数据中提取与训练时相同的特征
# 必须使用与训练模型时相同的特征组合，否则预测会失败
x_fixed = extract_input(input_name, data_fixed)

# ============================================================================
# 集成预测
# ============================================================================

# 使用集成模型进行预测
# 参数说明:
#   x_fixed: 固定放电数据的特征矩阵
#   'variable-discharge': 模型来源实验（模型是用可变放电数据训练的）
#   input_name: 特征名称，用于定位模型文件
# 
# 函数会自动:
#   1. 加载 experiments/results/variable-discharge/models/ 中的所有模型
#   2. 每个模型分别进行预测
#   3. 计算所有模型预测的均值作为最终预测
#   4. 计算所有模型预测的标准差作为不确定性度量
# 
# 返回值:
#   pred_fixed: 预测均值，shape (n_samples,)
#   pred_fixed_err: 预测标准差，shape (n_samples,)
pred_fixed, pred_fixed_err = ensemble_predict(x_fixed, 'variable-discharge', input_name)

# ============================================================================
# 保存预测结果
# ============================================================================

# 定义结果保存目录
dts = 'experiments/results/fixed-discharge'

# 保存预测均值
# 文件名格式: pred_mn_{特征名}.npy
# 用途: 后续分析和可视化
np.save('{}/predictions/pred_mn_{}.npy'.format(dts, input_name), pred_fixed)

# 保存预测标准差
# 文件名格式: pred_std_{特征名}.npy
# 用途: 不确定性分析，标准差大表示预测不确定性高
np.save('{}/predictions/pred_std_{}.npy'.format(dts, input_name), pred_fixed_err)

# 保存真实值
# 文件名格式: true_{特征名}.npy
# 用途: 与预测值对比，计算评估指标（R²、RMSE等）
np.save('{}/predictions/true_{}.npy'.format(dts, input_name), cap_ds_fixed)

# ============================================================================
# 完成
# ============================================================================
# 预测结果已保存到 experiments/results/fixed-discharge/predictions/
# 可以使用以下代码读取和分析结果:
#
# import numpy as np
# from sklearn.metrics import r2_score, mean_squared_error
# 
# pred = np.load('experiments/results/fixed-discharge/predictions/pred_mn_eis-actions.npy')
# true = np.load('experiments/results/fixed-discharge/predictions/true_eis-actions.npy')
# 
# r2 = r2_score(true, pred)
# rmse = np.sqrt(mean_squared_error(true, pred))
# print(f'R² = {r2:.4f}, RMSE = {rmse:.4f} Ah')
