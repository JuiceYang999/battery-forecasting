# ============================================================================
# 文件名: 1n-step-lookahead.py
# 功能: N步前瞻预测 - 预测未来N个循环的容量
# 研究问题: 模型能够准确预测多远的未来？
# 方法: 使用当前状态预测未来第N个循环的容量
# ============================================================================

import sys
sys.path.append('.')  # 将当前目录添加到Python路径
import time  # 用于计算训练耗时
import pickle  # 序列化模块（未使用但保留）

import numpy as np  # 数值计算库

# 从工具模块导入数据提取和模型类
from utils.exp_util import extract_data, extract_n_step_data, extract_input, identify_cells
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
# 日志文件将记录不同预测步数下的性能指标
log_name = 'experiments/results/{}/log-n-step-lookahead.txt'.format(experiment)

# ============================================================================
# 特征选择
# ============================================================================

# 定义输入特征类型
# 'eis-final-actions': EIS特征 + 最终循环的充放电速率
# 只使用目标循环的动作，不使用中间循环的动作
input_name = 'eis-final-actions'

# ============================================================================
# 数据提取
# ============================================================================

# 提取N步预测所需的数据
# 返回值:
#   states: 字典，键为电池ID，值为状态矩阵（EIS特征等）
#   actions: 字典，键为电池ID，值为动作矩阵（充放电速率）
#   cycles: 字典，键为电池ID，值为循环编号数组
#   cap_ds: 字典，键为电池ID，值为放电容量数组
(states, actions, cycles, cap_ds) = extract_n_step_data(experiment, channels)

# 获取电池-通道映射关系
# 返回字典：{通道号: [电池ID列表]}
cell_map = identify_cells(experiment)

# ============================================================================
# 定义预测步数列表
# ============================================================================

# 定义要测试的预测步数
# 1步: 预测下一个循环
# 2步: 预测第2个循环
# 4步: 预测第4个循环
# ...
# 40步: 预测第40个循环
n_steps = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40]

# ============================================================================
# N步前瞻预测主循环
# ============================================================================

# 遍历每个预测步数
for step in n_steps:
    # 构建实验名称
    # 格式: {特征名}_n{步数}_xgb
    # 例如: eis-final-actions_n1_xgb, eis-final-actions_n4_xgb
    experiment_name = '{}_n{}_xgb'.format(input_name, step)
    
    # 构建实验信息字符串
    experiment_info = '\nInput: {} \tOutput: Q_n+{} \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(
        input_name,                # 输入特征名称
        step,                      # 预测步数
        params['max_depth'],       # 树的最大深度
        params['n_estimators'],    # 树的数量
        params['n_ensembles'],     # 集成模型数量
        params['n_splits']         # 交叉验证折数
    )
    print(experiment_info)
    
    # 记录开始时间
    t0 = time.time()

    # ------------------------------------------------------------------------
    # 构建N步预测数据集
    # ------------------------------------------------------------------------
    
    # nl: n-lookahead的缩写，表示前瞻步数减1
    # 例如：预测第4个循环时，nl=3（需要跳过3个循环）
    nl = step - 1
    
    # 初始化数据列表
    nl_states = []   # 存储所有样本的状态特征
    nl_actions = []  # 存储所有样本的动作特征
    nl_caps = []     # 存储所有样本的目标容量
    nl_idx = []      # 存储所有样本的电池标识
    
    # 遍历每个通道
    for channel in channels:
        # 获取该通道对应的电池列表
        cells = cell_map[channel]
        
        # 遍历该通道的每个电池
        for cell in cells:
            # 获取该电池的数据
            cell_states = states[cell]    # 状态矩阵 (n_cycles, n_state_features)
            cell_actions = actions[cell]  # 动作矩阵 (n_cycles, n_action_features)
            cell_cap_ds = cap_ds[cell]    # 容量数组 (n_cycles,)
            
            # 获取该电池的循环数量
            ns = cell_states.shape[0]
            
            # 特殊情况：nl=0表示预测下一个循环（1步预测）
            if nl == 0:
                # 直接使用所有数据，不需要跳过循环
                nl_actions.append(cell_actions)
                nl_states.append(cell_states)
                nl_caps.append(cell_cap_ds)
            
            # 一般情况：nl>0表示多步预测
            else:
                # 根据特征类型选择不同的动作提取方式
                if input_name == 'eis-actions':
                    # 使用所有中间循环的动作
                    # 例如：预测第4个循环时，使用第1、2、3、4个循环的动作
                    for i in range(ns - nl):
                        # 提取从第i个到第i+nl个循环的所有动作
                        # reshape(1, -1)将其展平为一行
                        nl_actions.append(cell_actions[i:i+nl+1, :].reshape(1, -1))
                        
                elif input_name == 'eis-final-actions':
                    # 只使用目标循环的动作
                    # 例如：预测第4个循环时，只使用第4个循环的动作
                    for i in range(ns - nl):
                        # 只提取第i+nl个循环的动作
                        nl_actions.append(cell_actions[i+nl, :].reshape(1, -1))
                
                # 状态特征：使用前面的循环，去掉最后nl个循环
                # 例如：预测第4个循环时，使用第1、2、...、n-3个循环的状态
                nl_states.append(cell_states[:-nl, :])
                
                # 目标容量：跳过前nl个循环
                # 例如：预测第4个循环时，目标是第4、5、...、n个循环的容量
                nl_caps.append(cell_cap_ds[nl:])
            
            # 记录电池标识
            # 每个样本都标记其所属的电池
            nl_idx.append([cell] * (ns - nl))

    # ------------------------------------------------------------------------
    # 合并所有电池的数据
    # ------------------------------------------------------------------------
    
    # 将电池标识列表展平为一维数组
    nl_idx = np.array([item for sublist in nl_idx for item in sublist])
    
    # 垂直堆叠所有电池的状态特征
    # 结果shape: (total_samples, n_state_features)
    nl_states = np.vstack(nl_states)
    
    # 垂直堆叠所有电池的动作特征
    # 结果shape: (total_samples, n_action_features)
    nl_actions = np.vstack(nl_actions)
    
    # 水平拼接所有电池的目标容量
    # 结果shape: (total_samples,)
    nl_caps = np.hstack(nl_caps)

    # ------------------------------------------------------------------------
    # 构建输入特征和目标变量
    # ------------------------------------------------------------------------
    
    # 将状态和动作水平拼接，形成完整的输入特征矩阵
    # shape: (total_samples, n_state_features + n_action_features)
    x = np.concatenate((nl_states, nl_actions), axis=1)
    
    # 目标变量：未来第step个循环的放电容量
    y = nl_caps

    # ------------------------------------------------------------------------
    # 模型训练和评估
    # ------------------------------------------------------------------------
    
    # 创建XGBoost模型对象
    regressor = XGBModel(
        x, y,           # 输入特征和目标变量
        nl_idx,         # 电池标识，用于分组交叉验证
        experiment, 
        experiment_name, 
        n_ensembles=params['n_ensembles'],
        n_splits=params['n_splits'], 
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators']
    )
    
    # 执行完整的分析流程
    # 包括：交叉验证、训练、预测、评估、保存结果
    regressor.analysis(log_name, experiment_info)
    
    # 计算并打印训练耗时
    elapsed_time = time.time() - t0
    print('Time taken = {:.2f}'.format(elapsed_time))

# ============================================================================
# 完成
# ============================================================================

print('Done.')  # 所有预测步数的实验完成

# ============================================================================
# 输出文件说明
# ============================================================================
# 
# 1. 模型文件:
#    experiments/results/variable-discharge/models/
#    ├── eis-final-actions_n1_xgb_*.pkl
#    ├── eis-final-actions_n2_xgb_*.pkl
#    └── ... (每个预测步数的模型)
# 
# 2. 预测结果:
#    experiments/results/variable-discharge/predictions/
#    ├── pred_mn_eis-final-actions_n1_xgb_{cell}.npy
#    ├── pred_mn_eis-final-actions_n2_xgb_{cell}.npy
#    └── ... (每个预测步数和电池的预测结果)
# 
# 3. 日志文件:
#    experiments/results/variable-discharge/log-n-step-lookahead.txt
#    包含不同预测步数的性能对比:
#    - 1步: R²=?, RMSE=? Ah
#    - 2步: R²=?, RMSE=? Ah
#    - 4步: R²=?, RMSE=? Ah
#    - ...
# 
# 4. 预期趋势:
#    预测步数增加 → R²降低，RMSE增加
#    短期预测（1-4步）通常较准确
#    长期预测（>16步）误差显著增大
# 
# 5. 应用场景:
#    - 1-4步: 短期维护计划
#    - 4-16步: 中期性能预测
#    - >16步: 长期寿命估计（不确定性较高）
