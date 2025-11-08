# ============================================================================
# 文件名: 1data-efficiency.py
# 功能: 分析不同训练数据量对模型性能的影响（数据效率分析）
# 研究问题: 需要多少电池数据才能训练出性能良好的模型？
# 方法: 使用不同数量的训练电池，评估模型性能
# ============================================================================

import sys
sys.path.append('.')  # 将当前目录添加到Python路径
import time  # 用于计算训练耗时
import pickle  # 序列化模块（未使用但保留）

import numpy as np  # 数值计算库
from sklearn.metrics import r2_score  # R²评估指标

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
# 日志文件将记录不同训练数据量下的性能指标
log_name = 'experiments/results/{}/log-n-cells.txt'.format(experiment)

# 指定使用的特征组合
input_name = 'eis-actions'

# 定义要测试的训练电池数量列表
# 将分别使用2、4、8、16、20个电池进行训练，观察性能变化
n_cells_list = [2, 4, 8, 16, 20]

# 生成24个电池的随机排列
# 24是可变放电数据集中的总电池数量（PJ097-PJ152中的24个）
p = np.random.permutation(24)

# ============================================================================
# 数据提取
# ============================================================================

# 从原始数据文件中提取可变放电数据集
# 返回值:
#   cell_var: 电池标识数组
#   cap_ds_var: 放电容量数组（目标变量）
#   data_var: 特征数据元组
cell_var, cap_ds_var, data_var = extract_data(experiment, channels)

# 获取所有唯一的电池标识，并按照随机排列重新排序
# 这样可以确保每次实验使用不同的电池组合
cell_ids = np.unique(cell_var)[p]

# 从data_var中提取指定的输入特征
x = extract_input(input_name, data_var)

# ============================================================================
# 定义目标变量
# ============================================================================

# 目标变量: 下一循环的放电容量 (Ah)
y = cap_ds_var

# 获取交叉验证折数
n_splits = params['n_splits']

# ============================================================================
# 数据效率分析主循环
# ============================================================================

# 遍历每个训练电池数量
for n_cells in n_cells_list:
    # 构建实验名称，包含电池数量信息
    # 格式: {特征名}_{电池数}cells_xgb
    # 例如: eis-actions_2cells_xgb, eis-actions_4cells_xgb
    experiment_name = '{}_{}cells_xgb'.format(input_name, n_cells)
    
    # 构建实验信息字符串
    experiment_info = '\nInput: {} \tOutput: Q_n+1 \t{} cells \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(
        input_name,                # 输入特征名称
        n_cells,                   # 训练电池数量
        params['max_depth'],       # 树的最大深度
        params['n_estimators'],    # 树的数量
        params['n_ensembles'],     # 集成模型数量
        params['n_splits']         # 交叉验证折数
    )
    
    # 记录开始时间
    t0 = time.time()
    
    # 初始化性能指标列表
    r2s_tr = []   # 训练集R²列表
    r2s_te = []   # 测试集R²列表
    pes_tr = []   # 训练集相对误差列表
    pes_te = []   # 测试集相对误差列表

    # ------------------------------------------------------------------------
    # 交叉验证循环
    # ------------------------------------------------------------------------
    
    # 进行n_splits次交叉验证
    # 每次使用不同的电池组合作为训练集和测试集
    for split in range(n_splits):
        # 选择当前折的电池
        # 前n_cells+2个电池：前2个作为测试，后n_cells个作为训练
        cell_ids_s = cell_ids[0:n_cells+2]
        
        # 更新实验信息（这里重新定义了experiment_info，会覆盖之前的）
        experiment_info = '\nInput: {} \tOutput: c(discharge)_n+1 \nMax depth: {}\tSplits:{}\n'.format(
            input_name, 
            params['max_depth'], 
            n_cells
        )
        
        # 打印当前折使用的测试电池
        print(cell_ids[0])  # 测试电池1
        print(cell_ids[1])  # 测试电池2
        
        # 分配测试和训练电池
        cell_test1 = cell_ids[0]              # 第一个测试电池
        cell_test2 = cell_ids[1]              # 第二个测试电池
        cell_train = cell_ids[2:n_cells+2]   # n_cells个训练电池
        
        # 根据电池标识筛选数据索引
        # np.isin检查cell_var中的每个元素是否在指定的电池列表中
        # np.where返回满足条件的索引
        idx_test1 = np.where(np.isin(cell_var, cell_test1))
        idx_test2 = np.where(np.isin(cell_var, cell_test2))
        idx_train = np.where(np.isin(cell_var, cell_train))
        
        # 根据索引提取训练和测试数据
        x_train = x[idx_train]
        print('Number of datapoints = {}'.format(x_train.shape[0]))
        y_train = cap_ds_var[idx_train]
        
        x_test1 = x[idx_test1]
        y_test1 = cap_ds_var[idx_test1]
        
        x_test2 = x[idx_test2]
        y_test2 = cap_ds_var[idx_test2]

        # --------------------------------------------------------------------
        # 模型训练和预测
        # --------------------------------------------------------------------
        
        # 创建XGBoost模型对象
        # 注意: X和y参数传入None，因为在train_and_predict中会传入实际数据
        regressor = XGBModel(
            None, None,  # X和y设为None
            cell_ids_s,  # 电池标识列表
            experiment, 
            experiment_name, 
            n_ensembles=params['n_ensembles'],
            n_splits=params['n_splits'], 
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators']
        )
        
        # 训练模型并进行预测
        # 返回10个值：
        #   - y_pred_tr, y_pred_tr_err: 训练集预测均值和标准差
        #   - y_pred_te1, y_pred_te1_err: 测试集1预测均值和标准差
        #   - y_pred_te2, y_pred_te2_err: 测试集2预测均值和标准差
        #   - y_pred_te3, y_pred_te3_err: 测试集3预测均值和标准差（未使用）
        #   - y_pred_te4, y_pred_te4_err: 测试集4预测均值和标准差（未使用）
        y_pred_tr, y_pred_tr_err, y_pred_te1, y_pred_te1_err, y_pred_te2, y_pred_te2_err, y_pred_te3, y_pred_te3_err, y_pred_te4, y_pred_te4_err = regressor.train_and_predict(
            x_train, y_train,  # 训练数据
            x_test1, cell_test1,  # 测试集1
            x_test2, cell_test2   # 测试集2
        )

        # --------------------------------------------------------------------
        # 保存预测结果
        # --------------------------------------------------------------------
        
        # 定义结果保存目录
        dts = 'experiments/results/{}'.format(experiment)
        
        # 保存测试电池1的预测结果
        np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, experiment_name, cell_test1), y_pred_te1)
        np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, experiment_name, cell_test1), y_pred_te1_err)
        np.save('{}/predictions/true_{}_{}.npy'.format(dts, experiment_name, cell_test1), y_test1)
        
        # 保存测试电池2的预测结果
        np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, experiment_name, cell_test2), y_pred_te2)
        np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, experiment_name, cell_test2), y_pred_te2_err)
        np.save('{}/predictions/true_{}_{}.npy'.format(dts, experiment_name, cell_test2), y_test2)

        # --------------------------------------------------------------------
        # 计算性能指标
        # --------------------------------------------------------------------
        
        # 计算并保存训练集R²
        r2s_tr.append(r2_score(y_train, y_pred_tr))
        
        # 计算并保存两个测试集的R²
        r2s_te.append(r2_score(y_test1, y_pred_te1))
        r2s_te.append(r2_score(y_test2, y_pred_te2))
        
        # 计算相对误差: |预测值 - 真实值| / 真实值
        pes_tr.append(np.abs(y_train - y_pred_tr) / y_train)
        pes_te.append(np.abs(y_test1 - y_pred_te1) / y_test1)
        pes_te.append(np.abs(y_test2 - y_pred_te2) / y_test2)
        
        # 滚动电池列表，为下一折准备
        # np.roll将数组循环移位2个位置
        # 这样下一折会使用不同的电池组合
        cell_ids = np.roll(cell_ids, 2)

    # ------------------------------------------------------------------------
    # 汇总统计结果
    # ------------------------------------------------------------------------
    
    # 计算训练集R²的中位数
    r2_tr = np.median(np.array(r2s_tr))
    
    # 计算测试集R²的中位数
    r2_te = np.median(np.array(r2s_te))
    
    # 计算训练集相对误差的中位数（转换为百分比）
    pe_tr = 100 * np.median(np.hstack(pes_tr).reshape(-1))
    
    # 计算测试集相对误差的中位数（转换为百分比）
    pe_te = 100 * np.median(np.hstack(pes_te).reshape(-1))
    
    # 打印当前训练电池数量的性能结果
    print('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}'.format(r2_tr, pe_tr, r2_te, pe_te))
    
    # 将结果写入日志文件
    with open(log_name, 'a+') as file:  # 'a+'模式：追加写入，如果文件不存在则创建
        file.write(experiment_info)
        file.write('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}\n'.format(r2_tr, pe_tr, r2_te, pe_te))
        file.flush()  # 立即将缓冲区内容写入文件

# ============================================================================
# 完成
# ============================================================================

print('Done.')  # 所有训练电池数量的实验完成

# ============================================================================
# 输出文件说明
# ============================================================================
# 
# 1. 预测结果文件:
#    experiments/results/variable-discharge/predictions/
#    ├── pred_mn_eis-actions_2cells_xgb_{cell}.npy
#    ├── pred_mn_eis-actions_4cells_xgb_{cell}.npy
#    └── ... (每个n_cells和测试电池的预测结果)
# 
# 2. 日志文件:
#    experiments/results/variable-discharge/log-n-cells.txt
#    包含不同训练数据量的性能对比:
#    - 2个电池: Train R²=?, Test R²=?, Train error=?%, Test error=?%
#    - 4个电池: Train R²=?, Test R²=?, Train error=?%, Test error=?%
#    - 8个电池: Train R²=?, Test R²=?, Train error=?%, Test error=?%
#    - 16个电池: Train R²=?, Test R²=?, Train error=?%, Test error=?%
#    - 20个电池: Train R²=?, Test R²=?, Train error=?%, Test error=?%
# 
# 3. 预期趋势:
#    训练电池数量增加 → 测试R²提高，测试误差降低
#    但增益会逐渐递减（边际效应递减）
