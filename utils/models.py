# ============================================================================
# 文件名: models.py
# 功能: XGBoost模型封装类，用于电池容量预测
# 包含: 模型训练、预测、评估和结果保存功能
# ============================================================================

import os  # 操作系统接口，用于文件和目录操作
import sys  # 系统相关参数和函数
import time  # 时间相关函数

import numpy as np  # 数值计算库
import xgboost as xgb  # XGBoost梯度提升树库
from sklearn.metrics import r2_score, mean_squared_error  # 评估指标

sys.path.append('../')  # 添加父目录到Python路径
import pickle  # 用于序列化和保存模型

"""


from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, least_squares


import re
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

from scipy.integrate import simps
from scipy.stats import iqr
from collections import deque
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from scipy.stats import iqr
import copy
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.gridspec as gridspec

from d3pg.battery.util import path_to_file, find_cells, find_cycle, search_files, reverse_action
from d3pg.battery.features import reward_function, cv_features, form_state, normalise_state
import pdb
"""

class XGBModel:
    """
    XGBoost模型封装类
    
    功能:
    ----
    1. 训练集成XGBoost模型
    2. 进行预测并提供不确定性估计
    3. 交叉验证评估
    4. 保存模型和预测结果
    
    主要方法:
    --------
    - train_no_predict: 训练模型但不预测
    - train_and_predict: 训练并预测
    - analysis: 完整的交叉验证分析
    - analysis_vd2: 针对vd2数据集的分析
    """
    
    def __init__(self, X, y, cell_nos, experiment, experiment_name, n_ensembles=10, n_splits=5, start_seed=1, max_depth=25, n_estimators=100):
        """
        初始化XGBoost模型
        
        参数:
        ----
        X : ndarray, shape (n_samples, n_features)
            输入特征矩阵
        y : ndarray, shape (n_samples,)
            目标变量（放电容量）
        cell_nos : ndarray, shape (n_samples,)
            每个样本对应的电池标识，用于分组交叉验证
        experiment : str
            实验类型（如'variable-discharge', 'fixed-discharge'等）
        experiment_name : str
            实验名称，用于文件命名（如'eis-actions_n1_xgb'）
        n_ensembles : int, 默认10
            集成学习的模型数量，更多模型可降低预测方差
        n_splits : int, 默认5
            交叉验证折数
        start_seed : int, 默认1
            随机种子起始值，用于可重复性
        max_depth : int, 默认25
            XGBoost树的最大深度，控制模型复杂度
        n_estimators : int, 默认100
            XGBoost中树的数量
        """
        # 设置随机种子，确保结果可重复
        np.random.seed(start_seed+10)
        
        # 保存随机种子起始值
        self.start_seed = start_seed
        
        # 集成学习参数：每个训练/测试分割训练的模型数量
        self.n_ensembles = n_ensembles
        
        # XGBoost模型参数：树的最大深度
        self.max_depth = max_depth
        
        # XGBoost模型参数：每个模型的树数量
        self.n_estimators = n_estimators
        
        # 交叉验证参数：训练/测试分割的数量
        self.n_splits = n_splits
        
        # 当前训练/测试分割的索引（初始为0）
        self.n_split = 0
        
        # 完整数据集：输入特征矩阵，shape (n_samples, n_features)
        self.X = X
        
        # 完整数据集：输出目标变量，shape (n_samples,)
        self.y = y
        
        # 完整数据集：每个数据点对应的电池标识，shape (n_samples,)
        # 用于在交叉验证时按电池分组
        self.cell_nos = cell_nos
        
        # 识别数据集中所有唯一的电池标识
        self.cell_idx = np.unique(self.cell_nos)
        
        # 实验名称（如'variable-discharge'）
        # 用于确定数据来源和结果保存路径
        self.experiment = experiment
        
        # 实验标识名称（如'eis-actions_n1_xgb'）
        # 用于模型和结果文件的命名
        self.experiment_name = experiment_name

        # 特殊实验类型处理：'both'表示同时使用固定和可变放电数据
        if self.experiment == 'both':
            # 只保留固定放电的电池
            cell_idx_new = []
            for i in range(len(self.cell_idx)):
                if experiment_map[self.cell_idx[i]] == 'fixed-discharge':
                    cell_idx_new.append(self.cell_idx[i])
            cell_idx_new = np.array(cell_idx_new)
            self.cell_idx = cell_idx_new
            
        # 特殊实验类型处理：'both-variable'表示只使用可变放电数据
        elif self.experiment == 'both-variable':
            # 只保留可变放电的电池
            cell_idx_new = []
            for i in range(len(self.cell_idx)):
                if experiment_map[self.cell_idx[i]] == 'variable-discharge':
                    cell_idx_new.append(self.cell_idx[i])
            cell_idx_new = np.array(cell_idx_new)
            self.cell_idx = cell_idx_new
            
        # 对电池标识进行排序，确保顺序一致
        self.cell_idx.sort()
        
        # 打印电池标识列表，用于确认数据集
        print(self.cell_idx)

        """
        self.n = X.shape[0] # number of datapoints
        self.n_te = self.n // n_splits + 1 # number of test points
        self.idx = np.random.permutation(self.n)
        """
    def split_into_four(self):
        """
        将数据分成4批进行交叉验证（用于chemistry2数据集）
        
        功能:
        ----
        将32个电池分成4组，每组8个电池
        每次留出一组作为测试集，其余3组作为训练集
        
        返回:
        ----
        idx_tests : list
            测试集索引列表（每个电池的索引）
        idx_train : ndarray
            训练集索引数组
        split_map[self.n_split] : list
            当前折的测试电池列表
        """
        
        # 定义4组电池的分割映射
        # 每组8个电池，尽量保证分布均匀
        split_map = {
            0: ['PJ248', 'PJ249', 'PJ250', 'PJ251', 'PJ264', 'PJ265', 'PJ266', 'PJ267'],
            1: ['PJ252', 'PJ253', 'PJ254', 'PJ255', 'PJ268', 'PJ269', 'PJ270', 'PJ271'],
            2: ['PJ256', 'PJ257', 'PJ258', 'PJ259', 'PJ272', 'PJ273', 'PJ274', 'PJ275'],
            3: ['PJ260', 'PJ261', 'PJ262', 'PJ263', 'PJ276', 'PJ277', 'PJ278', 'PJ279'],
        }

        # 打印当前折的测试电池
        print('Split {}: Test cells'.format(self.n_split))
        print(split_map[self.n_split])

        # 收集所有测试电池的数据索引
        idx_tests = []

        # 遍历当前折的每个测试电池
        for j, cell in enumerate(split_map[self.n_split]):
            # 找到该电池的所有数据点索引
            idx = np.array(np.where(self.cell_nos == cell)).reshape(-1)
            idx_tests.append(idx)

        # 将所有测试电池的索引合并为一维数组
        idx_test = np.hstack(idx_tests).reshape(-1)
        
        # 识别训练集数据点索引
        # 从所有索引中删除测试集索引，剩余的就是训练集索引
        idx_train = np.delete(np.arange(self.X.shape[0]), idx_test)

        return idx_tests, idx_train, split_map[self.n_split]

    def split_by_cell(self):
        """
        按电池分割数据（留二电池出交叉验证）
        
        功能:
        ----
        每次留出2个电池作为测试集，其余电池作为训练集
        两个测试电池的选择：
        - 第一个：按顺序选择
        - 第二个：间隔n_splits个位置选择（循环）
        
        返回:
        ----
        X_train : ndarray
            训练集特征
        y_train : ndarray
            训练集目标
        X_test1 : ndarray
            测试集1特征
        y_test1 : ndarray
            测试集1目标
        X_test2 : ndarray
            测试集2特征
        y_test2 : ndarray
            测试集2目标
        cell_test1 : str
            测试电池1的ID
        cell_test2 : str
            测试电池2的ID
        """
        
        # 选择第一个测试电池：按当前折索引顺序选择
        cell_test1 = self.cell_idx[self.n_split]
        
        # 选择第二个测试电池：间隔n_splits个位置（循环选择）
        # 使用模运算确保索引不超出范围
        cell_test2 = self.cell_idx[(int(self.n_splits + self.n_split) % len(self.cell_idx))]
        
        # 打印当前折的测试电池信息
        print('Split {}: Test cells {} and {}'.format(self.n_split, cell_test1, cell_test2))

        # 识别测试电池的数据点索引
        # np.where找到所有属于该电池的数据点
        idx_test1 = np.array(np.where(self.cell_nos == cell_test1)).reshape(-1)
        idx_test2 = np.array(np.where(self.cell_nos == cell_test2)).reshape(-1)
        
        # 合并两个测试电池的索引
        idx_test = np.hstack([idx_test1, idx_test2]).reshape(-1)

        # 识别训练集数据点索引
        # 从所有索引中删除测试集索引
        idx_train = np.delete(np.arange(self.X.shape[0]), idx_test)

        # 根据索引提取训练和测试数据
        X_test1 = self.X[idx_test1, :]  # 测试集1特征
        y_test1 = self.y[idx_test1]      # 测试集1目标
        X_test2 = self.X[idx_test2, :]  # 测试集2特征
        y_test2 = self.y[idx_test2]      # 测试集2目标
        X_train = self.X[idx_train, :]  # 训练集特征
        y_train = self.y[idx_train]      # 训练集目标

        return X_train, y_train, X_test1, y_test1, X_test2, y_test2, cell_test1, cell_test2

    def train_and_predict(self, X_train, y_train, X_test1, cell_test1, X_test2=None, cell_test2=None,
                          X_test3=None, cell_test3=None, X_test4=None, cell_test4=None):
        """
        训练集成模型并进行预测
        
        功能:
        ----
        1. 使用Bootstrap采样创建多个训练子集
        2. 对每个子集训练一个XGBoost模型
        3. 保存所有模型
        4. 对训练集和测试集进行预测
        5. 计算预测均值和标准差
        
        参数:
        ----
        X_train : ndarray
            训练集特征
        y_train : ndarray
            训练集目标
        X_test1 : ndarray
            测试集1特征（必需）
        cell_test1 : str
            测试电池1的ID（必需）
        X_test2 : ndarray, optional
            测试集2特征
        cell_test2 : str, optional
            测试电池2的ID
        X_test3 : ndarray, optional
            测试集3特征
        cell_test3 : str, optional
            测试电池3的ID
        X_test4 : ndarray, optional
            测试集4特征
        cell_test4 : str, optional
            测试电池4的ID
        
        返回:
        ----
        10个值（5对预测均值和标准差）:
        - y_pred_tr, y_pred_tr_err: 训练集预测
        - y_pred_te1, y_pred_te1_err: 测试集1预测
        - y_pred_te2, y_pred_te2_err: 测试集2预测
        - y_pred_te3, y_pred_te3_err: 测试集3预测
        - y_pred_te4, y_pred_te4_err: 测试集4预测
        """
        
        # 计算Bootstrap采样大小：使用90%的训练数据
        # 这样可以增加数据多样性，降低过拟合风险
        n_bootstrap = int(0.9 * X_train.shape[0])
        
        # 生成每个集成模型的随机种子
        # 确保每个模型使用不同的随机状态
        states = self.n_ensembles * np.arange(1, self.n_ensembles + 1, 1) + self.n_split + self.start_seed
        
        # 定义结果保存目录
        dts = 'experiments/results/{}'.format(self.experiment)

        # 初始化预测结果列表
        y_pred_trs = []   # 训练集预测列表
        y_pred_te1s = []  # 测试集1预测列表
        y_pred_te2s = []  # 测试集2预测列表
        y_pred_te3s = []  # 测试集3预测列表
        y_pred_te4s = []  # 测试集4预测列表

        # 训练n_ensembles个集成模型
        for i, ensemble_state in enumerate(states):

            # --------------------------------------------------------------------
            # Bootstrap采样和模型训练
            # --------------------------------------------------------------------
            
            # 设置随机种子，确保可重复性
            np.random.seed(ensemble_state)
            
            # Bootstrap采样：从训练集中随机抽取n_bootstrap个样本（有放回）
            idx = np.random.permutation(X_train.shape[0])[0:n_bootstrap]
            
            # 创建XGBoost回归器
            # max_depth: 树的最大深度
            # n_estimators: 树的数量
            # random_state: 随机种子（加上n_split确保每折使用不同的种子）
            regr = xgb.XGBRegressor(
                max_depth=self.max_depth, 
                n_estimators=self.n_estimators, 
                random_state=ensemble_state + self.n_split
            )
            
            # 使用Bootstrap采样的数据训练模型
            regr.fit(X_train[idx], y_train[idx])

            # --------------------------------------------------------------------
            # 保存模型
            # --------------------------------------------------------------------
            
            # 保存训练好的模型到文件
            # 文件名格式: {实验名}_{集成索引}_{测试电池}.pkl
            with open('{}/models/{}_{}_{}.pkl'.format(dts, self.experiment_name, i, cell_test1), 'wb') as f:
                pickle.dump(regr, f)

            # --------------------------------------------------------------------
            # 进行预测
            # --------------------------------------------------------------------
            
            # 对训练集进行预测
            y_pred_tr = regr.predict(X_train)
            # 将预测结果reshape为3维数组 (1, n_samples, 1)，方便后续堆叠
            y_pred_trs.append(y_pred_tr.reshape(1, y_pred_tr.shape[0], -1))
            
            # 对测试集1进行预测（必需）
            y_pred_te1 = regr.predict(X_test1)
            y_pred_te1s.append(y_pred_te1.reshape(1, y_pred_te1.shape[0], -1))

            # 对测试集2进行预测（如果提供）
            if X_test2 is not None:
                y_pred_te2 = regr.predict(X_test2)
                y_pred_te2s.append(y_pred_te2.reshape(1, y_pred_te2.shape[0], -1))
                # 保存测试电池2的模型
                with open('{}/models/{}_{}_{}.pkl'.format(dts, self.experiment_name, i, cell_test2), 'wb') as f:
                    pickle.dump(regr, f)

            # 对测试集3进行预测（如果提供）
            if X_test3 is not None:
                y_pred_te3 = regr.predict(X_test3)
                y_pred_te3s.append(y_pred_te3.reshape(1, y_pred_te3.shape[0], -1))
                # 保存测试电池3的模型
                with open('{}/models/{}_{}_{}.pkl'.format(dts, self.experiment_name, i, cell_test3), 'wb') as f:
                    pickle.dump(regr, f)

            # 对测试集4进行预测（如果提供）
            if X_test4 is not None:
                y_pred_te4 = regr.predict(X_test4)
                y_pred_te4s.append(y_pred_te4.reshape(1, y_pred_te4.shape[0], -1))
                with open('{}/models/{}_{}_{}.pkl'.format(dts, self.experiment_name, i, cell_test4), 'wb') as f:
                    pickle.dump(regr, f)

        # --------------------------------------------------------------------
        # 聚合所有集成模型的预测结果
        # --------------------------------------------------------------------
        
        # 将训练集预测列表垂直堆叠为3维数组 (n_ensembles, n_samples, 1)
        y_pred_trs = np.vstack(y_pred_trs)
        # 将测试集1预测列表垂直堆叠
        y_pred_te1s = np.vstack(y_pred_te1s)
        
        # 计算训练集预测的均值（沿第0轴，即集成模型维度）
        y_pred_tr = np.mean(y_pred_trs, axis=0)
        # 计算训练集预测的标准差（不确定性度量）
        y_pred_tr_err = np.sqrt(np.var(y_pred_trs, axis=0))
        
        # 计算测试集1预测的均值
        y_pred_te1 = np.mean(y_pred_te1s, axis=0)
        # 计算测试集1预测的标准差
        y_pred_te1_err = np.sqrt(np.var(y_pred_te1s, axis=0))

        # 处理测试集2（如果提供）
        if X_test2 is not None:
            y_pred_te2s = np.vstack(y_pred_te2s)
            y_pred_te2 = np.mean(y_pred_te2s, axis=0)
            y_pred_te2_err = np.sqrt(np.var(y_pred_te2s, axis=0))
        else:
            # 如果没有提供测试集2，返回None
            y_pred_te2 = None
            y_pred_te2_err = None

        # 处理测试集3（如果提供）
        if X_test3 is not None:
            y_pred_te3s = np.vstack(y_pred_te3s)
            y_pred_te3 = np.mean(y_pred_te3s, axis=0).reshape(-1)
            y_pred_te3_err = np.sqrt(np.var(y_pred_te3s, axis=0)).reshape(-1)
        else:
            y_pred_te3 = None
            y_pred_te3_err = None

        # 处理测试集4（如果提供）
        if X_test4 is not None:
            y_pred_te4s = np.vstack(y_pred_te4s)
            y_pred_te4 = np.mean(y_pred_te4s, axis=0).reshape(-1)
            y_pred_te4_err = np.sqrt(np.var(y_pred_te4s, axis=0)).reshape(-1)
        else:
            y_pred_te4 = None
            y_pred_te4_err = None

        # 返回10个值：5对预测均值和标准差
        return y_pred_tr.reshape(-1), y_pred_tr_err.reshape(-1), y_pred_te1.reshape(-1), y_pred_te1_err.reshape(-1), y_pred_te2.reshape(-1), y_pred_te2_err.reshape(-1), y_pred_te3, y_pred_te3_err, y_pred_te4, y_pred_te4_err

    def train_and_predict_vd2(self, idx_train, idx_tests, cell_tests):
        """
        训练并预测vd2数据集（chemistry2数据）
        
        功能:
        ----
        与train_and_predict类似，但支持多个测试电池（用于4折交叉验证）
        
        参数:
        ----
        idx_train : ndarray
            训练集索引
        idx_tests : list of ndarray
            测试集索引列表（每个元素对应一个测试电池的索引）
        cell_tests : list of str
            测试电池ID列表
        
        返回:
        ----
        y_pred_tr : ndarray
            训练集预测均值
        y_pred_tr_err : ndarray
            训练集预测标准差
        y_pred_tes : list of ndarray
            测试集预测均值列表
        y_pred_te_errs : list of ndarray
            测试集预测标准差列表
        """
        
        # 根据索引提取训练数据
        X_train = self.X[idx_train, :]
        y_train = self.y[idx_train]

        # 打印训练数据形状，用于调试
        print(X_train.shape)
        print(y_train.shape)

        # 计算Bootstrap采样大小：使用90%的训练数据
        n_bootstrap = int(0.9 * X_train.shape[0])
        
        # 生成每个集成模型的随机种子
        states = self.n_ensembles * np.arange(1, self.n_ensembles + 1, 1) + self.n_split + self.start_seed
        
        # 定义结果保存目录
        dts = 'experiments/results/{}'.format(self.experiment)

        # 初始化预测结果列表
        y_pred_trs = []  # 训练集预测
        pred_tes = dict()  # 测试集预测字典

        # 为每个测试电池初始化预测列表
        for j in range(len(cell_tests)):
            pred_tes[j] = []
        print(pred_tes)
        
        # 训练n_ensembles个集成模型
        for i, ensemble_state in enumerate(states):

            # Bootstrap采样并训练XGBoost模型
            np.random.seed(ensemble_state)
            idx = np.random.permutation(X_train.shape[0])[0:n_bootstrap]
            regr = xgb.XGBRegressor(
                max_depth=self.max_depth, 
                n_estimators=self.n_estimators, 
                random_state=ensemble_state + self.n_split
            )
            regr.fit(X_train[idx], y_train[idx])

            # 对每个测试电池进行预测
            for j, cell in enumerate(cell_tests):
                # 提取测试数据
                X_test = self.X[idx_tests[j]]
                print(X_test.shape)
                
                # 保存模型
                with open('{}/models/{}_{}_{}.pkl'.format(dts, self.experiment_name, i, cell), 'wb') as f:
                    pickle.dump(regr, f)
                
                # 进行预测
                pred = regr.predict(X_test)
                print(pred.shape)
                pred_tes[j].append(pred.reshape(1, pred.shape[0], -1))

            # 对训练集进行预测
            pred = regr.predict(X_train)
            y_pred_trs.append(pred.reshape(1, pred.shape[0], -1))

        # 聚合所有集成模型的预测结果
        y_pred_trs = np.vstack(y_pred_trs)
        y_pred_tr = np.mean(y_pred_trs, axis=0).reshape(-1)
        y_pred_tr_err = np.sqrt(np.var(y_pred_trs, axis=0)).reshape(-1)

        # 聚合每个测试电池的预测结果
        y_pred_tes = []
        y_pred_te_errs = []

        for j in range(len(cell_tests)):
            y_pred_te = np.vstack(pred_tes[j])
            y_pred_tes.append(np.mean(y_pred_te, axis=0).reshape(-1))
            y_pred_te_errs.append(np.sqrt(np.var(y_pred_te, axis=0)).reshape(-1))

        return y_pred_tr, y_pred_tr_err, y_pred_tes, y_pred_te_errs

    def analysis(self, log_name, experiment_info):
        """
        完整的交叉验证分析流程
        
        功能:
        ----
        1. 对每个交叉验证折进行训练和预测
        2. 保存每个测试电池的预测结果
        3. 计算并记录性能指标（R²、相对误差）
        4. 将结果写入日志文件
        
        参数:
        ----
        log_name : str
            日志文件路径
        experiment_info : str
            实验信息字符串，会写入日志文件
        """
        
        # 初始化性能指标列表
        r2s_tr = []   # 训练集R²列表
        r2s_te = []   # 测试集R²列表
        pes_tr = []   # 训练集相对误差列表
        pes_te = []   # 测试集相对误差列表

        # 遍历每个交叉验证折
        for n_split in range(self.n_splits):
            # 设置当前折索引
            self.n_split = n_split

            # 按电池分割数据：留出2个测试电池
            X_train, y_train, X_test1, y_test1, X_test2, y_test2, cell_test1, cell_test2 = self.split_by_cell()

            # 训练模型并对训练集和测试集进行预测
            y_pred_tr, y_pred_tr_err, y_pred_te1, y_pred_te1_err, y_pred_te2, y_pred_te2_err, _, _, _, _ = self.train_and_predict(X_train, y_train, X_test1, cell_test1=cell_test1, X_test2=X_test2, cell_test2=cell_test2)

            # 定义结果保存目录
            dts = 'experiments/results/{}'.format(self.experiment)
            
            # 保存测试电池1的预测结果
            np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, self.experiment_name, cell_test1), y_pred_te1)
            np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, self.experiment_name, cell_test1), y_pred_te1_err)
            np.save('{}/predictions/true_{}_{}.npy'.format(dts, self.experiment_name, cell_test1), y_test1)
            
            # 保存测试电池2的预测结果
            np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, self.experiment_name, cell_test2), y_pred_te2)
            np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, self.experiment_name, cell_test2), y_pred_te2_err)
            np.save('{}/predictions/true_{}_{}.npy'.format(dts, self.experiment_name, cell_test2), y_test2)

            # 计算并保存训练集R²
            r2s_tr.append(r2_score(y_train, y_pred_tr))
            
            # 计算并保存两个测试电池的R²
            r2s_te.append(r2_score(y_test1, y_pred_te1))
            r2s_te.append(r2_score(y_test2, y_pred_te2))
            
            # 计算相对误差: |预测值 - 真实值| / 真实值
            pes_tr.append((np.abs(y_train - y_pred_tr) / y_train))
            pes_te.append(np.abs(y_test1 - y_pred_te1) / y_test1)
            pes_te.append(np.abs(y_test2 - y_pred_te2) / y_test2)
        
        # 计算所有折的中位数性能指标
        r2_tr = np.median(np.array(r2s_tr))  # 训练集R²中位数
        r2_te = np.median(np.array(r2s_te))  # 测试集R²中位数
        pe_tr = 100 * np.median(np.hstack(pes_tr).reshape(-1))  # 训练集相对误差中位数（百分比）
        pe_te = 100 * np.median(np.hstack(pes_te).reshape(-1))  # 测试集相对误差中位数（百分比）
        
        # 打印性能指标
        print('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}'.format(r2_tr, pe_tr, r2_te, pe_te))
        
        # 将实验信息和性能指标写入日志文件
        with open(log_name, 'a+') as file:
            file.write(experiment_info)
            file.write('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}\n'.format(r2_tr, pe_tr, r2_te, pe_te))
        return

    def train_no_predict(self, X_train, y_train):
        """
        训练模型但不进行预测（用于生产环境）
        
        功能:
        ----
        训练n_ensembles个集成模型并保存，但不进行预测
        用于只需要训练模型的场景（如生成用于部署的模型）
        
        参数:
        ----
        X_train : ndarray
            训练集特征
        y_train : ndarray
            训练集目标
        """
        
        # 计算Bootstrap采样大小：使用90%的训练数据
        n_bootstrap = int(0.9 * X_train.shape[0])
        
        # 生成每个集成模型的随机种子
        states = self.n_ensembles * np.arange(1, self.n_ensembles + 1, 1) + self.n_split + self.start_seed
        
        # 定义结果保存目录
        dts = 'experiments/results/{}'.format(self.experiment)

        # 训练n_ensembles个集成模型
        for i, ensemble_state in enumerate(states):

            # Bootstrap采样并训练XGBoost模型
            np.random.seed(ensemble_state)
            idx = np.random.permutation(X_train.shape[0])[0:n_bootstrap]
            regr = xgb.XGBRegressor(
                max_depth=self.max_depth, 
                n_estimators=self.n_estimators, 
                random_state=ensemble_state + self.n_split
            )
            regr.fit(X_train[idx], y_train[idx])

            # 保存模型
            # 文件名格式: {实验名}_{集成索引}.pkl
            with open('{}/models/{}_{}.pkl'.format(dts, self.experiment_name, i), 'wb') as f:
                pickle.dump(regr, f)

        return

    def analysis_vd2(self, log_name, experiment_info):
        """
        vd2数据集的完整交叉验证分析流程
        
        功能:
        ----
        与analysis方法类似，但使用4折交叉验证（每折8个测试电池）
        专门用于chemistry2数据集（32个电池）
        
        参数:
        ----
        log_name : str
            日志文件路径
        experiment_info : str
            实验信息字符串，会写入日志文件
        """
        
        # 初始化性能指标列表
        r2s_tr = []   # 训练集R²列表
        r2s_te = []   # 测试集R²列表
        pes_tr = []   # 训练集相对误差列表
        pes_te = []   # 测试集相对误差列表

        # 遍历每个交叉验证折（4折）
        for n_split in range(self.n_splits):
            # 设置当前折索引
            self.n_split = n_split

            # 将数据分成4批：留出8个测试电池
            idx_tests, idx_train, cells_test = self.split_into_four()

            # 训练模型并对训练集和所有测试电池进行预测
            y_pred_tr, y_pred_tr_err, y_pred_tes, y_pred_te_errs = self.train_and_predict_vd2(idx_train, idx_tests, cells_test)

            # 计算训练集性能指标
            y_train = self.y[idx_train]
            r2s_tr.append(r2_score(y_train, y_pred_tr))
            pes_tr.append(np.abs(y_train - y_pred_tr) / y_train)
            
            # 定义结果保存目录
            dts = 'experiments/results/{}'.format(self.experiment)
            
            # 遍历每个测试电池，保存预测结果并计算性能指标
            for j, cell in enumerate(cells_test):
                # 提取该测试电池的真实值和预测值
                y_test = self.y[idx_tests[j]]
                pred_test = y_pred_tes[j]
                pred_test_err = y_pred_te_errs[j]
                
                # 打印形状信息（用于调试）
                print(pred_test.shape)
                print(y_test.shape)
                print(pred_test_err.shape)
                
                # 保存测试电池的预测结果
                np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, self.experiment_name, cell), pred_test)
                np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, self.experiment_name, cell), pred_test_err)
                np.save('{}/predictions/true_{}_{}.npy'.format(dts, self.experiment_name, cell), y_test)
                
                # 计算并保存该测试电池的性能指标
                r2s_te.append(r2_score(y_test, pred_test))
                pes_te.append((np.abs(y_test - pred_test) / y_test).reshape(-1))

        # 计算所有折的中位数性能指标
        r2_tr = np.median(np.array(r2s_tr))  # 训练集R²中位数
        r2_te = np.median(np.array(r2s_te))  # 测试集R²中位数
        pe_tr = 100 * np.median(np.hstack(pes_tr).reshape(-1))  # 训练集相对误差中位数（百分比）
        pe_te = 100 * np.median(np.hstack(pes_te).reshape(-1))  # 测试集相对误差中位数（百分比）
        
        # 打印性能指标
        print('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}'.format(r2_tr, pe_tr, r2_te, pe_te))
        print(r2s_te)  # 打印所有测试电池的R²（用于详细分析）
        
        # 将实验信息和性能指标写入日志文件
        with open(log_name, 'a+') as file:
            file.write(experiment_info)
            file.write('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}\n'.format(r2_tr, pe_tr, r2_te, pe_te))
        return
