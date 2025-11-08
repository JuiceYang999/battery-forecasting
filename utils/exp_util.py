# ============================================================================
# 文件名: exp_util.py
# 功能: 标准数据集（variable-discharge和fixed-discharge）的工具函数集
# 包含: 数据提取、特征工程、EIS分析、等效电路模型拟合等功能
# 数据集: PJ097-PJ152 (可变放电), PJ121-PJ136 (固定放电)
# ============================================================================

import os  # 操作系统接口，用于文件操作
import sys  # 系统相关参数和函数
sys.path.append('../')  # 添加父目录到Python路径
import pickle  # 用于序列化和保存数据
import time  # 时间相关函数

# 科学计算库
from scipy.interpolate import interp1d  # 一维插值函数
from scipy.optimize import curve_fit, least_squares  # 曲线拟合和最小二乘优化
from scipy.integrate import simpson  # 辛普森积分法
from scipy.stats import iqr  # 四分位距（Interquartile Range）
import numpy as np  # 数值计算库
import pandas as pd  # 数据分析库


# ============================================================================
# 电池-实验类型映射表
# ============================================================================
# 将每个电池ID映射到其对应的实验类型（数据目录）
experiment_map = {'PJ097':'variable-discharge',
                  'PJ098':'variable-discharge',
                  'PJ099':'variable-discharge',
                  'PJ100':'variable-discharge',
                  'PJ101':'variable-discharge',
                  'PJ102':'variable-discharge',
                  'PJ103':'variable-discharge',
                  'PJ104':'variable-discharge',
                  'PJ105':'variable-discharge',
                  'PJ106':'variable-discharge',
                  'PJ107':'variable-discharge',
                  'PJ108':'variable-discharge',
                  'PJ109':'variable-discharge',
                  'PJ110':'variable-discharge',
                  'PJ111':'variable-discharge',
                  'PJ112':'variable-discharge',
                  'PJ121':'fixed-discharge',
                  'PJ122':'fixed-discharge',
                  'PJ123':'fixed-discharge',
                  'PJ124':'fixed-discharge',
                  'PJ125':'fixed-discharge',
                  'PJ126':'fixed-discharge',
                  'PJ127':'fixed-discharge',
                  'PJ128':'fixed-discharge',
                  'PJ129':'fixed-discharge',
                  'PJ130':'fixed-discharge',
                  'PJ131':'fixed-discharge',
                  'PJ132':'fixed-discharge',
                  'PJ133':'fixed-discharge',
                  'PJ134':'fixed-discharge',
                  'PJ135':'fixed-discharge',
                  'PJ136':'fixed-discharge',
                  'PJ145':'variable-discharge',
                  'PJ146':'variable-discharge',
                  'PJ147':'variable-discharge',
                  'PJ148':'variable-discharge',
                  'PJ149':'variable-discharge',
                  'PJ150':'variable-discharge',
                  'PJ151':'variable-discharge',
                  'PJ152':'variable-discharge',
                  }

# ============================================================================
# 数据文件列名映射
# ============================================================================
# 定义不同测试类型的数据文件列名
column_map = {
    'GCPL': ['time', 'ewe', 'i', 'capacity', 'power', 'ox/red', 'unnamed'],  # 恒流充放电数据列名
    'GEIS': ['freq', 're_z', '-im_z', 'time', 'unnamed']  # 电化学阻抗谱数据列名
}

# 颜色映射名称列表（用于可视化，未在当前代码中使用）
cmap_names = ['Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r', 'RdPu', 'BuPu', 'GnBu', 'YlOrRd']


# ============================================================================
# 主要数据提取函数
# ============================================================================

def extract_data(experiment, channels):
    """
    从原始数据文件中提取电池循环数据
    
    功能:
    ----
    1. 读取指定实验类型和通道的所有电池数据
    2. 提取EIS特征、容量-电压曲线、充放电速率等
    3. 计算SOH、累积吞吐量等衍生特征
    4. 返回组织好的特征矩阵和目标变量
    
    参数:
    ----
    experiment : str
        实验类型，如'variable-discharge'或'fixed-discharge'
    channels : list of int
        要处理的测试通道列表，如[1, 2, 3, 4, 5, 6, 7, 8]
    
    返回:
    ----
    cell_idx : ndarray
        电池标识数组，shape (n_samples,)
    cap_ds : ndarray
        放电容量数组（目标变量），shape (n_samples,)
    data : tuple
        特征数据元组，包含9个元素:
        (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, 
         d_rates, c1_rates, c2_rates)
        - last_caps: 上一循环容量
        - sohs: 健康状态
        - eis_ds: EIS特征（200维）
        - cvfs: 容量-电压曲线特征（1000维）
        - ocvs: 开路电压
        - cap_throughputs: 累积吞吐量
        - d_rates: 放电速率
        - c1_rates: 第一阶段充电速率
        - c2_rates: 第二阶段充电速率
    """

    cell_map = identify_cells(experiment)
    n_repeats = 1
    n_steps = 4
    new_log_freq = np.linspace(-1.66, 3.9, 100)

    freq1 = np.log10(2.16)
    freq2 = np.log10(17.8)
    idx_freq1 = np.argmin(np.abs(new_log_freq-freq1))
    idx_freq2 = np.argmin(np.abs(new_log_freq-freq2))

    nl = 0

    c1_rates = []
    c2_rates = []
    d_rates = []
    eis_cs = []
    eis_ds = []
    t_charges = []
    t1_charges = []
    t2_charges = []
    ocvs = []
    cap_cs = []
    cap_ds = []
    cap_nets = []
    cap_throughputs = []
    cap_inits = []
    eis_inits = []
    cycles = []
    cell_idx = []
    cvfs = []
    last_caps = []
    sohs = []

    column_map = {
        'GCPL': ['time', 'ewe', 'i', 'capacity', 'power', 'ox/red', 'unnamed'],
        'GEIS': ['freq', 're_z', '-im_z', 'time', 'unnamed']
    }

    geis_discharge_no = 1
    gcpl_charge_no = 2
    geis_charge_no = 3
    gcpl_discharge_no = 4

    for channel in channels:

        cells = cell_map[channel]

        for cell in cells:

            cell_no = int(cell[-3:])
            #cmap = plt.get_cmap(name, 70)
            #cell_ars = []
            dir = 'raw-data/{}/'.format(experiment_map[cell])

            # First get the initial EIS and GCPL
            cycle = 0
            path = path_to_file(channel, cell, cycle, dir)
            ptd = '{}{}/'.format(dir, cell)
            filestart = '{}_{:03d}a_'.format(cell, cycle)

            if cell_no in [145, 146, 147, 148]:
                start_cycle = 3
            else:
                start_cycle = 2

            ptf_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, 4, 'GEIS', channel)
            ptf_cv = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, 3, 'GCPL', channel)

            # Get features of discharge capacity-voltage curve

            cap0, cvf0, e_out, d_rate, cap_curve_norm = discharge_features(ptf_cv, cycle)

            print('Cell PJ{:03d}\t C0 {:.2f}\t Start cycle {}'.format(cell_no, cap0, start_cycle))
            oldcap = e_out

            # Get initial features of discharge EIS spectrum
            x_eis = eis_features(ptf_eis, new_log_freq=new_log_freq, n_repeats=n_repeats)
            eis0 = x_eis

            cell_c1_rates = []
            cell_c2_rates = []
            cell_d_rates = []
            cell_eis_cs = []
            cell_eis_ds = []
            cell_t_charges = []
            cell_t1_charges = []
            cell_t2_charges = []
            cell_ocvs = []
            cell_cap_cs = []
            cell_cap_ds = []
            cell_cycles = []
            cell_cap_nets = []
            cell_cap_throughputs = []
            cell_cvfs = []
            cell_cvfs.append(cvf0)

            cell_last_caps = []
            cell_sohs = []
            cell_last_caps.append(cap0)
            cell_sohs.append(1)

            cap_net = 0
            cap_throughput = 0
            cell_cap_nets.append(cap_net)
            cell_cap_throughputs.append(cap_throughput)

            for cycle in range(start_cycle, start_cycle+30):
                path = path_to_file(channel, cell, cycle, dir)
                ptd = '{}{}/'.format(dir, cell)
                filestart = '{}_{:03d}_'.format(cell, cycle)

                for step in range(n_steps):
                    true_cycle = 1+(cycle-2)*n_steps + step

                    ptf_discharge_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, geis_discharge_no + step*4, 'GEIS', channel)
                    ptf_charge_gcpl = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, gcpl_charge_no + step*4, 'GCPL', channel)
                    ptf_charge_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, geis_charge_no + step*4, 'GEIS', channel)
                    ptf_discharge_gcpl = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, gcpl_discharge_no + step*4, 'GCPL', channel)

                    ptf_files = [ptf_discharge_eis, ptf_charge_gcpl, ptf_charge_eis, ptf_discharge_gcpl]

                    #check_files(ptf_files)

                    # Get features of discharge EIS spectrum
                    try:
                        eis_d = eis_features(ptf_discharge_eis, new_log_freq=new_log_freq, n_repeats=1)
                    except:
                        eis_d = None

                    # Compute the time to charge
                    try:
                        t_charge, cap_c, c1_rate, c2_rate, t1_charge, t2_charge, ocv = charge_features(ptf_charge_gcpl)

                    except:
                        t_charge = None
                        cap_c = None
                        c1_rate = None
                        c2_rate = None

                    # Get features of discharge capacity-voltage curve
                    try:
                        cap_d, cvf, _, d_rate, _ = discharge_features(ptf_discharge_gcpl, true_cycle, cap_curve_norm)
                    except:
                        cap_d = None
                        cvf = None
                        d_rate = None

                    if any(elem is None for elem in (eis_d, t_charge, cap_c, cap_d)):
                        #print('{:02d}\t{} {} {}'.format(true_cycle, t_charge, cap_c, cap_d))
                        continue
                    else:
                        cap_net += cap_c - cap_d
                        cap_throughput += cap_c + cap_d
                        cell_cvfs.append(cvf)
                        cell_c1_rates.append(c1_rate)
                        cell_c2_rates.append(c2_rate)
                        cell_d_rates.append(d_rate)
                        cell_eis_ds.append(eis_d)
                        cell_t_charges.append(t_charge)
                        cell_t1_charges.append(t1_charge)
                        cell_t2_charges.append(t2_charge)
                        cell_ocvs.append(ocv)
                        cell_cap_cs.append(cap_c)
                        cell_cap_ds.append(cap_d)
                        cell_sohs.append(cap_d / cap0)
                        cell_last_caps.append(cap_d)
                        cell_cap_nets.append(cap_net)
                        cell_cap_throughputs.append(cap_throughput)
                        cell_cycles.append(1+(cycle-2)*n_steps + step)
                        #print('{:02d}\t{:.1f}\t{:.1f}\t{:.3f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(1+(cycle-2)*n_steps + step, c1_rate, c2_rate, ocv, t_charge, t1, t2, t_charge-t1-t2))

            cell_c1_rates = np.array(cell_c1_rates)
            cell_c2_rates = np.array(cell_c2_rates)
            cell_d_rates = np.array(cell_d_rates)
            cell_eis_ds = np.vstack(np.array(cell_eis_ds))
            cell_cvfs = np.vstack(np.array(cell_cvfs))
            cell_t_charges = np.array(cell_t_charges)
            cell_t1_charges = np.array(cell_t1_charges)
            cell_t2_charges = np.array(cell_t2_charges)
            cell_ocvs = np.array(cell_ocvs)
            cell_cap_cs = np.array(cell_cap_cs)
            cell_cap_ds = np.array(cell_cap_ds)
            cell_cap_nets = np.array(cell_cap_nets)
            cell_cap_throughputs = np.array(cell_cap_throughputs)
            cell_cycles = np.array(cell_cycles)
            cell_last_caps = np.array(cell_last_caps)
            cell_sohs = np.array(cell_sohs)

            if nl == 0:
                cell_idx.append([cell,]*cell_t_charges.shape[0])
                cycles.append(cell_cycles)
                c1_rates.append(cell_c1_rates.reshape(-1, 1))
                c2_rates.append(cell_c2_rates.reshape(-1, 1))
                d_rates.append(cell_d_rates.reshape(-1, 1))
                eis_ds.append(cell_eis_ds)
                t_charges.append(cell_t_charges)
                t1_charges.append(cell_t1_charges)
                t2_charges.append(cell_t2_charges)
                ocvs.append(cell_ocvs)
                cap_cs.append(cell_cap_cs)
                cap_ds.append(cell_cap_ds)
                last_caps.append(cell_last_caps[:-1])
                sohs.append(cell_sohs[:-1])
                cap_nets.append(cell_cap_nets[:-1])
                cap_throughputs.append(cell_cap_throughputs[:-1])
                cvfs.append(cell_cvfs[:-1, :])
                cap_inits.append(cap0*np.ones(cell_t_charges.shape[0]))
                eis_inits.append(np.tile(eis0, (cell_t_charges.shape[0], 1)))
            elif nl == 1:
                cell_idx.append([cell,]*(cell_t_charges.shape[0]-1))
                cycles.append(cell_cycles[:-nl])
                c1_rates.append(np.concatenate((cell_c1_rates[:-nl].reshape(-1, 1), cell_c1_rates[nl:].reshape(-1, 1)), axis=1))
                c2_rates.append(np.concatenate((cell_c2_rates[:-nl].reshape(-1, 1), cell_c2_rates[nl:].reshape(-1, 1)), axis=1))
                d_rates.append(np.concatenate((cell_d_rates[:-nl].reshape(-1, 1), cell_d_rates[nl:].reshape(-1, 1)), axis=1))
                eis_ds.append(cell_eis_ds[:-nl])
                t_charges.append(cell_t_charges)
                t1_charges.append(cell_t1_charges)
                t2_charges.append(cell_t2_charges)
                ocvs.append(cell_ocvs[:-nl])
                cap_cs.append(cell_cap_cs)
                cap_ds.append(cell_cap_ds[nl:])
                cap_nets.append(cell_cap_nets[:-1])
                cap_throughputs.append(cell_cap_throughputs[:-1])
                cvfs.append(cell_cvfs[:-1-nl, :])
                cap_inits.append(cap0*np.ones(cell_t_charges.shape[0]-1))
                eis_inits.append(np.tile(eis0, (cell_t_charges.shape[0]-1, 1)))

    cycles = np.hstack(cycles)
    c1_rates = np.vstack(c1_rates)
    c2_rates = np.vstack(c2_rates)
    d_rates = np.vstack(d_rates)
    eis_ds = np.vstack(eis_ds)
    eis_inits = np.vstack(eis_inits)
    t_charges = np.hstack(t_charges)
    t1_charges = np.hstack(t1_charges)
    t2_charges = np.hstack(t2_charges)
    ocvs = np.hstack(ocvs)
    cap_cs = np.hstack(cap_cs)
    cap_ds = np.hstack(cap_ds)
    last_caps = np.hstack(last_caps)
    sohs = np.hstack(sohs)
    cap_nets = np.hstack(cap_nets)
    cap_throughputs = np.hstack(cap_throughputs)
    cap_inits = np.hstack(cap_inits)
    cell_idx = np.array([item for sublist in cell_idx for item in sublist])
    cvfs = np.vstack(cvfs)
    """
    np.save('cell_variable.npy', cell_idx)
    np.save('cap_ds_variable.npy', cap_ds)
    np.save('last_caps_variable.npy', last_caps)
    np.save('soh_variable.npy', sohs)
    np.save('eis_variable.npy', eis_ds)
    np.save('eis_inits_variable.npy', eis_inits)
    np.save('cvfs_variable.npy', cvfs)
    np.save('ocvs_variable.npy', ocvs)
    np.save('cap_throughputs_variable.npy', cap_throughputs)
    np.save('d_rates_variable.npy', d_rates)
    np.save('c1_rates_variable.npy', c1_rates)
    np.save('c2_rates_variable.npy', c2_rates)
    print('Saved data')
    """
    data = (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates)

    return cell_idx, cap_ds, data

# ============================================================================
# 文件路径辅助函数
# ============================================================================

def path_to_file(channel, cell, cycle, dir='raw-data/variable-discharge/'):
    """
    构建数据文件路径前缀
    
    参数:
    ----
    channel : int
        测试通道号
    cell : str
        电池ID，如'PJ097'
    cycle : int
        循环编号
    dir : str
        数据目录路径
    
    返回:
    ----
    str : 文件路径前缀，如'raw-data/variable-discharge/PJ097_1/PJ097_002_'
    """
    sub_dir = '{}{}_{}/'.format(dir, cell, channel)
    file_start = '{}_{:03d}_'.format(cell, cycle)
    return sub_dir + file_start

def check_files(ptf_files):
    """
    检查文件是否存在，打印不存在的文件路径
    
    参数:
    ----
    ptf_files : list of str
        文件路径列表
    """
    for ptf in ptf_files:
        if not os.path.isfile(ptf):
            print(ptf)  # 打印不存在的文件路径
        else:
            continue
    return



# ============================================================================
# 特征提取函数
# ============================================================================

def discharge_features(ptf, cycle, cap_curve_norm=None):
    """
    从放电数据中提取特征
    
    功能:
    ----
    1. 读取GCPL放电数据文件
    2. 提取放电容量、电压特征
    3. 拟合S型曲线（sigmoid）到容量-电压曲线
    4. 计算容量曲线的变化特征（方差、四分位距）
    5. 计算放电速率和输出能量
    
    参数:
    ----
    ptf : str
        数据文件路径
    cycle : int
        循环编号（用于判断是否为初始循环）
    cap_curve_norm : ndarray, optional
        归一化的容量曲线（用于计算变化）
    
    返回:
    ----
    cap : float
        放电容量 (Ah)
    features : ndarray
        特征向量，包含7个元素:
        [sigmoid_a, sigmoid_b, sigmoid_c, v0, v1, log_var, log_iqr]
    e_out : float
        输出能量 (Wh)
    d_rate : float
        放电速率 (A)
    cap_curve : ndarray
        容量曲线（1000个点）
    
    如果文件不存在，返回None
    """

    if os.path.isfile(ptf):
        # open txt file
        df = pd.read_csv(ptf, delimiter='\t')
        while len(df.columns) != 7:
            print('Wrong parameters exported for {}. Re-export txt file using rl-gcpl template.'.format(ptf))
            time.sleep(120)
            df = pd.read_csv(ptf, delimiter='\t')

        df.columns = column_map['GCPL']
        dfd = df.loc[(df['i'] < 0) & (df['ox/red']==0)]

        voltage = dfd['ewe'].to_numpy()
        capacity = dfd['capacity'].to_numpy()

        # extract starting (resting) charge voltage, v0
        v0 = np.max(voltage)

        v_mid = 0.5*(np.max(voltage) + np.min(voltage))
        v = voltage - v_mid
        cap = capacity[-1]

        # Initial guess at parameters - helps find the optimal solution
        p0 = [cap, 1.0, 0.01]

        try:
            sigmoid_params, _ = curve_fit(general_sigmoid, v, capacity, p0)
        except RuntimeError:
            sigmoid_params = np.array(p0)

        # extract features from the capacity-voltage discharge curve
        cap_curve = cv_features(capacity, v)

        if cycle == 0:
            log_var = -1.0
            log_iqr = -1.0

        else:
            delta_c = cap_curve - cap_curve_norm
            if np.var(delta_c) <= 1.0e-01:
                log_var = -1.0
                log_iqr = -1.0
            else:
                log_var = np.log10(np.var(delta_c))
                log_iqr = np.log10(iqr(delta_c))

        # extract final (resting) discharge voltage, v1
        t1 = dfd['time'].to_numpy().max()
        dfr = df.loc[df['time']>t1]
        v1 = dfr['ewe'].to_numpy().max()

        # compute discharge rate
        d_rate = dfd['i'].to_numpy()[-10]*-1

        # compute energy outputted
        power = dfd['power'].to_numpy()
        time = dfd['time'].to_numpy()
        e_out = -simpson(power, time)

        features = np.hstack([sigmoid_params, np.array([v0, v1, log_var, log_iqr])])

        return cap, features, e_out, d_rate, cap_curve

    else:
        return None

def general_sigmoid(x, a, b, c):
    """
    广义S型（sigmoid）函数
    
    功能:
    ----
    用于拟合容量-电压曲线的S型关系
    
    参数:
    ----
    x : ndarray
        输入变量（电压）
    a : float
        振幅参数（最大容量）
    b : float
        陡度参数（曲线斜率）
    c : float
        中点参数（拐点位置）
    
    返回:
    ----
    ndarray : S型函数值
    """
    return a / (1.0 + np.exp(b * (x - c)))

def cv_features(capacity, v):
    """
    从容量-电压曲线中提取特征
    
    功能:
    ----
    通过样条插值将容量-电压曲线标准化为1000个均匀分布的点
    这样可以比较不同循环之间的曲线变化
    
    参数:
    ----
    capacity : ndarray
        容量数组
    v : ndarray
        电压数组（相对于中点）
    
    返回:
    ----
    c : ndarray
        插值后的容量曲线（1000个点）
    
    注:
    ---
    该方法参考了P. Attia等人的工作
    """
    # 在最小和最大电压之间生成1000个均匀分布的点
    x = np.linspace(np.min(v), np.max(v), 1000)
    # 创建插值函数
    f = interp1d(v, capacity)
    # 计算插值后的容量值
    c = f(x)

    return c

def charge_features(ptf):
    """
    从充电数据中提取特征
    
    功能:
    ----
    1. 读取GCPL充电数据文件
    2. 识别两阶段充电协议（CC-CV：恒流-恒压）
    3. 计算每个阶段的充电时间、速率和容量
    4. 提取开路电压（OCV）
    
    参数:
    ----
    ptf : str
        充电数据文件路径
    
    返回:
    ----
    t_charge : float
        总充电时间（小时）
    charge_cap : float
        充电容量 (Ah)
    c1_rate : float
        第一阶段充电速率 (A)
    c2_rate : float
        第二阶段充电速率 (A)
    t1 : float
        第一阶段充电时间（小时）
    t2 : float
        第二阶段充电时间（小时）
    ocv : float
        开路电压 (V)
    
    如果文件不存在，返回None
    """
    # 打开并读取txt文件
    if os.path.isfile(ptf):
        df = pd.read_csv(ptf, delimiter='\t')
        # 检查列数是否正确，如果不正确则等待重新导出
        while len(df.columns) != 7:
            print('Wrong parameters exported for {}. Re-export txt file using rl-gcpl template.'.format(ptf))
            time.sleep(120)
            df = pd.read_csv(ptf, delimiter='\t')

        # 设置列名
        df.columns = column_map['GCPL']
        
        # 提取开路电压（前5个数据点的平均值）
        try:
            ocv = np.mean(df.iloc[0:5].ewe.to_numpy())
        except:
            ocv = None

        # 计算第一和第二充电阶段的时间
        # 找到电流为0的索引（表示充电阶段之间的休息期）
        idx = df.loc[df.i == 0.0].index.to_numpy()
        # 找到索引不连续的位置（表示充电阶段的边界）
        change = np.array(np.where((idx[1:] - idx[:-1]) != 1)).reshape(-1)
        assert change.shape[0] == 2, "Error in computing time to change"
        
        # 识别两个充电阶段的起始和结束索引
        idx_start1 = idx[change[0]]      # 第一阶段开始前的休息结束
        idx_change1 = idx[change[0]+1]   # 第一阶段结束后的休息开始
        idx_start2 = idx[change[1]]      # 第二阶段开始前的休息结束
        idx_change2 = idx[change[1]+1]   # 第二阶段结束后的休息开始

        # 计算充电阶段的时间（这里计算的是休息时间，后面会重新计算）
        t1 = df.iloc[idx_change1].time - df.iloc[idx_start1].time
        t2 = df.iloc[idx_change2-1].time - df.iloc[idx_start2].time
        
        # 提取充电数据（电流 > 0）
        df_charge = df.loc[df.i > 0.0]
        # 第一阶段充电数据
        df1_charge = df.loc[(df.i > 0.0) & (df.time < df.iloc[idx_change1].time)]
        # 第二阶段充电数据
        df2_charge = df.loc[(df.i > 0.0) & (df.time > df.iloc[idx_start2].time)]

        # 重新计算实际充电时间
        t1 = df1_charge.time.max() - df1_charge.time.min()
        t2 = df2_charge.time.max() - df2_charge.time.min()

        # 计算平均充电速率
        c1_rate = np.mean(df1_charge.i.to_numpy())  # 第一阶段（恒流）
        c2_rate = np.mean(df2_charge.i.to_numpy())  # 第二阶段（恒压）

        # 提取各阶段的容量
        cap1 = df1_charge.capacity.max()
        cap2 = df2_charge.capacity.max()
        computed_cap = (c1_rate * t1 + c2_rate * t2) / 3600  # 计算的总容量

        # 计算总充电时间（扣除两阶段之间的休息时间）
        t_charge = df_charge.time.max() - df_charge.time.min() - (df.iloc[idx_start2].time - df.iloc[idx_change1].time)
        # 实际测量的充电容量
        charge_cap = df_charge.capacity.max()

        # 返回充电特征（时间转换为小时）
        return t_charge / 3600, charge_cap, c1_rate, c2_rate, t1/3600, t2/3600, ocv

    else:
        return None

# ============================================================================
# 等效电路模型（ECM）拟合函数
# ============================================================================

def overall_fitness_er(p, freq, re_z, im_z):
    """
    扩展Randles模型的总体拟合函数
    
    功能:
    ----
    计算模型预测阻抗与实际测量阻抗之间的误差
    用于最小二乘优化
    
    参数:
    ----
    p : tuple
        模型参数 (r1, r2, c2, r3, c3, A)
        - r1: 串联电阻
        - r2: RC并联电路的电阻
        - c2: RC并联电路的电容
        - r3: Randles电路的电阻
        - c3: Randles电路的电容
        - A: Warburg系数（扩散阻抗）
    freq : ndarray
        频率数组 (Hz)
    re_z : ndarray
        实际测量的实部阻抗
    im_z : ndarray
        实际测量的虚部阻抗
    
    返回:
    ----
    penalty : ndarray
        误差惩罚（包括实部、虚部、模和相位的误差）
    """
    (r1, r2, c2, r3, c3, A) = p

    # 计算模型预测的阻抗
    computed_re_z = real_z_er(freq, r1, r2, c2, r3, c3, A)
    computed_im_z = imaginary_z_er(freq, r2, c2, r3, c3, A)

    # 计算阻抗的模和相位
    computed_mod_z = np.sqrt(computed_re_z**2 + computed_im_z**2)
    computed_phase_z = np.arctan(computed_im_z / computed_re_z)
    mod_z = np.sqrt(re_z**2 + im_z**2)
    phase_z = np.arctan(im_z / re_z)

    # 计算总误差（实部、虚部、模、相位的平方误差之和）
    penalty = (re_z - computed_re_z)**2 + (im_z - computed_im_z)**2 + (mod_z - computed_mod_z)**2 + (phase_z - computed_phase_z)**2

    return penalty

def real_z_er(freq, r1, r2, c2, r3, c3, A):
    """
    扩展Randles模型的实部阻抗
    
    模型结构: R1 + R2||C2 + R3||CPE
    """
    w = 2 * np.pi * freq  # 角频率

    return r1 + r2 / (1 + (w*r2*c2)**2) + (r3 + A/w**0.5) / ((1 + A*c3*w**0.5)**2 + (c3*w*(r3 + A/w**0.5))**2)

def imaginary_z_er(freq, r2, c2, r3, c3, A):
    """
    扩展Randles模型的虚部阻抗
    """
    w = 2 * np.pi * freq  # 角频率

    return w*r2**2*c2 / (1 + (w*r2*c2)**2) + (w*c3*(r3 + A/w**0.5)**2 + A**2*c3 + A/w**0.5) / ((1 + A*c3*w**0.5)**2 + (c3*w*(r3 + A/w**0.5))**2)

def general_sigmoid(x, a, b, c):
    """广义S型函数（重复定义，用于向后兼容）"""
    return a / (1.0 + np.exp(b * (x - c)))

def overall_fitness_r(p, freq, re_z, im_z):
    """
    Randles模型的总体拟合函数
    
    功能:
    ----
    计算模型预测阻抗与实际测量阻抗之间的归一化误差
    用于最小二乘优化
    
    参数:
    ----
    p : tuple
        模型参数 (r1, r3, c3, A)
        - r1: 串联电阻
        - r3: Randles电路的电阻
        - c3: Randles电路的电容
        - A: Warburg系数
    freq : ndarray
        频率数组 (Hz)
    re_z : ndarray
        实际测量的实部阻抗
    im_z : ndarray
        实际测量的虚部阻抗
    
    返回:
    ----
    penalty : ndarray
        归一化误差惩罚
    """
    (r1, r3, c3, A) = p

    # 计算模型预测的阻抗
    computed_re_z = real_z_r(freq, r1, r3, c3, A)
    computed_im_z = imaginary_z_r(freq, r3, c3, A)

    # 计算阻抗的模和相位
    computed_mod_z = np.sqrt(computed_re_z**2 + computed_im_z**2)
    computed_phase_z = np.arctan(computed_im_z / computed_re_z)
    mod_z = np.sqrt(re_z**2 + im_z**2)
    phase_z = np.arctan(im_z / re_z)

    # 计算归一化误差（除以各自的均值进行归一化）
    penalty = (re_z - computed_re_z)**2 / (np.mean(re_z))**2 + \
              (im_z - computed_im_z)**2 / (np.mean(im_z))**2 + \
              (mod_z - computed_mod_z)**2 / (np.mean(mod_z))**2 + \
              (phase_z - computed_phase_z)**2 / (np.mean(phase_z))**2

    return penalty

def real_z_r(freq, r1, r3, c3, A):
    """
    Randles模型的实部阻抗
    
    模型结构: R1 + R3||CPE
    """
    w = 2 * np.pi * freq  # 角频率

    return r1 + (r3 + A/w**0.5) / ((1 + A*c3*w**0.5)**2 + (c3*w*(r3 + A/w**0.5))**2)

def imaginary_z_r(freq, r3, c3, A):
    """
    Randles模型的虚部阻抗
    """
    w = 2 * np.pi * freq  # 角频率

    return (w*c3*(r3 + A/w**0.5)**2 + A**2*c3 + A/w**0.5) / ((1 + A*c3*w**0.5)**2 + (c3*w*(r3 + A/w**0.5))**2)

def extract_features(log_freq, re_z, im_z):
    """
    从EIS数据中提取等效电路模型参数（旧版本，未使用）
    
    参数:
    ----
    log_freq : ndarray
        对数频率
    re_z : ndarray
        实部阻抗
    im_z : ndarray
        虚部阻抗
    
    返回:
    ----
    popt1 : ndarray
        实部阻抗拟合参数
    popt2 : ndarray
        虚部阻抗拟合参数
    """
    assert log_freq.shape == re_z.shape == im_z.shape

    freq = 10**log_freq

    popt2, pcov2 = curve_fit(imaginary_z, freq, im_z)
    popt1, pcov1 = curve_fit(real_z, freq, re_z, p0=np.insert(popt2, 0, re_z.min()))

    return popt1, popt2

def eis_features(path, new_log_freq=np.linspace(-1.66, 3.9, 100), n_repeats=1):
    """
    从EIS数据文件中提取特征
    
    功能:
    ----
    1. 读取GEIS数据文件
    2. 提取频率、实部和虚部阻抗
    3. 对多次重复测量取平均
    4. 插值到统一的对数频率网格
    5. 返回200维特征向量（100个实部 + 100个虚部）
    
    参数:
    ----
    path : str
        GEIS数据文件路径
    new_log_freq : ndarray
        新的对数频率网格，默认从-1.66到3.9的100个点
        对应频率范围约0.02 Hz到8000 Hz
    n_repeats : int
        重复测量次数，默认为1
    
    返回:
    ----
    ndarray : EIS特征向量，shape (200,)
        前100个元素为实部阻抗，后100个元素为虚部阻抗
    """
    # 读取EIS数据文件
    df = pd.read_csv(path, delimiter='\t')
    df.columns = column_map['GEIS']

    # 提取阻抗数据
    re_z = df['re_z'].to_numpy()  # 实部阻抗
    im_z = df['-im_z'].to_numpy()  # 虚部阻抗
    freq = df['freq'].to_numpy()  # 频率

    # 如果有多次重复测量，取平均值
    re_z = np.mean(re_z.reshape(n_repeats, -1), axis=0)
    im_z = np.mean(im_z.reshape(n_repeats, -1), axis=0)
    freq = np.mean(freq.reshape(n_repeats, -1), axis=0)
    log_freq = np.log10(freq)  # 转换为对数频率

    # 使用三次样条插值到新的频率网格
    f1 = interp1d(log_freq, re_z, kind='cubic')
    f2 = interp1d(log_freq, im_z, kind='cubic')

    # 计算新频率点的阻抗值
    re_z = f1(new_log_freq)
    im_z = f2(new_log_freq)

    # 拼接实部和虚部，返回200维特征向量
    return np.hstack((re_z, im_z)).reshape(-1)


def eis_to_ecm(eis, new_log_freq, feature_type='randles'):
    """
    将EIS特征转换为等效电路模型（ECM）参数
    
    功能:
    ----
    使用最小二乘法将EIS阻抗谱拟合到等效电路模型
    将200维EIS特征降维到4维或6维ECM参数
    
    参数:
    ----
    eis : ndarray, shape (n_samples, 200)
        EIS特征矩阵（前100个为实部，后100个为虚部）
    new_log_freq : ndarray
        对数频率网格
    feature_type : str
        模型类型，可选：
        - 'randles': Randles模型（4参数：r1, r3, c3, A）
        - 'extended-randles': 扩展Randles模型（6参数：r1, r2, c2, r3, c3, A）
    
    返回:
    ----
    x : ndarray, shape (n_samples, n_params)
        ECM参数矩阵
        - Randles模型: (n_samples, 4)
        - 扩展Randles模型: (n_samples, 6)
    """
    # 分离实部和虚部阻抗
    n_features = eis.shape[1] // 2
    re_z = eis[:, :n_features]  # 前100个为实部
    im_z = eis[:, n_features:]  # 后100个为虚部
    assert re_z.shape == im_z.shape

    if feature_type == 'extended-randles':
        # 扩展Randles模型：6个参数
        x = np.zeros((re_z.shape[0], 6))
        for i in range(re_z.shape[0]):
            # 使用最小二乘法拟合
            # x0: 初始猜测值
            # bounds: 参数边界（所有参数必须为正）
            ls = least_squares(
                overall_fitness_er, 
                x0=(1, 0.1, 0.1, 1, 1, 0.3), 
                bounds=([0, 0, 0, 0, 0, 0], [100, 100, 100, 100, 100, 100]), 
                args=(10**new_log_freq, re_z[i, :], im_z[i, :])
            )
            x[i, :] = ls.x.reshape(-1)

    elif feature_type == 'randles':
        # Randles模型：4个参数
        x = np.zeros((re_z.shape[0], 4))
        for i in range(re_z.shape[0]):
            # 使用最小二乘法拟合
            ls = least_squares(
                overall_fitness_r, 
                x0=(1, 0.1, 0.1, 0.3), 
                bounds=([0, 0, 0, 0], [100, 100, 100, 100]), 
                args=(10**new_log_freq, re_z[i, :], im_z[i, :])
            )
            x[i, :] = ls.x.reshape(-1)
    return x


def extract_input(input_name, data):
    """
    根据特征名称提取输入特征矩阵
    
    功能:
    ----
    根据指定的特征组合名称，从数据元组中提取并组合相应的特征
    支持多种特征组合，用于不同的实验配置
    
    参数:
    ----
    input_name : str
        特征组合名称，支持的选项包括：
        - 'eis-actions': EIS + 充放电速率
        - 'cvfs-actions': 容量-电压曲线 + 充放电速率
        - 'eis-cvfs-actions': EIS + 容量-电压曲线 + 充放电速率
        - 'eis-cvfs-ct-actions': EIS + 容量-电压曲线 + 累积吞吐量 + 充放电速率
        - 'eis-cvfs-ct-c-actions': EIS + 容量-电压曲线 + 累积吞吐量 + 上一容量 + 充放电速率
        - 'ecmr-actions': Randles模型参数 + 充放电速率
        - 'ecmer-actions': 扩展Randles模型参数 + 充放电速率
        - 'ecmr-cvfs-actions': Randles + 容量-电压曲线 + 充放电速率
        - 'ecmer-cvfs-actions': 扩展Randles + 容量-电压曲线 + 充放电速率
        - 'ecmr-cvfs-ct-c-actions': Randles + 容量-电压曲线 + 累积吞吐量 + 上一容量 + 充放电速率
        - 'ecmer-cvfs-ct-c-actions': 扩展Randles + 容量-电压曲线 + 累积吞吐量 + 上一容量 + 充放电速率
        - 'c-actions': 上一容量 + 充放电速率
        - 'soh-actions': SOH + 充放电速率
        - 'eis-c-actions': EIS + 上一容量 + 充放电速率
        - 'eis-soh-actions': EIS + SOH + 充放电速率
        - 'actions': 仅充放电速率
        - 'eis': 仅EIS
        - 'cvfs': 仅容量-电压曲线
        - 'ocv': 仅开路电压
        - 'ct': 仅累积吞吐量
        - 'c': 仅上一容量
        - 'soh': 仅SOH
    data : tuple
        特征数据元组，包含9个元素：
        (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, 
         d_rates, c1_rates, c2_rates)
    
    返回:
    ----
    x : ndarray, shape (n_samples, n_features)
        组合后的输入特征矩阵
        如果input_name不在支持列表中，返回None
    """
    # 解包数据元组
    (c, soh, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates) = data

    # 构建动作特征：放电速率 + 第一阶段充电速率 + 第二阶段充电速率
    actions = np.hstack([d_rates.reshape(-1, 1), c1_rates.reshape(-1, 1), c2_rates.reshape(-1, 1)])

    if input_name == 'eis-cvfs-ct-actions':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)
    elif input_name == 'eis-cvfs-ct-c-actions':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        states = np.concatenate((states, c.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-cvfs-actions':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-ct-actions':
        states = np.concatenate((eis_ds, cap_throughputs.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-actions':
        states = eis_ds
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'cvfs-actions':
        states = cvfs
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ct-actions':
        states = cap_throughputs.reshape(-1, 1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'cvfs-ct-actions':
        states = np.concatenate((cvfs, cap_throughputs.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-cvfs-ct':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        x = states

    elif input_name == 'eis-cvfs':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        x = states

    elif input_name == 'actions':
        x = actions

    elif input_name == 'eis':
        x = eis_ds

    elif input_name == 'ecmr-cvfs-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='randles')
        states = np.concatenate((states, cvfs), axis=1)
        print(states.shape)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmr-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='randles')
        print(states.shape)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmer-cvfs-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='extended-randles')
        states = np.concatenate((states, cvfs), axis=1)
        print(states.shape)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmer-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='extended-randles')
        print(states.shape)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmr-cvfs-ct-c-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='randles')
        states = np.concatenate((states, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        states = np.concatenate((states, c.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmer-cvfs-ct-c-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='extended-randles')
        states = np.concatenate((states, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        states = np.concatenate((states, c.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'cvfs':
        x = cvfs

    elif input_name == 'ocv':
        x = ocvs.reshape(-1, 1)

    elif input_name == 'ct':
        x = cap_throughputs.reshape(-1, 1)

    elif input_name == 'c':
        x = c.reshape(-1, 1)

    elif input_name == 'soh':
        x = soh.reshape(-1, 1)

    elif input_name == 'c-actions':
        states = c.reshape(-1, 1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'soh-actions':
        states = soh.reshape(-1, 1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-c-actions':
        states = np.concatenate((eis_ds, c.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-soh-actions':
        states = np.concatenate((eis_ds, soh.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    else:
        print('Choose different input name')
        x = None

    return x

# ============================================================================
# 电池识别和N步数据提取函数
# ============================================================================

def identify_cells(experiment):
    """
    识别实验对应的电池-通道映射关系
    
    功能:
    ----
    根据实验类型返回通道号到电池ID列表的映射
    
    参数:
    ----
    experiment : str
        实验类型，可选：
        - 'variable-discharge': 可变放电实验
        - 'fixed-discharge': 固定放电实验
        - 'both': 同时包含两种实验
        - 'both-variable': 仅可变放电电池
    
    返回:
    ----
    cell_map : dict or None
        通道号到电池ID列表的映射
        格式: {通道号: [电池ID列表]}
        如果实验类型不支持，返回None
    """
    if experiment == 'variable-discharge':
        cell_map = {1:['PJ097','PJ105','PJ145'],
                     2:['PJ098','PJ106','PJ146'],
                     3:['PJ099','PJ107','PJ147'],
                     4:['PJ100','PJ108','PJ148'],
                     5:['PJ101','PJ109','PJ149'],
                     6:['PJ102','PJ110','PJ150'],
                     7:['PJ103','PJ111','PJ151'],
                     8:['PJ104','PJ112','PJ152']}

    elif experiment == 'fixed-discharge':
        cell_map = {1:['PJ121', 'PJ129'],
                2:['PJ122', 'PJ130'],
                3:['PJ125', 'PJ131'],
                4:['PJ126', 'PJ132'],
                5:['PJ123', 'PJ133'],
                6:['PJ124', 'PJ134'],
                7:['PJ127', 'PJ135'],
                8:['PJ128', 'PJ136'],}

    elif experiment == 'both':
        cell_map = {1:['PJ097','PJ105', 'PJ121', 'PJ129', 'PJ145'],
                2:['PJ098','PJ106', 'PJ122', 'PJ130', 'PJ146'],
                3:['PJ099','PJ107', 'PJ125', 'PJ131', 'PJ147'],
                4:['PJ100','PJ108','PJ126', 'PJ132', 'PJ148'],
                5:['PJ101','PJ109','PJ123', 'PJ133', 'PJ149'],
                6:['PJ102','PJ110','PJ124', 'PJ134', 'PJ150'],
                7:['PJ103','PJ111','PJ127', 'PJ135', 'PJ151'],
                8:['PJ104','PJ112','PJ128', 'PJ136', 'PJ152'],}
    elif experiment == 'both-variable':
        cell_map = {1:['PJ097','PJ105', 'PJ121', 'PJ129', 'PJ145'],
                2:['PJ098','PJ106', 'PJ122', 'PJ130', 'PJ146'],
                3:['PJ099','PJ107', 'PJ125', 'PJ131', 'PJ147'],
                4:['PJ100','PJ108','PJ126', 'PJ132', 'PJ148'],
                5:['PJ101','PJ109','PJ123', 'PJ133', 'PJ149'],
                6:['PJ102','PJ110','PJ124', 'PJ134', 'PJ150'],
                7:['PJ103','PJ111','PJ127', 'PJ135', 'PJ151'],
                8:['PJ104','PJ112','PJ128', 'PJ136', 'PJ152'],}

    else:
        cell_map = None

    return cell_map

def extract_n_step_data(experiment, channels):
    """
    提取N步前瞻预测所需的数据
    
    功能:
    ----
    与extract_data类似，但返回字典格式的数据，按电池组织
    用于N步前瞻预测实验，其中需要预测未来第N个循环的容量
    
    参数:
    ----
    experiment : str
        实验类型
    channels : list of int
        要处理的测试通道列表
    
    返回:
    ----
    data : tuple of dicts
        包含4个字典的元组 (states, actions, cycles, cap_ds)
        每个字典的键为电池ID，值为对应的数据数组：
        - states: EIS特征矩阵，shape (n_cycles, 200)
        - actions: 充放电速率矩阵，shape (n_cycles, 3)
        - cycles: 循环编号数组，shape (n_cycles,)
        - cap_ds: 放电容量数组，shape (n_cycles,)
    """

    # 获取电池-通道映射
    cell_map = identify_cells(experiment)
    n_repeats = 1  # EIS测量重复次数
    n_steps = 4    # 每个大循环包含的子步骤数
    new_log_freq = np.linspace(-1.66, 3.9, 100)  # 对数频率网格

    freq1 = np.log10(2.16)
    freq2 = np.log10(17.8)
    idx_freq1 = np.argmin(np.abs(new_log_freq-freq1))
    idx_freq2 = np.argmin(np.abs(new_log_freq-freq2))

    nl = 0

    states = {}
    actions = {}
    cycles = {}
    cap_ds = {}

    for channel in channels:

        cells = cell_map[channel]

        for cell in cells:

            cell_states = []
            cell_actions = []
            cell_cycles = []
            cell_cap_ds = []

            cell_no = int(cell[-3:])
            #cmap = plt.get_cmap(name, 70)
            #cell_ars = []
            dir = 'raw-data/{}/'.format(experiment_map[cell])

            # First get the initial EIS and GCPL
            cycle = 0
            path = path_to_file(channel, cell, cycle, dir)
            ptd = '{}{}/'.format(dir, cell)
            filestart = '{}_{:03d}a_'.format(cell, cycle)

            if cell_no in [145, 146, 147, 148]:
                start_cycle = 3
            else:
                start_cycle = 2

            ptf_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, 4, 'GEIS', channel)
            ptf_cv = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, 3, 'GCPL', channel)

            # Get features of discharge capacity-voltage curve

            cap0, cvf0, e_out, d_rate, cap_curve_norm = discharge_features(ptf_cv, cycle)

            print('Cell PJ{:03d}\t C0 {:.2f}'.format(cell_no, cap0))
            oldcap = e_out

            # Get initial features of discharge EIS spectrum
            x_eis = eis_features(ptf_eis, new_log_freq=new_log_freq, n_repeats=n_repeats)
            eis0 = x_eis

            for cycle in range(start_cycle, start_cycle+30):
                path = path_to_file(channel, cell, cycle, dir)
                ptd = '{}{}/'.format(dir, cell)
                filestart = '{}_{:03d}_'.format(cell, cycle)

                geis_discharge_no = 1
                gcpl_charge_no = 2
                geis_charge_no = 3
                gcpl_discharge_no = 4

                for step in range(n_steps):
                    true_cycle = 1+(cycle-2)*n_steps + step

                    ptf_discharge_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, geis_discharge_no + step*4, 'GEIS', channel)
                    ptf_charge_gcpl = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, gcpl_charge_no + step*4, 'GCPL', channel)
                    ptf_charge_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, geis_charge_no + step*4, 'GEIS', channel)
                    ptf_discharge_gcpl = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, gcpl_discharge_no + step*4, 'GCPL', channel)

                    ptf_files = [ptf_discharge_eis, ptf_charge_gcpl, ptf_charge_eis, ptf_discharge_gcpl]

                    #check_files(ptf_files)

                    # Get features of discharge EIS spectrum
                    try:
                        eis_d = eis_features(ptf_discharge_eis, new_log_freq=new_log_freq, n_repeats=1)
                    except:
                        eis_d = None

                    # Compute the time to charge
                    try:
                        t_charge, cap_c, c1_rate, c2_rate, t1_charge, t2_charge, ocv = charge_features(ptf_charge_gcpl)

                    except:
                        t_charge = None
                        cap_c = None
                        c1_rate = None
                        c2_rate = None

                    # Get features of discharge capacity-voltage curve
                    try:
                        cap_d, cvf, _, d_rate, _ = discharge_features(ptf_discharge_gcpl, true_cycle, cap_curve_norm)
                    except:
                        cap_d = None
                        cvf = None
                        d_rate = None

                    if any(elem is None for elem in (eis_d, t_charge, cap_c, cap_d)):
                        #print('{:02d}\t{} {} {}'.format(true_cycle, t_charge, cap_c, cap_d))
                        continue
                    else:
                        cell_states.append(eis_d)
                        cell_actions.append(np.array([d_rate, c1_rate, c2_rate]).reshape(1, -1))
                        cell_cap_ds.append(cap_d)
                        cell_cycles.append(1+(cycle-2)*n_steps + step)
                        #print('{:02d}\t{:.1f}\t{:.1f}\t{:.3f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(1+(cycle-2)*n_steps + step, c1_rate, c2_rate, ocv, t_charge, t1, t2, t_charge-t1-t2))

            cell_states = np.vstack(np.array(cell_states))
            cell_actions = np.vstack(np.array(cell_actions))
            cell_cycles = np.array(cell_cycles)
            cell_cap_ds = np.array(cell_cap_ds)

            states[cell] = cell_states
            actions[cell] = cell_actions
            cycles[cell] = cell_cycles
            cap_ds[cell] = cell_cap_ds

    data = (states, actions, cycles, cap_ds)

    return data

def ensemble_predict(x, experiment, input_name, n_ensembles=10):
    y_preds = []
    dts = 'experiments/results/{}'.format(experiment)
    experiment_name = '{}_n1_xgb'.format(input_name)
    
    # Auto-detect actual number of model files
    actual_ensembles = 0
    for i in range(n_ensembles):
        model_path = '{}/models/{}_{}.pkl'.format(dts, experiment_name, i)
        if os.path.isfile(model_path):
            actual_ensembles = i + 1
        else:
            break
    
    if actual_ensembles == 0:
        raise FileNotFoundError('No model files found for {}'.format(experiment_name))
    
    print('Loading {} ensemble models...'.format(actual_ensembles))
    
    for i in range(actual_ensembles):
        with open('{}/models/{}_{}.pkl'.format(dts, experiment_name, i), 'rb') as f:
            regr = pickle.load(f)
            y_pred = regr.predict(x)
            y_preds.append(y_pred.reshape(1, y_pred.shape[0], -1))
    y_preds = np.vstack(y_preds)
    y_pred = np.mean(y_preds, axis=0)
    y_pred_err = np.sqrt(np.var(y_preds, axis=0))
    return y_pred.reshape(-1), y_pred_err.reshape(-1)
