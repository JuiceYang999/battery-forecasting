import subprocess
import sys
import time
import os

# --- [新代码] 自动定位项目根目录 ---
# 1. 获取 'run_all_experiments.py' 脚本所在的目录 (即 '.../experiments')
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取该目录的父目录 (即 '.../（跑代码）'，这是我们的项目根)
project_root = os.path.dirname(script_dir)

# 3. 切换 Python 的当前工作目录到项目根目录
os.chdir(project_root)
print(f"--- [系统] 已自动切换工作目录至项目根目录: {project_root} ---")
# --- [新代码] 自动定位完成 ---


# --- 脚本执行清单 ---
# [修改] 列表中所有脚本都必须包含 'experiments/' 路径
scripts_to_run = [
    # --- 1. 训练 "variable-discharge" (Type 1) 模型 ---
    # 对应论文 Table 1, Fig 3, Fig 6 的基础模型
    "experiments/1variable-discharge-train.py",
    
    # --- 2. 运行 "fixed-discharge" (Fig 6 鲁棒性) 预测 ---
    #    (依赖于 1variable-discharge-train.py 生成的模型)
    "experiments/1fixed-discharge-predict.py",
    
    # --- 3. 运行 "next-cycle" (Table 1) 核心性能复现 ---
    #    (这是一个完整的训练+评估脚本)
    "experiments/1next-cycle-capacity.py",
    
    # --- 4. 运行 "data-efficiency" (Fig 5) 数据效率复现 ---
    #    (这是一个完整的训练+评估脚本)
    "experiments/1data-efficiency.py",
    
    # --- 5. 运行 "n-step-lookahead" (Fig 4) 长期预测复现 ---
    #    (这是一个完整的训练+评估脚本)
    "experiments/1n-step-lookahead.py",
    
    # --- 6. 训练 "vd2" (Type 2) 模型 ---
    #    (对应论文 Table 2)
    "experiments/2vd2-train.py",
    
    # --- 7. 运行 "vd2 next-cycle" (Table 2) 核心性能复现 ---
    #    (这是一个完整的训练+评估脚本)
    "experiments/2next-cycle-capacity-vd2.py",
    
    # --- 8. 运行 "vd2-35C" (Table 3 温度鲁棒性) 预测 ---
    #    (依赖于 1variable-discharge-train.py 生成的模型)
    "experiments/2vd2-predict.py"
]

def run_experiment(script_name):
    """
    使用当前的 Python 解释器运行一个指定的脚本。
    """
    print("="*70)
    print(f"--- [开始] 正在运行: {script_name} ---")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # [修改] 现在脚本名称是 'experiments/...'
        # 因为我们 CWD 到了项目根目录，这个相对路径是正确的
        # 并且 Python 启动时会将根目录加入 sys.path，'import utils' 就能成功
        subprocess.run([sys.executable, script_name], check=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        print("="*70)
        print(f"--- [成功] 完成: {script_name} (用时: {duration:.2f} 秒) ---")
        print("="*70)
        print("\n" * 2)
        return True
        
    except subprocess.CalledProcessError as e:
        print("="*70)
        print(f"!!! [错误] 运行 {script_name} 失败! !!!")
        print(f"返回代码: {e.returncode}")
        print(f"输出:\n{e.stdout}")
        print(f"错误信息:\n{e.stderr}")
        print("="*70)
        print("!!! 自动执行已终止 !!!")
        print("\n" * 2)
        return False
    except FileNotFoundError as e:
        print("="*70)
        print(f"!!! [错误] 找不到文件: {e.filename} !!!")
        print(f"Python 解释器路径: {sys.executable}")
        print(f"当前工作目录: {os.getcwd()}")
        print("!!! 自动执行已终止 !!!")
        print("\n" * 2)
        return False

if __name__ == "__main__":
    print("--- 启动论文复现自动执行流程 ---")
    print(f"将要按顺序执行 {len(scripts_to_run)} 个实验脚本...")
    print("请确保：")
    print("1. 你已激活了正确的 Conda 环境。")
    print("2. 你的数据路径已在 utils/ 文件夹中配置正确。")
    print("-" * 70 + "\n")
    
    overall_start_time = time.time()
    
    for i, script in enumerate(scripts_to_run):
        print(f"--- 任务 {i+1} / {len(scripts_to_run)} ---")
        success = run_experiment(script)
        if not success:
            print("--- 流程因错误而提前终止 ---")
            break
            
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    
    print("--- 所有任务执行完毕 ---")
    print(f"总计用时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")