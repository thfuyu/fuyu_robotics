import os
import pandas as pd
import sim_analysis  # 确保 sim_analysis 已经导入
import numpy as np
import matplotlib.pyplot as plt

# 创建输出文件夹
output_dir = r'D:\desktop1\2-12-surgecast\plume\visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取数据文件
puff_filename = r'D:\desktop1\2-12-surgecast\plume\puff_data_test.pickle'
wind_filename = r'D:\desktop1\2-12-surgecast\plume\wind_data_test.pickle'

# 加载数据
data_puffs = pd.read_pickle(puff_filename)
data_wind = pd.read_pickle(wind_filename)

# 获取时间序列
t_min = data_wind['time'].min()  # 获取最小时间戳（即起始时间）
t_max = data_wind['time'].max()  # 获取最大时间戳（即结束时间）

# 设置时间步长为0.01秒
time_steps = np.arange(t_min, t_max, 0.01)

# 计算浓度
data_puffs = sim_analysis.calculate_concentrations(data_puffs)

# 绘制每个时间步的图像
for idx, t_val in enumerate(time_steps):
    # 绘制图像
    fig, ax = sim_analysis.plot_puffs_and_wind_vectors(
        data_puffs, 
        data_wind, 
        t_val, 
        fname='', 
        plotsize=(8,8)
    )
    # 设置坐标范围
    ax.set_xlim(-1, 12)  # 设置 x 轴范围
    ax.set_ylim(-4, 4)  # 设置 y 轴范围

    # 保存图像
    filename = os.path.join(output_dir, f'puff_wind_vectors_t{t_val:3.3f}.png')
    fig.savefig(filename)

    # 输出文件编号和保存路径
    print(f"保存图像 {idx+1}，文件名: {filename}")

    # 清理图像以释放内存
    plt.close(fig)

print(f"所有图像已保存至: {output_dir}")
