import gas_diffusion_good as gd
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import surgecast as sc
import threading
import time
import os
import logging  # 用于日志记录
import parameters  # 导入参数模块

# 设置区域和源范围
box_size = parameters.box_size  # 模拟空间的尺寸
source_box_size = parameters.source_box_size  # 源范围的尺寸（未在代码中使用）

# 环境参数
max_particles = parameters.max_particles  # 最大粒子数量
release_rate = parameters.release_rate  # 每次迭代释放的粒子数
diffusion_coefficient = parameters.diffusion_coefficient  # 扩散系数
wind_speed = parameters.wind_speed  # 基础风速向量
buoyancy = parameters.buoyancy  # 浮力系数
concentration_decay = parameters.concentration_decay  # 浓度衰减系数
turbulence_strength = parameters.turbulence_strength  # 湍流强度
lateral_wind_strength = parameters.lateral_wind_strength  # 侧风强度
global_concentration = parameters.global_concentration  # 新粒子初始浓度
max_steps = parameters.max_steps  # 最大步数（未在代码中使用）

# 创建无人机实例
initial_position = np.array(parameters.initial_position)  # 无人机初始位置
drone = sc.SurgeCastAgent(initial_position=initial_position)

# 创建环境
source_position = np.array(parameters.source_position)  # 污染源位置
env = gd.GasDiffusion3D(
    max_particles=max_particles, release_rate=release_rate, box_size=box_size,
    diffusion_coefficient=diffusion_coefficient, wind_speed=wind_speed,
    buoyancy=buoyancy, concentration_decay=concentration_decay,
    turbulence_strength=turbulence_strength, lateral_wind_strength=lateral_wind_strength,
    global_concentration=global_concentration, source_position=source_position
)

# 绘图函数
def plot_data():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter_plot = ax.scatter([], [], [], c=[], cmap='viridis', s=20, alpha=0.5, vmin=0, vmax=1)
    colorbar = plt.colorbar(scatter_plot, ax=ax, label='Concentration')

    ax.set_xlim(0, box_size[0])
    ax.set_ylim(0, box_size[1])
    ax.set_zlim(0, box_size[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    anim = FuncAnimation(fig, gd.animate, frames=200, interval=50, blit=False, fargs=(env, ax, scatter_plot, colorbar, drone))

    plt.show()

def update_data():
    drone.position = np.array(initial_position).copy()
    drone.trace_positions = [drone.position.copy()]
    time.sleep(10)
    
    for step in range(1000):
        time.sleep(0.5)
        drone.update(env)
        detection_threshold_squared = parameters.detection_threshold ** 2  # 检测阈值的平方
        if np.sum((drone.position - source_position) ** 2) <= detection_threshold_squared:
            print("The gas source is located nearby")
            break

# 创建并启动数据更新线程
data_thread = threading.Thread(target=update_data)
data_thread.daemon = True
data_thread.start()

# 主线程处理绘图
plot_data()
