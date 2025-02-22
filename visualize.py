import os
import pandas as pd
import sim_analysis  # 确保 sim_analysis 已经导入
import numpy as np
import matplotlib.pyplot as plt
import config
import environment
import gc

def plot_wind_vectors(wind_directions, t_val, ax):
    """
    绘制风向标。
    
    :param wind_directions: 风向数据
    :param t_val: 当前时间戳
    :param ax: 当前绘图的坐标轴
    """
    # 获取当前时间戳对应的风向
    idx = np.argmin(np.abs(np.array(time_stamps) - t_val))
    wind_direction = wind_directions[idx]

    # 使用风向计算风速的方向向量
    angle = np.radians(wind_direction)  # 转换为弧度
    v_x = np.cos(angle)  # x方向分量
    v_y = np.sin(angle)  # y方向分量

    # 设定箭头的长度，scale 值较小来控制箭头的大小
    arrow_length = 0.5  # 适当调整长度

    # 绘制风向标（箭头）
    x, y = -0.1, 3.5  # 箭头的中心位置
    ax.quiver(x, y, v_x * arrow_length, v_y * arrow_length, color='black', scale=1)

    # 绘制源的位置（虚线圈）
    ax.scatter(x, y, s=500, facecolors='none', edgecolors='k', linestyle='--')

def plot_source(source_position, radius, ax):
    """
    绘制源的虚线圆，并标注为 'source circle'。
    
    :param source_position: 气味源的二维坐标 (x, y)
    :param radius: 圆的半径
    :param ax: 当前绘图的坐标轴
    """
    # 绘制虚线圆
    circle = plt.Circle(source_position, radius, color='r', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle)

    # 添加图例
    ax.legend(['Source Circle'], loc='upper right')

def plot_drone_with_trajectory(ax, position, trace_positions, time_stamps, t_val, arm_length=config.arm_length):
    """
    绘制无人机模型，包括轨迹和四个桨叶。
    """
    # 获取与当前时间最接近的索引
    idx = np.argmin(np.abs(np.array(time_stamps) - t_val))
    
    # 提取该时间点的x, y坐标
    x_position, y_position = trace_positions[idx]

    # 绘制轨迹（到当前时间为止）
    trace_x = [pos[0] for pos in trace_positions[:idx+1]]
    trace_y = [pos[1] for pos in trace_positions[:idx+1]]
    ax.plot(trace_x, trace_y, color='blue', label='Drone Trajectory')

    # 绘制当前时间点的无人机位置
    ax.scatter(x_position, y_position, color='red', s=100, label='Drone Position')

    # 绘制四个桨叶，形成 "X" 形布局
    x, y = position
    angle_45 = np.pi / 4  # 45度角
    arm_positions = np.array([
        [x + arm_length * np.cos(angle_45), y + arm_length * np.sin(angle_45)],  
        [x - arm_length * np.cos(angle_45), y - arm_length * np.sin(angle_45)],  
        [x + arm_length * np.cos(angle_45), y - arm_length * np.sin(angle_45)],  
        [x - arm_length * np.cos(angle_45), y + arm_length * np.sin(angle_45)]   
    ])

    # 通过 ax.plot() 绘制无人机的轴臂
    for i in range(4):
        ax.plot([x, arm_positions[i, 0]], [y, arm_positions[i, 1]], 'k')

    ax.legend()


def plot_sensor_concentration(sensor_concentrations, t_val, ax, trace_positions, time_stamps, concentration_data):
    """
    绘制传感器浓度的折线图。
    """
    # 获取当前时间戳对应的索引
    idx = np.argmin(np.abs(np.array(time_stamps) - t_val))

    # 获取传感器浓度（直接从sensor_concentrations获取，而不是从concentration_data）
    sensor_concentration = sensor_concentrations[idx]  # 使用来自 CSV 的浓度数据
    
    # 将浓度数据加入到concentration_data列表中
    concentration_data.append((t_val, sensor_concentration))

    # 解包times和concentrations用于绘制
    times, concentrations = zip(*concentration_data)
    ax.plot(times, concentrations, label="Sensor Concentration", color='blue')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concentration')
    ax.legend()


def plot_wind_speed(wind_speeds, t_val, ax, trace_positions, time_stamps, wind_data):
    """
    绘制风速大小的折线图。
    """
    # 获取当前时间戳对应的索引
    idx = np.argmin(np.abs(np.array(time_stamps) - t_val))  # 获取最接近的时间戳的索引
    
    # 获取风速数据（直接从wind_speeds获取，而不是从wind_data中获取）
    wind_speed = wind_speeds[idx]  # 使用来自 CSV 的风速数据
    
    # 将风速数据加入到wind_data列表中
    wind_data.append((t_val, wind_speed))

    # 解包time和wind_speed用于绘制
    times, speeds = zip(*wind_data)
    ax.plot(times, speeds, label="Wind Speed", color='orange')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.legend()

def plot_wind_direction(wind_directions, t_val, ax, trace_positions, time_stamps, wind_direction_data):
    """
    绘制风速方向的折线图。
    """
    # 获取当前时间戳对应的索引
    idx = np.argmin(np.abs(np.array(time_stamps) - t_val))  # 获取最接近的时间戳的索引
    
    # 获取风向数据（直接从wind_directions获取，而不是从wind_direction_data中获取）
    wind_direction = wind_directions[idx]  # 使用来自 CSV 的风向数据
    if wind_direction > 180:
        wind_direction -= 360  # 确保风向在0到360度之间
    #最sb的一段
    # 将风向数据加入到wind_direction_data列表中
    wind_direction_data.append((t_val, wind_direction))

    # 解包times和directions用于绘制
    times, directions = zip(*wind_direction_data)
    ax.plot(times, directions, label="Wind Direction", color='green')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wind Direction (°)')
    ax.legend()


def plot_project_visualization(output_dir, trace_positions, time_stamps, wind_speeds, wind_directions, sensor_concentrations, data_puffs, data_wind, time_step=config.dt):
    """
    绘制项目可视化图像，包括风速、风向、无人机轨迹、传感器浓度等内容，并保存。
    """
    # 获取时间序列
    t_min, t_max = time_stamps.min(), time_stamps.max()
    time_steps = np.arange(t_min, t_max, time_step)

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 存储浓度数据
    concentration_data, wind_data, wind_direction_data = [], [], []

    for idx, t_val in enumerate(time_steps):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1])

        # 第一个子图：绘制无人机轨迹
        ax1 = fig.add_subplot(gs[0, 0:2])
        # 使用data_puffs和data_wind绘制烟羽和风速矢量
        sim_analysis.plot_puffs_and_wind_vectors(data_puffs, data_wind, t_val, ax1)
        ax1.set_xlim(config.xlim1)  # 设置 x 轴范围
        ax1.set_ylim(config.ylim1)  # 设置 y 轴范围

        # 绘制无人机模型（轨迹和四个桨叶）
        drone_position = trace_positions[idx]
        plot_drone_with_trajectory(ax1, drone_position, trace_positions, time_stamps, t_val)

        # 绘制风向标
        plot_wind_vectors(wind_directions, t_val, ax1)

        # 绘制气味源
        plot_source(source_position=config.source_position, radius=config.success_threshold, ax=ax1)

        # 第二个子图：绘制传感器浓度
        ax2 = fig.add_subplot(gs[1, 0])
        # 使用CSV文件的浓度数据
        plot_sensor_concentration(sensor_concentrations, t_val, ax2, trace_positions, time_stamps, concentration_data)
        ax2.set_xlim(config.xlim2)  # 设置 x 轴范围
        ax2.set_ylim(config.ylim2)  # 设置 y 轴范围

        # 第三个子图：绘制风速
        ax3 = fig.add_subplot(gs[1, 1:3])
        # 使用CSV文件的风速数据
        plot_wind_speed(wind_speeds, t_val, ax3, trace_positions, time_stamps, wind_data)
        ax3.set_xlim(config.xlim3)  # 设置 x 轴范围
        ax3.set_ylim(config.ylim3)  # 设置 y 轴范围

        # 第四个子图：绘制风速方向
        ax4 = fig.add_subplot(gs[2, 0:2])
        # 使用CSV文件的风向数据
        plot_wind_direction(wind_directions, t_val, ax4, trace_positions, time_stamps, wind_direction_data)
        ax4.set_xlim(config.xlim4)  # 设置 x 轴范围
        ax4.set_ylim(config.ylim4)  # 设置 y 轴范围

        # 保存图像
        filename = os.path.join(output_dir, f'project_visualization_t{t_val:3.3f}.png')
        fig.savefig(filename)
        plt.close(fig)

    print(f"所有图像已保存至: {output_dir}")


    

# --- 主程序 ---
if __name__ == "__main__":
    # 配置参数
    config.datadir = r'D:\desktop1\2-12-surgecast\plume'  # 数据文件夹路径
    output_folder = r"D:\desktop1\2-12-surgecast\photos"  # 设置输出文件夹路径

    source_position = config.source_position  # 二维坐标
    max_steps = config.max_steps
    success_threshold = config.success_threshold

    # --- 读取CSV文件并解析轨迹数据 ---
    trajectory_df = pd.read_csv(r"D:\desktop1\2-12-surgecast\drone_trajectory_with_wind_and_concentration.csv")

    trace_positions = list(zip(trajectory_df['x_position'], trajectory_df['y_position']))  # 轨迹坐标
    time_stamps = trajectory_df['time'].values  # 时间戳
    wind_speeds = trajectory_df['wind_speed'].values  # 风速
    wind_directions = trajectory_df['wind_direction'].values  # 风向
    sensor_concentrations = trajectory_df['sensor_concentration'].values  # 传感器浓度
    print("CSV文件加载完毕！")

    # --- 数据加载与处理 ---
    # 读取烟羽数据和风速数据（puff_data_test.pickle 和 wind_data_test.pickle）
    puff_data = pd.read_pickle(os.path.join(config.datadir, "puff_data_test.pickle"))
    wind_data = pd.read_pickle(os.path.join(config.datadir, "wind_data_test.pickle"))
    
    # 时间筛选与预处理
    puff_data = puff_data[(puff_data['time'] >= config.start_time) & (puff_data['time'] <= config.end_time)].copy()
    wind_data = wind_data[(wind_data['time'] >= config.start_time) & (wind_data['time'] <= config.end_time)].copy()
    print("PICKLE文件加载完毕！")

    # 对puff_data进行其他预处理
    puff_data['x_minus_radius'] = puff_data['x'] - puff_data['radius']
    puff_data['x_plus_radius'] = puff_data['x'] + puff_data['radius']
    puff_data['y_minus_radius'] = puff_data['y'] - puff_data['radius']
    puff_data['y_plus_radius'] = puff_data['y'] + puff_data['radius']
    puff_data = sim_analysis.calculate_concentrations(puff_data)  # 计算浓度
    print("浓度预处理完毕！")
    # --- 传递数据给绘制函数 ---
    plot_project_visualization(
        trace_positions=trace_positions,
        time_stamps=time_stamps,
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        sensor_concentrations=sensor_concentrations,
        data_puffs=puff_data,
        data_wind=wind_data,
        output_dir=output_folder,
        time_step=config.dt
    )

