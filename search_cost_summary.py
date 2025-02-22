import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import euclidean
from scipy import interpolate
import config
import surgecast as sc
import sim_analysis
import environment
import visualize
from config import trajectory_config
import utils

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

    # 初始化环境
    env = environment.PlumeEnvironment2D(puff_data, wind_data, source_position, start_time=config.start_time, end_time=config.end_time)
    print("环境初始化完毕！")

    # 用于记录所有搜索的时间和步长
    all_search_times = []
    all_search_distances = []

    # 进行10轮搜索
    for search_round in range(10):
        print(f"开始第 {search_round + 1} 轮搜索")

        # 重置参数
        trace_positions = []  # 存储轨迹的列表
        wind_speeds = []  # 存储风速数据
        wind_directions = []  # 存储风速方向数据
        sensor_concentrations = []  # 存储气体浓度数据
        search_success_time = None  # 记录搜索成功的时间戳

        # 每一轮都重新初始化SurgeCastAgent实例
        initial_position = [8, 1]  # 假设无人机起始位置为 (8, 1)，可以根据需要修改
        surgecast_agent = sc.SurgeCastAgent(initial_position)  # 创建 SurgeCastAgent 实例

        # 获取所有时间戳
        time_stamps = sorted(puff_data['time'].unique())  # 获取所有时间戳

        for t_val in time_stamps:
            # 更新环境中的当前时间
            env.current_time = t_val

            # 更新SurgeCastAgent的位置
            surgecast_agent.update(env)

            # 存储当前位置
            trace_positions.append(surgecast_agent.position.copy())

            # 获取风速和风速方向
            wind_speed, wind_direction = env.calculate_wind_with_error(t_val, error_magnitude=config.error_magnitude, error_angle=config.error_angle)
            wind_speeds.append(wind_speed)
            wind_directions.append(wind_direction)

            # 获取气体浓度
            x_position, y_position = surgecast_agent.position
            sensor_concentration = env.sensor_with_error(t_val, x_position, y_position, sensor_stddev=config.sensor_stddev)
            sensor_concentrations.append(sensor_concentration)

            # 检查搜索是否成功
            if surgecast_agent.check_success(source_position, success_threshold):
                print(f"搜索成功！无人机到达目标位置。时间: {t_val}")
                search_success_time = t_val  # 记录搜索成功的时间戳
                break  # 如果成功，退出循环

        # 计算搜索成本
        if search_success_time is not None:
            total_time, total_distance = utils.calculate_search_cost(trace_positions, time_stamps, search_success_time)
            all_search_times.append(total_time)
            all_search_distances.append(total_distance)

    # 将所有搜索的时间和步长保存到一个新的CSV文件
    search_summary_df = pd.DataFrame({
        'search_round': [i + 1 for i in range(10)],
        'search_time': all_search_times,
        'search_distance': all_search_distances
    })

    search_summary_df.to_csv('search_summary.csv', index=False)
    print("所有搜索的总结数据已保存到 'search_summary.csv'")

