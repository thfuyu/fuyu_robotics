
#环境的一些参数上的设置
duration = 10  # 模拟时长（秒）
cores = 24  # 使用的核心数
dataset_name = 'constant'  # 数据集名称（可以选择 'constant', 'noisy3' 等）
fname_suffix = ''  # 文件名后缀
dt = 0.01  # 每步的时间（秒）
wind_magnitude = 1  # 风速大小（m/s）
wind_y_varx = 1  # 风速的垂直变异性
birth_rate = 20  # Poisson出生率
source_position = (0, 0)
success_threshold = 0.4 #源附近绘制一个圈，判定是否搜索成功


# --- 初始化系统 ---
start_time = 5.0 #从puff和wind两个文件中中截取的一段时间上的片段
end_time = 10.0
max_steps = 500



#添加一下绘图时候的图像显示范围，一共四个子图，可以好的展示一下
xlim1 = (-1, 12)
ylim1 = (-4, 4)
xlim2 = (5, 10)  # 设置 x 轴范围
ylim2 = (0, 5)  # 设置 y 轴范围
xlim3 = (5, 10)  # 设置 x 轴范围
ylim3 = (0, 2)  # 设置 y 轴范围
xlim4 = (5, 10)  # 设置 x 轴范围
ylim4 = (-90, 90)  # 设置 y 轴范围


# 无人机随机轨迹参数配置
trajectory_config = {
    "n_pts": 100,          # 采样点数，建议增加到100提高平滑度
    "xmin": -1,            # x轴最小值
    "xmax": 12,           # x轴最大值
    "ymin": -4,            # y轴最小值
    "ymax": 4,           # y轴最大值
    "tstart": start_time,        # 起始时间，保持与仿真一致
    "tend": end_time,          # 终止时间，保持与仿真一致
    "dt": dt,             # 时间步长，建议0.1提高效率
    "local_state": None    # 随机数生成器的状态
}



# 无人机参数配置
cast_speed = 0.3    # 横向搜索速度
surge_speed = 0.3    # 逆风搜索速度
cast_angle = 5      # 横向搜索角度
surge_angle = 15     # 逆风搜索偏角
lod = 0.1           # 浓度检测阈值，超过则开始启动SURGE算法
concentration_gap = 0.1  # 探测器单边距

initial_position = (2, 0)  # 无人机的出生位置

cast_border = 1  # 每次CAST行为最大的运行距离，超过就开始转向

sensor_stddev = 0.01 #传感器的误差情况
error_magnitude = 0.2 #风速传感器的大小误差
error_angle = 5 #风速传感器的角度误差

arm_length = 0.1 #无人机绘制桨叶部分的长度

#逆风偏置项


# 绘制无人机轨迹时候的颜色选择
traj_colormap = {
    'SURGE': 'seagreen',
    'CAST': 'blue',
    'RECAST': 'slateblue'
}

traj_colormap = { 
    'on': 'seagreen',
    'off': 'blue',
}
