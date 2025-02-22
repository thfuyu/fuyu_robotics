import numpy as np
import sim_analysis
import numpy as np
import pandas as pd
import random

class PlumeEnvironment2D:
    def __init__(self, puff_data, wind_data, source_position, start_time, end_time):
        """
        初始化环境类，存储烟羽数据，风速数据和源的位置
        :param puff_data: 烟羽数据
        :param wind_data: 风速数据
        :param source_position: 源的二维位置 (x, y)
        :param start_time: 起始时间
        :param end_time: 终止时间
        """
        self.puff_data = puff_data
        self.wind_data = wind_data
        self.source_position = np.array(source_position[:2])  # 只取前两维
        self.current_time = start_time  # 当前时间
        self.start_time = start_time
        self.end_time = end_time

    def calculate_wind_with_error(self, t_val, error_magnitude, error_angle):
        """
        计算当前时刻的风速，并加入指定的随机误差。
        
        :param t_val: 当前时间戳
        :param error_magnitude: 误差的风速大小
        :param error_angle: 误差的风速方向最大角度（相对于原风速方向的角度，单位：度）
        
        :return: final_wind_speed（带误差的风速大小），final_wind_direction（带误差的风速方向）
        """
        # 获取数据中的所有时间戳
        time_stamps = self.wind_data['time'].unique()
        
        # 找到与 t_val 最接近的时间戳
        closest_time = min(time_stamps, key=lambda x: abs(x - t_val))
        
        # 筛选出与最近时间戳匹配的数据
        data_at_t = self.wind_data[self.wind_data.time == closest_time]
        
        # 获取当前时刻的风速分量
        v_x, v_y = data_at_t.wind_x.mean(), data_at_t.wind_y.mean()

        # 计算原风速的大小和方向
        wind_speed = np.sqrt(v_x**2 + v_y**2)
        wind_direction = np.arctan2(v_y, v_x) * (180 / np.pi)  # 计算风速方向，单位为度
        wind_direction = (wind_direction + 360) % 360  # 确保角度在0-360度之间

        # 生成一个在[-error_angle/2, error_angle/2]范围内的随机误差角度
        random_error_angle = random.uniform(-error_angle / 2, error_angle / 2)
        

        # 将误差的方向转化为弧度
        error_radian = np.deg2rad(random_error_angle)

        # 计算误差风速的x和y分量
        random_error_magnitude = random.uniform(0, error_magnitude)  # 随机生成一个误差风速大小
        error_v_x = random_error_magnitude * np.cos(error_radian)
        error_v_y = random_error_magnitude * np.sin(error_radian)

        # 计算原风速的x和y分量
        wind_radian = np.deg2rad(wind_direction)
        wind_v_x = wind_speed * np.cos(wind_radian)
        wind_v_y = wind_speed * np.sin(wind_radian)

        # 加入误差后的风速分量
        final_v_x = wind_v_x + error_v_x
        final_v_y = wind_v_y + error_v_y

        # 计算带有误差后的风速大小和风速方向
        final_wind_speed = np.sqrt(final_v_x**2 + final_v_y**2)
        final_wind_direction = np.arctan2(final_v_y, final_v_x) * (180 / np.pi)
        final_wind_direction = (final_wind_direction + 360) % 360  # 确保角度在0-360度之间

        return final_wind_speed, final_wind_direction



    def sensor_with_error(self, t_val, x_val, y_val, sensor_stddev):
        """
        模拟传感器测量误差，计算带有误差的浓度值。
        
        参数：
        - t_val: 时间戳。
        - x_val, y_val: 二维坐标。
        - sensor_stddev: 传感器的标准差（噪声水平）。
        
        返回：
        - 带误差的浓度值。
        """
        
        # 计算无误差的浓度
        concentration = sim_analysis.get_concentration_at_point_in_time_pandas(self.puff_data, t_val, x_val, y_val)
        
        # 为浓度添加标准差误差，假设误差服从正态分布
        noise = np.random.normal(0, sensor_stddev)
        concentration_with_error = concentration + noise
        
        return concentration_with_error

    def update_time(self, dt):
        """
        更新当前时间
        :param dt: 时间步长
        """
        self.current_time = np.clip(self.current_time + dt, self.start_time, self.end_time)


#绘图的函数放到另外的一个地方来完成