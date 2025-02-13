"""
3D气体扩散与无人机移动仿真环境

模块结构：
- GasDiffusion3D类：模拟气体粒子扩散过程，包括释放、运动、衰减等
- Drone类：模拟无人机移动，包含运动控制、传感器检测、状态获取等功能
- 辅助函数：animate（动画更新）、on_key_press（键盘控制）

主要参数说明：
================================================================================
GasDiffusion3D 参数：
--------------------------------------------------------------------------------
max_particles: int            最大粒子数量（默认：1000）
release_rate: int             每次迭代释放粒子数（默认：10）
box_size: tuple               模拟空间尺寸(x,y,z)（默认：(500,500,300)）
diffusion_coefficient: float  扩散系数（默认：0.5）
wind_speed: tuple             基础风速向量(x,y,z)（默认：(1,0,0)）
buoyancy: float               浮力系数（z轴方向，默认：0.1）
concentration_decay: float    浓度衰减系数（默认：0.01）
turbulence_strength: float    湍流强度（默认：0.1）
lateral_wind_strength: float  侧风强度（默认：2.0）
global_concentration: float   新粒子初始浓度（默认：2）
source_position: tuple        污染源位置(x,y,z)（默认：(0,0,0)）

Drone 参数：
--------------------------------------------------------------------------------
initial_position: tuple       初始位置(x,y,z)
box_size: tuple               活动空间边界
uav_speed: float             移动速度（默认：5）

关键方法说明：
================================================================================
GasDiffusion3D:
- set_environment()           动态设置环境参数（污染源、风速、全局浓度）
- release_particles()         按设定速率释放新粒子
- update()                    更新粒子状态（运动、衰减、边界检测）
- get_sensor_data()           获取指定位置传感器数据（浓度、风速）

Drone:
- move()                      根据动作指令移动（6方向控制）
- position_move()             向指定坐标移动
- detect_sensor()             获取当前位置传感器数据
- get_state()                 返回状态向量（位置+速度+浓度+风速）
- plot()                     可视化无人机及运动轨迹

动画控制：
- animate()                   每帧更新函数（气体扩散+无人机状态）
- on_key_press()              键盘事件处理（方向键控制无人机）
"""

import numpy as np  # 导入numpy库，用于处理数组和数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘制图形
from mpl_toolkits.mplot3d import Axes3D  # 从mpl_toolkits中导入3D绘图工具
from matplotlib.animation import FuncAnimation  # 导入动画工具类，用于实现动态演示
import matplotlib
import time

# 定义一个3D气体扩散模拟类
class GasDiffusion3D:
    # 初始化函数，定义各种参数
    def __init__(self, max_particles=1000, release_rate=10, box_size=(500, 500, 300), 
                 diffusion_coefficient=0.5, wind_speed=(1, 0, 0), buoyancy=0.1, 
                 concentration_decay=0.01, turbulence_strength=0.1, lateral_wind_strength=2.0, 
                 global_concentration=2, source_position=(0,0,0)):
        self.max_particles = max_particles  # 最大粒子数
        self.release_rate = release_rate  # 每次释放的粒子数
        self.box_size = np.array(box_size)  # 模拟空间的尺寸
        self.diffusion_coefficient = diffusion_coefficient  # 扩散系数
        self.wind_speed = np.array(wind_speed)  # 风速向量
        self.buoyancy = buoyancy  # 浮力系数
        self.concentration_decay = concentration_decay  # 浓度衰减系数
        self.turbulence_strength = turbulence_strength  # 湍流强度
        self.lateral_wind_strength = lateral_wind_strength  # 侧风强度
        self.global_concentration = global_concentration  # 全局浓度

        # 初始化粒子的位置、浓度和年龄
        self.positions = np.empty((0, 3))  # 粒子的位置
        self.concentrations = np.empty((0,))  # 粒子的浓度
        self.ages = np.empty((0,))  # 粒子的年龄
        self.source_position = source_position  # 粒子释放的位置

    # 设置环境参数，如释放源位置、风速等
    def set_environment(self, source_position=None, wind_speed=None, global_concentration=None):
        if source_position is not None:
            self.source_position = np.array(source_position)  # 设置释放源位置
        if wind_speed is not None:
            self.wind_speed = np.array(wind_speed)  # 设置风速
        if global_concentration is not None:
            self.global_concentration = global_concentration  # 设置全局浓度

    # 释放粒子，模拟粒子生成
    def release_particles(self):
        new_particles = self.source_position + np.random.normal(0, 1, (self.release_rate, 3))  # 生成新粒子的位置
        new_concentrations = np.full(self.release_rate, self.global_concentration)  # 生成新粒子的浓度
        new_ages = np.zeros(self.release_rate)  # 生成新粒子的初始年龄

        self.positions = np.vstack((self.positions, new_particles))  # 将新粒子加入现有粒子列表
        self.concentrations = np.hstack((self.concentrations, new_concentrations))  # 将新粒子浓度加入现有浓度列表
        self.ages = np.hstack((self.ages, new_ages))  # 将新粒子年龄加入现有年龄列表

    # 更新粒子位置、浓度等属性
    def update(self):
        self.release_particles()  # 释放新粒子

        # 限制粒子的最大数量，删除最旧的粒子
        if len(self.positions) > self.max_particles:
            self.positions = self.positions[-self.max_particles:]  # 保留最近的粒子
            self.concentrations = self.concentrations[-self.max_particles:]
            self.ages = self.ages[-self.max_particles:]

        # 布朗运动，模拟粒子的随机扩散
        brownian_motion = np.random.normal(0, np.sqrt(2*self.diffusion_coefficient), self.positions.shape)
        # 确定性运动，由风速和浮力决定
        deterministic_motion = self.wind_speed + np.array([0, 0, self.buoyancy])
        # 湍流运动，模拟随机扰动
        turbulence = np.random.normal(0, self.turbulence_strength, self.positions.shape)

        # 侧风的影响
        lateral_wind = np.zeros_like(self.positions)
        left_side_mask = self.positions[:, 0] < self.box_size[0] / 2  # 判断粒子是否在左侧
        right_side_mask = self.positions[:, 0] >= self.box_size[0] / 2  # 判断粒子是否在右侧
        lateral_wind[left_side_mask, 1] = self.lateral_wind_strength  # 左侧粒子受侧风向上影响 (0, 1, 0)
        lateral_wind[right_side_mask, 1] = -self.lateral_wind_strength  # 右侧粒子受侧风向下影响 (0, -1, 0)

        # 更新粒子位置
        self.positions += brownian_motion + deterministic_motion + turbulence + lateral_wind
        self.ages += 1  # 更新粒子年龄
        # 随着年龄增长，粒子的浓度会指数衰减
        self.concentrations *= np.exp(-self.concentration_decay * self.ages)

        # 筛选出仍在模拟空间中的粒子
        mask = np.all((self.positions >= 0) & (self.positions < self.box_size), axis=1) & (self.concentrations > 0.01)
        self.positions = self.positions[mask]
        self.concentrations = self.concentrations[mask]
        self.ages = self.ages[mask]

    def plot(self, ax, scatter_plot, colorbar):
            # 更新散点图的位置和浓度数据
            scatter_plot._offsets3d = (self.positions[:, 0], self.positions[:, 1], self.positions[:, 2])
            scatter_plot.set_array(self.concentrations)
            colorbar.update_normal(scatter_plot)  # 更新颜色条

    def get_concentration_at_position(self, position, distance_threshold=20):
        try:
            # 计算每个粒子到指定位置的距离
            distances = np.linalg.norm(self.positions - position, axis=1)

            # 使用 numpy 的 where 方法来找到符合条件的粒子索引
            valid_indices = np.where(distances < distance_threshold)[0]

            # 如果没有粒子符合条件，则返回 0.0 并记录日志
            if valid_indices.size == 0:
                #print(f"No valid particles found within {distance_threshold} units of position {position}.")
                return 0.0

            # 获取符合条件的粒子位置和浓度
            valid_distances = distances[valid_indices]
            valid_concentrations = self.concentrations[valid_indices]

            # 计算加权系数
            weights = np.exp(-valid_distances / (2 * self.diffusion_coefficient))

            # 如果加权系数之和为 0，则返回 0 浓度
            if np.sum(weights) == 0:
                print(f"Sum of weights is zero for position {position}.")
                return 0.0

            # 计算加权平均浓度
            weighted_concentration = np.sum(valid_concentrations * weights) / np.sum(weights)

            # 限制粒子浓度的位数为小数点后三位
            return round(weighted_concentration, 5)
        except Exception as e:
            # 捕获任何潜在的错误并记录
            print(f"Error in get_concentration_at_position: {e}")
            return 0.0



    def get_wind_vector(self, position):
        if position[0] < self.box_size[0] / 2 : # 判断粒子是否在左侧
            wind_vector = self.wind_speed+(0,self.lateral_wind_strength,0)
        elif position[0] >= self.box_size[0] / 2 : # 判断粒子是否在右侧
            wind_vector = self.wind_speed+(0,-self.lateral_wind_strength,0)

        return wind_vector

    # 获取传感器数据，包括当前位置的浓度和风速
    def get_sensor_data(self, position):
        concentration = self.get_concentration_at_position(position)  # 获取当前位置的浓度
        wind_vector = self.get_wind_vector(position)

        return {
            'position': position,  # 传感器位置
            'concentration': concentration,  # 气体浓度
            'wind_vector': wind_vector  # 风速、风向
        
        }

# 定义无人机类，用于模拟无人机在环境中移动
class Drone:
    def __init__(self, initial_position, box_size,uav_speed=5):
        self.position = np.array(initial_position, dtype=np.float64)  # 确保位置是 float64 类型
        self.uav_speed = uav_speed  # 无人机速度
        
        self.arm_lines = []  # 保存轴臂的线条对象，用于更新时清除旧的
        self.trace_line = None  # 保存轨迹线的对象
        self.trace_positions = [initial_position]  # 保存轨迹点
        self.box_size = box_size
        self.yaw = 0
    # 根据动作更新无人机的位置
    def move(self, action):
        """
        根据动作更新无人机的位置。
        动作空间：0 - 向前, 1 - 向后, 2 - 向左, 3 - 向右, 4 - 向上, 5 - 向下
        """
        if action == 0:
            self.position[0] -= self.uav_speed  # 向前移动
        elif action == 1:
            self.position[0] += self.uav_speed  # 向后移动
        elif action == 2:
            self.position[1] -= self.uav_speed  # 向左移动
        elif action == 3:
            self.position[1] += self.uav_speed  # 向右移动
        elif action == 4:
            self.position[2] += self.uav_speed  # 向上移动
        elif action == 5:
            self.position[2] -= self.uav_speed  # 向下移动

        # 获取该位置的状态，包括气体浓度
        #state = drone.get_state(gas)
        
        # 打印结果
        #print(f"Position :{state[0:3]} --Speed: {state[3:4]}--Concentration:{state[4:5]}--Wind:{state[5:8]}")

    # 根据动作更新无人机的位置
    def position_move(self, next_position):
        """
        根据 Transformer 输出的下一个位置更新无人机的位置。
        :param next_position: 下一个目标坐标 (x, y, z)
        """
        target_position = np.array(next_position, dtype=np.float64)  # 将目标位置转换为 float64 类型
        position = np.array(self.position, dtype=np.float64)  # 将目标位置转换为 float64 类型
        speed = np.array(self.uav_speed, dtype=np.float64)  # 将目标位置转换为 float64 类型
        # 计算方向并按速度移动
        direction = target_position - position
        distance = np.linalg.norm(direction)
        
        if distance > speed:
            direction = direction / distance  # 归一化方向向量
            position += direction * speed  # 按速度移动
        else:
            position = target_position  # 如果距离小于速度，直接移动到目标位置

        # 确保无人机在边界内
        position = np.clip(position, [0, 0, 0], self.box_size)
        self.position = position
        # 更新轨迹
        self.trace_positions.append(position.copy())


    # 检测当前位置的传感器数据
    def detect_sensor(self, env):
        """
        检测当前位置的气体浓度。
        """
        return env.get_sensor_data(self.position)
    
    # 获取状态
    def get_state(self, env):
        sensor_data = env.get_sensor_data(self.position)

        # 将位置（3D坐标）、速度（1D标量）和传感器数据拼接成一个状态向量
        state = np.concatenate([
            self.position,  # 3D 坐标 (x, y, z)
            np.array([self.uav_speed]),  # 无人机速度 (1D)
            np.array([sensor_data['concentration']]),  # 气体浓度 (1D)
            np.array(sensor_data['wind_vector'])  # 风向和风速 (3D)
        ])

        return state



    # 绘制无人机模型
    def plot(self, ax, arm_length=5):
        """
        绘制四轴无人机模型，"X" 形布局。
        
        :param ax: Matplotlib 3D 坐标系
        :param arm_length: 无人机每个轴臂的长度
        """
        # 无人机中心位置
        x, y, z = self.position
        
        # 定义旋翼的相对位置，形成 "X" 字型
        angle_45 = np.pi / 4  # 45度对应的弧度
        arm_positions = np.array([
            [x + arm_length * np.cos(angle_45), y + arm_length * np.sin(angle_45), z],  # 前左
            [x - arm_length * np.cos(angle_45), y - arm_length * np.sin(angle_45), z],  # 后右
            [x + arm_length * np.cos(angle_45), y - arm_length * np.sin(angle_45), z],  # 前右
            [x - arm_length * np.cos(angle_45), y + arm_length * np.sin(angle_45), z]   # 后左
        ])

        # 清除旧的轴臂线条
        for line in self.arm_lines:
            line.remove()  # 移除旧的线条对象
        self.arm_lines = []  # 清空列表

        # 通过 ax.plot() 绘制无人机的轴臂并保存线条对象
        self.arm_lines.append(ax.plot([x, arm_positions[0, 0]], [y, arm_positions[0, 1]], [z, arm_positions[0, 2]], 'k')[0])
        self.arm_lines.append(ax.plot([x, arm_positions[1, 0]], [y, arm_positions[1, 1]], [z, arm_positions[1, 2]], 'k')[0])
        self.arm_lines.append(ax.plot([x, arm_positions[2, 0]], [y, arm_positions[2, 1]], [z, arm_positions[2, 2]], 'k')[0])
        self.arm_lines.append(ax.plot([x, arm_positions[3, 0]], [y, arm_positions[3, 1]], [z, arm_positions[3, 2]], 'k')[0])

        # 更新无人机的移动轨迹
        self.trace_positions.append(self.position.copy())  # 保存当前位置到轨迹点
        
        # 如果轨迹线存在，更新其数据；否则，创建新的轨迹线
        trace_x = [pos[0] for pos in self.trace_positions]
        trace_y = [pos[1] for pos in self.trace_positions]
        trace_z = [pos[2] for pos in self.trace_positions]
        
        if self.trace_line is None:
            # 创建轨迹线
            self.trace_line = ax.plot(trace_x, trace_y, trace_z, 'r')[0]  # 红色轨迹线
        else:
            # 更新轨迹线的数据
            self.trace_line.set_data(trace_x, trace_y)
            self.trace_line.set_3d_properties(trace_z)


def animate(frame, gas, ax, scatter_plot, colorbar, drone):
    # 更新气体扩散
    gas.update()
    gas.plot(ax, scatter_plot, colorbar)

    # 清除旧的无人机模型，确保无人机不会在图形上重叠
    if len(ax.collections) > 1:
        del ax.collections[1:]

    drone.plot(ax)  # 绘制无人机

    return scatter_plot,ax

# 键盘控制函数
def on_key_press(event, drone):
    if event.key == 'up':          # 上箭头
        drone.move(0)  # 向前
    elif event.key == 'down':      # 下箭头
        drone.move(1)  # 向后
    elif event.key == 'left':      # 左箭头
        drone.move(2)  # 向左
    elif event.key == 'right':     # 右箭头
        drone.move(3)  # 向右
    elif event.key == 'w':
        drone.move(4)  # 向上
    elif event.key == 'e':
        drone.move(5)  # 向下


