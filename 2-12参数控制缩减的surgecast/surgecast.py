import gas_diffusion_good as gd
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import threading
import time
import os
import logging  # 用于日志记录
import random


class SurgeCastAgent():

    def __init__(self, initial_position, surge_speed=20, cast_speed=50, cast_angle=5, cast_border=400, lod=0 ):
        # 初始化参数
        #self.recastnum = 5  # 重找烟羽精度
        
        # cast 参数
        self.cast_speed = cast_speed
        self.cast_angle = cast_angle  # 横向扫描偏移角度
        self.cast_border = cast_border  # 扫描时间
        self.lod = lod  # 最低浓度检出限
        self.cast_direction = np.array([0,0,0])  #cast方向
        self.anti_direction = np.array([0,0,0])  #cast逆方向
        self.windvector = np.array([0,0,0])  #风向向量
        self.windpoint = 0  #风向计算
        
        # surge 参数
        self.surge_speed = np.array(surge_speed, dtype=np.float64)
        self.surge_direction = np.array([0, 0, 0])  # 初始 surge 方向
        self.surge_angle = 15  # surge偏移角度
        self.concentration_gap = 30 #探测器单边距离
        self.surge_maxc = 0  # 最大浓度
        
        # 全局变量
        self.position = np.array(initial_position, dtype=np.float64)
        self.angle = np.array([0, 0, 0])
        self.current_mode = "CAST"  # 初始模式为 CAST
        self.border = 0  # 用于搜索边界计量
        self.dir = 1  # 碰壁检测，用于判断是否撞墙
        self.arm_lines = []  # 保存轴臂的线条对象，用于更新时清除旧的
        self.trace_line = None  # 保存轨迹线的对象
        self.trace_positions = [initial_position]  # 保存轨迹点

    def detect_gas_concentration(self, environment):
        # 获取一定距离两侧的浓度， 0为逆时针，1为顺时针
        return np.array([environment.get_concentration_at_position(self.position + self.concentration_gap * np.array([-self.angle[1],self.angle[0],0])),environment.get_concentration_at_position(self.position + self.concentration_gap * np.array([self.angle[1],-self.angle[0],0]))])


    def levy_flight(self):
        """Levy 飞行算法，用于 Z 轴随机移动"""
        step = np.random.standard_cauchy() * 2
        self.position[2] += step
        # 限制 Z 轴范围
        self.position[2] = np.clip(self.position[2], 10, 50)

    def surge(self, environment):
        
        # 两个一定距离的探测器，向浓度偏大的探测器一定夹角沿风向前进
        
        # 更新风向
        if self.position.dtype != np.float64:
            self.position = self.position.astype(np.float64)
        self.windpoint = self.windpoint + 1
        self.windvector = self.windvector * ((self.windpoint - 1) / self.windpoint) + environment.get_wind_vector(self.position) * (1 / self.windpoint)
        
        # 向逆风方向而不是气味浓度增加的方向移动
        gas_concentration = self.detect_gas_concentration(environment)
        surgevector = -environment.get_wind_vector(self.position)
        surgevector = surgevector / np.linalg.norm(surgevector)
        print(gas_concentration)
        if gas_concentration[0] > gas_concentration[1]:
            self.surge_direction = np.array([surgevector[0]*np.cos(np.radians(self.surge_angle))-surgevector[1]*np.sin(np.radians(self.surge_angle)),surgevector[0]*np.sin(np.radians(self.surge_angle))+surgevector[1]*np.cos(np.radians(self.surge_angle)),0])
        else:
            self.surge_direction = np.array([surgevector[0]*np.cos(np.radians(-self.surge_angle))-surgevector[1]*np.sin(np.radians(-self.surge_angle)),surgevector[0]*np.sin(np.radians(-self.surge_angle))+surgevector[1]*np.cos(np.radians(-self.surge_angle)),0])
        
        # 接近源头下调夹角与步长
        self.surge_maxc = gas_concentration[0]*gas_concentration[1]
        if(self.surge_maxc * 0.95 > gas_concentration[0]*gas_concentration[1]):
            self.surge_angle *= 0.8
            self.surge_speed *= 0.8
        
        if self.position.dtype != np.float64:
            self.position = self.position.astype(np.float64)
        self.position += self.surge_direction * self.surge_speed
        self.angle = self.surge_direction
        # print(gas_concentration)
        if gas_concentration[0] == 0 and gas_concentration[1] == 0:
            self.current_mode = "CAST"
            print("转换为CAST模式")
            self.border = 0

    # 重找烟羽，还待重写
    def recast(self, environment):
        
        #目前仅应用于二维
         if self.position.dtype != np.float64:
             self.position = self.position.astype(np.float64)
         self.windpoint = self.windpoint + 1
         self.windvector = self.windvector * ((self.windpoint - 1) / self.windpoint) + environment.get_wind_vector(self.position) * (1 / self.windpoint)
         print(self.windvector)
        #圆点设置为上一个搜寻点
        # center_point = self.position - self.surge_direction * self.surge_speed
        # angle = np.pi()/self.recastnum
        
        #surge_direction + angle * i
        
        # np.array([self.surge_direction[0]*np.cos(angle*i)-self.surge_direction[1]*np.sin(angle*i), self.surge_direction[0]*np.sin(angle*i)+self.surge_direction[1]*np.cos(angle*i), 0])
        # radius = abs(self.surge_direction * self.surge_speed)
        # angle = np.pi()/self.recastnum
        #for i in range(self.recastnum):
        #    self.position = center_point + radius * np.array([cos()])
        
        #待完善，可以先用cast
        
        
    def cast(self, environment):
        
        #用测定的所有风向数据，设定一个大概方向，避免风向变化幅度过大导致踌躇不前
        if self.position.dtype != np.float64:
            self.position = self.position.astype(np.float64)
        self.windpoint = self.windpoint + 1
        self.windvector = self.windvector * ((self.windpoint - 1) / self.windpoint) + environment.get_wind_vector(self.position) * (1 / self.windpoint)
        #print(self.windvector)
        
        #搜寻角度设定为风向偏转90度
        castvector = np.array([-self.windvector[1],self.windvector[0],0]) * self.dir / np.linalg.norm(self.windvector)
        #转化为向量
        angle_rad = np.array([np.cos(np.radians(self.cast_angle * self.dir)),np.sin(np.radians(self.cast_angle * self.dir)),0]) 
        
        #保存cast逆向方向，方便碰壁回弹
        anticastvector = np.array([-self.windvector[1],self.windvector[0],0]) * self.dir * -1
        antiangle_rad = np.array([np.cos(np.radians(self.cast_angle * self.dir * -1)),np.sin(np.radians(self.cast_angle * self.dir * -1)),0]) 
        self.antidirection = np.array([antiangle_rad[0]*anticastvector[0]-antiangle_rad[1]*anticastvector[1], antiangle_rad[0]*anticastvector[1]+antiangle_rad[1]*anticastvector[0], 0])
        
        self.cast_direction = np.array([angle_rad[0]*castvector[0]-angle_rad[1]*castvector[1], angle_rad[0]*castvector[1]+angle_rad[1]*castvector[0], 0])
        #测试用句
        #print(self.direction)
        #print(self.dir)
        
        self.position += self.cast_direction * self.cast_speed
        #print(direction)
        # 每次扫描尝试获取气味
        self.angle = self.cast_direction
        gas_concentration = self.detect_gas_concentration(environment)
        if gas_concentration[0] > self.lod or gas_concentration[1] > self.lod:
            self.current_mode = "SURGE"
            self.surge_maxc = 0
            print("转换为SURGE模式")
        else:
            self.border += self.cast_speed
            if self.border > self.cast_border:
                # 降低步长，可以改进为关于风速的函数，风速越大，步长越小
                self.cast_speed = self.cast_speed * 0.8
                self.border = 0

    def update(self, environment):
        if self.current_mode == "SURGE":
            self.surge(environment)
        elif self.current_mode == "CAST":
            self.cast(environment)
        elif self.current_mode == "RECAST":
            self.recast(environment)
        # 边界检测与镜像反向修补
        hit = False
        for i in range(3):  # 检查每个维度（x, y, z）
            if self.position[i] < 0:
                # 位置镜像处理，将越界的负值反射到正方向的对称位置
                self.position[i] = self.position[i]/self.cast_direction[i]*self.anti_direction[i]
                #self.position[i] = -self.position[i]
                # 方向镜像
                self.surge_direction[i] = -self.surge_direction[i]
                hit = True
                
            elif self.position[i] > environment.box_size[i]:
                # 位置镜像处理，将越界的正值反射到负方向的对称位置
                self.position[i] = environment.box_size[i] + (self.position[i] - environment.box_size[i])/self.cast_direction[i]*self.anti_direction[i]
                #self.position[i] = 2 * environment.box_size[i] - self.position[i]
                # 方向镜像
                self.surge_direction[i] = -self.surge_direction[i]
                hit = True
            
        if hit == True:
            self.dir=-self.dir
        # 更新无人机位置和轨迹
        # self.position += self.surge_direction * self.surge_speed

        # 添加 Z 轴的 Levy 移动
        self.levy_flight()

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
        