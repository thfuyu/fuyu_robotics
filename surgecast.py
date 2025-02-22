import numpy as np
import config  # 导入参数模块
import sim_analysis
import environment as env

class SurgeCastAgent():
    def __init__(self, initial_position):
        # 初始化参数
        self.cast_speed = config.cast_speed
        self.cast_angle = config.cast_angle
        self.cast_border = config.cast_border
        self.lod = config.lod  # 最低浓度检出限
        self.cast_direction = np.array([0, 0])  # CAST方向
        self.anti_direction = np.array([0, 0])  # CAST逆方向
        self.windvector = np.array([0, 0])  # 风向向量
        self.windpoint = 0  # 风向计算
        
        self.surge_speed = config.surge_speed
        self.surge_direction = np.array([0, 0])  # 初始 SURGE 方向
        self.surge_angle = config.surge_angle
        self.concentration_gap = config.concentration_gap
        self.surge_maxc = 0  # 最大浓度
        
        self.position = np.array(initial_position[:2], dtype=np.float64)  # 只取前两个坐标 (x, y)
        self.angle = np.array([0, 0])
        self.current_mode = "CAST"  # 初始模式为 CAST
        self.border = 0  # 用于搜索边界计量
        self.dir = 1  # 碰壁检测，用于判断是否撞墙
        self.arm_lines = []  # 保存轴臂的线条对象，用于更新时清除旧的
        self.trace_line = None  # 保存轨迹线的对象
        self.trace_positions = [initial_position[:2]]  # 保存轨迹点，只记录 (x, y)
        # 新增计数器
        self.zero_concentration_count = 0  # 记录气体浓度为零的次数


    def detect_gas_concentration(self, environment):
        """
        获取当前无人机位置两侧的浓度值（逆时针和顺时针方向），使用传感器误差模型计算浓度。
        """
        # 获取当前无人机位置
        current_position = self.position

        # 计算两侧的检测点位置
        left_position = current_position + self.concentration_gap * np.array([-self.angle[1], self.angle[0]])
        right_position = current_position + self.concentration_gap * np.array([self.angle[1], -self.angle[0]])

        # 获取当前时间点的浓度，使用传感器误差模型
        current_time = environment.current_time

        left_concentration = environment.sensor_with_error(current_time, left_position[0], left_position[1], sensor_stddev=config.sensor_stddev)
        right_concentration = environment.sensor_with_error(current_time, right_position[0], right_position[1], sensor_stddev=config.sensor_stddev)

        return np.array([left_concentration, right_concentration])

    def surge(self, env):  # env 是 PlumeEnvironment2D 的实例
        """
        SURGE 模式：向浓度偏高的方向前进。
        如果检测到气体浓度为零，则根据计数器决定切换到 CAST 或 RECAST 模式。
        """
        print("启动SURGE")
        if self.position.dtype != np.float64:
            self.position = self.position.astype(np.float64)
        self.windpoint += 1

        # 使用 env 作为环境实例，调用方法
        wind_speed, wind_direction = env.calculate_wind_with_error(
            env.current_time,  # 使用 env 实例的 current_time
            error_magnitude=config.error_magnitude,
            error_angle=config.error_angle
        )

        self.windvector = np.array([wind_speed * np.cos(np.radians(wind_direction)), wind_speed * np.sin(np.radians(wind_direction))])

        gas_concentration = self.detect_gas_concentration(env)  # 这里使用 env

        surgevector = -self.windvector
        surgevector = surgevector / np.linalg.norm(surgevector)

        # 判断浓度的偏高方向并移动
        if gas_concentration[0] > gas_concentration[1]:
            self.surge_direction = np.array([
                surgevector[0] * np.cos(np.radians(self.surge_angle)) - surgevector[1] * np.sin(np.radians(self.surge_angle)),
                surgevector[0] * np.sin(np.radians(self.surge_angle)) + surgevector[1] * np.cos(np.radians(self.surge_angle))
            ])
        else:
            self.surge_direction = np.array([
                surgevector[0] * np.cos(np.radians(-self.surge_angle)) - surgevector[1] * np.sin(np.radians(-self.surge_angle)),
                surgevector[0] * np.sin(np.radians(-self.surge_angle)) + surgevector[1] * np.cos(np.radians(-self.surge_angle))
            ])

        self.surge_maxc = gas_concentration[0] * gas_concentration[1]
        if self.surge_maxc * 0.95 > gas_concentration[0] * gas_concentration[1]:
            self.surge_angle *= 0.8
            self.surge_speed *= 0.8

        self.position += self.surge_direction * self.surge_speed
        self.angle = self.surge_direction

        if gas_concentration[0] < self.lod and gas_concentration[1] < self.lod:
            if self.zero_concentration_count == 0:
                self.current_mode = "CAST"
                print("开始 CAST 模式")
            else:
                self.current_mode = "RECAST"
                print("开始 RECAST 模式")
            self.zero_concentration_count += 1


    def cast(self, env):
        """
        CAST 模式：沿着风向的垂直方向移动，每次移动的距离不超过 cast_border。
        每次方向反转后重新计数，但每次反向时搜索范围会逐步增大。
        """
        if self.position.dtype != np.float64:
            self.position = self.position.astype(np.float64)
        self.windpoint += 1

        # 使用环境类中的 calculate_wind_with_error 计算带误差的风速和风向
        wind_speed, wind_direction = env.calculate_wind_with_error(
            env.current_time,  # 传入环境实例的 current_time
            error_magnitude=config.error_magnitude,
            error_angle=config.error_angle
        )
        # 计算出来风的二维向量，第一个元素是风的水平方向分量
        self.windvector = np.array([
            wind_speed * np.cos(np.radians(wind_direction)),
            wind_speed * np.sin(np.radians(wind_direction))
        ])

        # 计算与风向垂直的单位向量（逆时针 90 度旋转）
        castvector = np.array([-self.windvector[1], self.windvector[0]]) / np.linalg.norm(self.windvector)
        # 计算与风向垂直并且朝向逆风的方向
        angle_rad = np.array([
            np.cos(np.radians(self.cast_angle)),  # 计算与垂直风向的夹角的x分量
            np.sin(np.radians(self.cast_angle))   # 计算与垂直风向的夹角的y分量
        ])

        # 逆风向上搜索（垂直风向的方向和夹角结合）
        direction_upwind_up = castvector * angle_rad

        # 逆风向下搜索：第一个坐标不变，第二个坐标反向
        direction_upwind_down = np.array([direction_upwind_up[0], -direction_upwind_up[1]])
        
        direction_upwind_up = direction_upwind_up / np.linalg.norm(direction_upwind_up)
        direction_upwind_up[0] -= 0.1  # 给第一个坐标减去偏置量0.1
        direction_upwind_down = direction_upwind_down / np.linalg.norm(direction_upwind_down)
        direction_upwind_down[0] -= 0.1  # 给第一个坐标减去偏置量0.1

        # 只在第一次计算时才更新方向，后续的方向反转通过更新方向标志位来改变
        if not hasattr(self, 'initialized_direction') or not self.initialized_direction:
            # 设置初始方向，默认为逆风向上搜索
            self.cast_direction = direction_upwind_up
            self.initialized_direction = True  # 标记方向已经初始化
            print(f"初始方向设置为: {self.cast_direction}")  # 调试输出初始方向

        # 更新无人机的位置
        self.position += self.cast_direction * self.cast_speed
        self.angle = self.cast_direction  # 更新方向

        self.border += self.cast_speed
        print(f"当前 border: {self.border}")  # 打印当前边界值

        # 判断是否达到边界，触发反向
        if self.border > config.cast_border * self.dir:  # 每次反向时增加搜索范围
            self.dir += 1  # 增加 dir 值，扩大搜索范围
            print(f"方向反转，更新后的 dir: {self.dir}")  # 调试输出方向反转后的 dir 值
            # 判断当前方向，反向切换到另一个方向
            # 计算self.cast_direction到direction_upwind_up和direction_upwind_down的距离
            distance_up = np.linalg.norm(self.cast_direction - direction_upwind_up)  # 到逆风向上方向的距离
            distance_down = np.linalg.norm(self.cast_direction - direction_upwind_down)  # 到逆风向下方向的距离

            # 判断哪个方向距离当前方向更近
            if distance_up < distance_down:
                print("当前方向更接近逆风向上，切换到逆风向下搜索")
                self.cast_direction = direction_upwind_down  # 切换为逆风向下
            else:
                print("当前方向更接近逆风向下，切换到逆风向上搜索")
                self.cast_direction = direction_upwind_up  # 切换为逆风向上
            self.border = 0  # 重置边界值

        # 获取当前浓度值
        gas_concentration = self.detect_gas_concentration(env)
        if gas_concentration[0] > self.lod or gas_concentration[1] > self.lod:
            self.current_mode = "SURGE"
            self.surge_maxc = 0
            print("转换为SURGE模式")


    def recast(self, environment):
        """
        RECAST 模式：用于应对风向变化，类似CAST，但更加灵活
        """
        if self.position.dtype != np.float64:
            self.position = self.position.astype(np.float64)
        self.windpoint += 1

        # 使用环境类中的 calculate_wind_with_error 计算带误差的风速和风向
        wind_speed, wind_direction = environment.calculate_wind_with_error(self.position, error_magnitude=config.error_magnitude, error_angle=config.error_angle)

        self.windvector = np.array([wind_speed * np.cos(np.radians(wind_direction)), wind_speed * np.sin(np.radians(wind_direction))])

        castvector = np.array([-self.windvector[1], self.windvector[0]]) * self.dir / np.linalg.norm(self.windvector)
        angle_rad = np.array([np.cos(np.radians(self.cast_angle * self.dir)), np.sin(np.radians(self.cast_angle * self.dir))])

        anticastvector = np.array([-self.windvector[1], self.windvector[0]]) * self.dir * -1
        antiangle_rad = np.array([np.cos(np.radians(self.cast_angle * self.dir * -1)), np.sin(np.radians(self.cast_angle * self.dir * -1))])
        self.anti_direction = np.array([
            antiangle_rad[0] * anticastvector[0] - antiangle_rad[1] * anticastvector[1],
            antiangle_rad[0] * anticastvector[1] + antiangle_rad[1] * anticastvector[0]
        ])

        self.cast_direction = np.array([
            angle_rad[0] * castvector[0] - angle_rad[1] * castvector[1],
            angle_rad[0] * castvector[1] + angle_rad[1] * castvector[0]
        ])

        self.position += self.cast_direction * self.cast_speed
        self.angle = self.cast_direction

        distance_from_start = np.linalg.norm(self.position - np.array(self.trace_positions[0]))

        if distance_from_start >= config.cast_border:
            self.dir *= -1
            self.border = 0

        gas_concentration = self.detect_gas_concentration(environment)
        if gas_concentration[0] > 0.000001 or gas_concentration[1]  > 0.000001 :
            self.current_mode = "SURGE"
            self.surge_maxc = 0
            print("转换为SURGE模式")
        else:
            if self.border > self.cast_border:
                self.cast_speed *= 0.8
                self.border = 0

    def check_success(self, source_position, success_threshold):
            """
            判定搜索是否成功：判断无人机当前位置是否小于阈值范围
            :param source_position: 源的二维坐标
            :param success_threshold: 成功阈值（距离源的最大允许距离）
            :return: bool 是否成功
            """
            distance_to_source = np.linalg.norm(self.position - np.array(source_position))
            if distance_to_source <= success_threshold:
                return True  # 搜索成功
            return False  # 搜索未成功


    def update(self, environment):
        """
        更新无人机的状态，计算下一帧的位置。
        """
        if self.current_mode == "SURGE":
            self.surge(environment)
        elif self.current_mode == "CAST":
            self.cast(environment)
        elif self.current_mode == "RECAST":
            self.recast(environment)

        # 更新当前位置
        self.trace_positions.append(self.position.copy())

        # 返回当前的无人机位置作为下一帧的起始点
        return self.position


