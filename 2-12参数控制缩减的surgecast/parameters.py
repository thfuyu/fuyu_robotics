# 参数文件：parameters.py

# 模拟空间的尺寸 (x, y, z)
box_size = (500, 500, 300)

# 源范围的尺寸 (x, y, z)（未在代码中使用）
source_box_size = (50, 50, 210)

# 最大粒子数量
max_particles = 1000

# 每次迭代释放的粒子数
release_rate = 10

# 扩散系数
diffusion_coefficient = 0.6

# 基础风速向量 (x, y, z)
wind_speed = (5, 0, 0)

# 浮力系数（z轴方向）
buoyancy = 0.1

# 浓度衰减系数
concentration_decay = 0.001

# 湍流强度
turbulence_strength = 0

# 侧风强度
lateral_wind_strength = 2.0

# 新粒子初始浓度
global_concentration = 2

# 最大步数（未在代码中使用）
max_steps = 20

# 无人机初始位置 (x, y, z)
initial_position = [500, 0, 30]

# 污染源位置 (x, y, z)
source_position = [0, 250, 30]

# 检测阈值（用于判断无人机是否接近污染源）
detection_threshold = 50
