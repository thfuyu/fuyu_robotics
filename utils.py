import numpy as np

def calculate_search_cost(trace_positions, time_stamps, search_success_time):
    """
    计算搜索过程的总时间和总步长。

    :param trace_positions: 无人机的轨迹位置列表 [(x1, y1), (x2, y2), ...]
    :param time_stamps: 时间戳列表
    :param search_success_time: 搜索成功时的时间戳
    :return: total_time, total_distance
    """
    # 计算总时间
    # 使用搜索成功的时间戳减去开始时间戳来计算总时间
    total_time = search_success_time - time_stamps[0]

    # 计算总步长
    total_distance = 0
    for i in range(1, len(trace_positions)):
        distance = np.linalg.norm(np.array(trace_positions[i]) - np.array(trace_positions[i-1]))
        total_distance += distance

    # 输出结果
    print(f"总时间: {total_time:.2f} 秒")
    print(f"总步长: {total_distance:.2f} 米")
    
    return total_time, total_distance
