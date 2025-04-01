import math

def calculate_convergence_time(mu, initial_distance, s, epsilon):
    """
    计算稳定时间（迭代次数）的函数。

    参数:
    mu (float): 收敛因子 (0 < mu < 1)
    initial_distance (float): 初始距离 ||θ(0) - θ∞1_n||
    s (float): 系数 (s ∈ [0.8, 1.2])
    epsilon (float): 目标误差

    返回:
    int: 达到目标误差所需的迭代次数
    """
    mu = 0.84
    initial_distance = 1.0
    s = 1.0
    epsilon = 1e-6
    # 计算初始误差的缩放值
    scaled_initial_distance = s * initial_distance

    # 计算所需的迭代次数
    k = math.log(scaled_initial_distance / epsilon) / abs(math.log(mu))

    # 返回向上取整的迭代次数
    return math.ceil(k)

# 示例参数


# 计算稳定时间
convergence_time = calculate_convergence_time(mu, initial_distance, s, epsilon)
print(f"所需的迭代次数: {convergence_time}")