import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import alpha
from tqdm import tqdm

# 参数设置
n = 100  # 节点数
delta1 = 1.0  # 敏感度
epsilon = 0.1  # 隐私预算
s_values = np.logspace(np.log10(0.8), np.log10(1.2), num=20) # s在[0.8, 1.2]对数均匀采样
num_runs = 10000 # 每个s的仿真次数
tolerance = 1e-3  # 收敛判据
max_iter = 500  # 最大迭代次数
alpha=1e-6

# 存储结果
variances = []
settling_times = []

for s in tqdm(s_values):
    # 初始化参数
    q=1e-6
    # q = alpha+(1-alpha)*abs(s-1)  # 接近0，模拟单次扰动
    c = delta1 * q / (epsilon * (q - abs(s - 1)))  # 根据式(32)
    print(c)
    c = max(c, 0)  # 确保c非负

    # 生成每个节点的参数
    s_i = np.full(n, s)  # 每个节点的s_i
    c_i = np.full(n, c)  # 每个节点的c_i
    q_i = np.full(n, q)  # 每个节点的q_i

    # 检查q_i是否合理（确保q_i < 1）
    if np.any(q_i >= 1):
        raise ValueError("q_i must be less than 1 to avoid division by zero or negative values.")

    # 多次仿真
    theta_inf_list = []
    time_list = []

    for _ in range(num_runs):
        # 初始化状态（随机生成初始值）
        theta0 = np.random.normal(50, 10, n)
        true_avg = np.mean(theta0)

        # 生成噪声（仅初始时刻注入）
        eta = np.random.laplace(0, c, n)

        # 算法迭代
        theta = theta0 + s * eta  # 初始扰动
        converged = False

        # 拉普拉斯矩阵L（随机图，此处简化为全连接图）
        L = np.eye(n) * n - np.ones((n, n))  # 全连接图拉普拉斯矩阵

        for t in range(max_iter):
            # 状态更新（式12，无后续噪声）
            theta = theta - 0.1 * L @ theta  # 步长h=0.1

            # 检查收敛
            avg_estimate = np.mean(theta)
            if np.abs(avg_estimate - true_avg) < tolerance:
                time_list.append(t)
                converged = True
                break

        if not converged:
            time_list.append(max_iter)

        theta_inf_list.append(avg_estimate)

    # 根据公式计算方差
    # variance=2*delta1**2/(n**2)*np.sum(1/epsilon**2)
    variance=delta1**2/(n**2)*np.sum((s_i**2) * (q_i**2) /epsilon**2*(q_i - abs(s - 1))*(1 - q_i**2))
    variances.append(variance)

    print(variance)
    # 计算平均收敛时间
    settling_times.append(np.mean(time_list))

# 绘制图3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 图3a：方差 vs s
ax1.plot(s_values, variances, 'bo-', linewidth=2)

# ax1.semilogx(s_values, variances, 'bo-', linewidth=2)
ax1.set_xlabel('Noise-to-State Gain (s)', fontsize=12)
ax1.set_ylabel('Variance of $\\theta_\\infty$', fontsize=12)
ax1.grid(True, which="both", linestyle='--')
# ax1.axvline(x=1, color='r', linestyle='--', label='Optimal (s=1)')
ax1.legend()

# 图3b：收敛时间 vs s
ax2.plot(s_values, settling_times, 'ro-', linewidth=2)
# ax2.semilogx(s_values, settling_times, 'ro-', linewidth=2)
ax2.set_xlabel('Noise-to-State Gain (s)', fontsize=12)
ax2.set_ylabel('Settling Time (iterations)', fontsize=12)
ax2.grid(True, which="both", linestyle='--')
# ax2.axvline(x=1, color='r', linestyle='--', label='Optimal (s=1)')
ax2.legend()

plt.suptitle('Figure 3: Impact of Parameter $s$ on Algorithm Performance', fontsize=14)
plt.tight_layout()
plt.show()