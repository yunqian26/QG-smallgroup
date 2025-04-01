import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义函数φ(α, s)
def phi(alpha, s):
    numerator = s**2 * (alpha + (1 - alpha) * np.abs(s - 1))**2
    denominator = alpha**2 * (1 - np.abs(s - 1))**2 * (1 - (alpha + (1 - alpha) * np.abs(s - 1))**2)
    return numerator / denominator

# 创建网格数据
alpha = np.linspace(0.01, 0.99, 100)  # 避免α=0和α=1
s = np.linspace(0.01, 1.99, 100)     # 避免s=0和s=2
alpha, s = np.meshgrid(alpha, s)

# 计算φ值
phi_values = phi(alpha, s)

# 限制φ值的范围以避免过大值
phi_values[phi_values > 7] = 7

# 绘制3D曲面图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(alpha, s, phi_values, cmap='jet', edgecolor='none')

# 设置标签和标题
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$s$')
ax.set_zlabel(r'$\phi(\alpha, s)$')
ax.set_title(r'3D Surface Plot of $\phi(\alpha, s)$')

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=10)

# 调整视角
ax.view_init(elev=15, azim=0)  # 设置俯仰角为30度，方位角为120度

# 显示图形
plt.show()