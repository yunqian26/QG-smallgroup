import numpy as np
import matplotlib.pyplot as plt
# 加载 NPY 文件
file_path = "D:\\Desktop\\中期考核\\数据集\\B\\init_positions.npy"  # 替换为你的文件路径
data = np.load(file_path)

# 打印加载的数据
print(data)
# 检查数组的形状（代理数量，维度）
print("形状:", data.shape)

# 检查数组的数据类型v
print("数据类型:", data.dtype)
# 获取第一个代理的状态
agent_1_state = data[0]
print("第一个代理的状态:", agent_1_state)

# 获取第二个代理的状态
agent_2_state = data[1]
print("第二个代理的状态:", agent_2_state)

# 获取第一个代理的 x 坐标
x1 = data[0, 0]
print("第一个代理的 x 坐标:", x1)

# 获取第一个代理的 y 坐标
y1 = data[0, 1]
print("第一个代理的 y 坐标:", y1)
for i, agent_state in enumerate(data):
    x, y = agent_state
    print(f"代理 {i+1}: x = {x}, y = {y}")

# 计算第一个代理和第二个代理之间的欧几里得距离
distance = np.linalg.norm(data[0] - data[1])
print(f"代理 1 和代理 2 之间的距离: {distance}")



# 提取 x 和 y 坐标
x_coords = data[:, 0]
y_coords = data[:, 1]

# 绘制代理位置
plt.scatter(x_coords, y_coords)
plt.xlabel("x")
plt.ylabel("y")
plt.title("代理位置")
plt.show()