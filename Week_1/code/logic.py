import numpy as np
import matplotlib.pyplot as plt

# Sigmoid 函数
def sigmoid(z):
    return 1/(1+np.exp(-z))#用于逻辑回归的函数

#损失函数
def compute_cost(X,y,theta):#逻辑回归的损失函数
    a=len(y)
    h=sigmoid(X @ theta)#逻辑回归的结果
    cost=(-1/a) * (y.T@np.log(h)+(1 - y).T@np.log(1 - h))#损失函数
    return cost


# 梯度下降
def gradient_descent(X,y,theta,alpha,num_iterations):
    m=len(y)
    cost_history=np.zeros(num_iterations)#用于存储损失的矩阵
    for i in range(num_iterations):#开始下降
        h=sigmoid(X@theta)#首次的结果
        gradient=(1/m) * (X.T@(h-y))#得到残差
        theta-=alpha * gradient#根据学习率不断下降
        cost_history[i]=compute_cost(X,y,theta)#将本次学习的结果存入矩阵中
    return theta, cost_history

'''
因为实在没什么头绪所以下面的数据就找了ai来帮我写qwq
'''
# 数据准备
def prepare_data():
    # 创建一个简单的二分类数据集
    np.random.seed(0)
    X = np.random.rand(100, 2) * 10
    y = (X[:, 0] + X[:, 1] > 10).astype(int)

    # 添加截距项
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    return X, y

def plot_data(X, y):
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], label='Class 0')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# 主函数
def main():
    # 准备数据
    X, y = prepare_data()

    # 初始化参数
    theta = np.zeros(X.shape[1])
    alpha = 0.1
    num_iterations = 1000

    # 训练模型
    theta, cost_history = gradient_descent(X, y, theta, alpha, num_iterations)

    # 打印训练后的参数
    print("训练后的参数:", theta)

    # 可视化损失函数
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost History')
    plt.show()

    # 可视化决策边界


if __name__ == "__main__":
    main()