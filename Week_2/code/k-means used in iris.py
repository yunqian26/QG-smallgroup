import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def k_means(X, K, maxiters=100, tol=1e-6):
    centers = X[np.random.choice(X.shape[0], K, replace=False)]#随机选取中心点
    for i in range(maxiters):#开始迭代
        distance = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)#计算中心点与其余点的距离
        cluster = np.argmin(distance, axis=1)#找到距离最小的中心点
        new_centers = np.array([X[cluster == k].mean(axis=0) for k in range(K)])#重新分配中心点
        if np.linalg.norm(new_centers - centers) < tol:#若两次中心点的移动距离过小，则视为迭代结束
            break
        centers = new_centers#更新中心点
    return centers, cluster#返回中心点与距离


if __name__ == '__main__':
    url = 'https://gairuo.com/file/data/dataset/iris.data'
    dataraw = pd.read_csv(url)
    print(dataraw.head()) #检查
    X = dataraw.iloc[:, :-1] #假设最后一列是标签，只取特征
    mean = np.mean(X, axis=0)#计算列平均值
    std = np.std(X, axis=0)#计算列标准差
    X_clean = (X - mean) / std#对数据标准化处理

    K = 3#聚类中心个数
    centers, cluster = k_means(X_clean.values, K)#使用kmeans
    print("中心点：")
    print(centers)
    print("聚类结果：")
    print(cluster)#文字输出结果

    plt.scatter(X_clean.iloc[:, 0], X_clean.iloc[:, 1], c=cluster)#绘制iris数据点图
    plt.scatter(centers[:, 0], centers[:, 1], c='r', s=200, alpha=0.75)#绘制中心点
    plt.title('K-means Clustering on IRIS Dataset')#图名（这个ai跑的）
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()#绘图