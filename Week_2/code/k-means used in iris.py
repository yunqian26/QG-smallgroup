import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def k_means(X, K, maxiters=100, tol=1e-5):
    centers = X[np.random.choice(X.shape[0], K, replace=False)]
    for i in range(maxiters):
        distance = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        cluster = np.argmin(distance, axis=1)
        new_centers = np.array([X[cluster == k].mean(axis=0) for k in range(K)])
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    return centers, cluster


if __name__ == '__main__':
    url = 'https://gairuo.com/file/data/dataset/iris.data'
    dataraw = pd.read_csv(url)

    print(dataraw.head())  # 打印前几行数据以检查列名
    X = dataraw.iloc[:, :-1]  # 假设最后一列是标签，我们只取特征
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_clean = (X - mean) / std

    K = 3
    centers, cluster = k_means(X_clean.values, K)
    print("Centers:")
    print(centers)
    print("Cluster labels:")
    print(cluster)

    plt.scatter(X_clean.iloc[:, 0], X_clean.iloc[:, 1], c=cluster)
    plt.scatter(centers[:, 0], centers[:, 1], c='r', s=200, alpha=0.75)
    plt.title('K-means Clustering on IRIS Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()