import numpy as np
import matplotlib.pyplot as plt
def k_means(X, K, maxiters=100, tol=1e-4):
    centers=X[np.random.choice(X.shape[0], K, replace=False)]
    for i in range(maxiters):
        distance = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        cluster=np.argmin(distance,axis=1)
        new_centers = np.array([X[cluster == k].mean(axis=0) for k in range(K)])
        plt.scatter(X[:, 0], X[:, 1], c=cluster)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.show()
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers=new_centers
    return centers,cluster

if __name__=='__main__':
    np.random.seed(999)
    X = np.vstack([
        np.random.normal(loc=[0, 0], scale=1.0, size=(100, 2)),
        np.random.normal(loc=[5, 5], scale=1.0, size=(100, 2)),
        np.random.normal(loc=[0, 5], scale=1.0, size=(100, 2))
    ])
    K=4
    centers,cluster=k_means(X, K)
    print(centers)
    print(cluster)

    plt.scatter(X[:, 0], X[:, 1], c=cluster)
    plt.scatter(centers[:,0] ,centers[:,1],c='r')
    plt.show()