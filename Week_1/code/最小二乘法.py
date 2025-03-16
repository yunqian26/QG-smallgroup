import numpy as np
import matplotlib.pyplot as plt
def minplus2(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    coefficients, residuals, rank, s = np.linalg.lstsq(A, y, rcond=-1)
    x_coeff, y_coeff = coefficients
    print(f"拟合的直线方程为：y = {x_coeff:.2f}x + {y_coeff:.2f}")
    plt.scatter(x, y, color='b')
    x_fit = np.linspace(min(x), max(x), num=20)
    y_fit = x_coeff * x_fit + y_coeff
    plt.plot(x_fit, y_fit, color='red', label=f'拟合曲线: y = {x_coeff:.2f}x+ {y_coeff:.2f}')
    plt.show()

if __name__ == '__main__':
    a=np.array([1, 2, 6, 7, 5,2])
    b = np.array([2, 1, 5, 4, 5,8])
    minplus2(a,b)