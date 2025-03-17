import numpy as np
import matplotlib.pyplot as plt
def step_descent(x,y,learningrate=0.02,iterations=100):
    a1,b1=x.shape
    opening=np.zeros(b1)
    drops=[]
    for iteration in range(iterations):
        prediction =x @ opening
        diff=prediction-y
        drop=(1/(2*a1))*np.sum(diff**2)
        drops.append(drop)
        step=(1/a1)*(x.T @ diff)
        opening =opening-learningrate*step
    return opening,drops
def minplus2(x, y,learningrate=0.02,iterations=100):
    e = np.vstack([x, np.ones(len(x))]).T
    coefficients,drops = step_descent(e,y,learningrate,iterations)
    x_coeff, y_coeff = coefficients
    print(f"拟合的直线方程为：y = {x_coeff:.2f}x + {y_coeff:.2f}")
    x_fit = np.linspace(min(x), max(x), num=20)
    y_fit = x_coeff * x_fit + y_coeff
    plt.plot(x_fit, y_fit, color='red', label=f'拟合曲线: y = {x_coeff:.2f}x+ {y_coeff:.2f}')
    plt.show()
    predict_y=e@coefficients
    mse=np.mean((y-predict_y)**2)#均方误差
    mae=np.mean(np.abs(y-predict_y))
    rmse=np.sqrt(mse/len(y))
    r_2=1-(np.sum((y-predict_y)**2)/np.sum((y-np.mean(y))**2))
    print('回归系数：',coefficients)
    print(f'均方误差：{mse:.4f}')
    print(f'绝对误差：{mae:.4f}')
    print(f'均方根误差：{rmse:.4f}')
    print(f'绝对系数：{r_2:.4f}')


if __name__ == '__main__':
    a=np.array([1,2,6,7,5,3])
    b = np.array([2,1,5,4,5,8])
    plt.scatter(a, b, color='b')
    minplus2(a,b)