import numpy as np
import matplotlib.pyplot as plt
def fx(x):
    return (x-2)**2+10
def dx(x):
    return 2*(x-2)
def stepdown(x,learningrate,iterations):
    for i in range(iterations):
        temdown=dx(x)
        x=x-learningrate*temdown
    return x
if __name__ == '__main__':
    x=np.arange(-10,11,1)
    y=np.arange(-10,11,1)
    learningrate=0.1
    iteration=30
    finalx=stepdown(x,learningrate,iteration)
    print(finalx)
    x_values = np.linspace(-10, 10, 400)
    plt.plot(x_values, fx(x_values), label='f(x)')
    plt.scatter(finalx, fx(finalx), color='red', label="Optimal Point")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Gradient Descent Optimization")
    plt.show()

















