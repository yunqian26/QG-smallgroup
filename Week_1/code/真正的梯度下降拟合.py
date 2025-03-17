import numpy as np
def step_descent(x,y,learningrate=0.02,iterations=100):
    a,b=x.shape
    opening=np.zeros(b)
    droping=[]
    for iteration in range(iterations):
        prediction=x @ opening
        diff=prediction-y
        drop=(1/(2*a))*np.sum(diff**2)
        droping.append(drop)
        step=(1/x)*(a.T @ diff)
        opening=opening*(1-step)
    return opening,droping