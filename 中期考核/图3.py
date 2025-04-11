import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from numpy.linalg import norm

np.random.seed(80)

#参数设置
n=200#代理数量
p=0.1#伯努利分布参数
delta=1#隐私参数δ
epsilon=0.1*np.ones(n)#隐私预算ε
num_times=10000#模拟次数
tolerance=1e-2#收敛容差
num_iterations=500#最大迭代次数
Pi_n=np.ones((n, n))/n#投影矩阵


# 3. 构建拉普拉斯矩阵
def construct_laplacian(G):#手动构建拉普拉斯矩阵
    n=G.number_of_nodes()#获取节点数量
    L=np.zeros((n, n))#构建一个同样大的空矩阵用于存储拉普拉斯矩阵的值
    for i in range(n):#遍历全部节点
        for j in range(n):#再次遍历
            if i==j:#对于对角线
                L[i,j]=sum(G[i][j]['weight'] for j in G.neighbors(i))#将节点i相连的所有边的权重之和存放到L[i,j]
            elif G.has_edge(i, j):#对于非对角线
                L[i,j]=-G[i][j]['weight']#权重取负存入，因为拉普拉斯矩阵需要作为对角矩阵的度矩阵减去邻接矩阵，取负即直接取到结果
    return L#返回拉普拉斯矩阵

#分布式共识算法实现
def DP_consensus(L,initial_states,s,alpha=1e-6,delta=1,epsilon=0.1,max_iter=num_iterations,tol=tolerance):
    """
    实现差分隐私平均共识算法
    参数:
        L: 拉普拉斯矩阵
        initial_states: 初始状态
        s: 噪声增益参数(论文中的s)[16]
        alpha: 噪声衰变参数(接近0)[16]
    """
    mu=0.84
    n=len(initial_states)
    theta=initial_states.copy()
    theta_history=[theta.copy()]
    q_i=alpha+(1-alpha)*abs(s-1)#计算q_i参数
    c=delta*q_i/(epsilon*(q_i-abs(s-1)))#公式32
    var=(2*(delta**2)/(n**2))*np.sum((s**2)*(q_i**2)/((epsilon**2)*((q_i-abs(s-1))**2)*(1-q_i**2)))#公式30
    A=np.eye(n)-0.5*L
    B=s*np.eye(n)-0.5*L#s对应的单位矩阵
    for k in range(max_iter):
        noise=np.random.laplace(0,c,size=n)#生成拉普拉斯噪声
        theta_new=A@theta+B@noise#更新状态,公式16
        theta_history.append(theta_new.copy())#将更新状态记录
        theta=theta_new
        if norm(theta-np.mean(theta))<tol:#检查是否符合收敛
            break#若收敛则中断迭代
    km=np.log(var/tolerance)/np.log(1/mu)#计算迭代速率
    return var,theta,theta_history,km#返回方差、状态、迭代速率

# 5. 模拟实验（扫描s参数）
def run_sweep_experiment(s_values,num_runs=num_times):
    """运行参数扫描实验"""
    # 固定图和初始条件
    G=nx.erdos_renyi_graph(n,p)
    for u,v in G.edges():#遍历边，边权是两个i.i.d伯努利变量之和
        G[u][v]['weight']=np.random.binomial(1, p)+np.random.binomial(1, p)#设置边权，生成一个服从伯努利分布（二项分布）的随机变量两次并将其相加，增加权重的多样性，模拟不同的连接强度
    L = construct_laplacian(G)#生成拉普拉斯矩阵

    try:
        positions = np.load("init_positions.npy")#从本地数据集文件读取位置坐标
        if len(positions)!=n:
            raise ValueError("文件中的位置坐标数量与n不匹配")
        initial_states=positions[:,0]#使用x坐标作为初始状态
    except Exception as e:
        print(f"读取文件失败:{e}")#如果读取失败，使用随机生成的初始状态
        initial_states=np.random.normal(50, 10, size=n)
        print("正在使用随机生成的初始状态")

    # 存储结果
    std_devs=[]#方差
    settling_times=[]#平均迭代次数

    for s in tqdm(s_values):#有多少个s_value则迭代多少次
        conv_points=[]#收敛点
        times=[]#每次的迭代次数

        for _ in range(num_runs):#迭代指定次数
            var1,final_theta,_,num_iter=DP_consensus(L,initial_states,s)#调用平均共识算法
            conv_points.append(np.mean(final_theta))#将最后一点存入
            times.append(num_iter)#将迭代次数存入
        # 计算样本标准差和平均收敛时间
        std_devs.append(var1)
        settling_times.append(np.mean(times))

    return std_devs, settling_times,conv_points

# 6. 生成s值对数尺度范围[16]
s_values = np.logspace(np.log10(0.8), np.log10(1.2), num=50)#[0.8,1.2]中按对数取50个点
# 7. 运行实验
std_devs,settling_times,conv_points= run_sweep_experiment(s_values, num_times)

# 图3(a):
plt.figure(figsize=(18,8))#画布
plt.subplot(1, 2, 1)#分布
plt.semilogy(s_values, np.array(std_devs) ** 2, 'o-')#曲线图
plt.xlabel('s')
plt.ylabel('Empirical Var(θ∞)')
plt.title('Fig 3.a')
plt.grid(True)#网格

# 图3(b):
plt.subplot(1, 2, 2)
plt.plot(s_values, settling_times, 'o-')
plt.xlabel('s')
plt.ylabel('Ave .Settling time')
plt.title('Fig 3.b')
plt.grid(True)

plt.tight_layout()#自动调整
plt.show()
