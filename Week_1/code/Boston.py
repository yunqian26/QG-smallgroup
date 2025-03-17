import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy.physics.units import length

'''
CRIM：城镇人均犯罪率。
ZN：占地面积超过25,000平方英尺的住宅用地比例。
INDUS：城镇非零售业务地区的比例。
CHAS：查尔斯河虚拟变量（如果靠近查尔斯河则为1，否则为0）。
NOX：一氧化氮浓度（每千万份）。
RM：每栋住宅的平均房间数。
AGE：1940年之前建造的自住单位比例。
DIS：与波士顿五个就业中心的加权距离。
RAD：径向高速公路的可达性指数。
TAX：每10,000美元的全额物业税率。
PIRATIO：城镇的学生与教师比例。
B：1000×(Bk - 0.63)^2，其中Bk是城镇黑人的比例。
LSTAT：低收入阶层人口比例。
MEDV:自有住房的中位数价格，单位为1000美元
'''

def minplus2(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    coefficients, residuals, rank, s = np.linalg.lstsq(A, y, rcond=-1)
    x_coeff, y_coeff = coefficients
    print(f"拟合的直线方程为：y = {x_coeff:.2f}x + {y_coeff:.2f}")
    x_fit = np.linspace(min(x), max(x), num=20)
    y_fit = x_coeff * x_fit + y_coeff
    plt.plot(x_fit, y_fit, color='red', label=f'拟合曲线: y = {x_coeff:.2f}x+ {y_coeff:.2f}')
    plt.show()

if __name__== '__main__':
    dataraw=pd.read_csv("D:\\document\\GitHub\\QG-smallgroup\\Week_1\\code\\boston.csv")
    lengthofdata=np.arange(len(dataraw))
    shuff=dataraw.sample(frac=1).reset_index(drop=True)
    ratio=0.2
    ratio_point=int(len(shuff)*ratio)
    data=shuff[:ratio_point]
    print(data.head())
    print(data.isnull().sum())
    print(data.describe())
    print(data.shape)
    print(dataraw.head)
    # # print(data.dtypes)
    # print(data["PIRATIO"])
    # # plt.figure(figsize=(8, 4))
    # # plt.boxplot(data["CRIM"])
    # # plt.title("Boxplot of CRIM")
    # # plt.show()
    # plt.figure(figsize=(8, 4))
    # plt.boxplot(data["PIRATIO"])
    # plt.title("Boxplot of PIRATIO")
    # plt.show()
    # # plt.figure(figsize=(8, 4))
    # # plt.boxplot(data["CHAS"])
    # # plt.title("Boxplot of CHAS")
    # # plt.show()
    # # plt.figure(figsize=(8, 4))
    # # plt.boxplot(data["INDUS"])
    # # plt.title("Boxplot of INDUS")
    # # plt.show()

    # #数据预处理
    #线箱法
    Q1_piratio = data["PIRATIO"].quantile(0.25)
    Q3_piratio = data["PIRATIO"].quantile(0.75)
    diff=Q3_piratio-Q1_piratio
    low=Q1_piratio-diff*1.5
    high=Q3_piratio+diff*1.5
    data=data[(data["PIRATIO"]>=low)&(data["PIRATIO"]<=high)]
    #z分数法
    mean_CRIM=data['CRIM'].mean()
    std_CRIM=data['CRIM'].std()
    lowline_CRIM=mean_CRIM-3*std_CRIM
    highline_CRIM=mean_CRIM+3*std_CRIM
    data=data[(data['CRIM']>=lowline_CRIM)&(data['CRIM']<=highline_CRIM)]
    zero_of_crim=data[data['CRIM']<=0.5]
    aver_of_price=zero_of_crim['MEDV'].mean()
    #



    # # print(data.head())
    # # print(data.isnull().sum())
    # # print(data.describe())
    # # print(data.shape)
    # # print(data.dtypes)
    # print(data["PIRATIO"])
    # plt.figure(figsize=(8, 4))
    # plt.boxplot(data["PIRATIO"])
    # plt.title("Boxplot of PIRATIO")
    # plt.show()

    # #绘图
    #房价（MEDV）与犯罪率（CRIM）,散点图
    data1=data.copy()
    data1.loc[data['CRIM']<=0.5,'MEDV']=aver_of_price
    data1=data1[data1['MEDV']<48]
    plt.scatter(data1['CRIM'], data1['MEDV'],c='r',alpha=0.5,label='CRIM')
    plt.title('Price Crimrates')
    plt.xlabel('CRIM')
    plt.ylabel('Price')
    minplus2(data1['CRIM'], data1['MEDV'])
    plt.show()

    #房价（MEDV)与零售INDUS，折线图
    # price_indus=data.groupby('INDUS')['MEDV'].mean()
    # price_indus.plot(color=['b'])
    # plt.title('Price by Indus')
    # plt.xlabel('Indus')
    # plt.ylabel('Price')
    # plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10,11,13,15,17,19])
    # minplus2(data['INDUS'], data['MEDV'])
    # plt.show()


    #房价(MEDV)与学生与教师比例(PTRATIO),折线图
    price_ptratip=data.groupby('PIRATIO')['MEDV'].mean()
    price_ptratip.plot(color=['b'])
    plt.title('Price by Piratio')
    plt.xlabel('Piratio')
    plt.ylabel('Price')
    minplus2(data['PIRATIO'], data['MEDV'])
    plt.show()

    #
    # #房价(MEDV)与高速公路可达指数(RAD),折线图
    # price_highway=data.groupby('RAD')['MEDV'].mean()
    # price_highway.plot(color=['b'])
    # plt.title('Price by Highway')
    # plt.xlabel('RAD')
    # plt.ylabel('Price')
    # minplus2(data['RAD'], data['MEDV'])
    # plt.show()

    #
    # #房价(MEDV)与是否靠近河流(CHAS)
    # price_chas=data.groupby('CHAS')['MEDV'].mean()
    # price_chas.plot(kind='bar',color=['y'])
    # plt.title('Price by Chas')
    # plt.xlabel('CHAS')
    # plt.ylabel('Price')
    # minplus2(data['CHAS'], data['MEDV'])
    # plt.show()
