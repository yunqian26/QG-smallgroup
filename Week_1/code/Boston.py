import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyexpat import features

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

def minplus2(x, y,learningrate=0.02,iterations=100,type='none'):
    e = np.vstack([x, np.ones(len(x))]).T
    if type=='none':
        coefficients, residuals, rank, s = np.linalg.lstsq(e, y, rcond=-1)
    elif type=='step':
        coefficients,drops = step_descent(e,y,learningrate=learningrate,iterations=iterations)
    x_coeff, y_coeff = coefficients
    print(f"拟合的直线方程为：y = {x_coeff:.2f}x + {y_coeff:.2f}")
    x_fit = np.linspace(min(x), max(x), num=20)
    y_fit = x_coeff * x_fit + y_coeff
    plt.plot(x_fit, y_fit, color='red', label=f'拟合曲线: y = {x_coeff:.2f}x+ {y_coeff:.2f}')

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

if __name__== '__main__':
    dataraw=pd.read_csv("D:\\OneDrive\\文档\\GitHub\\QG-smallgroup\\Week_1\\code\\boston.csv")
    lengthofdata=np.arange(len(dataraw))
    shuff=dataraw.sample(frac=1).reset_index(drop=True)
    ratio=0.8
    ratio_point=int(len(shuff)*ratio)
    data=shuff[:ratio_point]
    # print(data.head())
    # print(data.isnull().sum())
    # print(data.describe())
    # print(data.shape)
    # print(dataraw.head)
    # print(data.dtypes)
    # print(data["PIRATIO"])
    # plt.figure(figsize=(8, 4))
    # plt.boxplot(data["CRIM"])
    # plt.title("Boxplot of CRIM")
    # plt.show()
    # plt.figure(figsize=(8, 4))
    # plt.boxplot(data["PIRATIO"])
    # plt.title("Boxplot of PIRATIO")
    # plt.show()
    # plt.figure(figsize=(8, 4))
    # plt.boxplot(data["CHAS"])
    # plt.title("Boxplot of CHAS")
    # plt.show()
    # plt.figure(figsize=(8, 4))
    # plt.boxplot(data["INDUS"])
    # plt.title("Boxplot of INDUS")
    # plt.show()

    #数据预处理
    # plt.figure(figsize=[10, 10])  # 设置整个图的大小
    # for i in range(data.shape[1]):  # 遍历每一列
    #     plt.subplot(4,4, i + 1)  # 使用 5×5 的网格布局
    #     plt.boxplot(data.iloc[:, i], showmeans=True, meanline=False)  # 绘制箱线图并显示均值线
    #     plt.xlabel(data.columns[i])  # 使用列名作为 x 轴标签
    # plt.tight_layout()  # 自动调整子图布局，避免重叠
    # plt.show()
    #
    # plt.figure(figsize=[10, 10])
    #
    # # 遍历每个特征
    # for j in range(data.shape[1]):
    #     # 获取当前特征的数据
    #     x = data.iloc[:, j]
    #     # 绘制散点图
    #     plt.scatter([i for i in range(data.shape[0])], x, label='data point', alpha=0.6)
    #     # 计算分位数
    #     quantiles = np.quantile(x, [0.25, 0.5, 0.75])
    #     # 绘制分位数点
    #     plt.scatter([round(data.shape[0] / 4), round(data.shape[0] / 2), round(data.shape[0] / 4 * 3)],
    #                 quantiles, color='red', label='fen wei shu dian')
    #     # 添加标签和标题
    #     features_name=data.columns[j]
    #     plt.xlabel('data suoyin')
    #     plt.ylabel('MEDV')
    #     plt.title(f'tezheng:{features_name}')
    #     plt.legend()
    #     plt.show()


    num_features = data.shape[1] - 1  # 减去目标变量 'MEDV'
    for k in range(num_features):
        x = data.iloc[:, k]  # 获取当前特征列
        features = data.columns[k]  # 获取当前特征名称
        plt.scatter(x=data[features], y=data['MEDV'], label='data point', alpha=0.5)
        plt.title(f'{features} vs MEDV')
        plt.xlabel(features)
        plt.ylabel('MEDV')
        plt.legend()
        plt.show()















    #z分数法
    # mean_CRIM=data['CRIM'].mean()
    # std_CRIM=data['CRIM'].std()
    # lowline_CRIM=mean_CRIM-3*std_CRIM
    # highline_CRIM=mean_CRIM+3*std_CRIM
    # data=data[(data['CRIM']>=lowline_CRIM)&(data['CRIM']<=highline_CRIM)]
    # zero_of_crim=data[data['CRIM']<=0.5]
    # aver_of_price=zero_of_crim['MEDV'].mean()




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
    # data1=data.copy()
    # data1.loc[data['CRIM']<=0.5,'MEDV']=aver_of_price
    # data1=data1[data1['MEDV']<45]
    # plt.scatter(data1['CRIM'], data1['MEDV'],c='r',alpha=0.5,label='CRIM')
    # plt.title('Price Crimrates')
    # plt.xlabel('CRIM')
    # plt.ylabel('Price')
    # minplus2(data1['CRIM'], data1['MEDV'],type='step')
    # plt.show()
    #
    # # 房价（MEDV)与零售INDUS，折线图
    # plt.scatter(data1['INDUS'], data1['MEDV'],c='r',alpha=0.5,label='INDUS')
    # plt.title('Price by Indus')
    # plt.xlabel('Indus')
    # plt.ylabel('Price')
    # plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10,11,13,15,17,19])
    # minplus2(data['INDUS'], data['MEDV'],learningrate=0.01,type='step')
    # plt.show()


    #房价(MEDV)与学生与教师比例(PTRATIO),折线图
    # 线箱法
    # data2=data.copy()
    # Q1_piratio = data2["PIRATIO"].quantile(0.25)
    # Q3_piratio = data2["PIRATIO"].quantile(0.75)
    # diff = Q3_piratio - Q1_piratio
    # low = Q1_piratio - diff * 1.5
    # high = Q3_piratio + diff * 1.5
    # data2 = data2[(data2["PIRATIO"] >= low) & (data2["PIRATIO"] <= high)]
    # plt.scatter(data1['PIRATIO'], data1['MEDV'],c='r',alpha=0.5,label='INDUS')
    # plt.title('Price by Piratio')
    # plt.xlabel('Piratio')
    # plt.ylabel('Price')
    # minplus2(data2['PIRATIO'], data2['MEDV'])
    # plt.show()

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
