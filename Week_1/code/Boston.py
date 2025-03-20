import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#导入所需库
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
def step_descent(x,y,learningrate=0.01,iterations=1000):                                        #调用梯度下降
    a1,b1=x.shape                                                                               #将特征矩阵的行列分别输出到a1与b1中，用于获取数据个数
    opening=np.zeros(b1)                                                                        #初始模型拟合出的参数，初始为0用于开始学习
    drops=[]                                                                                    #用于记录损失值
    for iteration in range(iterations):                                                         #开始下降
        prediction =x @ opening                                                                 #计算预测值
        diff=prediction-y                                                                       #计算预测值与真实值差值
        drop=(1/(2*a1))*np.sum(diff**2)                                                         #损失函数
        drops.append(drop)                                                                      #将本次损失值加入到损失值列表中
        step=(1/a1)*(x.T @ diff)                                                                #每次下降的梯度
        opening =opening-learningrate*step                                                      #更新拟合参数
        if np.isnan(opening).any() or np.isinf(opening).any():                                  #如果参数出现无效值
            print("梯度下降过程中出现无效值，停止迭代。")
            break                                                                               #发出警报并停止迭代
    return opening,drops                                                                        #返回参数与损失函数的值

def minplus2(x, y,learningrate=0.01,iterations=1000,type='none'):
    e = np.vstack([x, np.ones(len(x))]).T                                                       #将特征插入一个全为1的矩阵中并转至，以便后续调用
    if type=='none':                                                                            #若无需梯度下降
        coefficients, residuals, rank, s = np.linalg.lstsq(e, y, rcond=-1)                      #利用numpy中自带的求二乘法函数计算参数
    elif type=='step':                                                                          #若需要梯度下降
        coefficients,drops = step_descent(e,y,learningrate=learningrate,iterations=iterations)  #调用梯度下降求参数
    x_coeff, y_coeff = coefficients                                                             #提取系数
    predict_y=e@coefficients                                                                    #用系数与特征矩阵计算价格预测值
    print(f"拟合的直线方程为：y = {x_coeff:.2f}x + {y_coeff:.2f}")                                 #用格式化输出拟合出的直线
    x_fit = np.linspace(min(x), max(x), num=20)                                                 #横轴
    y_fit = x_coeff * x_fit + y_coeff                                                           #纵轴，因为是线性回归，遂使用一次函数
    plt.plot(x_fit, y_fit, color='red', label=f'拟合曲线: y = {x_coeff:.2f}x+ {y_coeff:.2f}')#将拟合的直线可视化
    mse=np.mean((y-predict_y)**2)#均方误差                                                        #以下为查看拟合是否足够好的函数
    mae=np.mean(np.abs(y-predict_y))
    rmse=np.sqrt(mse/len(y))
    r_2=1-(np.sum((y-predict_y)**2)/np.sum((y-np.mean(y))**2))

    print('回归系数：',coefficients)
    print(f'均方误差：{mse:.4f}')
    print(f'绝对误差：{mae:.4f}')
    print(f'均方根误差：{rmse:.4f}')
    print(f'绝对系数：{r_2:.4f}')



if __name__== '__main__':                                                                         #程序入口
    dataraw=pd.read_csv("boston.csv")                                                             #数据导入（从源文件夹）
    print(dataraw.head())
    print(dataraw.isnull().sum())
    print(dataraw.describe())
    print(dataraw.shape)
    print(dataraw.dtypes)                                                                         #查看数据基本数据
    lengthofdata=np.arange(len(dataraw))                                                          #读取数据长度
    shuff=dataraw.sample(frac=1).reset_index(drop=True)                                           #数据打乱（frac=1表示全数据打乱）并替换掉原来的
    ratio=0.8
    ratio_reflect=1-ratio
    ratio_point=int(len(shuff)*ratio)
    ratio_repoint=int(len(shuff)*ratio_reflect)
    data=shuff[:ratio_point]
    data_reflect=shuff[ratio_repoint:]                                                            #分成80%和20%两个数据集

    # 全体数据箱线图
    for i in range(data.shape[1]):                                                                #遍历每一个特征
        plt.boxplot(data.iloc[:, i], showmeans=True, meanline=False)                              #绘制箱线图并显示均值线
        plt.xlabel(data.columns[i])                                                               #将x轴改为特征名
        plt.show()                                                                                #显示图片
    # 全体特征与MEDV的散点图
    num_features = data.shape[1] - 1                                                              #减去目标变量 'MEDV'
    for k in range(num_features):
        x = data.iloc[:, k]                                                                       #获取每一个列/特征
        features = data.columns[k]                                                                #获取特征名称
        plt.scatter(x=data[features], y=data['MEDV'], label='data point', alpha=0.5)              #画出该特征与MEDV的散点图
        plt.title(f'{features} vs MEDV')                                                          #图名
        plt.xlabel(features)                                                                      #x轴名称
        plt.ylabel('MEDV')                                                                        #y轴名称
        plt.legend()                                                                              #图例
        plt.show()                                                                                #画图

    '''
    LSTAT
    RM
    DIS
    AGE
    CRIM
    以上为数据中看起来像是线性的特征，接下来对他们进行数据清洗与可视化
    '''


    data1 = data.copy()                                                                           #复制数据集，方便后续更改不会出现交错
    Q1_LSTAT = data1["LSTAT"].quantile(0.25)                                                      #计算25%所在的点
    Q3_LSTAT = data1["LSTAT"].quantile(0.75)                                                      #计算75%所在的点
    diff = Q3_LSTAT - Q1_LSTAT                                                                    #计算25%到75%的差值
    low = Q1_LSTAT - diff * 1.5                                                                   #计算所需要保留的下限
    high = Q3_LSTAT + diff * 1.5                                                                  #计算所需要保留的上限
    data1 = data1[(data1["LSTAT"] >= low) & (data1["LSTAT"] <= high)]                             #筛除无需的数据（异常点）
    # 以下为3σ方法
    mean_LSTAT = data1['LSTAT'].mean()                                                            #该特征的平均值
    std_LSTAT = data1['LSTAT'].std()                                                              #该特征的标准差
    lowline_LSTAT = mean_LSTAT - 3 * std_LSTAT                                                    #平均值减去3σ计算下界
    highline_LSTAT = mean_LSTAT + 3 * std_LSTAT                                                   #平均值加上3σ计算上届
    data1 = data1[(data1['LSTAT'] >= lowline_LSTAT) & (data1['LSTAT'] <= highline_LSTAT)]         #筛除无需的数据（异常点）

    # data1=data1[(data1["MEDV"]<50)]
    plt.scatter(data1['LSTAT'], data1['MEDV'],c='b',alpha=0.5,label='LSTAT')                      #画出数据清洗后的数据
    plt.title('Price LSTATrates')                                                                 #图名
    plt.xlabel('LSTAT')                                                                           #x轴名字
    plt.ylabel('Price')                                                                           #y轴名字
    minplus2(data1['LSTAT'], data1['MEDV'],learningrate=0.009,iterations=20000,type='step')       #调用最小二乘法计算拟合直线
    plt.show()                                                                                    #打印


#以下就是重复上面的过程，仅更改参数，不再过多加上注释
#每一个数据的学习率与迭代次数均为多次尝试得出较为优秀表现的参数

    data1 = data.copy()
    Q1_AGE = data1["AGE"].quantile(0.25)
    Q3_AGE = data1["AGE"].quantile(0.75)
    diff = Q3_AGE - Q1_AGE
    low = Q1_AGE - diff * 1.5
    high = Q3_AGE + diff * 1.5
    data1 = data1[(data1["AGE"] >= low) & (data1["AGE"] <= high)]

    mean_AGE = data1['AGE'].mean()
    std_AGE = data1['AGE'].std()
    lowline_AGE = mean_AGE - 3 * std_AGE
    highline_AGE = mean_AGE + 3 * std_AGE
    data1 = data1[(data1['AGE'] >= lowline_AGE) & (data1['AGE'] <= highline_AGE)]

    zero_of_AGE = data1[data1['AGE'] >=100]
    aver_of_price = zero_of_AGE['MEDV'].mean()
    data1.loc[data1['AGE'] >= 100, 'MEDV'] = aver_of_price

    # data1 = data1[(data1['MEDV'] < 35)&(data1['MEDV'] > 10)]
    plt.scatter(data1['AGE'], data1['MEDV'], c='r', alpha=0.5, label='AGE')
    plt.title('Price AGETrates')
    plt.xlabel('AGE')
    plt.ylabel('Price')
    minplus2(data1['AGE'], data1['MEDV'], learningrate=0.0001, iterations=150000, type='step')
    plt.show()




    data1 = data.copy()
    Q1_AGE = data1["RM"].quantile(0.25)
    Q3_AGE = data1["RM"].quantile(0.75)
    diff = Q3_AGE - Q1_AGE
    low = Q1_AGE - diff * 1.5
    high = Q3_AGE + diff * 1.5
    data1 = data1[(data1["RM"] >= low) & (data1["RM"] <= high)]

    mean_AGE = data1['RM'].mean()
    std_AGE = data1['RM'].std()
    lowline_AGE = mean_AGE - 3 * std_AGE
    highline_AGE = mean_AGE + 3 * std_AGE
    data1 = data1[(data1['RM'] >= lowline_AGE) & (data['RM'] <= highline_AGE)]

    zero_of_AGE = data1[data1['RM'] <= 0.5]
    aver_of_price = zero_of_AGE['MEDV'].mean()
    data1.loc[data['RM'] <= 0.5, 'MEDV'] = aver_of_price

    # data1 = data1[(data1['MEDV'] < 45)&(data1['MEDV'] > 10)]
    plt.scatter(data1['RM'], data1['MEDV'], c='b', alpha=0.4, label='RM')
    plt.title('Price AGETrates')
    plt.xlabel('RM')
    plt.ylabel('Price')
    minplus2(data1['RM'], data1['MEDV'], learningrate=0.001, iterations=20000, type='step')
    plt.show()





    data1 = data.copy()
    Q1_DIS = data1["DIS"].quantile(0.25)
    Q3_DIS = data1["DIS"].quantile(0.75)
    diff = Q3_DIS - Q1_DIS
    low = Q1_DIS - diff * 1.5
    high = Q3_DIS + diff * 1.5
    data1 = data1[(data1["DIS"] >= low) & (data1["DIS"] <= high)]

    mean_DIS = data1['DIS'].mean()
    std_DIS = data1['DIS'].std()
    lowline_DIS = mean_DIS - 3 * std_DIS
    highline_DIS = mean_DIS + 3 * std_DIS
    data1 = data1[(data1['DIS']>=lowline_DIS)&(data['DIS']<=highline_DIS)]

    zero_of_AGE = data1[data1['DIS'] <= 0.5]
    aver_of_price = zero_of_AGE['MEDV'].mean()
    data1.loc[data['DIS'] <= 0.5, 'MEDV'] = aver_of_price

    data1 = data1[(data1['MEDV']<45)&(data1['DIS']<10)]
    plt.scatter(data1['DIS'], data1['MEDV'], c='b', alpha=0.4, label='DIS')
    plt.title('Price DISTrates')
    plt.xlabel('DIS')
    plt.ylabel('Price')
    minplus2(data1['DIS'], data1['MEDV'], learningrate=0.001, iterations=20000, type='step')
    plt.show()



    data1 = data.copy()
    Q1_CRIM = data1["CRIM"].quantile(0.25)
    Q3_CRIM = data1["CRIM"].quantile(0.75)
    diff = Q3_CRIM - Q1_CRIM
    low = Q1_CRIM - diff * 1.5
    high = Q3_CRIM + diff * 1.5
    data1 = data1[(data1["CRIM"] >= low) & (data1["CRIM"] <= high)]

    mean_CRIM = data1['CRIM'].mean()
    std_CRIM = data1['CRIM'].std()
    lowline_CRIM = mean_CRIM - 3 * std_CRIM
    highline_CRIM = mean_CRIM + 3 * std_CRIM
    data1 = data1[(data1['CRIM'] >= lowline_CRIM) & (data['CRIM'] <= highline_CRIM)]
    #
    zero_of_CRIM = data1[data1['CRIM'] <= 0.1]
    aver_of_price = zero_of_CRIM['MEDV'].mean()
    data1.loc[data['CRIM'] <= 0.1, 'MEDV'] = aver_of_price

    data1 = data1[(data1['MEDV'] < 45)&(data1['MEDV'] > 10)]
    plt.scatter(data1['CRIM'], data1['MEDV'], c='b', alpha=0.4, label='CRIM')
    plt.title('Price CRIMTrates')
    plt.xlabel('CRIM')
    plt.ylabel('Price')
    minplus2(data1['CRIM'], data1['MEDV'], learningrate=0.001, iterations=20000, type='step')
    plt.show()
