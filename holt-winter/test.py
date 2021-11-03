import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import time
#指数平滑算法
def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    s_temp = [0 for i in range(len(s))]
    s_temp[0] = s[0]#( s[0] + s[1] + s[2] ) / 3
    for i in range(1, len(s)):
        s_temp[i] = alpha * s[i] + (1 - alpha) * s_temp[i-1]
    return s_temp

def compute_single(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    return exponential_smoothing(alpha, s)

def compute_double(alpha, s):
    '''
    二次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回二次指数平滑模型参数a, b， list
    '''
    s_single = compute_single(alpha, s)
    s_double = compute_single(alpha, s_single)

    a_double = [0 for i in range(len(s))]
    b_double = [0 for i in range(len(s))]

    for i in range(len(s)):
        a_double[i] = 2 * s_single[i] - s_double[i]                    #计算二次指数平滑的a
        b_double[i] = (alpha / (1 - alpha)) * (s_single[i] - s_double[i])  #计算二次指数平滑的b

    return a_double, b_double

def compute_triple(alpha, s):
    '''
    三次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回三次指数平滑模型参数a, b, c， list
    '''
    s_single = compute_single(alpha, s)
    s_double = compute_single(alpha, s_single)
    s_triple = exponential_smoothing(alpha, s_double)

    a_triple = [0 for i in range(len(s))]
    b_triple = [0 for i in range(len(s))]
    c_triple = [0 for i in range(len(s))]

    for i in range(len(s)):
        a_triple[i] = 3 * s_single[i] - 3 * s_double[i] + s_triple[i]
        b_triple[i] = (alpha / (2 * ((1 - alpha) ** 2))) * ((6 - 5 * alpha) * s_single[i] - 2 * ((5 - 4 * alpha) * s_double[i]) + (4 - 3 * alpha) * s_triple[i])
        c_triple[i] = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single[i] - 2 * s_double[i] + s_triple[i])

    return a_triple, b_triple, c_triple
def mae(data):
    sum = 0
    for i in range(len(data)):
        sum += abs(data[i])
    return sum/len(data)
def mse(data):
    sum = 0
    for i in range(len(data)):
        sum += (data[i])**2
    return sum/len(data)
def mape(data_real , data_redisual):
    sum = 0
    for i in range(len(data_real)):
        sum += abs(data_redisual[i]/data_real[i])
    result = sum/len(data_real)
    return result*100
def window(histor_data):
    #####每一次都进行一次指数平滑,
    #sigle = compute_single(0.4,histor_data)
    a_doubel , b_double = compute_double(0.4 , histor_data)
    return (a_doubel[-1]+b_double[-1])

if __name__ == "__main__":
    '''df = pd.read_excel("Data_AgTD20210604.xlsx",usecols = [0,1,4])
    df.columns = ['date','open','close']
    split = int(len(df)*0.8)
    train, test = df.iloc[:split, 1], df.iloc[split:, 1]'''
###########################
    df = pd.read_excel("open_close.xlsx",usecols = [1])
    df.columns = ['open_close']
    split = int(len(df)*0.8)
    train, test = df.iloc[:split,0], df.iloc[split:,0]
###########################
    alpha = 0.4
    train = np.array(train)
    test = np.array(test)
    train_data = []
    test_data = []
    l = len(train)
    for i in range(l):
        train_data.append(train[i])
    sigle = compute_single(alpha, train)#计算的si
    print(sigle[-10:] , train_data[-10:])
    print(len(sigle) , len(train_data))
    a_doubel , b_double = compute_double(alpha , train)#计算的si和ti

    a_triple, b_triple, c_triple = compute_triple(alpha , train)
    sigle_data = []
    double_data = []
    triple_data = []
    count  = 0
    for i in range(len(a_doubel)):
        if i == 0:
            double_data.append(train[0])
            #continue
        count = i 
        d = a_doubel[i] + b_double[i]*1
        double_data.append(d)
    print(double_data[:10] , train_data[:10])
    print(len(double_data ),len(train_data))
    #二次平滑预测后面的三个
    pre_thre = []
    for i in range(1,4):
        d = a_doubel[-1]+b_double[-1]*i
        pre_thre.append(d)
    count = 0
#################holt winter
    for i in range(l):
        if i == 0:
            triple_data.append(train[0])
            continue
        count = i
        d = a_triple[i]+ b_triple[i] + c_triple[i]
        triple_data.append(d)
    l = 100
    plt.plot(range(l),sigle[:l],label ="sigel",color = 'r')
    plt.plot(range(l),train_data[0:l],label = "real",color = 'g')
    plt.plot(range(l),double_data[:l] , label = 'doubel',color = 'b')
    #plt.plot(triple_data[:l] , label = 'tripe_data',color = 'y')
    plt.legend()
    plt.show()
    #三次平滑预测后面的三个
    for i in range(3):
        d =  a_triple[-1]+ b_triple[-1]*i + c_triple[-1]*(i**2)
        triple_data.append(d)
    
    residual_doubel =[]
    residual_triple = []
    for i in range(len(train_data)):
        d_doubel = train_data[i] - double_data[i]
        d_triple = train_data[i] - triple_data[i]
        residual_doubel.append(d_doubel)
        residual_triple.append(d_triple)
    '''plt.plot(residual_doubel , label = 'residual_doubel')
    #plt.plot(residual_doubel_triple,label = 'residual_triple',color = 'y')
    plt.title('residual_doubel_triple')
    plt.show()'''
    ##################################################################################################################
    #统计量判断模型好坏
    #MAE
    double_mae = mae(residual_doubel)
    print('二次平滑指数mae:',double_mae)
    triple_mae = mae(residual_triple)
    print('holt_winter mae:',triple_mae)
    #MSE
    double_mse = mse(residual_doubel)
    print('二次平滑指数MSE:',double_mse)
    triple_mse = mse(residual_triple)
    print('holt_winter MSE:',triple_mse)
    #MAPE
    double_mape = mape(train_data[:l], residual_doubel)
    print('二次平滑指数MAPE:',double_mape)
    triple_mape = mape(train_data[:l] , residual_triple)
    print('holt_winter MAPE:',triple_mape)
    ###########################################################################################################
    #dn = pd.DataFrame({"residual_high": residual_doubel})
    #dn.to_excel('residual_double_open.xlsx')s
    '''origin_money = 10000
    result = 10000
    sell_gap = 0
    i = 0
    while i < len(train_data) - 3:
        for j in range(3):
            if (double_data[i+j] - train_data[i]) / train_data[i] > 0.01:
                sell_gap = j
                break
        if(sell_gap != 0 ):
            profit = (train_data[i+sell_gap]/train_data[i])*result
            cost = profit*0.008
            result = profit - cost
        i = i+1+sell_gap
        sell_gap = 0
    year_profit = ((result - 10000)/13)/10000
    print('最后资金：',result)
    print('平均年收益率:',year_profit , '%')'''
    pred = []
    calculation_data = list(train)
    print(len(calculation_data),len(test))
    for i in range(1000):
        print('第:{}个'.format(i))
        result = window(calculation_data)
        calculation_data.append(result)
        pred.append(result)
    forcast = calculation_data[len(train):]
    plt.plot(forcast[:1000],color = 'g')
    plt.plot(test[:1000],color = 'r')
    plt.legend()
    plt.show()
    print(forcast[:100] , test[:100])
