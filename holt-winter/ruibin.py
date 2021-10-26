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
    s_temp[0] = ( s[0] + s[1] + s[2] ) / 3
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

if __name__ == "__main__":
    df = pd.read_csv("Data_AgTD20210604.xlsx")#替换成你的数据 
    train = df.get('out').values(:)
    train, test = df.iloc[:, 1], df.iloc[2500:5000, 1]
    alpha = 0.4
    train = np.array(train)
    test = np.array(test)
    train_data = []
    test_data = []
    l = len(train)
    for i in range(l):
        train_data.append(train[i])
    sigle = compute_single(alpha, train)
    a_doubel , b_double = compute_double(alpha , train)
    a_triple, b_triple, c_triple = compute_triple(alpha , train)
    sigle_data = []
    double_data = []
    triple_data = []
    count  = 0
    for i in range(l):
        if i == 0:
            double_data.append(train[0])
            continue
        count = i 
        d = a_doubel[i] + b_double[i]*1
        double_data.append(d)
    #预测后面的三个
    for i in range(3):
        d = a_doubel[-1]+b_double[-1]*i
        double_data.append(d)
    count = 0
    for i in range(l):
        if i == 0:
            triple_data.append(train[0])
            continue
        count = i
        d = a_triple[i]+ b_triple[i] + c_triple[i]
        triple_data.append(d)
    #预测后面的三个
    for i in range(3):
        d =  a_triple[-1]+ b_triple[-1]*i + c_triple[-1]*(i**2)
        triple_data.append(d)