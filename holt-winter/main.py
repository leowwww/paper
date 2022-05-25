import matplotlib.pyplot as plt
from numpy.testing._private.utils import decorate_methods
from scipy.optimize._trustregion_constr.tr_interior_point import tr_interior_point
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import os
path ='E:\\项目\\论文实验\\stock_predict_with_LSTM-master\\data\\data_emd\\residual'#lstm_residual(emd0).xlsx
def RMSE(real , pred):
    sum = 0
    for i in range(len(real)):
        sum += (real[i] - pred[i])**2
    return (sum/len(real))**0.5
def MAPE(real , pred):
    sum = 0 
    for i in range(len(real)):
        sum += abs((pred[i] - real[i]) / real[i])
    return sum/len(real)
def run (roll_num , filelist , key):
    df = pd.read_excel(os.path.join(path , filelist[key]),usecols = [1])
    df.columns = ['residual']
    split = int(len(df)*0.5)
    train, test = df.iloc[:split,0], df.iloc[split:,0]
    train = np.array(train)
    test = np.array(test)
    for i in range(len(train)):#异常值处理
        if abs(train[i])>50:
            train[i] = 0
    #print(len(df) , len(train) , len(test))
    # Holt’s Method
    fit1= ExponentialSmoothing(train, trend="add",seasonal="add",  seasonal_periods=1000).fit()
    ########滚动预测
   
    test_num = len(test)
    print('数据长度：',test_num)
    test_star = test_num % roll_num
    train_data = train
    pred = []
    if test_star !=0:
        pred = []
        b = fit1.forecast(test_star)#start = 0 , end = len(test)
        for i in range(test_star):
            pred.append(b[i])
        #print(pred)
        train_data = np.append(train_data , test[:test_star])
    #print(len(train_data))
    for i in range(test_star , test_num , roll_num):
        #print(i,len(pred))
        b = ExponentialSmoothing(train_data, trend="add",seasonal="add",  seasonal_periods=1000).fit().forecast(roll_num)
        for j in range(roll_num):
            pred.append(b[j])
        train_data=np.append(train_data,test[i:i+roll_num])

    pred  = pd.DataFrame(pred)
    pred.to_excel('./emd_residual_data/roll_{}_result_{}.xlsx'.format(roll_num , filelist[key]))
    return pred
if __name__ =='__main__':   
    filelist = os.listdir(path)
    print(filelist)
    #pred = run(500, filelist , 0)
    sum = np.zeros(18010,dtype= float) 
    for i in range(len(filelist)):
        print('-----------第{}个---------'.format(i))
        pred = run(500 , filelist , i)
        pred = np.array(pred.iloc[:,0])
        sum += pred
    sum = pd.DataFrame(sum)
    sum.to_excel('./emd_residual_data/hw-sum.xlsx')
    exit(0)
    print('结束啦！！')


        
    
















    df = pd.read_excel("E:\\项目\\论文实验\\stock_predict_with_LSTM-master\\data\\data_emd\\residual\\lstm_residual(emd0).xlsx",usecols = [1])
    df.columns = ['residual']
    split = int(len(df)*0.8)
    train, test = df.iloc[:split,0], df.iloc[split:,0]
    train = np.array(train)
    test = np.array(test)
    '''for i in range(len(train)):#异常值处理
        if abs(train[i])>50:
            train[i] = 0
    print(len(df) , len(train) , len(test))
    ################归一化
    max_data = max(np.array(df))[0]
    min_data = min(np.array(df))[0]
    print(max_data,min_data)
    new_data = []
    df = np.array(df)
    for i in range(len(df)):
        new_data.append((df[i][0] - min_data))  
    print(len(new_data))
    train , test = new_data[:split] , new_data[split:]
    train = np.array(train)
    test = np.array(test)
    print(train[:10])
    # Holt’s Method
    
    fit1= ExponentialSmoothing(train, trend="add",seasonal="add",  seasonal_periods=1000).fit()
    #fit2 =  ExponentialSmoothing(train, trend="add").fit()
    ########滚动预测
    roll_num = 1
    test_num = len(test)
    test_star = test_num % roll_num
    train_data = train
    pred = []
    if test_star !=0:
        pred = []
        b = fit1.forecast(test_star)#start = 0 , end = len(test)
        for i in range(test_star):
            pred.append(b[i])
        print(pred)
        train_data = np.append(train_data , test[:test_star])
    print(len(train_data))
    for i in range(test_star , test_num , roll_num):
        print(i,len(pred))
        b = ExponentialSmoothing(train_data, trend="add",seasonal="add",  seasonal_periods=1000).fit().forecast(roll_num)
        for j in range(roll_num):
            pred.append(b[j])
        train_data=np.append(train_data,test[i:i+roll_num])'''
    #################################
    fit1= ExponentialSmoothing(train, trend="add",seasonal="add",  seasonal_periods=1000).fit()
    b = fit1.predict(start = len(train) , end = len(train)+len(test))
    a = fit1.forecast(len(test))
    #b.to_excel('normal_result.xlsx')
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(pred)
    plt.title('pred')
    plt.xlabel('time (s)')
    plt.subplot(2,1,2)
    plt.plot(a)
    plt.title('Decomposed modes')
    plt.xlabel('time (s)')
    plt.legend()
    plt.tight_layout()
    plt.show()  
    