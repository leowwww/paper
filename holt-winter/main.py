import matplotlib.pyplot as plt
from numpy.testing._private.utils import decorate_methods
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import time
def RMSE(real , pred):
    sum = 0
    for i in range(len(real)):
        sum += (real[i] - pred[i])**2
    return (sum/len(real))**0.5
if __name__ =='__main__':   
    df = pd.read_excel("data\\lstm_residual.xlsx",usecols = [1])
    df.columns = ['residual']
    split = int(len(df)*0.8)
    train, test = df.iloc[:split,0], df.iloc[split:,0]
    train = np.array(train)
    test = np.array(test)
    for i in range(len(train)):
        if abs(train[i])>50:
            train[i] = 0
    print(len(df) , len(train) , len(test))
    ################归一化
    '''max_data = max(np.array(df))[0]
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
    print(train[:10])'''
    # Holt’s Method
    
    fit1= ExponentialSmoothing(train, trend="add",seasonal="add",  seasonal_periods=10).fit()
    a = fit1.forecast(100)
    plt.plot(a[:100],label = 'predction')
    plt.plot(test[:100] , label = 'real')
    plt.legend()
    plt.show()
    p_d = []
    train_data = train[-10:]
    for i in range(100):
        fit1 = ExponentialSmoothing(train_data, trend="add").fit()
        #print(len(train_data))
        pred = fit1.forecast(1)
        p_d.append(pred[0])
        train_data = list(train_data)
        train_data.append(pred[0])
        train_data = np.array(train_data)
    plt.plot(p_d[:100] , color = 'r',label = 'predction')
    plt.plot(test[:100], color = 'g',label = 'real')
    plt.legend()
    plt.show()
    #fit2 = ExponentialSmoothing(data, trend="mul", seasonal=None).fit().fittedvalues

    #fit1= ExponentialSmoothing(train, trend="add",seasonal="add",  seasonal_periods=1000).fit()#seasonal="add",  seasonal_periods=100
    #fit2 = ExponentialSmoothing(train, trend="add").fit()
    train_data = train
    train_data = list(train_data)
    pred_data = []
    for i in range(50):
        fit = ExponentialSmoothing(train[:50+i], trend="add",seasonal="add",  seasonal_periods=10).fit()
        prd = fit.forecast(1)
        print(prd,test[i])
        pred_data.append(prd)
    plt.plot(pred_data[:50])
    plt.plot(train[50:50+50])
    plt.legend()
    plt.show()
    rmse = RMSE(test , pred_data)
    print('RMSE:{}'.format(rmse))
    pred = pd.DataFrame(pred_data)
    pred.to_excel('linear_open.xlsx')
