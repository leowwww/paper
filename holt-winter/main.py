import matplotlib.pyplot as plt
from numpy.testing._private.utils import decorate_methods
from scipy.optimize._trustregion_constr.tr_interior_point import tr_interior_point
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
def MAPE(real , pred):
    sum = 0 
    for i in range(len(real)):
        sum += abs((pred[i] - real[i]) / real[i])
    return sum/len(real)
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
    
    fit1= ExponentialSmoothing(train, trend="add",seasonal="add",  seasonal_periods=1000).fit()
    #fit2 =  ExponentialSmoothing(train, trend="add").fit()
    ########滚动预测
    test_num = len(test)
    test_star = test_num % 14
    train_data = train
    pred = []
    b = fit1.forecast(test_star)#start = 0 , end = len(test)
    for i in range(test_star):
        pred.append(b[i])
    print(pred)
    train_data = np.append(train_data , test[:test_star])
    print(len(train_data))
    for i in range(test_star , test_num , 14):
        print(i,len(pred))
        b = ExponentialSmoothing(train_data, trend="add",seasonal="add",  seasonal_periods=1000).fit().forecast(14)
        for j in range(14):
            pred.append(b[0])
        train_data=np.append(train_data,test[i:i+14])
    #################################
    b = fit1.predict(start = len(train) , end = len(train)+100)
    a = fit1.forecast(100)
    #b.to_excel('normal_result.xlsx')
    pred  = pd.DataFrame(pred)
    pred.to_excel('roll_result.xlsx')
    '''plt.plot(b[:100],label = 'forecast')
    plt.plot(test[:100], label = 'real')
    plt.plot(a , label = 'pred')
    plt.legend()
    plt.show()'''
    '''b = pd.DataFrame(b)
    b.to_excel('holt-winters-result.xlsx')'''

