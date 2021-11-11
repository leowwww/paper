import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa import holtwinters
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox as ljbox 

def adf (data_1 , data_2):
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_1 = data_1[:,0]
    data_2 = data_2[:,0]
    data = []
    for i in range(len(data_1)):
        data.append(data_1[i] - data_2[i])
    a = adfuller(data)
    return a
def white_noise(data):
    return ljbox(data , lags=1,boxpierce=True)
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
def FA(real , pred):
    result = MAPE(real , pred)

if __name__ == "__main__":
    #########时间序列平稳性检验
    '''df = pd.read_excel("Data_AgTD20210604.xlsx",usecols=[0,1,2,3,4],header= None)
    df.columns = ['time','open','high','low','close']
    data_close = df.loc[: ,['close']]
    data_open = df.loc[: ,['open']]
    result = adf(data_close , data_open)
    print('open-closeADF检验：',result)
    print('openADF检验:',adfuller(np.array(data_open)))
    print('closeADF检验:',adfuller(np.array(data_close)))
    data_close = np.array(data_close)
    data_close = data_close[:,0]
    data_open = np.array(data_open)
    data_open = data_open[:,0]
    data = []
    for i in range(len(data_open)):
        data.append(data_close[i] - data_open[i])
    print('open-close白噪声:',white_noise(data))'''
    ##############检验lstm的残差序列
    '''df = pd.read_excel('lstm_residual.xlsx',usecols=[1])
    data = np.array(df)
    print('adf:{}'.format(adfuller(data)))
    print('white:{}'.format(ljbox(data , lags=1,boxpierce=False)))'''
    ##############合并数据集
    holt_winter = pd.read_excel('holt-winters-result.xlsx',usecols=[1])
    lstm_data = pd.read_excel('lstm_result.xlsx',usecols=[1])[-6725:]
    real_data = pd.read_excel('lstm_origin_data.xlsx',usecols=[1])[-6725:]
    print(real_data.head())
    print(lstm_data.head())
    print(len(holt_winter) , len(lstm_data) , len(real_data))
    lstm_data = np.array(lstm_data)
    real_data = np.array(real_data)
    holt_winter = np.array(holt_winter)
    jiejie_data = np.array(lstm_data) + np.array(holt_winter)
    
    ##########生成excel数据
    '''a = pd.DataFrame({'lstm':list(lstm_data[:,0]) , 'holt_winter':list(holt_winter[:,0]) , 'real_data':list(real_data[:,0])})
    a.to_excel('hybrid_data.xlsx')'''

    plt.plot(real_data[:100] , label = 'real_data')
    plt.plot(lstm_data[:100], label = 'lstm')
    plt.plot(jiejie_data[:100], label = 'hybrid_model')
    plt.legend()
    plt.show()
    print('lstm rmse:{}'.format( RMSE(real_data , lstm_data)[0] ))
    print('hybird rmse:{}'.format( RMSE(real_data ,jiejie_data)[0] ))
    print('lstm MAPE:{}'.format( MAPE(real_data , lstm_data)[0] ))
    print('hybird MAPE:{}'.format( MAPE(real_data ,jiejie_data)[0] ))
    print('lstm FA:{}%'.format( 100 - 100*MAPE(real_data , lstm_data)[0] ))
    print('hybird FA:{}%'.format( 100 - 100*MAPE(real_data ,jiejie_data)[0] ))

