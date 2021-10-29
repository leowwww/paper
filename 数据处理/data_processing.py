import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    return ljbox(data , lags=[38,70],boxpierce=True)


if __name__ == "__main__":
    #########时间序列平稳性检验
    df = pd.read_excel("Data_AgTD20210604.xlsx",usecols=[0,1,2,3,4],header= None)
    df.columns = ['time','open','high','low','close']
    data_close = df.loc[: ,['close']]
    data_open = df.loc[: ,['open']]
    '''result = adf(data_close , data_open)
    print('open-closeADF检验：',result)
    print('openADF检验:',adfuller(np.array(data_open)))
    print('closeADF检验:',adfuller(np.array(data_close)))'''
    data_close = np.array(data_close)
    data_close = data_close[:,0]
    data_open = np.array(data_open)
    data_open = data_open[:,0]
    data = []
    for i in range(len(data_open)):
        data.append(data_close[i] - data_open[i])
    #print('open-close白噪声:',white_noise(data))
    ##############
    print(data)
    data = pd.DataFrame(data)
    print(len(data))
    data.to_excel('open_close.xlsx')