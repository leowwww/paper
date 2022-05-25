from pandas import read_excel
from pandas import datetime
from pandas import DataFrame
import pandas
from pandas.io.pytables import SeriesFixed
from scipy.stats.stats import ModeResult
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
import time
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import pandas as pd
from statsmodels.tsa.tsatools import lagmat2ds
def parser(x):
    return datetime.strptime(str(x), '%Y-%m')
series =read_excel('DATA_AU_TDX.xlsx',usecols=[2])
series = np.array(series).flatten()
series = pd.Series(series)
diff1 = series.diff(1)
fig = plt.figure(figsize=(12 , 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(series , lags = 40 , ax = ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(series , lags= 40 , ax = ax2)
plt.show()
########################



series =read_excel('DATA_AU_TDX.xlsx',usecols=[2])
split = int(len(series)*0.7)
train = diff1.iloc[:split]
test = diff1.iloc[split:]
train = np.array(train)
test = np.array(test)
######################################季节性arima
'''maxmodel = sm.tsa.statespace.SARIMAX(train,order=(1, 1, 1),seasonal_order=(1,1,1,12)).fit()
b = maxmodel.predict(start = 0 , end = len(test))
plt.plot(b[:100])
plt.plot(test[:100])
plt.legend()
plt.show()'''
######################################
model_fit = ARIMA(train, order=(1,1,1)).fit()
#model_fit = sm.tsa.statespace.SARIMAX(train,order=(1, 1, 1),seasonal_order=(10,10,10,100)).fit()
print(model_fit.aic , model_fit.bic ,model_fit.hqic)
resid = model_fit.resid
pred = model_fit.predict(start = len(train)  , end = len(train) + 100)
pred_1 = model_fit.predict(start = len(train) , end = len(train)+100)
for i in range(100):
    print(pred[i] - pred_1[i])
#pred = model_fit.forecast(len(test))
print(len(pred) , len(train))
plt.plot(pred[:100] , label = 'predction')
#plt.plot(train[:100] , label = 'real_train')
plt.plot(test[:100] , label = 'real_test')
from statsmodels.tsa.arima_model import ARIMA#statsmodels.tsa.arima.model 
import matplotlib.pyplot as plt

import time
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
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
series =read_excel('Data_AgTD20210604.xlsx',usecols=[1])
print(len(series))
split = int(len(series)*0.7)
train = series.iloc[:split]
test = series.iloc[split:]
train = np.array(train)
test = np.array(test)

plot_acf(train , lags = 34)
plt.show()
plot_pacf(train , lags = 34)
plt.show()
model_fit = ARIMA(train, order=(1,1,1)).fit()
resid = model_fit.resid
plot_acf(resid , lags = 34)
plt.show()
plot_pacf(resid , lags = 34)
plt.show()
print('##############')
pred = model_fit.forecast(len(test))
print(pred[0])
print(len(pred) , len(test))
plt.plot(pred[0][:100] , label = 'predction')
plt.plot(test[:100] , label = 'real')
plt.legend()
plt.show()
