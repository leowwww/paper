from pandas import read_excel
from pandas import datetime
from pandas import DataFrame
import pandas
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
import time
import numpy as np

def parser(x):
    return datetime.strptime(str(x), '%Y-%m')
series =read_excel('Data_AgTD20210604.xlsx',usecols=[1])
print(len(series))
split = int(len(series)*0.7)
train = series.iloc[:split]
test = series.iloc[split:]
train = np.array(train)
test = np.array(test)
model_fit = ARIMA(train, order=(1,1,0)).fit()
print('##############')
pred = model_fit.predict(start = 0 , end = len(test))
print(len(pred) , len(train))
pyplot.plot(pred , label = 'predction')
pyplot.plot(test , label = 'real')
pyplot.legend()
pyplot.show()
