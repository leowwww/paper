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
series =read_excel('open_close.xlsx',usecols=[0])
split = int(len(series)*0.8)
train = series.iloc[:split]
test = series.iloc[split:]
train = np.array(train)
test = np.array(test)
model_fit = ARIMA(train, order=(1,1,0)).fit()
print('##############')
pred = model_fit.predict()
print(len(pred) , len(train))
pyplot.plot(pred[:1000] , label = 'predction')
pyplot.plot(train[:1000] , label = 'real')
pyplot.legend()
pyplot.show()
a = model_fit.forecast(100)
pyplot.plot(a , label = 'predction')
pyplot.plot(test[:100] , label = 'real')
pyplot.legend()
pyplot.show()
print(a)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
print('residuals:',len(residuals))
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())