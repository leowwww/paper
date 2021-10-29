from pandas import read_excel
from pandas import datetime
from pandas import DataFrame
import pandas
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import time
import numpy as np

def parser(x):
    return datetime.strptime(str(x), '%Y-%m')
series =read_excel('open_close.xlsx')
 
# fit model
print(series)
split = int(len(series)*0.8)
train = series.iloc[:split]
test = series.iloc[split:]
print(len(series) , len(train) , len(test))
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())