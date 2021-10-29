import matplotlib.pyplot as plt
from numpy.testing._private.utils import decorate_methods
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_excel("open_close.xlsx",usecols = [1])
df.columns = ['open_close']
split = int(len(df)*0.8)
train, test = df.iloc[:split,0], df.iloc[split:,0]
train = np.array(train)
test = np.array(test)

##############分解
'''decompose_result = seasonal_decompose(train[:400], model="add", period=70)
print(decompose_result)
plt.rcParams.update({'figure.figsize': (10, 10)})
decompose_result.plot().suptitle('Multiplicative Decompose')
plt.show()'''
#####一次指数平滑
once = SimpleExpSmoothing(train[:100]).fit(smoothing_level=0.6).fittedvalues
print(len(once))
plt.plot(once,label = 'once')
plt.plot(train[:100] ,label = 'real')
plt.legend()
plt.show()
data = pd.Series(train[:100])
doubel = ExponentialSmoothing(data, trend='add' ).fit().fittedvalues
#doubel_2= ExponentialSmoothing(train[:100], trend="mul", seasonal=None).fit().fittedvalues
plt.plot(doubel,label = '2exp')
plt.plot(train[:100],label = 'real')
plt.legend()
plt.show()

third= ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=20).fit().fittedvalues
plt.plot(third,label = '3exp')
plt.plot(train[:100],label = 'real')
plt.legend()
plt.show()


# Simple Exponential Smoothing
data = train[:50]
fit1 = SimpleExpSmoothing(data).fit(smoothing_level=0.2,optimized=False)
# plot
l1, = plt.plot(list(fit1.fittedvalues) + list(fit1.forecast(5)), marker='o')

fit2 = SimpleExpSmoothing(data).fit(smoothing_level=0.6,optimized=False)
# plot
l2, = plt.plot(list(fit2.fittedvalues) + list(fit2.forecast(5)), marker='o')
fit3 = SimpleExpSmoothing(data).fit()   
# plot
l3, = plt.plot(list(fit3.fittedvalues) + list(fit3.forecast(5)), marker='o')
l4, = plt.plot(data, marker='o')
plt.legend(handles = [l1, l2, l3, l4], labels = ['a=0.2', 'a=0.6', 'auto', 'data'], loc = 'best', prop={'size': 7})
plt.show()
###########二次指数平滑
data_sr = pd.Series(data)
# Holt’s Method
fit1= ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=70).fit().fittedvalues

#fit2 = ExponentialSmoothing(data, trend="mul", seasonal=None).fit().fittedvalues
plt.plot(fit1,label = 'add')
plt.show()
fit_mo= ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=70).fit()
pred = fit_mo.forecast(100)
plt.plot(pred , label = 'pred')
plt.plot(test[:100] , label = 'real')
plt.legend()
plt.show()