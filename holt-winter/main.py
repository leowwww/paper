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
df = pd.read_excel("open_close.xlsx",usecols = [1])
df.columns = ['open_close']
split = int(len(df)*0.8)
train, test = df.iloc[:split,0], df.iloc[split:,0]
train = np.array(train)
test = np.array(test)
################归一化
'''max_data = max(np.array(df))[0]
min_data = min(np.array(df))[0]
print(max_data,min_data)
new_data = []
print(np.array(df).shape)
time.sleep(5)
df = np.array(df)
for i in range(len(df)):
    new_data.append((df[i][0] - min_data)/(max_data - min_data))
print(len(new_data))
train , test = new_data[:split] , new_data[split:]
train = np.array(train)
test = np.array(test)'''
##############分解
'''decompose_result = seasonal_decompose(train[:400], model="add", period=70)
print(decompose_result)
plt.rcParams.update({'figure.figsize': (10, 10)})
decompose_result.plot().suptitle('Multiplicative Decompose')
plt.show()'''
#####一次指数平滑
'''once = SimpleExpSmoothing(train[:100]).fit(smoothing_level=0.6)

once = once.fittedvalues
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
l4, = plt.plot(train[:55], marker='^')
plt.legend(handles = [l1, l2, l3, l4], labels = ['a=0.2', 'a=0.6', 'auto', 'data'], loc = 'best', prop={'size': 7})
plt.show()
###########二次指数平滑
data_sr = pd.Series(data)
# Holt’s Method
fit1= ExponentialSmoothing(train[:50], trend="add").fit()
l1, = plt.plot(list(fit1.fittedvalues) + list(fit1.forecast(5)), marker='o')
l4, = plt.plot(train[:55], marker='^')
#fit2 = ExponentialSmoothing(data, trend="mul", seasonal=None).fit().fittedvalues
plt.legend(handles = [l1, l4], labels = ['2exp','data'], loc = 'best', prop={'size': 7})
plt.show()'''
fit1= ExponentialSmoothing(train, trend="add",seasonal="add",  seasonal_periods=100).fit()#seasonal="add",  seasonal_periods=100
'''plt.plot(fit1.fittedvalues[:100])
plt.plot(train[:100])
plt.show()'''

'''l1, = plt.plot(list(fit1.forecast(100)[20:]), marker='o')
l4, = plt.plot(test[20:100], marker='^')
#fit2 = ExponentialSmoothing(data, trend="mul", seasonal=None).fit().fittedvalues
plt.legend(handles = [l1, l4], labels = ['holt_winter','data'], loc = 'best', prop={'size': 9})
plt.show()
####计算mse'''

pred = fit1.forecast(len(test))
rmse = RMSE(test , pred)
print('RMSE:{}'.format(rmse))
pred = pd.DataFrame(pred)
pred.to_excel('linear.xlsx')
