import math
import numpy as np
import pandas as pd

'''df = pd.read_excel("E:\项目\论文实验\holt-winter\Data_AgTD20210604.xlsx",usecols = [0,1,2,3,4,5])
df.columns = ['date','open','high','low','close',"Volume"]
#a = df.ix[:100 ,['open','Volume']]
a = df.get(['open','Volume']).values[:100]
print(a)


dataframe = pd.read_csv('E:\项目\LSTM-Neural-Network-for-Time-Series-Prediction-master\data\sp500.csv')
#dataframe = pd.read_excel(filename , usecols=[0,1,2,3,4,5]) #新加
#dataframe.columns = ['date','open','high','low','close',"Volume"]#新加
cols = ['open']
i_split = 100
a = dataframe.get(cols).values[:100]
print(a)'''
a = [1,2,3,4]
b = [2,3,4,5]
c= []
c.append(a)
c.append(b)
c= np.array(c)
print(c.shape[1])
print(c.shape)
