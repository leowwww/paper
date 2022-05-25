import pandas as pd
import numpy as np 
import os

path ='E:\\项目\\论文实验\\stock_predict_with_LSTM-master\\data\\data_vmd'
filelist = os.listdir(path)

def RMSE(residual):
    sum = np.sum(residual)
    return (sum/len(residual))**0.5
def MAPE(real , pred):
    sum = 0 
    for i in range(len(real)):
        sum += abs((pred[i] - real[i]) / real[i])
    return sum/len(real)
sum = np.zeros(37119,dtype= float)
for i in range(len(filelist)):
    file_path = os.path.join(path , filelist[i])
    if 'residual' in file_path:
        print(file_path)
        data = np.array(pd.read_excel(file_path,usecols=[1]).iloc[:,0])
        sum += data
        print(data)
        #exit(0)
print(sum)
print('RMSE:',RMSE(sum))