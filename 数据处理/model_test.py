import pandas as pd
import numpy as np
import scipy.io as scio
import os

def RMSE(real , pred):
    sum = 0
    for i in range(len(real)):
        sum += (real[i] - pred[i])**2
    return (sum/len(real))**0.5

path_lstm = './emd/emd_lstm_result_sum.xlsx'
path_hw = './emd/hw-sum.xlsx'
path_test_original = './DATA_AU2010_2022.xlsx'

lstm_data = pd.read_excel(path_lstm).iloc[:,1]
hw_data = pd.read_excel(path_hw).iloc[:,1]
original_data = pd.read_excel(path_test_original).iloc[-(len(lstm_data)):,1]
print(len(lstm_data) , len(hw_data) , len(original_data))
lstm_data = np.array(lstm_data)
hw_data = np.array(hw_data)
original_data = np.array(original_data)

fuse_data = lstm_data + hw_data
print(lstm_data)
print(hw_data)
print(fuse_data)
print(original_data)


print('lstmRMSE:{}'.format(RMSE(lstm_data , original_data)))
print('fuseRMSE:{}'.format(RMSE(fuse_data , original_data)))
