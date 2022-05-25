import pandas as pd
import numpy as np
import scipy.io as scio
import os
'''data_y = pd.read_excel('DATA_AU_TDX.xlsx',usecols=[1]).iloc[-37120: , 0]
data_y = pd.DataFrame(data_y)
data_y.to_excel('./test_original.xlsx')
print(data_y)'''

###lstm数据sum(vmd && emd)
path ='E:\\项目\\论文实验\\stock_predict_with_LSTM-master\\data\\data_emd'
filelist = os.listdir(path)
sum = np.zeros(18010,dtype= float)
for i in range(len(filelist)):
    if 'result' in filelist[i]:
        filename = os.path.join(path , filelist[i])
        data = pd.read_excel(filename).iloc[-18010:,1]
        data = np.array(data)
        sum += data
sum = pd.DataFrame(sum)
sum.to_excel('./emd/emd_lstm_result_sum.xlsx')

