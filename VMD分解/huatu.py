import os
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np  
df = pd.read_excel('./vmd-imfs.xlsx',usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
pb = pd.read_excel('./vmd_res.xlsx' , usecols=[1])
au = pd.read_excel('./DATA_AU_TDX.xlsx',usecols=[1])
plt.figure(figsize=(2,2) , dpi = 200)
plt.xticks([])  # 去掉横坐标值
plt.yticks([])  # 去掉纵坐标值
plt.plot(np.array(au).T[0,:], linewidth=0.2, c='b')
plt.show()
# data_res =np.array(pb).T
# data= np.array(df).T
# print(data.shape)
# K = 17
# plt.figure(figsize=(7,7), dpi=200)
# for i in range(K):
#     plt.subplot(K+1,1,i+1)
#     plt.xticks([])  # 去掉横坐标值 
#     plt.yticks([])  # 去掉纵坐标值
#     plt.plot(data[i,:], linewidth=0.2, c='b')


# plt.subplot(18 , 1 , 18)
# plt.xticks([])  # 去掉横坐标值
# plt.yticks([])  # 去掉纵坐标值
# plt.plot(data_res[0,:], linewidth=0.2, c='b')#####添加res进去
# plt.show()


