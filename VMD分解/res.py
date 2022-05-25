from VMD import VMD
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.fftpack import fft

original_data = pd.read_excel('DATA_AU2010_2022.xlsx',usecols=[1]).iloc[:,0]
original_data = np.array(original_data)
vmd_data = pd.read_excel('./vmd-imfs.xlsx' , usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]).iloc[:,:]
vmd_data = np.array(vmd_data)
vmd_sum = np.sum(vmd_data , axis=1)
res_data = original_data - vmd_sum
res_data = pd.DataFrame(res_data)
res_data.to_excel('./vmd_res.xlsx')
'''print(original_data)
print(original_data.shape)
print(vmd_sum)
print(vmd_sum.shape)
print(res_data)'''

    