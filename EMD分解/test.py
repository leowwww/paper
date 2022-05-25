# 导入工具库
import numpy as np
from PyEMD import EMD, Visualisation
import pandas as pd

#读取数据
data_y = pd.read_excel('DATA_AU2010_2022.xlsx',usecols=[1])
data_y = np.array(data_y)
data_y = data_y.flatten()
data_y = np.array(data_y)
print(data_y.shape)
# 提取imfs和剩余信号res
emd = EMD()
emd.emd(data_y)
imfs, res = emd.get_imfs_and_residue()
print(type(imfs))

new_imfs = imfs.T
new_imfs = pd.DataFrame(new_imfs)
new_res = res.T
new_res = pd.DataFrame(new_res)
# print(new_imfs)
# print(imfs)
new_imfs.to_excel('./imfs.xlsx')
new_res.to_excel('./res.xlsx')

'''new_res = pd.DataFrame(res)
new_res.to_excel('./res.xlsx')'''

# 绘制 IMF
vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=np.arange(len(data_y)), include_residue=True)
vis.show()
