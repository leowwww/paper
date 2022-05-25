from VMD import VMD
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.fftpack import fft


data_y = pd.read_excel('DATA_AU2010_2022.xlsx',usecols=[1])
data_y = np.array(data_y)
data_y = data_y.flatten()
data_y = np.array(data_y)
data_y = data_y.astype(np.float32)
data_y = list(data_y)
######VMD分解


alpha = 2000       # moderate bandwidth constraint  
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 15             # 15
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-6  


#. Run VMD 
u, u_hat, omega = VMD(data_y, alpha, tau, K, DC, init, tol) 
print(u.shape , u_hat.shape , omega.shape)


#画图
#. Visualize decomposed modes
plt.figure()
plt.subplot(2,1,1)
plt.plot(data_y)
plt.title('Original signal')
plt.xlabel('time (s)')
plt.subplot(2,1,2)
plt.plot(u.T)
plt.title('Decomposed modes')
plt.xlabel('time (s)')
plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
plt.tight_layout()
plt.show()

plt.show()
result = np.array(u.real).T
result = pd.DataFrame(result)
result.to_excel('./vmd-imfs.xlsx')