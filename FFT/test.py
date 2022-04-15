
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import pandas as pd
import numpy.fft as nf

x = [i for i in range(92815)]

#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）

data_y = pd.read_excel('DATA_AU_TDX.xlsx',usecols=[1])
data_y = np.array(data_y)
data_y = data_y.flatten()[:64970] ########只对训练集进行滤波
print(data_y.shape)
yy=fft(data_y) #快速傅里叶变换
yreal = yy.real    # 获取实数部分
yimag = yy.imag    # 获取虚数部分

yf=abs(yy)    # 取绝对值
test_y = yy
for i in range(len(yy)):
    if i >len(yy) - 1:
        print(i)
        test_y[i] = 0
test = ifft(test_y).real
test = pd.DataFrame(test)
test.to_excel('./open_trian(no).xlsx')
exit(0)

############ifft 
residul = data_y - test



plt.subplot(221)
plt.plot(np.arange(100),data_y[0:100]) 
plt.title('Original wave')

plt.subplot(222)
plt.plot(np.arange(100),test[0:100])
plt.title('ifft wave')

plt.subplot(223)
plt.plot(np.arange(len(residul)),residul[:]) 
plt.title('ifft wave')
plt.show()