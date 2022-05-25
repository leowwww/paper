import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa import holtwinters
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox as ljbox 

def adf (data_1 , data_2):
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_1 = data_1[:,0]
    data_2 = data_2[:,0]
    data = []
    for i in range(len(data_1)):
        data.append(data_1[i] - data_2[i])
    a = adfuller(data)
    return a
def white_noise(data):
    return ljbox(data , lags=1,boxpierce=True)
def RMSE(real , pred):
    sum = 0
    for i in range(len(real)):
        sum += (real[i] - pred[i])**2
    return (sum/len(real))**0.5
def MAPE(real , pred):
    sum = 0 
    for i in range(len(real)):
        sum += abs((pred[i] - real[i]) / real[i])
    return sum/len(real)
def FA(real , pred):
    result = MAPE(real , pred)

if __name__ == "__main__":
    ######################vmd处理后的统计检验
    df = pd.read_excel("./data/vmd-imfs.xlsx",usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],header= None)
    for i in range(17):
        data = np.array(df.iloc[:,i])
        print(len(data))
        print('{}adf检验：{}'.format(i , adfuller(data)))
        print('{} 白噪声检验:{}'.format(i , ljbox(data)))
        
        time.sleep(5)
    ###############################
    ######################emd——>lstm之后的数据平稳性检验############## 编辑于2022-4-19
    path ='E:\\项目\\论文实验\\stock_predict_with_LSTM-master\\data\\data_emd\\residual'
    filelist = os.listdir(path)
    for i in range(len(filelist)):
        file_path = os.path.join(path , filelist[i])
        data = np.array(pd.read_excel(file_path,usecols=[1]).iloc[:,0])
        print('{}adf检验：{}'.format(filelist[i] , adfuller(data)))
        #平稳的 完美
    print('---------------------------------------------------------')
    for i in range(len(filelist)):
        file_path = os.path.join(path , filelist[i])
        data = np.array(pd.read_excel(file_path,usecols=[1]).iloc[:,0])
        print('{} 白噪声检验:{}'.format(filelist[i] , ljbox(data)))
        #拒绝是白噪声 完美
    exit(0)

    ##############################
    df = pd.read_excel("DATA_AU_TDX.xlsx",usecols=[0,1,2,3,4],header= None)
    df.columns = ['time','open','high','low','close']
    data_close = df.loc[: ,['close']]
    data_open = df.loc[: ,['open']]
    result = adf(data_close , data_open)

    print('open-closeADF检验：',result)
    print('openADF检验:',adfuller(np.array(data_open)))
    print('closeADF检验:',adfuller(np.array(data_close)))
    close = []
    data_close = np.array(data_close)
    print(data_close[0][0])
    for i in range(len(data_close) - 1):
        close.append(np.array(data_close[i+1][0]))
    print('close:',adfuller(close))
    print(data_open.shape)
    print('open-closeADF检验：',result)
    #print('openADF检验:',adfuller(np.array(data_open)))
    #print('closeADF检验:',adfuller(np.array(data_close)))
    data_close = np.array(data_close)
    data_close = data_close[:,0]
    data_open = np.array(data_open)
    data_open = data_open[:,0]
    data = []
    for i in range(len(data_open)):
        data.append(data_close[i] - data_open[i])
    print('open-close白噪声:',white_noise(data))
    ##############检验lstm的残差序列
    '''df = pd.read_excel('lstm_residual.xlsx',usecols=[1])
    data = np.array(df)
    print('adf:{}'.format(adfuller(data)))
    print('white:{}'.format(ljbox(data , lags=1,boxpierce=False)))'''
    ###############检查holt_winters的效果
    nor = pd.read_excel('AU_data\\normal_result.xlsx' , usecols=[1])
    ro = pd.read_excel('AU_data\\roll_result.xlsx' , usecols=[1])
    ro_14 = pd.read_excel('AU_data\\roll_14_result.xlsx' , usecols=[1])
    ro_4 = pd.read_excel('AU_data\\roll_4_result.xlsx' , usecols=[1])
    real = pd.read_excel('AU_data\\lstm_residual.xlsx',usecols=[1])[-5568:]
    print(len(nor) , len(ro) , len(real),len(ro_4) , len(ro_14))
    nor = np.array(nor)
    ro = np.array(ro)
    real = np.array(real)
    ro_14 = np.array(ro_14)
    ro_4 = np.array(ro_4)
    print("nor_rmse:{}".format(RMSE(real , nor)))
    print("ro_rmse:{}".format(RMSE(real , ro)))
    print("ro_14_rmse:{}".format(RMSE(real , ro_14)))
    print("ro_4_rmse:{}".format(RMSE(real , ro_4)))
    print('####################################')
    ##############合并数据集
    holt_winter = pd.read_excel('AU_data\\roll_result.xlsx' , usecols=[1])#效果没有roll_7好
    holt_winter_4 = pd.read_excel('AU_data\\roll_4_result.xlsx',usecols=[1])
    lstm_data = pd.read_excel('AU_data\\lstm_result.xlsx',usecols=[1])[-5569:-1]#最后一个是明天的值
    real_data = pd.read_excel('AU_data\\lstm_origin_data.xlsx',usecols=[1])[-5568:]
    print(len(holt_winter) , len(lstm_data) , len(real_data),len(holt_winter_4))
    lstm_data = np.array(lstm_data)
    real_data = np.array(real_data)
    holt_winter = np.array(holt_winter)
    holt_winter_4 = np.array(holt_winter_4)
    jiejie_data = np.array(lstm_data) + np.array(holt_winter)
    print(jiejie_data.shape)
    
    ##########生成excel数据
    '''a = pd.DataFrame({'lstm':list(lstm_data[:,0]) , 'holt_winter':list(holt_winter[:,0]) , 'real_data':list(real_data[:,0])})
    a.to_excel('hybrid_data.xlsx')'''

    plt.plot(real_data[:100] , label = 'real_data')
    plt.plot(lstm_data[:100], label = 'lstm')
    plt.plot(jiejie_data[:100], label = 'hybrid_model')
    plt.legend()
    plt.show()
    print('lstm rmse:{}'.format( RMSE(real_data , lstm_data)[0] ))
    print('hybird rmse:{}'.format( RMSE(real_data ,jiejie_data)[0]))
    print('lstm MAPE:{}'.format( MAPE(real_data , lstm_data)[0] ))
    print('hybird MAPE:{}'.format( MAPE(real_data ,jiejie_data)[0] ))
    print('lstm FA:{}%'.format( 100 - 100*MAPE(real_data , lstm_data)[0] ))
    print('hybird FA:{}%'.format( 100 - 100*MAPE(real_data ,jiejie_data)[0] ))
    ##############结论是ro_7效果最好

