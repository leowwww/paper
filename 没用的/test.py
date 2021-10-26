import pandas as pd
import mpl_finance as mpf
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import numpy as np
import os
from scipy import stats

df = pd.read_excel("Data_AgTD20210604.xlsx",usecols = [0,1,2,3,4]) #十五分钟数据
df.columns = ['date','open','high','low','close']
dn = pd.read_excel('六月new_day.xlsx' , usecols = [1,2,3,4,5])#只用开始和结束的列坐标
dn.columns = ['date','start_point' , 'end_1_point','jiudian_point','end_2_point']

def draw(data , count):
    plt.rcParams['font.family'] = 'SimHei' ## 设置字体
    fig = plt.figure()
    fig.set_size_inches(15,15)
    fig.subplots_adjust(bottom=0.2) ## 调整底部距离
    ax = plt.subplot()
    plt.xticks(rotation=45 , size = 20) ## 设置X轴刻度线并旋转45度      
    plt.yticks(size = 20) ## 设置Y轴刻度线
    '''plt.xlabel("时间" , fontdict={'weight':'normal','size': 20}) ##设置X轴标题
    plt.ylabel("股价（元）" , fontdict={'weight':'normal','size': 20}) ##设置Y轴标题'''
    plt.axis('off')  #去掉坐标轴
    mpf.candlestick_ohlc(ax,data,width=0.7,colorup='r',colordown='green', alpha=1)##设置利用mpf画股票K线图
    file_name ='E://项目//实验//图//'+str(count) + ".png"
    plt.savefig( file_name, dpi = 100) ## 保存图片"做多/1/k线/1.png"
    plt.close() ## 关闭plt，释放内存

if __name__ == '__main__':
    count = 0
    for i in range(len(dn)):
        day = dn.loc[i]['date'].day
        month = dn.loc[i]['date'].month
        year = dn.loc[i]['date'].year
        next_day = dn.loc[i+1]['date'].day
        start = dn.loc[i]['start_point']
        end = dn.loc[i]['end_2_point']
        data = []
        print('匹配：',dn.loc[i]['date'])
        for j  in range(start , end-2):
            data = []
            if df.loc[j]['high'] < df.loc[j+1]['high'] and df.loc[j + 1]['high'] < df.loc[j+2]['high'] : #连续三个点往上
                if (j-start) > 6:
                    #画图
                    count += 1
                    jishu = 0
                    for k in range(j-6 , j+3):#画六个点
                        jishu += 1
                        d = (jishu , df.loc[k]['open'],df.loc[k]['high'],df.loc[k]['low'],df.loc[k]['close'])
                        data.append(d)
                    draw(data , count)

