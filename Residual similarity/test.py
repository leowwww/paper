from matplotlib.pyplot import close
import numpy as np
import pandas as pd

def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    '''if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)'''

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

def normal(data):
    open_mean = np.mean(data, axis=0)              # 数据的均值和方差
    open_std = np.std(data, axis=0)
    norm_open = (data - open_mean)/open_std
    return norm_open
if __name__ == '__main__':
    data = pd.read_excel('DATA_AU_TDX.xlsx',usecols=[1,2,3,4])

    open = np.array(data.iloc[:,0])
    hight = np.array(data.iloc[:,1])
    low = np.array(data.iloc[:,2])
    close = np.array(data.iloc[:,3])

    norm_open = normal(open)
    norm_hight = normal(hight)
    norm_low = normal(low)
    norm_close = normal(close)
    print(cosine_similarity(norm_open, norm_hight))