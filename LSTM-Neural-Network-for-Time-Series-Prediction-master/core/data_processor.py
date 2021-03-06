import math
import numpy as np
import pandas as pd
import time

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        #dataframe = pd.read_csv(filename)
        dataframe = pd.read_excel(filename , usecols=[1,2,3]) #新加 , usecols=[0,1,2,3,4,5]
        #dataframe.columns = ['date','open','high','low','close',"Volume"]#新加
        dataframe.columns = ["lstm","holt_winter","real_data"] #"lstm","holt_winter","real_data"
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        self.a = 0
        print(dataframe.shape , self.len_train , self.len_test)

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        #没改
        data_windows = []
        for i in range(self.len_test - seq_len):
            
            data_windows.append(self.data_test[i:i+seq_len])
        data_windows = np.array(data_windows).astype(float)#转换数组元素类型
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
        x = data_windows[:, :-1]
        y = data_windows[:, -1, [-1]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        #print(type(window))
        #print(window)
        #print('################')
        
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window#window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [1]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        ##################改了
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        a = []
        for window in window_data:
            normalised_window = []
            ##################改了
            '''window_aver = max(np.array(window))
            window_min = min(np.array(window))'''
            for col_i in range(window.shape[1]):#window.shape[1]
                ################改了
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                #print(window[0,0] , window[0,1])
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)