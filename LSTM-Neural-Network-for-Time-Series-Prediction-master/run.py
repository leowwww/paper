import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import numpy as np

def plot_results(predicted_data, true_data):
    residual = []
    for i in range(len(true_data)):
        d = predicted_data[i] - true_data[i]
        residual.append(d)
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    #ax.plot(true_data, label='True Data')
    plt.scatter(range(len(residual)),residual, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + [data], label='Prediction')
        plt.legend()
    plt.show()

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

def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
	
    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    print(data.len_train , data.len_test , data.len_train + data.len_test)
    print (data.len_train)
    print(steps_per_epoch)
    print("#############################################")
    
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )
    #model.load_model('saved_models\\04112021-102516-e2.h5')

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    #predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    #predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    
    predictions = model.predict_point_by_point(x_test)
    print(len(predictions) , len(x_test))
    #plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
   ####????????????
    '''real_pre = []
    real_y = []
    for i in range(len(y_test)):
        real_pre.append((predictions[i]+1)*data_test[i][0])
        real_y.append((y_test[i][0]+1)*data_test[i][0])
    for i in range(10):
       print( real_y[i],data_test[49+i][0])'''
    plot_results(predictions[:100], y_test[:100])
    print(len(predictions))
    plt.plot(predictions)
    plt.plot(y_test)
    plt.show()
    print('RMSE:',RMSE(y_test , predictions))
    print('MAPE:',MAPE(y_test , predictions))  
    print('FA:{}%'.format(100 - 100*MAPE(y_test , predictions)))


if __name__ == '__main__':
    main()