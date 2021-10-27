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

'''def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()'''


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

    """
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)"""
	
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
    #model.load_model('saved_models\\27102021-101842-e2.h5')

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    #predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    #predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    
    predictions = model.predict_point_by_point(x_test)

    #plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    data_test = data.data_test
   ####反归一化
    real_pre = []
    real_y = []
    for i in range(len(y_test)):
        real_pre.append((predictions[i]+1)*data_test[i][0])
        real_y.append((y_test[i][0]+1)*data_test[i][0])
    for i in range(10):
       print( real_y[i],data_test[49+i][0])
    plot_results(predictions[:100], y_test[:100])
    plt.plot(real_pre)
    plt.plot(real_y)
    plt.show()
    print(len(predictions),len(y_test))
    print('MSE:',0.5*(np.sum (( np.array(y_test) - np.array(predictions))**2)))
    print('MAD:',(np.sum (abs( np.array(y_test) - np.array(predictions) ) ))/len(real_y))


if __name__ == '__main__':
    '''configs = json.load(open('config.json', 'r'))
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    x , y = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise= configs['data']['normalise']
    )
    print(x[:1])
    print('##################')
    print(y[:10])'''
    main()