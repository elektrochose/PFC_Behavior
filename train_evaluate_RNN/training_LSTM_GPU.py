#THIS IS EPSILON ROOT DIRECTORY !
ROOT = '/home/pablo/python/'
#standard modules
import sys
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
import numpy as np
import os
import pickle
import time
import itertools
import operator
import multiprocessing as mp


from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from behavioral_performance.utils import fileNames
from RNNmodule.SequenceClass import Sequences
idx = pd.IndexSlice

def create_model(sequence_length,
                 feature_dim,
                 RANDOM_STATE,
                 hd,
                 cell_type = 'RNN',
                 dropout = 0.2):

    noLayers = int(np.sum(hd != 0))
    #sequence_length, feature_dim = seqs.X_train[0].shape

    if cell_type == 'LSTM':
        RNNobj = LSTM
    elif cell_type == 'RNN':
        RNNobj = SimpleRNN

    callbacks = [ModelCheckpoint(
                    filepath = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5',
                    monitor = 'val_acc',
                    save_best_only = True)]


    #create model
    model = Sequential()
    if noLayers == 1:
        model.add(RNNobj(input_shape = (sequence_length, feature_dim),
                         units = hd[0]))
        model.add(Dropout(dropout))
    elif noLayers == 2:
        model.add(RNNobj(return_sequences = True,
                       input_shape = (sequence_length, feature_dim),
                       units = hd[0]))
        model.add(Dropout(dropout))
        model.add(RNNobj(hd[1]))
        model.add(Dropout(dropout))
    elif noLayers == 3:
        model.add(RNNobj(return_sequences = True,
                       input_shape = (sequence_length, feature_dim),
                       units = hd[0]))
        model.add(Dropout(dropout))
        model.add(RNNobj(return_sequences = True, units = hd[1]))
        model.add(Dropout(dropout))
        model.add(RNNobj(hd[2]))
        model.add(Dropout(dropout))
    model.add(Dense(feature_dim, activation='sigmoid'))
    if feature_dim == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(loss = loss,
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    model.RANDOM_STATE = RANDOM_STATE
    return model


def train_network(fileName, dataPrep, HDS):


    sequence_path = ROOT + \
                'DATA_structures/RNN_sequences/OneHotBinaryMinimal/' + \
                dataPrep + '/' + fileName

    #making sure we find right sequences
    if not os.path.isfile(sequence_path):
        print 'COULD NOT FIND SEQUENCES FILE: %s' %sequence_path
        return
    else:
        print 'loading sequences ...'
        seqs = pickle.load(open(sequence_path, 'rb'))
        print 'loaded successfully...'

    cellType_folders = {'RNN' : 'Models/RNN/OneHotBinaryMinimal/',
                        'LSTM' : 'Models/LSTM/Pablo/OneHotBinaryMinimal/'}

    for cell_type, hd in itertools.product(['RNN'], HDS):


        model_dir = ROOT + cellType_folders[cell_type] + dataPrep + '/'
        assert os.path.isdir(model_dir)


        network_name = '_D_'.join([fileName[:-2]] + [str(w) for w in hd if w > 0])
        model_dir += network_name


        #creating folder for model - need to save best iterations
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
            os.chdir(model_dir)
        else:
            os.chdir(model_dir)


        sequence_length, feature_dim = seqs.X_train[0].shape
        model = create_model(sequence_length, feature_dim, seqs.RANDOM_STATE,
                             hd, cell_type = 'RNN', dropout = 0.2)
        no_epochs = 100


        else:
            if not os.listdir(model_dir):
                os.chdir(model_dir)
            else:
                print 'MODEL ALREADY TRAINED - SKIPPING'
                continue


        dropout = 0.5
        no_epochs = 100
        layers = int(np.sum(hd != 0))
        sequence_length, feature_dim = seqs.X_train[0].shape

        if cell_type == 'LSTM':
            RNNobj = LSTM
        elif cell_type == 'RNN':
            RNNobj = SimpleRNN

        callbacks = [ModelCheckpoint(
                        filepath = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5',
                        monitor = 'val_acc',
                        save_best_only = True)]


        #create model
        model = Sequential()
        if layers == 1:
            model.add(RNNobj(input_shape = (sequence_length, feature_dim),
                             units = hd[0]))
            model.add(Dropout(dropout))
        elif layers == 2:
            model.add(RNNobj(return_sequences = True,
                           input_shape = (sequence_length, feature_dim),
                           units = hd[0]))
            model.add(Dropout(dropout))
            model.add(RNNobj(hd[1]))
            model.add(Dropout(dropout))
        elif layers == 3:
            model.add(RNNobj(return_sequences = True,
                           input_shape = (sequence_length, feature_dim),
                           units = hd[0]))
            model.add(Dropout(dropout))
            model.add(RNNobj(return_sequences = True, units = hd[1]))
            model.add(Dropout(dropout))
            model.add(RNNobj(hd[2]))
            model.add(Dropout(dropout))
        model.add(Dense(4, activation='sigmoid'))
        model.compile(loss = 'binary_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])

        print 'started training: %s' %model_dir
        History = model.fit(x = seqs.X_train,
                            y = seqs.y_train,
                            callbacks = callbacks,
                            validation_data = (seqs.X_validate, seqs.y_validate),
                            epochs = no_epochs,
                            batch_size = 256,
                            verbose = 0)

        print 'finished training model: %s' %model_dir
        pickle.dump(History.history, open('loss_acc_history.p','wb'))
                            batch_size = 512,
                            verbose = 0)

        print 'finished training model: %s' %model_dir
        pickle.dump(History, open('loss_acc_history.p','wb'))
    return



datatype = ['Full','Last', 'Med']
#adding artificial datasets
artificial_datasets = ['PSR_TbyT_Saline_Rigged.p',
                       'DSR_TbyT_Saline_Shuffled.p',
                       'PSR_TbyT_Saline_Shuffled.p']
for a_data in artificial_datasets:
    try:
        fileNames.index(a_data)
    except ValueError:
        fileNames.append(a_data)

print fileNames

#CREATING NETWORK DIMENSIONS__________
hidden_dimensions = [5, 20, 50, 100]
hidden_dimensions_red = [5, 50]
no_models = len(hidden_dimensions) + len(hidden_dimensions) ** 2 \
          + len(hidden_dimensions_red) ** 3
HDS = np.zeros([no_models, 3], dtype = int)
=======
datatype = ['Full', 'Last', 'Med']
#adding artificial datasets
fileNames.append('PSR_TbyT_Saline_Rigged.p')
fileNames.append('DSR_TbyT_Saline_Shuffled.p')
fileNames.append('PSR_TbyT_Saline_Shuffled.p')
fileNames.append('DSR_TbyT_Naive_Saline.p')


#CREATING NETWORK DIMENSIONS__________
hidden_dimensions = [5, 10, 20, 100]
hidden_dimensions_red = [5, 50]

HDS = np.zeros([69, 3], dtype = int)

HDS[:len(hidden_dimensions), 0] = hidden_dimensions
for index, (hd1, hd2) in enumerate(itertools.product(hidden_dimensions,
                                                     hidden_dimensions)):
    HDS[len(hidden_dimensions) + index, 0] = hd1
    HDS[len(hidden_dimensions) + index, 1] = hd2

counter = np.argmin(np.sum(HDS, axis = 1) != 0)
for index, (hd1, hd2, hd3) in enumerate(itertools.product(hidden_dimensions_red,
                                                          hidden_dimensions_red,
                                                          hidden_dimensions_red)):
    HDS[counter + index, 0] = hd1
    HDS[counter + index, 1] = hd2
    HDS[counter + index, 2] = hd3
#CREATING NETWORK DIMENSIONS__________

<<<<<<< HEAD
pool = mp.Pool(processes = 32)
=======

>>>>>>> 932cf9c99e452f2f015a7b03962c7893835b6af1
for dataPrep, fileName in itertools.product(datatype, fileNames):
    print '*' * 80
    print '%s - %s' %(dataPrep, fileName)
    print '*' * 80
<<<<<<< HEAD
    pool.apply_async(train_network, [fileName, dataPrep, HDS])
=======
    train_network(fileName, dataPrep, HDS)
>>>>>>> 932cf9c99e452f2f015a7b03962c7893835b6af1
