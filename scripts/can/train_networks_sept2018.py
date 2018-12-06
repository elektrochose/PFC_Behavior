import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import itertools
import multiprocessing as mp

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, SimpleRNN, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from behavioral_performance.tools import SequenceClass as SC
from behavioral_performance.compile_datasets import create_sequences as CS
ROOT = os.environ['HOME'] + '/python/'

Sequences = SC.Sequences
idx = pd.IndexSlice


def create_model(hd = [10, 0, 0],
                 sequence_length = 30,
                 feature_dim = 4,
                 output_dim = 2,
                 RANDOM_STATE = 10,
                 cell_type = 'RNN',
                 dropout = 0.2):

    noLayers = int(np.sum(hd != 0))
    #sequence_length, feature_dim = seqs.X_train[0].shape

    if cell_type == 'LSTM':
        RNNobj = LSTM
    elif cell_type == 'RNN':
        RNNobj = SimpleRNN

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
    model.add(Dense(output_dim, activation='sigmoid'))
    if output_dim == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(loss = loss,
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    model.RANDOM_STATE = RANDOM_STATE
    return model


def train_network(seq_dir, target_dir):
    #print 'loading sequences...',
    seqObj = pickle.load(open(seq_dir, 'rb'))
    #print 'done'
    for cell_type, hd in itertools.product(['RNN'], HDS):
        network_dir = '_'.join([str(w) for w in [cell_type] + list(hd)])
        model_dir = target_dir + network_dir
        if not os.path.isdir(model_dir):
            #print 'creating %s...' %model_dir,
            os.mkdir(model_dir)
            os.chdir(model_dir)
            #print 'done'
        else:
            print 'some training occured already...'
            print seq_dir
            print model_dir
            continue
        model = create_model(hd = hd,
                             sequence_length = seqObj.header['sequence_length'],
                             feature_dim = 4,
                             output_dim = 2,
                             RANDOM_STATE = seqObj.header['RANDOM_STATE'],
                             cell_type = cell_type,
                             dropout = 0.2)
        callbacks = \
            [ModelCheckpoint(filepath = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5',
                             monitor = 'val_acc',
                             save_best_only = True)]

        #print 'started training: %s' %model_dir
        History = model.fit(x = seqObj.X_train,
                            y = seqObj.y_train,
                            callbacks = callbacks,
                            validation_data = (seqObj.X_validate, seqObj.y_validate),
                            epochs = 100,
                            batch_size = 64,
                            verbose = 0)

        print 'finished training model: %s' %model_dir
        pickle.dump(History.history, open('loss_acc_history.p','wb'))

        #let's clean up - delete all models except the best one
        contents = [w for w in os.listdir('.') if w.startswith('w')]
        scores = {w: float(w[w.find('-') + 1:-5]) for w in contents}
        scores = sorted(scores.iteritems(), key=lambda (k,v):(v,k), reverse=True)
        for i, (key,val) in enumerate(scores):
            if i > 0: os.remove(key)


#CREATING NETWORK DIMENSIONS__________
hidden_dimensions = [5, 20, 50, 100]
hidden_dimensions_red = [10, 50]
no_models = len(hidden_dimensions) + len(hidden_dimensions) ** 2 \
          + len(hidden_dimensions_red) ** 3
HDS = np.zeros([no_models, 3], dtype = int)
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


if __name__ == '__main__':

    NO_SHUFFLES = 50


    SHUFFLE = range(51, 1 + NO_SHUFFLES)
    for CSHUFFLE in SHUFFLE:

        '''
        First we create the sequences, but with different random shuffling for
        separating train, validate, and test sets
        '''
        source = ROOT + 'DATA_structures/dataset_dataframes/'
        target = ROOT + \
                'DATA_structures/RNN_sequences/' + \
                'sequence_classification_mirror/SHUFFLE' + \
                str(CSHUFFLE) + '/'
        if not os.path.isdir(target): os.mkdir(target)
        for current_dataset in os.listdir(source):
            CS.create_sequences(source + current_dataset,
                                target + current_dataset,
                                RANDOM_STATE = CSHUFFLE)


        '''
        Once sequences are created, we start training.
        '''
        source = target
        target = ROOT + 'Models/RNN/SCM/SHUFFLE' + str(CSHUFFLE) + '/'
        if not os.path.isdir(target): os.mkdir(target)

        print 'initializing pool...'
        pool = mp.Pool(processes = 32)
        for seqs in os.listdir(source):
            seq_dir = source + seqs
            target_dir = target + seqs[:-2] + '/'
            if not os.path.isdir(target_dir): os.mkdir(target_dir)
            pool.apply_async(train_network, [seq_dir, target_dir])
        pool.close()
        pool.join()
