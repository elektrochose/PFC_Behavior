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




def train_network(sequence_path,
                  model_path,
                  hd,
                  RNNtype):

    #creating folder for model - need to save best iterations
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    else:
        if not os.listdir(model_path):
            os.chdir(model_path)
        else:
            print 'MODEL ALREADY TRAINED - SKIPPING'
            return

    #making sure we find right sequences
    if not os.path.isfile(sequence_path):
        print 'COULD NOT FIND SEQUENCES FILE: %s' %sequence_path
        return
    else:
        print 'loading sequences ...'
        seqs = pickle.load(open(sequence_path, 'rb'))
        print 'loaded successfully...'




    dropout = 0.5 #was .2
    no_epochs = 100
    layers = int(np.sum(hd != 0))
    sequence_length, feature_dim = seqs.X_train[0].shape

    if RNNtype == 'LSTM':
        RNNobj = LSTM
    elif RNNtype == 'RNN':
        RNNobj = SimpleRNN

    callbacks = [EarlyStopping(monitor = 'val_acc', patience = 10),
                 ModelCheckpoint(filepath = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5',
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
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    print 'started training: %s' %model_path
    model.fit(x = seqs.X_train,
              y = seqs.y_train,
              callbacks = callbacks,
              validation_data = (seqs.X_validate, seqs.y_validate),
              epochs = no_epochs,
              batch_size = 512,
              verbose = 0)

    print 'finished training model: %s' %model_path
    return


def retrieve_best_val_score(model_path):
    scores = {}
    for modelName in os.listdir(model_path):
        scores[modelName] = np.float(modelName[modelName.find('-') + 1:-5])
    return max(scores.iteritems(), key = operator.itemgetter(1))


cellType = ['RNN', 'LSTM']
cellType_folders = {'RNN' : 'Models/RNN/',
                    'LSTM' : 'Models/LSTM/Pablo/'}

datatype = ['binary', 'binaryMinimal', 'binaryOmni',
            'full_binary', 'full_binaryMinimal', 'full_binaryOmni']


#CREATING NETWORK DIMENSIONS__________
hidden_dimensions = [2, 5, 10, 20, 50, 100]
hidden_dimensions_red = [5, 20, 50]

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

#TRAINING NETWORKS IN PARALLEL_______________________
pool = mp.Pool(processes = 32)
for dataPrep, cell_type, fileName, hd in \
                    itertools.product(datatype, cellType, fileNames, HDS):

    sequence_path = ROOT + \
                'DATA_structures/RNN_sequences/' + dataPrep + '/' + fileName

    model_path = ROOT + cellType_folders[cell_type] + dataPrep + '/' + \
                '_D_'.join([fileName[:-2]] + [str(w) for w in hd if w > 0])


    pool.apply_async(train_network,
                    [sequence_path, model_path, hd, cell_type])
#we wait till every network is finished before next step
pool.close()
pool.join()
#TRAINING NETWORKS IN PARALLEL_______________________

# RETRIEVING VALIDATION SET SCORES________________

def propagate_networks(fileName):
    os.chdir(ROOT + 'Model_Evaluation/RNN/giant_network_grid')
    rows = pd.MultiIndex.from_product([fileNames, datatype],
                                       names = ['dataset', 'seq_type'])

    network_labels = ['_D_'.join([''] + [str(w) for w in hd if w > 0]) \
                      for hd in HDS]
    cols = pd.MultiIndex.from_product([cellType, network_labels],
                                       names = ['cell_type', 'network_dims'])
    val_scores = pd.DataFrame(np.full([len(rows), len(cols)], np.NaN),
                                        index = rows, columns = cols)
    test_scores = pd.DataFrame(np.full([len(rows), len(cols)], np.NaN),
                                         index = rows, columns = cols)
    #go thru each model we just trained
    for dataPrep in datatype:
        if not os.path.isfile(ROOT + \
        'Model_Evaluation/RNN/giant_network_grid/val' + fileName + dataPrep) \
        and not os.path.isfile(ROOT + \
        'Model_Evaluation/RNN/giant_network_grid/test' + fileName + dataPrep):

            print 'processing %s - %s' %(fileName, dataPrep)
            sequence_path = ROOT + \
                    'DATA_structures/RNN_sequences/' + dataPrep + '/' + fileName

            seqs = pickle.load(open(sequence_path, 'rb'))
            for cell_type, hd in itertools.product(cellType, HDS):

                model_label = '_D_'.join([''] + [str(w) for w in hd if w > 0])
                model_path = ROOT + cellType_folders[cell_type] + dataPrep + \
                                        '/' + '_D_'.join([fileName[:-2]] + \
                                                [str(w) for w in hd if w > 0])

                winner = retrieve_best_val_score(model_path)
                val_scores.loc[idx[fileName, dataPrep],
                           idx[cell_type, model_label]] = winner[1]

		model_path = model_path + '/' + winner[0]
                model = load_model(model_path)
		test_score = model.evaluate(seqs.X_test, seqs.y_test, verbose = 0)
                test_scores.loc[idx[fileName, dataPrep],
                                idx[cell_type, model_label]] =  test_score[1]


            pickle.dump(val_scores, open(ROOT + \
                    'Model_Evaluation/RNN/giant_network_grid/val' + \
                                                fileName + dataPrep, 'wb'))
            pickle.dump(test_scores, open(ROOT + \
                    'Model_Evaluation/RNN/giant_network_grid/test' + \
                                                fileName + dataPrep, 'wb'))
        else:
            print 'Networks already propagated'


pool = mp.Pool(processes = 16)
for fileName in fileNames:
    pool.apply_async(propagate_networks, [fileName])
pool.close()
pool.join()

#STITCH giant dataframes back together
val_DF = []
test_DF = []
for files in \
    itertools.product(os.listdir(
            ROOT + 'Model_Evaluation/RNN/giant_network_grid'), datatype):
    if files.startswith('val'):
        val_scores = pickle.load(open(ROOT + \
        'Model_Evaluation/RNN/giant_network_grid/' + files + dataPrep, 'rb'))
        val_DF.append(val_scores)
    elif files.startswith('test'):
        test_scores = pickle.load(open(ROOT + \
        'Model_Evaluation/RNN/giant_network_grid/' + files + dataPrep, 'rb'))
        test_DF.append(test_scores)

val_DF = pd.concat(val_DF, axis = 0)
val_DF.dropna()
val_DF.sort_index(axis = 0, inplace = True)
pickle.dump(val_DF,
        open(ROOT + 'Model_Evaluation/RNN/val_giant_network_grid.p', 'wb'))

test_DF = pd.concat(test_DF, axis = 0)
test_DF.dropna()
test_DF.sort_index(axis = 0, inplace = True)
pickle.dump(test_DF,
        open(ROOT + 'Model_Evaluation/RNN/test_giant_network_grid.p', 'wb'))


# RETRIEVING VALIDATION SET SCORES________________

scores = pickle.load(open(ROOT + \
                            'Model_Evaluation/RNN/giant_network_grid.p', 'rb'))
attr = ['sequences', 'model_path', 'val_score', 'test_score']
BEST_OF = pd.DataFrame(np.zeros([len(fileNames), len(attr)]),
                       index = fileNames,
                       columns = attr)

for ds_label, ds in scores.groupby(axis = 0, level = 'dataset'):
    d = dict()
    for st_label, seq_type in ds.groupby(axis = 0, level = 'seq_type'):
        network = seq_type.idxmax(axis = 1)
        val_score = seq_type.max(axis = 1)
        d[st_label] = network[0] + (val_score[0], )

    sequence_type = max(d, key = lambda x: d[x][2])
    cell_type, network_dims, val_score = d[sequence_type]
    model_path = ROOT + cellType_folders[cell_type] + \
                 sequence_type + '/' + \
                 ds_label[:-2] + network_dims
    model_path += '/' + retrieve_best_val_score(model_path)[0]


    BEST_OF.loc[ds_label, 'sequences'] = sequence_type
    BEST_OF.loc[ds_label, 'model_path'] = model_path
    BEST_OF.loc[ds_label, 'val_score'] = val_score
pickle.dump(BEST_OF,
            open(ROOT + 'Model_Evaluation/RNN/BEST_OF_GIANT_RUN.p', 'wb'))
