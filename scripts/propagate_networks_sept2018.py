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

ROOT = os.environ['HOME'] + '/python/'

Sequences = SC.Sequences
idx = pd.IndexSlice

datasets_ = ['Naive_DSR',
             'Naive_DSR_mPFC',
             'Naive_DSR_OFC',
             'Mid_Training_DSR',
             'Saline_DSR',
             'mPFC_DSR',
             'OFC_DSR',
             'IPSI_DSR',
             'CONTRA_DSR',
             'linear_rigged',
             'XOR_rigged',
             'shuffled',
             'Naive_PSR',
             'Mid_Training_PSR',
             'Saline_PSR',
             'mPFC_PSR',
             'OFC_PSR',
             'IPSI_PSR',
             'CONTRA_PSR']


def find_best_epoch_in_model_dir(current_dir):
    '''
    returns sorted list of models
    '''
    contents = os.listdir(current_dir)
    #will throw warning
    try:
        contents.pop(contents.index('loss_acc_history.p'))
    except ValueError:
        print("Training did not finish: %s" %current_dir)
        return
    scores = {w: float(w[w.find('-') + 1:-5]) for w in contents}
    scores = sorted(scores.iteritems(), key=lambda (k,v):(v,k), reverse=True)
    return scores

def propagate_network(dataset_label, model_dir, SEQUENCE_DIR):
    model = load_model(model_dir)
    scores = pd.Series(np.zeros(len(datasets_)),
                       index=datasets_, name=dataset_label)
    for dataset in datasets_:
        seqs = pickle.load(open(SEQUENCE_DIR + dataset + '.p', 'rb'))
        if model_dir.find(dataset) >= 0:
            X = seqs.X_test
            y = seqs.y_test
        else:
            X = np.concatenate([seqs.X_train, seqs.X_validate, seqs.X_test],
                                                                        axis=0)
            y = np.concatenate([seqs.y_train, seqs.y_validate, seqs.y_test],
                                                                        axis=0)
        test_scores = model.evaluate(X, y, verbose = 0)
        scores[dataset] = test_scores[1]
    pickle.dump(scores, open(ROOT + 'tmp/' + dataset_label + '.p', 'wb'))
    return


def glue_datasets_back():
    model_grid = []
    for dataset in datasets_:
        slice = pickle.load(open(ROOT + 'tmp/' + dataset + '.p', 'rb'))
        model_grid.append(slice)
    return pd.concat(model_grid, axis = 1).transpose()


def propagate_diagonal():
    prop_scores = pd.DataFrame(np.zeros((len(datasets_), 3)),
                               index = datasets_,
                               columns = ['train', 'validate', 'test'])
    #propagate network thru test sets (and train and validate sets)
    for dataset in os.listdir(SEQUENCES_DIR):
        print 'dataset: %s' %dataset
        print 'loading sequences...',
        seqs = pickle.load(open(SEQUENCES_DIR + dataset, 'rb'))
        print 'done'
        print 'loading model...',
        model = load_model(model_winners[dataset[:-2]])
        print 'done'

        train_scores = model.evaluate(seqs.X_train, seqs.y_train, verbose = 0)
        val_scores = model.evaluate(seqs.X_validate, seqs.y_validate, verbose = 0)
        test_scores = model.evaluate(seqs.X_test, seqs.y_test, verbose = 0)
        prop_scores.loc[dataset[:-2], 'train'] = train_scores[1]
        prop_scores.loc[dataset[:-2], 'validate'] = val_scores[1]
        prop_scores.loc[dataset[:-2], 'test'] = test_scores[1]
    return prop_scores

def find_model_winners(MODELS_DIR):
    #find the best models
    model_winners = {}
    for dataset_dir in os.listdir(MODELS_DIR):
        network_scores = {}
        for network_dir in os.listdir(MODELS_DIR + dataset_dir):
            current_dir = MODELS_DIR + dataset_dir + '/' + network_dir
            scores = find_best_epoch_in_model_dir(current_dir)
            network_scores[network_dir] = scores[0]
        network_scores = \
            sorted(network_scores.iteritems(),
                    key=lambda (k,v): v[1], reverse=True)
        model_winner_path = MODELS_DIR + dataset_dir \
                                              + '/' + network_scores[0][0] \
                                              + '/' + network_scores[0][1][0]
        assert os.path.isfile(model_winner_path)
        model_winners[dataset_dir] = model_winner_path
    return model_winners



if __name__ == '__main__':

    SEQUENCE_BASE_DIR = ROOT + 'DATA_structures/RNN_sequences/' \
                             + 'sequence_classification_mirror/'
    MODELS_BASE_DIR = ROOT + 'Models/RNN/SCM/'

    assert os.path.isdir(SEQUENCE_BASE_DIR)
    assert os.path.isdir(MODELS_BASE_DIR)

    for SHUFFLE in range(1, 101):
        MODELS_DIR = MODELS_BASE_DIR + 'SHUFFLE' + str(SHUFFLE) + '/'
        SEQUENCE_DIR = SEQUENCE_BASE_DIR + 'SHUFFLE' + str(SHUFFLE) + '/'

        if os.path.isdir(MODELS_DIR) and os.path.isdir(SEQUENCE_DIR):
            print 'finding networks that performed the ' \
                    + 'best for SHUFFLE %i...' %SHUFFLE,
            model_winners = find_model_winners(MODELS_DIR)
            print 'done'

            pool = mp.Pool(processes = 32)
            for dataset_label, model_file in model_winners.items():
                pool.apply_async(propagate_network,
                                 [dataset_label, model_file, SEQUENCE_DIR])
            pool.close()
            pool.join()
            MODEL_GRID = glue_datasets_back()
            MG_target = ROOT + 'Results/MODEL_GRIDS/'
            if not os.path.isdir(MG_target): os.mkdir(MG_target)
            MG_target += 'MODEL_GRID' + str(SHUFFLE) + '.p'
            pickle.dump(MODEL_GRID, open(MG_target, 'wb'))
            for tmp_file in os.listdir(ROOT + 'tmp/'):
                os.remove(ROOT + 'tmp/' + tmp_file)
            print 'completed model grid for SHUFFLE %i' %SHUFFLE
