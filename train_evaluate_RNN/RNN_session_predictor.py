'''
This function will take a Keras model and a session dataframe. It will take the
session and prepare it for the model to use, and spit out an accuracy value.
'''

import re
import sys
import pickle
import numpy as np
import pandas as pd
import itertools
import operator
import pysftp
from keras.models import load_model
from behavioral_performance.utils import fileNames, fileNameLabels

ROOT = '/Users/pablomartin/python/'
model_dirs = ROOT + 'Models/Winners/'
session_dirs = ROOT + 'DATA_structures/TbyT/'
seq_dirs = ROOT + 'DATA_structures/RNN_sequences/OneHotBinaryMinimal/'
seq_length_d = {1: 'Last', 30: 'Med', 200: 'Full'}
sets_dict = {'X_train' : 'y_train',
             'X_validate' : 'y_validate',
             'X_test' : 'y_test'}


def RNN_session_pred(session, model):
    noTrials = len(session)
    batch_size, seq_length, feature_dim = model.input_shape
    #checks whether model predicted choice only, not reward
    ch_red = lambda x,y: (x>1) == (y>1)
    #right now will only work with OneHotBinaryMinimal preparation
    combined = session['reward',0] + session['choice',0] * 2
    combined = combined.append(pd.Series([0,1,2,3]))
    combined = pd.get_dummies(combined)
    combined = combined.drop([0,1,2,3])

    sanity_check = 0
    exact_match, choice_match = 0,0
    info = {'choice' : [], 'prediction' : [], 'match' : []}
    x_seqs = []
    y_seqs = []
    for trial in range(1, noTrials - 1):
        if trial < seq_length:
            x = np.concatenate([np.zeros((seq_length - trial,4)),
                                combined.iloc[:trial].values], axis=0)
        else:
            x = combined.iloc[trial - 30: trial].values


        assert x.shape[0] == seq_length
        assert x.shape[1] == feature_dim
        if trial > 1:
            sanity_check += (np.argmax(x[-1]) == y)
        x_seqs.append(x)
        x = np.reshape(x, (1, seq_length, feature_dim))

        y_seqs.append(combined.iloc[trial])
        y = np.argmax(combined.iloc[trial])
        pred_x = np.argmax(model.predict(x))
        exact_match += (y == pred_x)
        choice_match += ch_red(pred_x, y)

        info['choice'].append(y)
        info['prediction'].append(pred_x)
        info['match'].append(ch_red(pred_x, y))
    x_seqs = np.asarray(x_seqs)
    y_seqs = np.asarray(y_seqs)
    values = model.evaluate(x = x_seqs, y = y_seqs, verbose = 0)

    exact_match_score = np.float(exact_match) / (noTrials - 2)
    choice_match_score = np.float(choice_match) / (noTrials - 2)
    sanity_check_score  = np.float(sanity_check) / (noTrials -3)
    evaluate_score = values[1]
    if sanity_check_score != 1:
        print 'sanity check failed: %.2f - should be 1.00' %(sanity_check_score)
    return [exact_match_score, choice_match_score, info, evaluate_score]




if __name__ == '__main__':
    fileNames = ['DSR_TbyT_OFC.p']
    for datasetIndex, fileName in enumerate(fileNames):
        print '*' * 80
        print 'working on dataset: %s' %(fileName)
        df = pickle.load(open(session_dirs + fileName, 'rb'))
        sessions = df.groupby(axis = 0, level = 'session')

        print 'loading model ...',
        model = load_model(model_dirs + fileName[:-2] + '.hdf5')
        print 'done'

        print 'loading sequences ...',
        dataPrep = seq_length_d[model.input_shape[1]] + '/'
        seqs = pickle.load(open(seq_dirs + dataPrep + fileName, 'rb'))
        print 'done'

        #determining what session belongs to what group
        print 'determining what set each session belonged to ...',
        TRAIN, VALIDATE, TEST = seqs.train_validate_test_split_by_session(df)
        TRAIN_SESSIONS = \
                [w for w, s in TRAIN.groupby(axis = 0, level = 'session')]
        VALIDATE_SESSIONS = \
                [w for w, s in VALIDATE.groupby(axis = 0, level = 'session')]
        TEST_SESSIONS = \
                [w for w, s in TEST.groupby(axis = 0, level = 'session')]
        print 'done'




        print("X_train: {}, {}".format(seqs.X_train.dtype, seqs.X_train.shape))
        print("y_train: {}, {}".format(seqs.y_train.dtype, seqs.y_train.shape))


        print 'Sequence scores using model.evaluate ...',
        results_train = \
            model.evaluate(x = seqs.X_train, y = seqs.y_train, verbose = 0)
        print 'training score : %.2f' %(results_train[1])

        print 'Sequence scores using model.predict ...',
        preds = np.argmax(model.predict(seqs.X_train), axis = 1)
        actual = np.argmax(seqs.y_train, axis=1)
        manual_score = np.float(np.sum(preds == actual)) / len(seqs.X_train)
        print 'training score : %.2f' %(manual_score)

        print 'changing dtype of y_train'
        seqs.y_train = seqs.y_train.astype(np.float64)
        print("X_train: {}, {}".format(seqs.X_train.dtype, seqs.X_train.shape))
        print("y_train: {}, {}".format(seqs.y_train.dtype, seqs.y_train.shape))


        print 'Sequence scores using model.evaluate ...',
        results_train = \
            model.evaluate(x = seqs.X_train, y = seqs.y_train, verbose = 0)

        print 'training score : %.2f' %(results_train[1])
        print model.metrics_names
        print 'Sequence scores using model.predict ...',
        preds = np.argmax(model.predict(seqs.X_train), axis = 1)
        actual = np.argmax(seqs.y_train, axis=1)
        manual_score = np.float(np.sum(preds == actual)) / len(seqs.X_train)
        print 'training score : %.2f' %(manual_score)


        match_by_hand = {}

        for setIndex, set in enumerate(sets_dict.keys()):
            tmp_match_by_hand = 0
            for t in xrange(len(seqs.__dict__[set])):
                pred = np.argmax(model.predict(
                        np.reshape(seqs.__dict__[set][t], (1,model.input_shape[1],4))))
                y_actual = np.argmax(seqs.__dict__[sets_dict[set]][t])
                tmp_match_by_hand += (pred == y_actual)
            match_by_hand[set] = \
                    np.float(tmp_match_by_hand) / len(seqs.__dict__[set])



        results_validate = \
            model.evaluate(x = seqs.X_validate, y = seqs.y_validate, verbose = 0)
        results_test = \
            model.evaluate(x = seqs.X_test, y = seqs.y_test, verbose = 0)
        print 'done'

        print 'evaluating score session by session ...',
        session_dict = {}
        for label, session in sessions:
            [exact_match_score, choice_match_score, info, evaluate_score] = \
                                                RNN_session_pred(session, model)

            try:
                tmp = TRAIN_SESSIONS.index(label)
                membership = 'train'
            except ValueError:
                pass
            try:
                tmp = VALIDATE_SESSIONS.index(label)
                membership = 'validate'
            except ValueError:
                pass
            try:
                tmp = TEST_SESSIONS.index(label)
                membership = 'test'
            except ValueError:
                pass


            session_dict[label] = \
            (exact_match_score, evaluate_score, membership, choice_match_score)
        print 'done'


        print '\n'
        print 'training score : %.2f' %(results_train[1])
        print 'validation score : %.2f' %(results_validate[1])
        print 'testing score : %.2f' %(results_test[1])
        print 'scores by hand: %s' %(match_by_hand)
        print '\n'
        for sess_label in session_dict.keys():
            print 'score for session %s : %.2f - from %s - evaluate: %.2f, choice: %.2f'\
                                                %(sess_label,
                                                session_dict[sess_label][0],
                                                session_dict[sess_label][2],
                                                session_dict[sess_label][1],
                                                session_dict[sess_label][3])
        print '\n'
        print '*' * 80
