
import os
import pickle
import numpy as np
import pandas as pd
from behavioral_performance.tools import SequenceClass as SC
Sequences = SC.Sequences
idx = pd.IndexSlice

ROOT = os.environ['HOME'] + '/python/'
source = ROOT + 'DATA_structures/dataset_dataframes/'
target = ROOT + 'DATA_structures/RNN_sequences/sequence_classification_mirror/'




def sess2sequences(sess, SEQ_LENGTH):
    #create X and y
    sess['X'] = sess['Choice'] * 2 + sess['AR']
    sess['y'] = sess.Choice.shift(-1)
    sess.dropna(inplace = True)
    sess['y'] = sess['y'].astype('int16')

    noTrials = len(sess)
    sess_X = np.zeros((noTrials, SEQ_LENGTH, 4), dtype = 'int16')
    sess_y = np.zeros((noTrials, 2), dtype = 'int16')
    for trial in range(noTrials):
        #the y sequence is trivial
        y = np.zeros(2, dtype = 'int16')
        y[sess['y'].iat[trial]] = 1

        X = np.zeros((SEQ_LENGTH, 4), dtype = 'int16')
        for past in range(SEQ_LENGTH):
            if trial - past >= 0:
                X[- 1 - past, sess['X'].iloc[trial - past]] = 1
        empty_trials = max(0, SEQ_LENGTH - trial - 1)
        assert empty_trials ==  np.sum(np.sum(X, axis = 1) == 0)
        sess_X[trial, :, :] = X
        sess_y[trial, :] = y
        if trial > 0:
            past_x = np.argmax(sess_X[trial, -1, :]) > 1
            assert past_x == np.argmax(sess_y[trial - 1, :])
    return sess_X, sess_y



if __name__ == '__main__':

    SEQ_LENGTH = 30
    for dataset_file in os.listdir(source):
        #load dataset
        dataset = pickle.load(open(source + dataset_file, 'rb'))
        #create sequences object
        seqObj = Sequences(dataset_file[:-2],
                           sequence_length = SEQ_LENGTH,
                           RANDOM_STATE = 10,
                           train_size = 0.5,
                           validate_size = 0.25,
                           test_size = 0.25)

        label_sets = set([(a, b) for a,b in zip(dataset.index.labels[0],
                                                dataset.index.labels[1])])
        label_transform = [(dataset.index.levels[0][a],
                            dataset.index.levels[1][b]) for a,b in label_sets]

        conv_sess = []
        for element in label_transform:
            sess = dataset.loc[element, :]
            sessID = '_'.join([str(w) for w in element])
            sess = pd.concat([sess], keys=[sessID], names=['SessionID'])
            conv_sess.append(sess)
        dataset_conv = pd.concat(conv_sess)
        dataset_conv.sort_index(axis = 0, inplace = True)
        session_list = [w for w in dataset_conv.index.levels[0].values]
        no_sessions = len(session_list)

        training_cutoff = \
                        int(np.floor(no_sessions * seqObj.header['train_size']))
        validate_cutoff = \
                    int(np.floor(no_sessions * seqObj.header['validate_size']))
        test_cutoff = \
                        int(np.floor(no_sessions * seqObj.header['test_size']))

        #set seed and shuffle the list
        np.random.seed(seqObj.header['RANDOM_STATE'])
        np.random.shuffle(session_list)
        validate_set = session_list[:validate_cutoff]
        test_set = session_list[validate_cutoff: validate_cutoff + test_cutoff]
        train_set = session_list[validate_cutoff + test_cutoff:]

        #make sure no session is in 2 sets
        assert len([t for val in test_set if val in train_set]) == 0
        assert len([t for val in test_set if val in validate_set]) == 0
        assert len([t for val in validate_set if val in train_set]) == 0
        #make sure all sessions are used!
        assert len(test_set) + len(train_set) + len(validate_set) == no_sessions

        train_df = dataset_conv.loc[idx[train_set, :, :], :]
        test_df = dataset_conv.loc[idx[test_set, :, :], :]
        validate_df = dataset_conv.loc[idx[validate_set, :, :], :]

        train_df.sort_index(axis = 0, inplace = True)
        test_df.sort_index(axis = 0, inplace = True)
        validate_df.sort_index(axis = 0, inplace = True)
        #make sure number of trials matches up
        assert len(dataset_conv) == \
                        (len(train_df) + len(test_df) + len(validate_df))

        '''
        Here we start making the sequences.
        We will go session by session, we do not want cross session leaking.
        For each session, we take the previous 30 trials, and attempt to predict
        the next choice. For each trial we will take the choice and reward info
        and turn it into a one-hot vector.

        Remember, the shape we want evenatually has the dimensions:
        samples X SEQ_LENGTH X number of features
        In our case:
        #samples X 30 X 4
        '''

        mirror = [train_df]
        for label, session in train_df.groupby(axis=0, level = 'SessionID'):
            session.loc[:,'AR'] = (session.loc[:,'AR'] + 1) % 2
            session.loc[:,'Choice'] = (session.loc[:,'Choice'] + 1) % 2
            index_tuples = [(x[0] + '_mirror', x[1], x[2]) for x in session.index]
            new_index = \
                pd.MultiIndex.from_tuples(
                        [(x[0] + '_mirror', x[1], x[2]) for x in session.index],
                        names = session.index.names)
            session.index = new_index
            mirror.append(session)
        train_with_mirror = pd.concat(mirror, axis = 0)
        train_with_mirror.sort_index(axis = 0, inplace = True)
        #to turn on mirroring, uncomment next line
        train_df = train_with_mirror

        #first we do training set
        dataset_X = []
        dataset_y = []
        for label, session in train_df.groupby(axis=0, level = 'SessionID'):
            X, y = sess2sequences(session, SEQ_LENGTH)
            dataset_X.append(X)
            dataset_y.append(y)
        X = np.concatenate(dataset_X)
        y = np.concatenate(dataset_y)
        assert X.shape[0] == y.shape[0]
        seqObj.X_train = X
        seqObj.y_train = y

        #validation set
        dataset_X = []
        dataset_y = []
        for label, session in validate_df.groupby(axis=0, level = 'SessionID'):
            X, y = sess2sequences(session, SEQ_LENGTH)
            dataset_X.append(X)
            dataset_y.append(y)
        X = np.concatenate(dataset_X)
        y = np.concatenate(dataset_y)
        assert X.shape[0] == y.shape[0]
        seqObj.X_validate = X
        seqObj.y_validate = y

        #testing set
        dataset_X = []
        dataset_y = []
        for label, session in test_df.groupby(axis=0, level = 'SessionID'):
            X, y = sess2sequences(session, SEQ_LENGTH)
            dataset_X.append(X)
            dataset_y.append(y)
        X = np.concatenate(dataset_X)
        y = np.concatenate(dataset_y)
        assert X.shape[0] == y.shape[0]
        seqObj.X_test = X
        seqObj.y_test = y

        pickle.dump(seqObj, open(target +  dataset_file, 'wb'))
