from __future__ import division, print_function

import os
import glob
import pickle
import numpy as np
import pandas as pd
idx = pd.IndexSlice


class SequenceIterator:
    def __init__(self,
                 data_label,
                 batch_size,
                 X,
                 y,
                 shuffle = 1,
                 SEQ_LENGTH = 30,
                 RANDOM_STATE = 6):

        assert train_size + validate_size + test_size == 1

        self.header = {'SEQ_LENGTH': SEQ_LENGTH,
                       'RANDOM_STATE' : RANDOM_STATE,
                       'data_label' : data_label}

        self.batch_size = batch_size
        self.X = X
        self.y = y

        self.i = 0
        self.shuffle = shuffle
        self.index = np.arange(len(self.X))
        self.steps_per_epoch = np.ceil(len(self.X) / self.batch_size)
        if shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        return self.next()


    def next(self):
        #if batch_size does not evenly divide the number of samples
        start = self.i
        end = min(start + self.batch_size, len(self.X))
        X_batch = self.X[start:end]
        y_batch = self.y[start:end]

        self.i += self.batch_size
        if self.i >= len(self.X):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        return X_batch, y_batch






def sess2sequences(sess, SEQ_LENGTH):
    '''
    Create X and y arrays given a session dataframe. X is encoded in one-hot
    vector of dimension 4, by combining choice and reward information into one
    number. 2 choices by 2 rewards = 4. Y is just choice information (of the
    next trial)
    '''
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




def shuffle_sessions(dataset, train_size, validate_size, test_size, RANDOM_STATE):

    #grouping dataset by session
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

    #calculating the cutoffs according to the size of the data splits
    training_cutoff = \
                    int(np.floor(no_sessions * train_size))
    validate_cutoff = \
                int(np.floor(no_sessions * validate_size))
    test_cutoff = \
                    int(np.floor(no_sessions * test_size))

    #set seed and shuffle the list
    np.random.seed(RANDOM_STATE)
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
    return train_df, test_df, validate_df




def mirror_dataset(dataset):
    mirror = [dataset]
    for label, session in dataset.groupby(axis=0, level = 'SessionID'):
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
    #mirroring is only done on the training data
    return train_with_mirror


def convert_by_session(train_df, SEQ_LENGTH):
    if len(train_df) > 0:
        dataset_X = []
        dataset_y = []
        for label, session in train_df.groupby(axis=0, level = 'SessionID'):
            X, y = sess2sequences(session, SEQ_LENGTH)
            dataset_X.append(X)
            dataset_y.append(y)
        X = np.concatenate(dataset_X)
        y = np.concatenate(dataset_y)
        assert X.shape[0] == y.shape[0]
    else:
        X = None
        y = None
    return X, y





def create_sequences(data_dir,
                     SEQ_LENGTH = 30,
                     RANDOM_STATE = 1,
                     BATCH_SIZE = 64,
                     MIRROR = 1,
                     SHUFFLE = 1,
                     train_size = 0.5,
                     validate_size = 0.25,
                     test_size = 0.25):

    '''
    Takes a dataset dataframe and creates SequenceIterator objects, given some
    batch size.

    Inputs -
    Sequence Length - (int) how long to make the sequence
    RANDOM_STATE - random seed - important to keep track for reproducibility
    BATCH_SIZE - how big each batch will be for training purposes
    MIRROR - whether we augment the training data
    SHUFFLE - whether we shuffle the data after every training epoch
    train_size - how much to devote to training
    validate_size - ditto
    test_size - ditto

    Output -
    Sequence Iterator - depending on how big training and validation
    and testing sets are. If they all exist, it returns separate iterators
    for each
    '''


    #load dataset
    try:
        dataset = pickle.load(open(data_dir, 'rb'))
    except IOError:
        print('File not found: {}...exiting'.format(data_dir))
        return


    #split and shuffle dataset by session
    train_df, test_df, validate_df = shuffle_sessions(dataset,
                                                      train_size,
                                                      validate_size,
                                                      test_size,
                                                      RANDOM_STATE)

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
    #mirroring is only done on the training data
    if MIRROR:
        train_df = mirror_dataset(train_df)

    #convert sessions into sequences
    X_train, y_train = convert_by_session(train_df, SEQ_LENGTH)
    X_validate, y_validate = convert_by_session(validate_df, SEQ_LENGTH)
    X_test, y_test = convert_by_session(test_df, SEQ_LENGTH)


    if not X_train is None and not y_train is None:
        #create iterator object
        SeqIterTrain = SequenceIterator(data_dir,
                                   batch_size = BATCH_SIZE,
                                   X = X_train,
                                   y = y_train,
                                   shuffle = SHUFFLE,
                                   SEQ_LENGTH = SEQ_LENGTH,
                                   RANDOM_STATE = RANDOM_STATE)
    else:
        SeqIterTrain = None


    if not X_validate is None and not y_validate is None:
        SeqIterVal = SequenceIterator(data_dir,
                                   batch_size = BATCH_SIZE,
                                   X = X_validate,
                                   y = y_validate,
                                   shuffle = SHUFFLE,
                                   SEQ_LENGTH = SEQ_LENGTH,
                                   RANDOM_STATE = RANDOM_STATE)
    else:
        SeqIterVal = None


    if not X_test is None and not y_test is None:
        SeqIterTest = SequenceIterator(data_dir,
                                  batch_size = BATCH_SIZE,
                                  X = X_test,
                                  y = y_test,
                                  shuffle = SHUFFLE,
                                  SEQ_LENGTH = SEQ_LENGTH,
                                  RANDOM_STATE = RANDOM_STATE)
    else:
        SeqIterTest = None

    return SeqIterTrain, SeqIterVal, SeqIterTest


if __name__ == '__main__':

    ROOT = os.environ['HOME'] + '/python/behavioral_performance/'
    data_dir = ROOT + 'data/dataset_dataframes/'
    target_dir = ROOT + 'data/sequences/'


    glob_search = os.path.join(data_dir, "*.p")
    dataset_dirs = sorted(glob.glob(glob_search))
    if len(dataset_dirs) == 0:
        raise Exception("No datasets found in {}".format(dataset_dirs))

    for data_dir in dataset_dirs:
        current_dataset = os.path.basename(data_dir)
        SeqIter = create_sequences(data_dir,
                             SEQ_LENGTH = 30,
                             RANDOM_STATE = 1,
                             MIRROR = 1,
                             SHUFFLE = 1,
                             train_size = 0.8,
                             validate_size = 0.2,
                             test_size = 0)
        pickle.dump(SeqIter, open(target_dir + current_dataset, 'wb'))
