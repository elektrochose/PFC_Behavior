import pickle
import copy
import numpy as np
import pandas as pd
from RNNmodule.SequenceClass import Sequences
ROOT = '/Users/pablomartin/python/'
idx = pd.IndexSlice
create = 0
check_raw = 0

if create == 1:
    #load saline datasets
    DSR_Saline = pickle.load(open(ROOT + 'DATA_structures/TbyT/DSR_TbyT_Saline.p', 'rb'))
    PSR_Saline = pickle.load(open(ROOT + 'DATA_structures/TbyT/PSR_TbyT_Saline.p', 'rb'))
    #get current choice label, but dereferenced
    DSR_choice = copy.deepcopy(DSR_Saline['choice', 0])
    PSR_choice = copy.deepcopy(PSR_Saline['choice', 0])
    DSR_reward = copy.deepcopy(DSR_Saline['reward', 0])
    PSR_reward = copy.deepcopy(PSR_Saline['reward', 0])

    #check if every entry is the same, it should
    assert np.sum(DSR_choice == DSR_Saline['choice', 0]) == len(DSR_Saline)
    assert np.sum(PSR_choice == PSR_Saline['choice', 0]) == len(PSR_Saline)

    #shuffle current choice - these are the eventual labels
    np.random.shuffle(DSR_choice)
    np.random.shuffle(PSR_choice)
    np.random.shuffle(DSR_reward)
    np.random.shuffle(PSR_reward)

    print 'after shuffle:'
    print '%i / %i match' \
            %(np.sum(DSR_choice == DSR_Saline['choice', 0]), len(DSR_Saline))
    print '%i / %i match' \
            %(np.sum(PSR_choice == PSR_Saline['choice', 0]), len(PSR_Saline))
    print '%i / %i match' \
            %(np.sum(DSR_reward == DSR_Saline['reward', 0]), len(DSR_Saline))
    print '%i / %i match' \
            %(np.sum(PSR_reward == PSR_Saline['reward', 0]), len(PSR_Saline))

    DSR_Saline['choice', 0] = DSR_choice
    PSR_Saline['choice', 0] = PSR_choice
    DSR_Saline['reward', 0] = DSR_reward
    PSR_Saline['reward', 0] = PSR_reward


    print 'after assigning:'
    print '%i / %i match' \
            %(np.sum(DSR_choice == DSR_Saline['choice', 0]), len(DSR_Saline))
    print '%i / %i match' \
            %(np.sum(PSR_choice == PSR_Saline['choice', 0]), len(PSR_Saline))
    print '%i / %i match' \
            %(np.sum(DSR_reward == DSR_Saline['reward', 0]), len(DSR_Saline))
    print '%i / %i match' \
            %(np.sum(PSR_reward == PSR_Saline['reward', 0]), len(PSR_Saline))

    #save result
    pickle.dump(DSR_Saline, \
        open(ROOT + 'DATA_structures/TbyT/DSR_TbyT_Saline_Shuffled.p', 'wb'))
    pickle.dump(PSR_Saline, \
        open(ROOT + 'DATA_structures/TbyT/PSR_TbyT_Saline_Shuffled.p', 'wb'))
    print 'created shuffled datasets'
elif create == 2:
    #load saline datasets
    DSR_Saline = pickle.load(open(ROOT + 'DATA_structures/TbyT/DSR_TbyT_Saline.p', 'rb'))
    PSR_Saline = pickle.load(open(ROOT + 'DATA_structures/TbyT/PSR_TbyT_Saline.p', 'rb'))

if check_raw == 1:
    '''
    First we check whether the raw structure contains the rigged pattern
    '''
    original_path = ROOT + 'DATA_structures/TbyT/DSR_TbyT_Saline.p'
    dataset_path = ROOT + 'DATA_structures/TbyT/DSR_TbyT_Saline_Shuffled.p'

    oDF = pickle.load(open(original_path, 'rb'))
    df = pickle.load(open(dataset_path, 'rb'))

    print '*' * 80
    #checking whether reward was shuffled
    print 'match between orginal and shuffled, reward: %i/%i' \
                        %(np.sum(oDF['reward', 0] == df['reward', 0]), len(oDF))
    if np.float(np.sum(oDF['reward', 0] == df['reward', 0])) / len(oDF) < 0.65:
        print 'TEST PASSED'
    else:
        print 'REWARD MAY HAVE NOT BEEN SHUFFLED'


    #checking whether choice was shuffled
    print 'match between orginal and shuffled, choice: %i/%i' \
                        %(np.sum(oDF['choice', 0] == df['choice', 0]), len(df))
    if np.float(np.sum(oDF['choice', 0] == df['choice', 0])) / len(oDF) < 0.65:
        print 'TEST PASSED'
    else:
        print 'CHOICE MAY HAVE NOT BEEN SHUFFLED'


    seqObject = Sequences(30, 'OneHotBinaryMinimal', RANDOM_STATE = 6)
    train, validate, test = \
                seqObject.train_validate_test_split_by_session(df,
                                                               validate_size = 0.25,
                                                               test_size = 0.25)
    trainO, validateO, testO = \
                seqObject.train_validate_test_split_by_session(oDF,
                                                               validate_size = 0.25,
                                                               test_size = 0.25)
    print 'training set:'
    print 'match between orginal and shuffled, reward: %i/%i' \
                        %(np.sum(trainO['reward', 0] == train['reward', 0]), len(train))
    print 'match between orginal and shuffled, choice: %i/%i' \
                        %(np.sum(trainO['choice', 0] == train['choice', 0]), len(train))

    print 'validation set:'
    print 'match between orginal and shuffled, reward: %i/%i' \
                        %(np.sum(validateO['reward', 0] == validate['reward', 0]), len(validate))
    print 'match between orginal and shuffled, choice: %i/%i' \
                        %(np.sum(validateO['choice', 0] == validate['choice', 0]), len(validate))

    print 'testing set:'
    print 'match between orginal and shuffled, reward: %i/%i' \
                        %(np.sum(testO['reward', 0] == test['reward', 0]), len(test))
    print 'match between orginal and shuffled, choice: %i/%i' \
                        %(np.sum(testO['choice', 0] == test['choice', 0]), len(test))


    print '*' * 80

    print 'Before mirroring:'
    for (label, session), (oLabel, oSession) in \
                                zip(train.groupby(axis = 0, level = 'session'),
                                    trainO.groupby(axis = 0, level = 'session')):
        shuffleCombined = \
            session.loc[:,idx['choice', 0]] * 2 + session.loc[:, idx['reward',0]]
        originalCombined = \
            oSession.loc[:,idx['choice', 0]] * 2 + oSession.loc[:, idx['reward',0]]
        percent_match = \
                float(np.sum(shuffleCombined == originalCombined)) / len(session)
        print 'match between original and shuffled: %.2f' %(percent_match)
        if label == oLabel and percent_match < 0.5:
            print 'session: %s - TEST PASSED' %(label)
        else:
            print 'session: %s - TEST FAILED' %(label)

    mirror = copy.deepcopy(train)
    mirror['choice',0] = (mirror['choice',0] + 1) % 2
    session_labels = [w + '_mirror' for w in mirror.index.levels[0]]
    mirror.index.set_levels(session_labels, level='session', inplace = True)
    train = pd.concat([train, mirror], axis=0)

    mirror = copy.deepcopy(trainO)
    mirror['choice',0] = (mirror['choice',0] + 1) % 2
    session_labels = [w + '_mirror' for w in mirror.index.levels[0]]
    mirror.index.set_levels(session_labels, level='session', inplace = True)
    trainO = pd.concat([trainO, mirror], axis=0)
    print '*' * 80
    print 'After mirroring:'
    for (label, session), (oLabel, oSession) in \
                                zip(train.groupby(axis = 0, level = 'session'),
                                    trainO.groupby(axis = 0, level = 'session')):
        shuffleCombined = \
            session.loc[:,idx['choice', 0]] * 2 + session.loc[:, idx['reward',0]]
        originalCombined = \
            oSession.loc[:,idx['choice', 0]] * 2 + oSession.loc[:, idx['reward',0]]
        percent_match = \
                float(np.sum(shuffleCombined == originalCombined)) / len(session)
        print 'match between original and shuffled: %.2f' %(percent_match)
        if label == oLabel and percent_match < 0.5:
            print 'session: %s - TEST PASSED' %(label)
        else:
            print 'session: %s - TEST FAILED' %(label)


print '*' * 80
'''
Last step is to check the actual sequences
'''

seq_dirs = 'DATA_structures/RNN_sequences/OneHotBinaryMinimal/'
dataType = ['Full', 'Med', 'Last']

normal = 'PSR_TbyT_Saline.p'
shuffle = normal[:-2] + '_Shuffled.p'
splits = ['X_test', 'X_validate', 'X_train']
y_splits = ['y_test', 'y_validate', 'y_train']

print 'Checking actual sequences:'
for dataPrep in dataType:
    seqs = pickle.load(open(ROOT + seq_dirs + dataPrep + '/' + shuffle,'rb'))
    original = pickle.load(open(ROOT + seq_dirs + dataPrep + '/' + normal, 'rb'))

    # print 'shuffling labels once more'
    # np.random.shuffle(original.X_train)
    # np.random.shuffle(original.X_validate)
    # np.random.shuffle(original.X_test)
    # np.random.shuffle(original.y_train)
    # np.random.shuffle(original.y_validate)
    # np.random.shuffle(original.y_test)
    # pickle.dump(original, open(ROOT + seq_dirs + dataPrep + '/' + shuffle, 'wb'))

    for split, y_split in zip(splits, y_splits):
        print '%s - %s' %(split, y_split)
        match = 0
        for trial in range(len(seqs.__dict__[split]) - 1):
            next_choice = np.argmax(seqs.__dict__[split][trial + 1][-1])
            y_label = np.argmax(seqs.__dict__[y_split][trial])
            match += (next_choice == y_label)
        print '# of label matches: %i/%i' %(match, len(seqs.__dict__[split]))

        match = 0
        x_seq_match = []
        for trial in range(len(seqs.__dict__[split])):
            shuffled_y = np.argmax(seqs.__dict__[y_split][trial])
            original_y = np.argmax(original.__dict__[y_split][trial])
            match += (shuffled_y == original_y)
            shuffled_X = np.argmax(seqs.__dict__[split][trial], axis=1)
            original_X = np.argmax(original.__dict__[split][trial], axis = 1)
            seq_ratio = np.float(np.sum(shuffled_X == original_X)) / len(shuffled_X)
            x_seq_match.append(seq_ratio)


        print 'ratio of shuffled y-labels: %.4f' %(float(match) / len(seqs.__dict__[split]))
        print 'expected ~ 0.25'
        print dataPrep
        print 'ratio of shuffled x-labels: %.4f' %(np.mean(x_seq_match))
        print '-' * 80
