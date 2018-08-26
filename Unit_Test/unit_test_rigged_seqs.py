import copy
import pickle
import pandas as pd
import numpy as np
from RNNmodule.SequenceClass import Sequences


'''
First we check whether the raw structure contains the rigged pattern
'''

dataset_path = '/Users/pablomartin/python/DATA_structures/TbyT/PSR_TbyT_Saline_Rigged.p'
df = pickle.load(open(dataset_path, 'rb'))

seqObject = Sequences(30, 'OneHotBinaryMinimal', RANDOM_STATE = 6)
train, validate, test = \
            seqObject.train_validate_test_split_by_session(df,
                                                           validate_size = 0.25,
                                                           test_size = 0.25)
def check_rigged(train):
    choice_match, reward_match = 0, 0
    noSessions = len(train.groupby(axis = 0, level='session'))
    for trial in range(2, len(train)):
        a = train['reward', 0].iloc[trial - 2] + train['choice', 0].iloc[trial - 2] * 2
        b = train['reward', 0].iloc[trial - 1] + train['choice', 0].iloc[trial - 1] * 2
        c = (a + b) % 4
        current_choice = c > 1
        current_reward = c % 2
        choice_match += (current_choice == train['choice',0].iloc[trial])
        reward_match += (current_reward == train['reward',0].iloc[trial])
    print 'choice match: %i/%i' %(choice_match, len(train))
    print 'reward match: %i/%i' %(reward_match, len(train))
    print '# of sessions: %i' %(noSessions)
    print '%i + %i = %i' %(choice_match, noSessions, len(train))

print 'training set:'
check_rigged(train)
print 'validation set:'
check_rigged(validate)
print 'testing set:'
check_rigged(test)

print 'We mirror the training data and check again:'
mirror = copy.deepcopy(train)
mirror['choice',0] = (mirror['choice',0] + 1) % 2
mirror['reward',0] = (mirror['reward',0] + 1) % 2
session_labels = [w + '_mirror' for w in mirror.index.levels[0]]
mirror.index.set_levels(session_labels, level='session', inplace = True)
train = pd.concat([train, mirror], axis=0)
check_rigged(train)



'''
OK, the problem was when we 'mirrored' the data. Just because we flip choice
and reward labels does not mean that the rule of adding them up carries over.
We reran the sequence creation step with mirroring off and got reasonable
results again
'''


print 'After Preprocessing:'
seqs = pickle.load(open('DATA_structures/RNN_sequences/' + \
                    'OneHotBinaryMinimal/Med/PSR_TbyT_Saline_Rigged.p','rb'))

samples = len(seqs.X_train)
sanity_check, match = 0, 0
for trial in range(samples - 1):
    #quick sanity check - y label of trial + 1 == x feature of trial
    sanity_check += (np.argmax(seqs.X_train[trial + 1][-1]) == \
                    np.argmax(seqs.y_train[trial]))

    x1 = np.argmax(seqs.X_train[trial][-1])
    x2 = np.argmax(seqs.X_train[trial][-2])
    rigged_choice = (x1 + x2) % 4

    actual_choice = np.argmax(seqs.y_train[trial])
    if trial < -100:
        print 'rigged:%i - actual:%i - trial:%i' %(rigged_choice, actual_choice, trial)
    match += (rigged_choice == actual_choice)

print '# of matches %i/%i' %(match, samples)
print '# of sanity_check %i/%i' %(sanity_check, samples)
