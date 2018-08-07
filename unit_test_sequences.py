import pickle
import copy
import numpy as np
import pandas as pd
from behavioral_performance.utils import fileNames

#adding artificial datasets
fileNames.append('PSR_TbyT_Saline_Rigged.p')
fileNames.append('DSR_TbyT_Saline_Shuffled.p')
fileNames.append('PSR_TbyT_Saline_Shuffled.p')
fileNames.append('DSR_TbyT_Naive_Saline.p')


ROOT = '/Users/pablomartin/python/'
seq_dirs = 'DATA_structures/RNN_sequences/OneHotBinaryMinimal/'
dataType = ['Full', 'Med', 'Last']

normal = 'DSR_TbyT_Saline.p'
shuffle = 'DSR_TbyT_Saline_Shuffled.p'

for dataPrep in dataType:
    normalSeqs = pickle.load(open(ROOT + seq_dirs + dataPrep + '/' + normal, 'rb'))
    shuffleSeqs = pickle.load(open(ROOT + seq_dirs + dataPrep + '/' + shuffle,'rb'))
    #check dimensions
    assert normalSeqs.X_train.shape == shuffleSeqs.X_train.shape
    assert normalSeqs.X_validate.shape == shuffleSeqs.X_validate.shape
    assert normalSeqs.X_test.shape == shuffleSeqs.X_test.shape
    assert normalSeqs.y_train.shape == shuffleSeqs.y_train.shape
    assert normalSeqs.y_validate.shape == shuffleSeqs.y_validate.shape
    assert normalSeqs.y_test.shape == shuffleSeqs.y_test.shape



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
pickle.dump(DSR_Saline, open(ROOT + 'DATA_structures/TbyT/DSR_TbyT_Saline_Shuffled.p', 'wb'))
pickle.dump(PSR_Saline, open(ROOT + 'DATA_structures/TbyT/PSR_TbyT_Saline_Shuffled.p', 'wb'))
print 'created shuffled datasets'


#rigged dataset - base will be PSR_Saline
PSR_Saline = pickle.load(open(ROOT + \
                            'DATA_structures/TbyT/PSR_TbyT_Saline.p', 'rb'))

A = copy.deepcopy(PSR_Saline['reward', 0])
np.random.shuffle(A)

print 'before assigning shuffled list to original'
print 'A: %i/%i' %(np.sum(A == PSR_Saline['reward', 0]), len(PSR_Saline))
print 'A: NAN values: %i' %np.sum(np.isnan(A))

PSR_Saline['reward',0] = A
print 'after assigning shuffled list to original'
print 'A: %i/%i' %(np.sum(A == PSR_Saline['choice', 0]), len(PSR_Saline))
print 'A: NAN values: %i' %np.sum(np.isnan(A))

for trial in range(1, len(PSR_Saline)):
    A = PSR_Saline['reward',0].iloc[trial - 1]
    B = PSR_Saline['choice',0].iloc[trial - 1]
    PSR_Saline['choice',0].iloc[trial] = A * (not B) + (not A) * B

pickle.dump(PSR_Saline, open(ROOT + \
                        'DATA_structures/TbyT/PSR_TbyT_Saline_Rigged.p', 'wb'))
print 'created rigged dataset'
