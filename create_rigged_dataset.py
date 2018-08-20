import pickle
import copy
import numpy as np
import pandas as pd

#rigged dataset - base will be PSR_Saline
PSR_Saline = pickle.load(open(ROOT + \
                            'DATA_structures/TbyT/PSR_TbyT_Saline.p', 'rb'))
fresh_copy = pickle.load(open(ROOT + \
                            'DATA_structures/TbyT/PSR_TbyT_Saline.p', 'rb'))
choice_ch = np.sum(PSR_Saline['choice',0] == fresh_copy['choice', 0])
reward_ch = np.sum(PSR_Saline['reward',0] == fresh_copy['reward', 0])
print 'matching values choice: %i/%i' %(choice_ch, len(PSR_Saline))
print 'matching values reward: %i/%i' %(reward_ch, len(PSR_Saline))
for label, session in PSR_Saline.groupby(axis = 0, level = 'session'):

    A = copy.deepcopy(session['reward', 0])
    B = copy.deepcopy(session['choice', 0])
    C = A + 2 * B
    for trial in range(2, len(session)):
        C.iloc[trial] = (C.iloc[trial - 1] + C.iloc[trial - 2]) % 4
    A = C % 2
    B = C > 1
    print 'before assigning rigged list to original'
    print 'A: %i/%i' %(np.sum(A == session['reward', 0]), len(session))
    print 'B: %i/%i' %(np.sum(B == session['choice', 0]), len(session))

    session['reward',0] = A
    session['choice',0] = B
    print 'after assigning rigged list to original'
    print 'A: %i/%i' %(np.sum(A == session['reward', 0]), len(session))
    print 'B: %i/%i' %(np.sum(B == session['choice', 0]), len(session))


    PSR_Saline.loc[idx[label,:,:],idx['choice',0]] = session['choice',0]
    PSR_Saline.loc[idx[label,:,:],idx['reward',0]] = session['reward',0]



choice_ch = np.sum(PSR_Saline['choice',0] == fresh_copy['choice', 0])
reward_ch = np.sum(PSR_Saline['reward',0] == fresh_copy['reward', 0])

print 'matching values choice: %i/%i' %(choice_ch, len(PSR_Saline))
print 'matching values reward: %i/%i' %(reward_ch, len(PSR_Saline))

pickle.dump(PSR_Saline, open(ROOT + \
                        'DATA_structures/TbyT/PSR_TbyT_Saline_Rigged.p', 'wb'))
print 'created rigged dataset'
