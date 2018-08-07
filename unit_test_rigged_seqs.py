import pickle
import numpy as np
from RNNmodule.SequenceClass import Sequences

seqs = pickle.load(open('DATA_structures/RNN_sequences/' + \
                    'OneHotBinaryMinimal/Med/PSR_TbyT_Saline_Rigged.p','rb'))

samples = len(seqs.X_train)
match = 0
for trial in range(samples):
    x = seqs.X_train[trial]
    y = seqs.y_train[trial]
    A = np.argmax(x[-1,:]) > 1
    B = (np.argmax(x[-2,:]) % 2) == 1
    rigged_choice = A * (not B) + (not A) * B
    actual_choice = np.argmax(y) > 1
    match += 1
print '# of matches %i/%i' %(match, samples)
