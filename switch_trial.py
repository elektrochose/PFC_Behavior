import config
import sys
import numpy as np
import pickle
from session_iterator import SessionIterator


def goal_curve(sess, movingWindow = 7):
    seed = sess.info['GA'][0]
    seq = (sess.info['Choice 2'] == sess.info['GA'][0] + 9)
    out = np.zeros([len(seq)], dtype=float)
    for trial in range(len(seq)):
        bottomWindow = trial - movingWindow
        if bottomWindow < 0:
            bottomWindow = 0
        out[trial] = float(np.sum(seq[bottomWindow : trial])) \
                                    / (1 + trial - bottomWindow)
    return out



regime = 'probabilistic'

saline = SessionIterator([regime, 'saline','PL','OFC'])
mpfc = SessionIterator([regime,'muscimol','PL'])
ofc = SessionIterator([regime,'muscimol','OFC'])

datasets = [saline, mpfc, ofc]
out = np.zeros([50, 3], dtype=float); out[:] = np.NaN

for dataSetIndex, dataset in enumerate(datasets):
    for sessIndex, sess in enumerate(dataset.sessionList):
        Rev1 = sess.get_block(2)
        LC = goal_curve(sess)
        Rev1LC = LC[Rev1.index[:]]
        if len(np.nonzero(Rev1LC < 0.5)[0]) > 0:
            out[sessIndex, dataSetIndex] = np.nonzero(Rev1LC < 0.5)[0][0]
        else:
            print "there was no switch trial here..."


np.savetxt('Results/tables/PSRswitchTrials.csv', out, delimiter=',')
