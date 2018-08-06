import pandas as pd
import numpy as np
from scipy.stats import kstest


idx = pd.IndexSlice

def my_train_test_split(df, RANDOM_STATE = 12, test_size = 0.2):
    sessions = df.groupby(axis = 0, level = 'session')
    noSessions = len(sessions)
    sessList = np.arange(noSessions)
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(sessList)
    cutoff = int(np.ceil(noSessions * test_size))
    test = ['S' + str(w) for w in sessList[:cutoff]]
    train = ['S' + str(w) for w in sessList[cutoff:]]
    return idx[train, :, :], idx[test, :, :]

def filename_to_label(fileName):
    task = fileName[0] + 'SR'
    fileName = fileName[:fileName.find('.')]
    regime = fileName[-fileName[::-1].find('_'):]
    return task, regime

def ks_test_model(decodingErrors, noTrials):
    sample = [np.float(trial)/noTrials for trial in decodingErrors]
    D, p = kstest(sample, 'uniform', args=(0, 1))
    return p


tasks = ['DSR','PSR']
dataLabels = ['FirstTraining', 'MidTraining', \
              'Saline', 'MPFC', 'OFC', \
              'Ipsi', 'Contra', ]

#retrieving directory and filenames
fileNameLabels = [task + '_' + dataLabel \
                  for task in tasks\
                  for dataLabel in dataLabels]

fileNameLabels[0] = 'DSR_Naive'
fileNameLabels.insert(1, 'DSR_Naive_mPFC')
fileNameLabels.insert(2, 'DSR_Naive_OFC')
fileNameLabels[9] = 'PSR_Naive'



fileNames = [task + '_TbyT_'  + dataLabel + '.p' \
             for task in tasks \
             for dataLabel in dataLabels]

fileNames.insert(1, 'DSR_TbyT_Naive_mPFC.p')
fileNames.insert(2, 'DSR_TbyT_Naive_OFC.p')
