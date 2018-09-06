
import os
import pickle
import numpy as np
import pandas as pd
idx = pd.IndexSlice

ROOT = os.environ['HOME'] + '/python/'
target = ROOT + 'DATA_structures/dataset_dataframes/'
'''
We will create a dataframe for each dataset. There will be 19 datasets total:
Naive - DSR
Naive - DSR - mPFC
Naive - DSR - OFC
Naive - PSR
Mid-training - DSR
Mid-training - PSR
Saline - DSR
Saline - PSR
mPFC - DSR
mPFC - PSR
OFC - DSR
OFC - PSR
IPSI - DSR
IPSI - PSR
CONTRA - DSR
CONTRA - PSR
Rigged Non-linear Dataset
Rigged Linear Dataset
Shuffled Dataset
'''

Naive = pickle.load(open(ROOT + \
        'DATA_structures/session_dataframes/Naive_SESSIONS_DATAFRAME.p', 'rb'))
DSR_TRAINING = pickle.load(open(ROOT + \
'DATA_structures/session_dataframes/DSR_TRAINING_SESSIONS_DATAFRAME.p', 'rb'))

#first we make the saline dataset
#from Naive it is quite simple
Naive_Saline = Naive.loc[idx[:,'Saline-Naive',:,:],:]
#excluding sessions from cohort 2 because they had training in other task first
Naive_first_session = \
                DSR_TRAINING.loc[idx[['41','43','44','45','46','47'],1,:,:],:]
Naive_DSR = pd.concat([Naive_Saline, Naive_first_session], axis=0)
pickle.dump(Naive_DSR, open(target + 'Naive_DSR.p', 'wb'))

#now we make Naive mPFC and OFC datasets
Naive_mPFC = Naive.loc[idx[:,'PL-muscimol',:,:],:]
Naive_OFC = Naive.loc[idx[:,'OFC-muscimol',:,:],:]
pickle.dump(Naive_mPFC, open(target + 'Naive_DSR_mPFC.p', 'wb'))
pickle.dump(Naive_OFC, open(target + 'Naive_DSR_OFC.p', 'wb'))

#same for PSR Naive - just take the first session of training df
PSR_TRAINING = pickle.load(open(ROOT + \
'DATA_structures/session_dataframes/PSR_TRAINING_SESSIONS_DATAFRAME.p', 'rb'))
Naive_first_session = PSR_TRAINING.loc[idx[:,1,:,:],:]
pickle.dump(Naive_first_session, open(target + 'Naive_PSR.p', 'wb'))

#Mid-training - DSR
out = []
for label, animal_data in DSR_TRAINING.groupby(axis = 0, level = 'rat'):
    animal_sessions = animal_data.groupby(axis = 0, level = 'training_session')
    no_sessions = len(animal_sessions)
    mid_session_index = int(np.floor(no_sessions / 2))
    out.append(animal_sessions.get_group(mid_session_index))
out = pd.concat(out, axis=0)
pickle.dump(out, open(target + 'Mid_Training_DSR.p', 'wb'))

#Mid-training - PSR
out = []
for label, animal_data in PSR_TRAINING.groupby(axis = 0, level = 'rat'):
    animal_sessions = animal_data.groupby(axis = 0, level = 'training_session')
    no_sessions = len(animal_sessions)
    mid_session_index = int(np.floor(no_sessions / 2))
    out.append(animal_sessions.get_group(mid_session_index))
out = pd.concat(out, axis=0)
pickle.dump(out, open(target + 'Mid_Training_PSR.p', 'wb'))

#All DSR-Trained datasets: saline, mpfc, ofc, ipsi, contra
DSR_TRAINED = pickle.load(open(ROOT + \
        'DATA_structures/session_dataframes/DSR_SESSIONS_DATAFRAME.p', 'rb'))
saline = []
for label, dataset in DSR_TRAINED.groupby(axis = 0, level = 'Infusion_String'):
    if label.find('saline') >= 0:
        saline.append(dataset)
    else:
        if label == 'CONTRA-muscimol':
            pickle.dump(dataset, open(target + 'CONTRA_DSR.p', 'wb'))
        elif label == 'IPSI-muscimol':
            pickle.dump(dataset, open(target + 'IPSI_DSR.p', 'wb'))
        elif label == 'OFC-muscimol':
            pickle.dump(dataset, open(target + 'OFC_DSR.p', 'wb'))
        elif label == 'PL-muscimol':
            pickle.dump(dataset, open(target + 'mPFC_DSR.p', 'wb'))
saline = pd.concat(saline, axis=0)
pickle.dump(saline, open(target + 'Saline_DSR.p', 'wb'))

#Shuffled dataset
choice = saline['Choice']
choice = np.random.permutation(choice)
saline.loc[:,'Choice'] = choice
pickle.dump(saline, open(target + 'shuffled.p', 'wb'))


#All PSR-Trained datasets: saline, mpfc, ofc, ipsi, contra
PSR_TRAINED = pickle.load(open(ROOT + \
        'DATA_structures/session_dataframes/PSR_SESSIONS_DATAFRAME.p', 'rb'))
saline = []
for label, dataset in PSR_TRAINED.groupby(axis = 0, level = 'Infusion_String'):
    if label.find('saline') >= 0:
        saline.append(dataset)
    else:
        if label == 'CONTRA-muscimol':
            pickle.dump(dataset, open(target + 'CONTRA_PSR.p', 'wb'))
        elif label == 'IPSI-muscimol':
            pickle.dump(dataset, open(target + 'IPSI_PSR.p', 'wb'))
        elif label == 'OFC-muscimol':
            pickle.dump(dataset, open(target + 'OFC_PSR.p', 'wb'))
        elif label == 'PL-muscimol':
            pickle.dump(dataset, open(target + 'mPFC_PSR.p', 'wb'))
saline = pd.concat(saline, axis=0)
pickle.dump(saline, open(target + 'Saline_PSR.p', 'wb'))

#rigged dataset - XOR
'''
We're gonna go session by session, and take the previous choice and reward,
and do an XOR operation to determine the next choice.
'''
rigged = []
for label, dataset in saline.groupby(axis = 0, level = 'Infusion_String'):
    for sess_label, sess in dataset.groupby(axis = 0, level = 'rat'):
        for trial in range(len(sess) - 1):
            sess['Choice'].iat[trial + 1] = \
                            sess['Choice'].iloc[trial] ^ sess['AR'].iloc[trial]
        rigged.append(sess)
rigged = pd.concat(rigged, axis = 0)
pickle.dump(rigged, open(target + 'XOR_rigged.p', 'wb'))
#rigged dataset - linear
'''
We're gonna go session by session, and do a "covert" pattern that a linear
model could pick up. Specifically, choice is gonna be EAST, WEST, WEST, WEST,
EAST, WEST, WEST, WEST, etc.
'''
rigged = []
for label, dataset in saline.groupby(axis = 0, level = 'Infusion_String'):
    for sess_label, sess in dataset.groupby(axis = 0, level = 'rat'):
        for trial in range(len(sess)):
            sess['Choice'].iat[trial] = int((trial % 4) > 0)
        rigged.append(sess)
rigged = pd.concat(rigged, axis = 0)
pickle.dump(rigged, open(target + 'linear_rigged.p', 'wb'))
