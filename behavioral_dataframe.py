import config
import sys
import numpy as np
import pickle
from session_iterator import SessionIterator
import pandas as pd


ROOT = config.ROOT
#defining some useful functions
perf = lambda sess: float(np.sum(sess['Correct'])) / np.shape(sess)[0]
cong = lambda sess: float(np.sum(sess['Correct'] == sess['AR'])) \
                        / np.shape(sess)[0]

def center_columns(dataframe, list_columns):
    for column in list_columns:
        dataframe[column] = dataframe[column] - np.nanmean(dataframe[column])
    return


#load datasets
for regime in ['deterministic','probabilistic']:

    saline = SessionIterator([regime, 'saline'])
    mpfc = SessionIterator([regime,'muscimol','PL'])
    ofc = SessionIterator([regime,'muscimol','OFC'])
    ipsi = SessionIterator([regime,'muscimol','ipsi'])
    contra = SessionIterator([regime,'muscimol','x'])


    infusionType = ['AASaline', 'mPFC', 'OFC', 'Ipsi', 'Contra']
    datasets = [saline, mpfc, ofc, ipsi, contra]
    categories = ['Performance',
                   'Overall_Perf',
                   'mPFC',
                   'OFC',
                   'ipsi',
                   'contra',
                   'Order',
                   'Rev1',
                   'GA',
                   'Uncertainty',
                   'Animal',
                   'Notes',
                   'Infusion_String']

    out = pd.DataFrame(np.full([300, len(categories)], np.NaN),
                                                    columns = categories)
    regimes = ['OFC-muscimol',
                 'OFC-saline',
                 'PL-muscimol',
                 'PL-saline',
                 'ipsiLeft-muscimol',
                 'ipsiLeft-saline',
                 'ipsiRight-muscimol',
                 'ipsiRight-saline',
                 'lOFCxrPL-muscimol',
                 'lOFCxrPL-saline',
                 'rOFCxlPL-muscimol',
                 'rOFCxlPL-saline']

    index = 0
    for condIndex, condition in enumerate(datasets):

        betas = [0, 0, 0, 0, 0]
        betas[condIndex] = 1

        for sessIndex, sess in enumerate(condition.sessionList):
            for block in range(1,3):
                B1 = sess.get_block(block)


                #info for linear regression
                out.loc[index,'Performance'] = perf(B1)
                out.loc[index,'Overall_Perf'] = perf(sess.info)
                out.loc[index,'mPFC']= betas[1]
                out.loc[index,'OFC'] = betas[2]
                out.loc[index,'ipsi'] = betas[3]
                out.loc[index,'contra'] = betas[4]
                out.loc[index,'Order'] = 0 #?????????
                out.loc[index,'Rev1'] = block - 1
                out.loc[index,'GA'] = B1['GA'].iloc[0]
                out.loc[index,'Uncertainty'] = 1 - cong(B1)
                out.loc[index,'Animal'] = sess.ratID
                out.loc[index,'Notes'] = sess.header['notes']
                out.loc[index,'Infusion_String'] = infusionType[condIndex]
                index += 1



    #manually input order - i know, it sucks
    if regime == 'deterministic':

        subjects = ['21', '23', '24', '25', '26',
                    '41', '43', '44', '45', '46', '47']

        ordering = np.full([len(subjects), len(regimes)], np.NaN)
        ordering[0,:] = [3,4,2,1,6,5,6,5,8,7,8,7]
        ordering[1,:] = [3,4,2,1,5,6,5,6,7,8,7,8]
        ordering[2,:] = [4,3,1,2,5,6,5,6,8,7,8,7]
        ordering[3,:] = [3,4,1,2,6,5,6,5,7,8,7,8]
        ordering[4,:] = [4,3,1,2,5,6,5,6,7,8,7,8]
        ordering[5,:] = [1,2,4,3,7,8,7,8,5,6,5,6]
        ordering[6,:] = [3,4,2,1,7,8,7,8,6,5,6,5]
        ordering[7,:] = [1,2,3,4,7,8,7,8,6,5,6,5]
        ordering[8,:] = [2,1,4,3,8,7,8,7,5,6,5,6]
        ordering[9,:] = [4,3,1,2,8,7,8,7,6,5,6,5]
        ordering[10,:] = [2,1,3,4,8,7,8,7,5,6,5,6]


    elif regime == 'probabilistic':
        #centering uncertainty
        center_columns(out, ['Uncertainty'])
        subjects = ['31', '32', '33', '34', '35', '36',
                    '51', '54', '55', '56', '57']

        ordering = np.full([len(subjects), len(regimes)], np.NaN)
        ordering[0,:] = [3,4,1,2,8,7,8,7,6,5,6,5]
        ordering[1,:] = [0,0,2,1,0,0,0,0,0,0,0,0]
        ordering[2,:] = [4,3,1,2,8,7,8,7,5,6,5,6]
        ordering[3,:] = [3,4,2,1,7,8,7,8,6,5,6,5]
        ordering[4,:] = [3,4,2,1,7,8,7,8,5,6,5,6]
        ordering[5,:] = [4,3,1,2,7,8,7,8,6,5,6,5]
        ordering[6,:] = [6,5,7,8,2,1,2,1,4,3,4,3]
        ordering[7,:] = [2,1,3,4,8,7,8,7,6,5,6,5]
        ordering[8,:] = [1,2,3,4,7,8,7,8,6,5,6,5]
        ordering[9,:] = [5,6,8,7,1,2,1,2,4,3,4,3]
        ordering[10,:] = [1,2,4,3,7,8,7,8,5,6,5,6]


    #fill in ordering
    for subjectIndex, subject in enumerate(subjects):
        for regimeIndex, noteString in enumerate(regimes):
            out.loc[(out['Animal'] == subject) & \
                    (out['Notes'] == noteString),
                    'Order'] = ordering[subjectIndex, regimeIndex]

    #trim the fat
    out = out[:index]


    #save the results
    if regime == 'deterministic':
        pickle.dump(out, open('DATA_structures/DSR_DATAFRAME.p','wb'))
    elif regime == 'probabilistic':
        pickle.dump(out, open('DATA_structures/PSR_DATAFRAME.p','wb'))
