import sys
sys.path.append('/Users/pablomartin/python/sutils/')
import os
import pickle
import pandas as pd
import numpy as np
import sutils.config
from sutils.session import session
from trial_by_trial_dataframe import session2dataframe

ROOT = sutils.config.ROOT
source = ROOT + 'cleanDATA/Naive/'
target = ROOT + 'DATA_structures/TbyT/'

labels = {'Saline-Naive' : 'Naive_Saline.p',
          'mPFC-Musc-Naive' : 'Naive_mPFC.p',
          'OFC-Musc-Naive': 'Naive_OFC.p'}

saline = []
mpfc = []
ofc = []
sal_index, mpfc_index, ofc_index = 0, 0, 0

for animal in [f for f in os.listdir(source) if not f.startswith('.')]:
    animal_dir = source + str(animal)
    for day in [g for g in os.listdir(animal_dir) if not g.startswith('.')]:
        sess = session('/'.join([animal_dir, day]))

        if sess.header['notes'] == 'Saline-Naive' :
            df_sess = session2dataframe(sess, sal_index)
            saline.append(df_sess)
            sal_index += 1
        if sess.header['notes'] == 'mPFC-Musc-Naive' :
            df_sess = session2dataframe(sess, mpfc_index)
            mpfc.append(df_sess)
            mpfc_index += 1
        if sess.header['notes'] == 'OFC-Musc-Naive' :
            df_sess = session2dataframe(sess, ofc_index)
            ofc.append(df_sess)
            ofc_index += 1

saline = pd.concat(saline, axis = 0)
mpfc = pd.concat(mpfc, axis = 0)
ofc = pd.concat(ofc, axis = 0)
saline.sort_index(axis = 0, inplace = True)
mpfc.sort_index(axis = 0, inplace = True)
ofc.sort_index(axis = 0, inplace = True)

pickle.dump(saline, open(target + 'DSR_TbyT_Naive_Saline.p', 'wb'))
pickle.dump(mpfc, open(target + 'DSR_TbyT_Naive_mPFC.p', 'wb'))
pickle.dump(ofc, open(target + 'DSR_TbyT_Naive_OFC.p', 'wb'))
