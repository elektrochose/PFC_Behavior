import config
import os
import sys

import numpy as np
import pickle
import pandas as pd
import datetime
from session import session

ROOT = config.ROOT
#defining some useful functions
perf = lambda sess: float(np.sum(sess['Correct'])) / np.shape(sess)[0]
criteria = lambda sess: np.sum(sess['Correct'][-12:])


def my_date_sorter(folderNames):
	dateList = []
	for dateIndex, date_string in enumerate(folderNames):
		DMY = [int(w) for w in date_string.split('_')]
		dateList.append(datetime.date(DMY[-1],DMY[-2],DMY[-3]))

	index = sorted(range(len(dateList)), key=lambda k: dateList[k])
	return index

def calculate_timing(info):
	return np.nanmean(info[['t1','t2','t3','t4']].apply(np.sum, axis=1))

DATA = ROOT + "/cleanDATA/Naive/"
TARGET = ROOT + "/DATA_structures/"
Naive = dict()


categories = ['Performance',
			   'Overall_Perf',
			   'TTC',
			   'Timing',
			   'Criteria',
			   'Order',
			   'Rev1',
			   'GA',
			   'Uncertainty',
			   'Animal',
			   'Notes',
			   'Infusion_String']

Naive = pd.DataFrame(np.full([300, len(categories)], np.NaN),
													columns = categories)

index = 0
#going thru available files
for folder in os.listdir(DATA):
	#figuring out the date order of infusions
	dateOrder = my_date_sorter(os.listdir(DATA + folder))
	for fileIndex, fileInFolder in enumerate(os.listdir(DATA + folder)):
		#load the session
		sess = session(DATA + folder + '/' + fileInFolder)
		for block in range(1,3):
			B1 = sess.get_block(block)
			if np.shape(B1)[0] > 10:

				Naive.loc[index,'Performance'] = perf(B1)
				Naive.loc[index,'Overall_Perf'] = perf(sess.info)
				Naive.loc[index,'TTC'] = np.shape(B1)[0]
				Naive.loc[index,'Criteria'] = criteria(B1)
				Naive.loc[index,'Order'] = dateOrder[fileIndex] + 1
				Naive.loc[index,'Rev1'] = block - 1
				Naive.loc[index,'GA'] = B1['GA'].iloc[0]
				Naive.loc[index,'Uncertainty'] = 0
				Naive.loc[index,'Animal'] = sess.ratID
				Naive.loc[index,'Timing'] = calculate_timing(B1)
				Naive.loc[index,'Notes'] = sess.header['notes']
				if sess.header['notes'][0] == 'S':
					sess.header['notes'] = 'AA' + sess.header['notes']
				Naive.loc[index,'Infusion_String'] = \
					sess.header['notes'][0:sess.header['notes'].find('-')]
				index += 1
Naive = Naive[:index]

pickle.dump(Naive, open(TARGET + 'Naive_DATAFRAME.p','wb'))
