import os
import sys
import pickle
import itertools
import datetime
import numpy as np
import pandas as pd
from behavioral_performance.tools import SessionIteratorClass as SI
from behavioral_performance.tools import SessionClass as SC
idx = pd.IndexSlice

'''
This script creates the block summary dataframes for DSR, PSR, and Naive datasets
A variable with creating these is whether and how to center continuous variables
such as timing and uncertainty
'''

ROOT = os.environ['HOME'] + '/python/'

#defining some useful functions
perf = lambda sess: float(np.sum(sess['Correct'])) / np.shape(sess)[0]
cong = lambda sess: float(np.sum(sess['Correct'] == sess['AR'])) \
						/ np.shape(sess)[0]

def center_columns(dataframe, list_columns):
	for column in list_columns:
		dataframe[column] = dataframe[column] - np.nanmean(dataframe[column])
	return



infusionType = ['Saline', 'mPFC', 'OFC', 'Ipsi', 'Contra']
blocks = ['B1', 'B2']
categories = ['Perf',
			   'OPerf',
			   'Trials',
			   'Errors',
			   'Timing',
			   'Criteria',
			   'Trial_Left_Criteria',
			   'Order',
			   'Rev1',
			   'GA',
			   'Uncertainty',
			   'Animal',
			   'Notes',
			   'Infusion_String',
			   'path']

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
notes_to_infusion_string = {'OFC-saline' : 'OFC-saline',
							'OFC-muscimol' : 'OFC-muscimol',
							'OFC-Musc-Naive' : 'OFC-muscimol',
							'PL-saline' : 'PL-saline',
							'PL-muscimol' : 'PL-muscimol',
							'mPFC-Musc-Naive' : 'PL-muscimol',
							'ipsiLeft-saline' : 'IPSI-saline',
							'ipsiRight-saline' : 'IPSI-saline',
							'ipsiLeft-muscimol' : 'IPSI-muscimol',
							'ipsiRight-muscimol' : 'IPSI-muscimol',
							'rOFCxlPL-saline' : 'CONTRA-saline',
							'lOFCxrPL-saline' : 'CONTRA-saline',
							'rOFCxlPL-muscimol' : 'CONTRA-muscimol',
							'lOFCxrPL-muscimol' : 'CONTRA-muscimol',
							'Saline-Naive' : 'Saline-Naive'}


#load datasets
datasets = [('deterministic', 'Trained','DSR'),
			('probabilistic', 'Trained','PSR'),
			('deterministic', 'Naive', 'Naive')]
for regime, training, savefile in datasets:

	sessions = SI.SessionIterator([regime, training])


	index = \
	pd.MultiIndex.from_product([blocks, range(len(sessions.sessionList))],
								names = ['block','ids'])
	out = pd.DataFrame(np.full([len(blocks) * len(sessions.sessionList),
								len(categories)], np.NaN),
								index = index, columns = categories)



	index = [0] * len(blocks)
	for sessIndex, sessPath in enumerate(sessions.sessionList):
		sess = SC.Session(sessPath)
		for block in range(min(len(blocks),
								len(sess.info.groupby(axis=0, level='block')))):
			B1 = sess.info.loc[block + 1]
			#was this block completed ? if not, how far away was the animal
			if len(B1) < 12:
				eight = (np.sum(B1['Correct'][-8:]) == 8)
				nine = (np.sum(B1['Correct'][-11:]) >= 9)
				criteria = (eight or nine)
				block_criteria = 9
			elif len(B1) < 15:
				eight = (np.sum(B1['Correct'][-8:]) == 8)
				nine = (np.sum(B1['Correct'][-11:]) >= 9)
				ten = (np.sum(B1['Correct'][-12:]) >= 10)
				criteria = (eight or nine or ten)
				if criteria:
					if nine: block_criteria = 9
					if ten: block_criteria = 10
			else:
				eight = (np.sum(B1['Correct'][-8:]) == 8)
				nine = (np.sum(B1['Correct'][-11:]) >= 9)
				ten = (np.sum(B1['Correct'][-12:]) >= 10)
				twelve = (np.sum(B1['Correct'][-15:]) >= 12)
				criteria = (eight or nine or ten or twelve)
				if criteria:
					if nine: block_criteria = 9
					if ten: block_criteria = 10
					if twelve: block_criteria = 12
			if criteria: TLC = 0
			else: TLC = block_criteria - np.sum(B1['Correct'][-12:])

			#info for linear regression
			out.loc[idx[blocks[block], index[block]],'Perf'] = perf(B1)
			out.loc[idx[blocks[block], index[block]],'OPerf'] = perf(sess.info)
			out.loc[idx[blocks[block], index[block]],'Trials']= len(B1)
			out.loc[idx[blocks[block], index[block]],'Errors'] = \
													np.sum(B1['Correct'] == 0)
			out.loc[idx[blocks[block], index[block]],'Timing'] = \
					np.nanmean(np.sum(B1.loc[:,['t1','t2','t3','t4']], axis=1))
			out.loc[idx[blocks[block], index[block]],'Criteria'] = criteria
			out.loc[idx[blocks[block], index[block]],'Trial_Left_Criteria'] = TLC
			out.loc[idx[blocks[block], index[block]],'Order'] = 0
			out.loc[idx[blocks[block], index[block]],'Rev1'] = block
			out.loc[idx[blocks[block], index[block]],'GA'] = B1['GA'].iloc[0]
			out.loc[idx[blocks[block], index[block]],'Uncertainty'] = 1 - cong(B1)
			out.loc[idx[blocks[block], index[block]],'Animal'] = sess.ratID
			out.loc[idx[blocks[block], index[block]],'Notes'] = sess.header['notes']
			out.loc[idx[blocks[block], index[block]],'Infusion_String'] = \
								notes_to_infusion_string[sess.header['notes']]
			out.loc[idx[blocks[block], index[block]],'path'] = sessPath + '/'
			#check label is found
			assert not notes_to_infusion_string[sess.header['notes']] == None
			index[block] += 1



	#manually input order - i know, it sucks
	if regime == 'deterministic' and training == 'Trained':

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


	elif regime == 'probabilistic' and training == 'Trained':
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


	#drop empty rows
	out.dropna(inplace=True)
	center_columns(out, ['Timing'])
	#fill in ordering
	if training == 'Trained':
		#assert proper infusion string length
		assert len(set(out['Infusion_String'])) == 8
		for subjectIndex, subject in enumerate(subjects):
			for regimeIndex, noteString in enumerate(regimes):
				out.loc[(out['Animal'] == subject) & \
						(out['Notes'] == noteString),
						'Order'] = ordering[subjectIndex, regimeIndex]
	elif training == 'Naive':
		#assert proper infusion string length
		assert len(set(out['Infusion_String'])) == 3
		for id in range(len(out)):
			path = out.iloc[id]['path']

			date = path[[i for i,c in enumerate(path) if c=='/'][-2]:].replace('/','')
			#converting date to datetime object
			DMY = [int(w) for w in date.split('_')]
			date = datetime.date(DMY[-1],DMY[-2],DMY[-3])

			folderNames = \
				os.listdir(path[:[i for i,c in enumerate(path) if c=='/'][-2]])
			#removing the annoying .DS_Store
			folderNames = [w for w in folderNames if not w.startswith('.')]

			dateList = []
			for dateIndex, date_string in enumerate(folderNames):
				DMY = [int(w) for w in date_string.split('_')]
				dateList.append(datetime.date(DMY[-1],DMY[-2],DMY[-3]))

			sorted_dates = sorted(range(len(dateList)), key=lambda k: dateList[k])
			sorted_dates = [dateList[w] for w in sorted_dates]

			Order = sorted_dates.index(date)
			out.iloc[id, out.columns.get_loc('Order')] = Order





	#convert appropriate columns to integers, no need for floats
	for column in ['Trials','Errors','Criteria','Trial_Left_Criteria',
				   'Order', 'Rev1', 'GA']:
		out[column] = out[column].astype('int16')

	#save the results
	pickle.dump(out, open('DATA_structures/block_dataframe/' \
										+ savefile + '_BLOCK_DATAFRAME.p','wb'))
