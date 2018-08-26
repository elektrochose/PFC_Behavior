
import config
import os
import sys
import pickle
import copy
import numpy as np
from session_iterator import SessionIterator
from session import session
from datetime import datetime as dt




ROOT = config.ROOT
TARGET = ROOT + '/DATA_structures/'


def gather_training_data():

	DATA = ROOT + "/cleanDATA/CohortX/"
	trainingData = dict()
	for cohort in range(2,6):
		#load the right cohort
		DATA = DATA[:-2] + str(cohort) + '/'
		#going thru available files
		for folder in os.listdir(DATA):
			for fileInFolder in os.listdir(DATA + folder):

				#load the session
				print DATA + folder + '/' + fileInFolder
				sess = session(DATA + folder + '/' + fileInFolder)

				#filer non-training session, must have at least 40 trials
				if np.shape(sess.info)[0] > 40 \
				and sess.header['notes']=='TRAINING':
					#making sure animal is in the dictionary
					if sess.ratID not in trainingData.keys():
						trainingData[sess.ratID] = []
					#add session to list!
					trainingData[sess.ratID].append(sess)


	badData = ['22','30','32','52','53']
	for rat in badData:
		trainingData.pop(rat,'None')

	#sort sessions by date
	for rat in trainingData.keys():
		trainingData[rat].sort(key=lambda x: x.dateObj)

	#save file
	pickle.dump(trainingData, open(TARGET + 'trainingData.p', 'wb'))
	return


def naive_experiment():
	DATA = ROOT + "/cleanDATA/Naive/"
	TARGET = ROOT + "/DATA_structures/"
	Naive = dict()
	#going thru available files
	for folder in os.listdir(DATA):
		for fileInFolder in os.listdir(DATA + folder):
			sess = session(DATA + folder + '/' + fileInFolder)

			if sess.ratID not in Naive.keys():
				Naive[sess.ratID] = []
			Naive[sess.ratID].append(sess)
	pickle.dump(Naive, open(TARGET + 'Naive.p', 'wb'))


if __name__ == '__main__':

	gather_training_data()
	naive_experiment()
	TD = pickle.load(open(TARGET + 'trainingData.p', 'rb'))
	detList = [w for w in TD.keys() if w[0]=='2' or w[0]=='4']
	probList = [w for w in TD.keys() if w[0]=='3' or w[0]=='5']
	whichList = [detList, probList]


	#28 is the max number of sessions for PSR, 21 have enough animals though

	category_labels = ['Training 1',
					   'Training Mid',
					   'Training Final',
					   'Saline',
					   'mPFC Muscimol',
					   'OFC Muscimol']



	dataset_labels = ['DSR_Half_Saline.p',
					  'DSR_Full_Saline.p',
					  'PSR_Half_Saline.p',
					  'PSR_Full_Saline.p']

	for DLI, regime in enumerate(['deterministic', 'probabilistic']):

		TRAINING_FIRST = []
		TRAINING_MIDDLE = []
		TRAINING_LAST = []
		#training only, no infusions (first, middle, and last session)
		for rat in whichList[regime=='probabilistic']:
			tmpSessionList = TD[rat]
			mid = int(np.floor(len(tmpSessionList)/2))
			TRAINING_FIRST.append(tmpSessionList[0])
			TRAINING_MIDDLE.append(tmpSessionList[mid])
			TRAINING_LAST.append(tmpSessionList[-1])


		#these are the infusion sessions
		SALINE_mPFC_OFC = SessionIterator(['PL','OFC',regime,'saline'])
		preSHUFFLE = SessionIterator(['PL','OFC',regime,'saline'])
		SALINE_COMBINED = SessionIterator([regime,'saline'])
		mPFC_MUSCIMOL = SessionIterator(['PL',regime,'muscimol'])
		OFC_MUSCIMOL = SessionIterator(['OFC',regime,'muscimol'])


		#shuffling the shuffled dataset
		SHUFFLE = []
		b1 = []; rev1 = []
		for sessIndex, sess in enumerate(preSHUFFLE.sessionList):
			tmpb1 = np.shape(sess.get_block(1))[0]
			tmprev1 = np.shape(sess.get_block(2))[0]
			if tmpb1 > 10:
				b1.append(tmpb1)
			if tmprev1 > 10:
				rev1.append(tmprev1)
			tmpInfo = sess.info.sample(frac = 1)
			tmpSess = copy.deepcopy(sess)
			tmpSess.info = tmpInfo
			tmpSess.shuffled = 1
			SHUFFLE.append(tmpSess)
		for sessIndex, sess in enumerate(SHUFFLE):
			sess.b1 = int(np.floor(np.nanmean(b1)))
			sess.rev1 = int(np.floor(np.nanmean(rev1)))


		out_half = [SHUFFLE,
				   TRAINING_FIRST,
				   TRAINING_MIDDLE,
				   TRAINING_LAST,
				   SALINE_mPFC_OFC.sessionList,
				   mPFC_MUSCIMOL.sessionList,
				   OFC_MUSCIMOL.sessionList]

		out_full = [SHUFFLE,
				   TRAINING_FIRST,
				   TRAINING_MIDDLE,
				   TRAINING_LAST,
				   SALINE_COMBINED.sessionList,
				   mPFC_MUSCIMOL.sessionList,
				   OFC_MUSCIMOL.sessionList]

		pickle.dump(out_half, open(TARGET + dataset_labels[0 + 2 * DLI], 'wb'))
		pickle.dump(out_full, open(TARGET + dataset_labels[1 + 2 * DLI], 'wb'))
