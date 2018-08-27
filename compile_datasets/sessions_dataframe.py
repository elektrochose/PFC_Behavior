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
SessionIterator = SI.SessionIterator
Session = SC.Session

'''
This script creates a dataframe with all sessions of a given datasets. The
sessions are concatenated into one pandas dataframe which can then be searched
or groupby either animal, or type of session. This script creates 5 datsets:

1) DSR NAIVE
2) DSR TRAINING
3) PSR TRAINING
4) DSR TRAINED
5) PSR TRAINED
'''

ROOT = os.environ['HOME'] + '/python/'
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


datasets = [('deterministic', 'Trained', 'DSR'),
			('probabilistic', 'Trained', 'PSR'),
			('deterministic', 'Naive', 'Naive'),
			('deterministic','Training', 'DSR_TRAINING'),
			('probabilistic','Training', 'PSR_TRAINING')]

for regime, training, savefile in datasets:

	print "loading %s sessions..." %regime,
	session_iterator = SessionIterator([regime, training])
	print "done"

	if regime == 'deterministic' and training == 'Trained':
		assert len(session_iterator.sessionList) == 88
	elif regime == 'probabilistic' and training == 'Trained':
		assert len(session_iterator.sessionList) == 82
	elif regime == 'deterministic' and training == 'Naive':
		assert len(session_iterator.sessionList) == 21


	sessions = [Session(w) for w in session_iterator.sessionList]
	rats = list(set([sess.ratID for sess in sessions]))

	out = []
	for rat in rats:
		rat_array = []
		infusion_type = []
		block_array = []
		trial_array = []
		tmp_sessions = [sess for sess in sessions if sess.ratID == rat]
		for sess_index, sess in enumerate(tmp_sessions):
			for block, trial in zip(*sess.info.index.labels):
				rat_array.append(rat)

				if training == 'Training':
					sessionID = 'training_session'
					val_to_add = 1 + sess_index
				else:
					sessionID = 'Infusion_String'
					val_to_add = notes_to_infusion_string[sess.header['notes']]

				infusion_type.append(val_to_add)
				block_array.append(sess.info.index.levels[0][block])
				trial_array.append(sess.info.index.levels[1][trial])

		rows = pd.MultiIndex.from_arrays([rat_array,
										  infusion_type,
										  block_array,
										  trial_array],
										  names=('rat', sessionID,
												 'block','trial'))
		vals = np.concatenate([sess.info.values for sess in tmp_sessions])
		out.append(pd.DataFrame(vals, index=rows, columns = sess.info.columns))

	out = pd.concat(out, axis = 0)
	#convert appropriate columns to integers, no need for floats
	for column in ['SA','GA','Correct','AR',
				   'Choice1', 'Choice2']:
		out.loc[:,column] =  out.loc[:,column].astype('int16')
	out.sort_index(axis = 0, inplace = True)
	out.sort_index(axis = 1, inplace = True)
	pickle.dump(out, open(ROOT + 'DATA_structures/session_dataframes/' \
							+ savefile + '_SESSIONS_DATAFRAME.p', 'wb'))
