import os
import pandas as pd
import numpy as np
ROOT = os.environ['HOME'] + '/python/'

'''
Given a log.txt file, it returns a dictionary with the header information and
a pandas dataframe with all the session data
'''


#default session is a representative session
def session_text_extractor(target = ROOT + "/cleanDATA/Cohort2/1/2_3_2017/"):
	if not os.path.isdir(target):
		print 'directory does not exist. exiting ...'
		return


	out = {'rat':'','cohort':'','task':'','notes':'','date':''};
	fields = ['SA:','GA:','C:','AR:','Ch1:','Ch2:',
			  'E:','S:','P:','t1:','t2:','t3:','t4:','TT:']

	with open(target + "log.txt") as file:
		text = file.readlines();

		noTrials = len(text);

		info = np.full([noTrials, 14], np.NaN)
		realTrial = 0

		for i in range(noTrials):

			#getting rat #
			if text[i][0:3] == 'Rat':
				out['rat'] = text[i][5:-1];
			#getting Cohort
			if text[i][0:5] == 'Exper':
				out['cohort'] = \
				text[i][str.find(text[i],'Cohort') + 6:str.find(text[i],'\n')]
			if text[i][0:4] == 'Regi':
				out['task'] = \
				text[i][str.find(text[i],'Regime')+8: str.find(text[i],'\n')]
			if text[i][0:5] == 'Notes':
				out['notes'] = \
				text[i][str.find(text[i],'Notes') + 7:str.find(text[i],'\n')]
			if text[i][0:5] == 'Saved':
				out['date'] = \
				text[i][-str.find(text[i][::-1],'/'):str.find(text[i],'\n')]


			#sorting thru the trial
			if text[i][0] == 't':
				#gets trial
				trial = text[i];
				#Start ARM

				for fieldIndex, field in enumerate(fields):
					if trial.find(field) > 0:
					 	info[realTrial][fieldIndex] = \
						float(trial[trial.find(field) + len(field) : \
							  trial.find(' ', trial.find(field))])
				#in DRL, AR is just C
				if np.isnan(info[realTrial][3]):
					info[realTrial][3] = info[realTrial][2];

				realTrial += 1


		#trimming data
		info = info[:realTrial, :]

		#create panda data frames
		dfINTinfo = pd.DataFrame(info[:,:-5],
								 columns = ['SA','GA','Correct','AR','Choice1',
								 			'Choice2', 'Error', 'Sensor',
											'Phase'],
								 dtype = 'int')
		dfFLOAT = pd.DataFrame(info[:,-5:],
							   columns = ['t1','t2','t3','t4','TT'],
							   dtype='float')

		dfINFO = pd.concat([dfINTinfo, dfFLOAT], axis = 1)


		return out, dfINFO
