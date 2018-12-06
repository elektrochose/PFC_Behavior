import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig

ROOT = os.environ['HOME'] + '/python/'

def plot_session_curve(window_size = 7,
					   title = 'Session Learning Curve',
					   large_plot = True,
					   DPI = 600,
					   savefig = 0,
					   plotErrors = 1,
					   **kwargs):

	'''
	Argument checking - there must be an argument session = sess, where sess is
	a dataframe containing a session. There can be additional arguments that
	must be specified via fields and must also be session files. This is done
	to compare model predictions right in the same plot. So future models will
	have to produce a dataframe with a 'Choice' field, and same row index as
	session, and it will be plotted alongside actual session.
	'''
	if not 'session' in kwargs.keys():
		print 'session dataframe must be included always as session = sess.'
		return
	else:
		if len(kwargs) > 1:
			for sess_included in range(len(kwargs) - 1):
				assert len(kwargs.values()[sess_included]) \
					== len(kwargs.values()[sess_included + 1])
			errorDecoding = \
				(kwargs.values()[0]['Choice'] != kwargs.values()[1]['Choice'])
			errorDecoding = np.nonzero(errorDecoding)[0]
		session = kwargs['session']



	noTrials = len(session)
	learningCurve = dict((key, []) for key in kwargs.keys())

	for trial in range(noTrials):
		b_index = int(max(0, trial - np.floor(window_size/2)))
		t_index = int(min(noTrials, trial + np.ceil(np.float(window_size)/2)))
		for key, session in kwargs.items():
			mov_avg = np.float(np.sum(
					session['Choice'].iloc[b_index:t_index])) / window_size
			learningCurve[key].append(mov_avg)

	#smoothing curves
	for key, curve in learningCurve.items():
		#smoothing with savgol filter
		smooth_signal = sig.savgol_filter(learningCurve[key], window_size, 2)
		#clipping below 0 and above 1
		smooth_signal = [max(0, w) for w in smooth_signal]
		smooth_signal = [min(1, w) for w in smooth_signal]
		learningCurve[key] = smooth_signal




	block_colors = ['green', 'purple']
	#determining where reversals happened
	blocks = session.groupby(axis = 0, level = 'block')
	block_lims = [(len(b),block_colors[list(set(b['GA']))[0]]) for l,b in blocks]
	abs_block_lims = [0] + list(np.cumsum([b[0] for b in block_lims]))
	block_lims = \
	[(abs_block_lims[i], abs_block_lims[i+1], b[1]) for i, b in enumerate(block_lims)]

	'''
	Plotting starts here
	'''
	color_palette = ['r', 'c', 'm', 'k']
	if large_plot:
		fig, ax = plt.subplots(figsize=(20,10))
	else:
		fig, ax = plt.subplots(figsize=(10,6))
	for key, curve in learningCurve.items():
		if key == 'session':
			curve_color = 'blue'
		else:
			curve_color = color_palette.pop(0)

		ax.plot(curve, color = curve_color, label = key)

	plt.xlim(0, noTrials)
	plt.ylim(0, 1)
	ylabels = ['%.1f' %w for w in np.arange(0,1.1,0.1)]
	plt.yticks(np.arange(0, 1, .1), ylabels)
	plt.xticks(np.arange(0, noTrials, 10),
		  list(np.arange(0, noTrials, 10)))
	ax.plot([0, noTrials],
			 [0.5, 0.5],
			 'k:', label = 'Chance')
	#plotting all the reversal lines
	colornax = ['green','purple']
	for start, end, colorblock in block_lims:
		ax.axvspan(start, end, facecolor=colorblock, alpha=0.2)

	#plot where the decoder made a mistake
	if len(kwargs) > 1 and plotErrors:
		for l in errorDecoding:
			if l >= 0 and l < noTrials:
				plt.text(l, 0.5, 'x', color = 'red',
						 verticalalignment = 'center',
						 horizontalalignment = 'center',
						 fontsize=16)

	plt.xlabel('Trial #')
	plt.ylabel('P(Choosing W)')
	plt.legend()
	plt.title(title, FontSize=24)
	if savefig: plt.savefig(title + '.jpg', dpi = DPI, bbox_inches = 'tight')
	plt.show()
