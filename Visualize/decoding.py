import numpy as np
import pandas as pd
from scipy import signal as sig
import matplotlib.pyplot as plt




def index_to_labels(index):
	if type(index) == pd.core.indexes.base.Index:
		return [w.replace('_',' ') for w in index.values]
	elif type(index) == pd.core.indexes.multi.MultiIndex:
		return [' '.join(w) for w in index.values]



def plot_model_grid_general(df,
 							vrange = [0.5, 0.8],
							title = 'Model Decoding',
							savefig = 0,
							DPI = 400):

	row_labels = index_to_labels(df.index)
	col_labels = index_to_labels(df.columns)

	fig, ax = plt.subplots(figsize=(10,6))
	grid = ax.imshow(df.values, vmin = vrange[0], vmax = vrange[1])

	plt.xticks(np.linspace(0, len(row_labels) - 1, len(row_labels)),
											row_labels, rotation=90)
	plt.yticks(np.linspace(0, len(col_labels) - 1, len(col_labels)),
											col_labels, rotation=0)
	plt.ylabel('Model Trained on', fontsize=14)
	plt.xlabel('Dataset Decoded', fontSize=14)
	ax.xaxis.tick_top()
	ax.yaxis.set_label_position("right")
	ax.yaxis.label.set_rotation(270)
	ax.yaxis.labelpad = 15
	cbar = fig.colorbar(grid)
	cbar.ax.get_yaxis().labelpad = 15
	cbar.ax.set_ylabel('Decoding Accuracy', rotation = 270)
	plt.title(title + '\n\n', fontSize=18, y=1.2)
	if savefig: plt.savefig('MODEL_GRID.jpg', dpi = DPI, bbox_inches = 'tight')
	plt.show()
	return
