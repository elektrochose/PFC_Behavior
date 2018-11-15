'''
We will analyze the results of the 100 Model Grids obtained from the 100 data
shuffles. First step, is to average results together and plot the *definitive*
plot. Next step will be to plot only the results of the simplified model grid.
This will only include saline, mPFC, Contra and Naive.

Next step will be devising some kind of statistic. Maybe some kind of clustering
analysis. We will have to do more research on this
'''

import os
import pickle
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from Visualize.decoding import plot_model_grid_general
import seaborn as sns



ROOT = os.environ['HOME'] + '/python/'
source = ROOT + 'behavioral_performance/Results/MODEL_GRIDS/'

GRID_DICT = {}
for mg in [w for w in os.listdir(source) if w.find('MODEL')==0]:
    print mg
    GRID = pickle.load(open(source + mg, 'rb'))
    GRID_DICT[mg] = GRID

panel = pd.Panel(GRID_DICT)
MASTER_GRID = panel.mean(axis=0)
plot_model_grid_general(MASTER_GRID, vrange=[0.5, 0.75])


'''
"mirrors" each matrix by taking average between 2 values opposite of the
diagonal. For example, PSR Saline decoding DSR mPFC and DSR mPFC decoding
PSR Saline.
'''
def transform_matrix(grid):
    M = 1 - grid.values
    D = np.full(M.shape, np.NaN)
    for i,j in itertools.product(range(len(M)), range(len(M))):
        D[i,j] = (M[i,j] + M[j,i]) / 2
    return D


X = 1 - panel.to_frame()
tsne = TSNE(n_components=2, verbose=1)
tsne_reduct = tsne.fit_transform(X.values)
tsne_reduct = pd.DataFrame(tsne_reduct, index=X.index)


#for master grid
M = transform_matrix(1 - MASTER_GRID)
tsne = TSNE(n_components=2, verbose=1, metric='precomputed')
tsne_reduct = tsne.fit_transform(M)
tsne_reduct = pd.DataFrame(tsne_reduct, index=MASTER_GRID.index)
tsne_reduct.index.set_names(['major'], inplace=True)


colors = get_spaced_colors(19)
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080', '#ffffff', '#000000']


fig, ax = plt.subplots(figsize=(20,12))
for i, (label, data) in enumerate(tsne_reduct.groupby('major')):
    #cc = tuple([np.float(w)/255 for w in colors[i]])
    cc = colors[i]
    ax.scatter(data.values[:,0], data.values[:,1],
                                color=cc,
                                label=label,
                                alpha = 0.9,
                                s=150)
ax.legend()
plt.xlabel('x-tSNE', fontSize=14)
plt.ylabel('y-tSNE', fontSize=14)
plt.title('t-SNE Visualization of Similarity Scores', fontSize=18)
plt.show()



'''
Let's try DSR only
'''
fig, ax = plt.subplots(figsize=(20,12))
j = 0
for i, (label, data) in enumerate(tsne_reduct.groupby('major')):
    if label.find('DSR')>=0:
        cc = colors[j]
        j += 1
        ax.scatter(data.values[:,0], data.values[:,1],
                                    color=cc,
                                    label=label,
                                    alpha = 0.8,
                                    s=150)
ax.legend()
plt.xlabel('x-tSNE', fontSize=14)
plt.ylabel('y-tSNE', fontSize=14)
plt.title('t-SNE Visualization of Similarity Scores', fontSize=18)
plt.show()
