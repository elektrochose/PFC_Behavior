import config
import sys
import numpy as np
import pickle
import pandas as pd
import time
import re
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


ROOT = config.ROOT
idx = pd.IndexSlice

LR1 = LogisticRegression(penalty = 'l1', solver = 'liblinear')
LR2 = LogisticRegression(penalty = 'l2', solver = 'liblinear')
SVM = svm.SVC()
SVMlinear = svm.LinearSVC()
treeModel = tree.DecisionTreeClassifier()

modelList = [LR1, LR2, SVM, SVMlinear, treeModel]






def coeff_plot(coeffs, features, T, title='Regression Coefficients'):
    grid = np.reshape(coeffs, [len(features), T]).transpose()
    plt.imshow(grid)
    plt.xticks(range(len(features)), features, rotation=90)
    plt.yticks(range(T), ['n - %i' %w for w in range(1, T + 1)])
    plt.colorbar()
    plt.title(title)
    plt.show()

def unique_string(task, dataset, perm, T):
    return "%s_%s_%s_%s" %(task, dataset, perm, T)
def inverse_unique_string(unique_string):
    stops = [m.start() for m in re.finditer('_',unique_string)]
    title = unique_string[:stops[1]]
    perm = int(unique_string[stops[1] + 1: stops[2]])
    T = int(unique_string[stops[2] + 1:])
    return title, perm, T





taskLabels = ['DSR','PSR']
dataLabels = ['Saline', 'MPFC', 'OFC', 'Ipsi', 'Contra']
modelLabels = ['LR1','LR2','SVM','SVMlinear','tree']
features = ['choice', 'reward','SA','RTs','choice_AR']

row_index = pd.MultiIndex.from_product([range(1, 11),
                                        range(0,2 ** len(features) - 1)],
                                        names = ['T','perm'])
column_index = pd.MultiIndex.from_product([taskLabels, dataLabels, modelLabels],
                                        names = ['task','dataset','model'])

modelDataFrame = \
pd.DataFrame(np.full([len(row_index),len(column_index)], np.NaN),
                                        index = row_index,
                                        columns=column_index)


perms5 = [format(w, "05b") for w in range(1, 2 ** len(features))]
featurePerms = []
for mybool in perms5:
    featurePerms.append([features[int(i)]  \
              for i,m in enumerate(list(mybool)) if int(m) > 0])


#fitting all models
coeffRep = dict()
for task in taskLabels:
    for dataLabel in dataLabels:
        start = time.time()
        print "loading %s-%s dataset..." %(task,dataLabel)
        df = pickle.load(open(config.ROOT
                              + '/DATA_structures/'
                              + task + '_TbyT_' \
                              + dataLabel + '.p', 'rb'))

        df.sort_index(axis = 0, inplace = True)
        df.sort_index(axis = 1, inplace = True)

        #X will vary depending on the model
        for slidingT in range(1, 11):
            sessionData = idx[:,:,slidingT:]
            #Y is always idx['choice',0] - we do not change
            Y = df.loc[sessionData, idx['choice',0]]
            for subsetIndex, subset in enumerate(featurePerms):
                currentSlice = idx[subset, 1:slidingT]
                #slice out the data
                X = df.loc[sessionData, currentSlice]
                for modelIndex, model in enumerate(modelList):

                    modelLabel = modelLabels[modelIndex]
                    modelDataFrame[task, dataLabel, modelLabel]\
                                    .loc[slidingT, subsetIndex]\
                    = np.mean(cross_val_score(model, X, Y, cv=10))

        end = time.time()
        print 'processing time: %1.2f minutes' %((end-start)/60)

modelDataFrame.sort_index(axis = 0, inplace=True)
modelDataFrame.sort_index(axis = 1, inplace=True)


pickle.dump(modelDataFrame, \
        open('DATA_structures/feature_selection_DF.p','wb'))


#retrieving the models that performed the best
dict_of_models = dict()
for modelLabel in modelLabels:
    best_models = dict()
    dict_of_models[modelLabel] = best_models
    for task in taskLabels:
        for dataLabel in dataLabels:

            a,b = modelDataFrame[task, dataLabel, modelLabel].idxmax()
            best_models[unique_string(task, dataLabel, b, a)] = \
                            modelDataFrame[task,dataLabel, modelLabel].max()

#printing results
for modelLabel in dict_of_models.keys():
    print "--------MODEL:%s ------" %modelLabel
    for key in dict_of_models[modelLabel].keys():
        print 'score:%1.4f for model %s' \
                %(dict_of_models[modelLabel][key], key)





# #plotting coefficients
# for modelIndex, model in enumerate(best_models.keys()):
#     coeffs = coeffRep[model]
#     title, b, T = inverse_unique_string(model)
#     coeff_plot(coeffs, featurePerms[b], T, title = title)





#mask = np.abs(coeffs) > np.mean(coeffs) + alpha * np.std(coeffs)
