import config
import sys
import numpy as np
import pickle
import pandas as pd
import time


import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn import svm
from sklearn import tree
from sklearn.feature_selection \
import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def plot_models(out, labels, title, range):
    noDataSets = len(labels)
    fig = plt.imshow(out, vmin = range[0], vmax = range[1])

    plt.xticks(np.linspace(0,noDataSets - 1, noDataSets),
                                            dataLabels, rotation=90)
    plt.yticks(np.linspace(0,noDataSets - 1, noDataSets),
                                            dataLabels, rotation=0)
    fig.axes.xaxis.tick_top()

    plt.colorbar()
    plt.title(title + '\n\n', fontSize=18, y=1)
    plt.show()
    return


def feature_labels(df, featureRanking, K):
    sortedIndex = np.argsort(featureRanking.scores_)[::-1]
    return [df.columns[w] for w in sortedIndex[:K]]

def preprocess_df(df):
    df.sort_index(axis = 0, inplace = True)
    df.sort_index(axis = 1, inplace = True)

    #X will vary depending on the model
    sessionData = idx[:,:,11:]
    #Y is always idx['choice',0] - we do not change
    Y = df.loc[sessionData, idx['choice',0]]
    #slice out the data
    past_slice = idx[['choice', 'reward', 'RTs', 'choice_AR'], 1:]
    current_slice = idx[['SA'],:]
    X1 = df.loc[sessionData, past_slice]
    X2 = df.loc[sessionData, current_slice]
    X = pd.concat([X1,X2], axis=1)
    #Sorting dataframe, again
    X.sort_index(axis = 0, inplace = True)
    X.sort_index(axis = 1, inplace = True)
    #centering reaction times
    X.loc[:,idx['RTs',:]] = scale(X.loc[:,idx['RTs',:]])
    return X,Y


def neural_networks(X,Y, hidden_layers, BS, EPS = 10):

    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = hidden_layers,
                         kernel_initializer = 'uniform',
                         activation = 'relu',
                         input_dim = X.shape[1]))

    # Adding the second hidden layer
    classifier.add(Dense(units = hidden_layers,
                         kernel_initializer = 'uniform',
                         activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1,
                         kernel_initializer = 'uniform',
                         activation = 'sigmoid'))
    # Compiling Neural Network
    classifier.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])

    classifier.fit(X, Y, batch_size = BS, epochs = EPS, verbose=0)
    return classifier



def epoch_size_test(X, Y):
    EPS_range = np.linspace(10,100,10)
    performance = np.full([2, len(EPS_range)], np.NaN)
    for EPS_index, EPS in enumerate(EPS_range):
        print 'on NN#%i...' %(EPS_index + 1)
        start = time.time()
        NN, score = neural_networks(X, Y, 10, 30, EPS = int(EPS))
        print 'score:%1.2f, by the way' %score
        end = time.time() - start
        performance[0, EPS_index] = EPS
        performance[1, EPS_index] = end
    plt.scatter(performance[0,:],performance[1,:])
    plt.show()
    return


def NN_predict(NN, X, Y):
    Y_prediction = NN.predict(X)
    Y_prediction = (Y_prediction > 0.5)
    return float(np.sum(Y_prediction.ravel() == Y))/len(Y)



ROOT = config.ROOT
idx = pd.IndexSlice
featureRanking = SelectKBest(mutual_info_classif, k='all')
model = SGDClassifier(penalty='elasticnet', l1_ratio=0.15)
model = LogisticRegression(penalty = 'l1', solver = 'liblinear')

model = XGBClassifier()
poly = PolynomialFeatures(2, interaction_only = True)




taskLabels = ['DSR','PSR']
dataLabels = ['FirstTraining', 'MidTraining', \
              'Saline', 'MPFC', 'OFC', \
              'Ipsi', 'Contra', ]

noShuffles = 40
OUT = np.full([noShuffles, len(dataLabels), len(dataLabels)], np.NaN)
units = [int(w) for w in np.r_[np.linspace(2,8,4),np.linspace(10,100,19)]]
batch_size =  [int(w) for w in np.linspace(5,50,10)]

for task in ['DSR']:
    for shuffle in range(noShuffles):
        start = time.time()
        print 'on shuffle %i...' %shuffle
        for datasetIndex, dataLabel in enumerate(dataLabels):


            #print "loading %s-%s dataset..." %(task,dataLabel)
            df = pickle.load(open(config.ROOT
                                  + '/DATA_structures/'
                                  + task + '_TbyT_' \
                                  + dataLabel + '.p', 'rb'))

            X,Y = preprocess_df(df)
            #separate into train, validate, and test
            X_train, X_test, Y_train, Y_test = \
                        train_test_split(X, Y, test_size=0.20, random_state=42)

            NN, score = neural_networks(X, Y,
                                 hardcoded_VARS[0][datasetIndex],
                                 hardcoded_VARS[1][datasetIndex])

            #give score along the diagonal
            OUT[shuffle, datasetIndex, datasetIndex] = score
            #now scroll thru all other datasets
            for NN_index, newLabel in enumerate(dataLabels):
                if NN_index != datasetIndex:
                    df = pickle.load(open(config.ROOT
                                          + '/DATA_structures/'
                                          + task + '_TbyT_' \
                                          + newLabel + '.p', 'rb'))
                    X,Y = preprocess_df(df)
                    OUT[shuffle, datasetIndex, NN_index] = \
                                            NN_predict(NN, X, Y)

        print 'processing time: %1.2f minutes' %((time.time() - start) / 60)

    pickle.dump(OUT, open('DATA_structures/DSR_NN_scores.p','wb'))
    plot_models(np.nanmean(OUT, axis=0), dataLabels, 'VALUES', [0.4,0.9])
    plot_models(np.nanstd(OUT, axis=0), dataLabels, 'VARIANCE', [0,0.2])
