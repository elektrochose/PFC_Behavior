import sys
sys.path.append('..')
import config
import numpy as np
import pickle
import pandas as pd
import time
import dill

import keras
from keras.models import Sequential
from keras.layers import Dense

import multiprocessing as mp


# from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.metrics import confusion_matrix
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

def split_df_XY(df):
    idx = pd.IndexSlice
    #Y is always idx['choice',0] - we do not change
    Y = df.loc[:, idx['choice',0]]
    #slice out the data
    past_slice = idx[['winStay', 'loseShift',
                      'RT2', 'choice', 'reward', 'choice_AR'], 1:]
    current_slice = idx[['SA','RT1'],:]
    X1 = df.loc[:, past_slice]
    X2 = df.loc[:, current_slice]
    X = pd.concat([X1,X2], axis=1)

    #Sorting dataframe, again
    X.sort_index(axis = 0, inplace = True)
    X.sort_index(axis = 1, inplace = True)

    return X,Y


def train_NN(X,Y, hidden_layers, BS, EPS = 50):

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



def predict_NN(NN, X, Y):
    Y_prediction = NN.predict(X)
    Y_prediction = (Y_prediction > 0.5)
    return float(np.sum(Y_prediction.ravel() == Y))/len(Y)




def hyperparameter_tuning(fileName):
    #start timer
    start = time.time()
    print 'started process %s' %fileName
    #ugly i know
    task = fileName[-fileName[::-1].find(
    '/'):][:fileName[-fileName[::-1].find('/'):].find('_')]
    dataLabel = fileName[-fileName[::-1].find(
    '_'):][:fileName[-fileName[::-1].find('_'):].find('.')]

    noShuffles = 10
    units = [int(w) for w in np.r_[np.linspace(2,8,4), np.linspace(10,100,19)]]
    batch_size =  [int(w) for w in np.linspace(5,50,10)]

    #load dataset
    df = pickle.load(open(fileName, 'rb'))
    #throw out first T trials, center RT columns
    df = prepare_dataframe(df)
    #split dataset
    X,Y = split_df_XY(df)


    scores = np.full([noShuffles, len(units), len(batch_size)], np.NaN)
    index = 0
    for shuffle in range(noShuffles):
        #separate into train, validate, and test
        X_train, X_test, Y_train, Y_test = \
                train_test_split(X, Y, test_size=0.20, random_state=shuffle)
        for unitIndex, unitNo in enumerate(units):
            for batchIndex, batchSize in enumerate(batch_size):
                #printing progress output
                if index%50 == 0 and index > 0:
                    print '-' * 80
                    print '%s' %dataLabel
                    print 'trained model: %i/%i' \
                        %(index + 1, len(units) * len(batch_size) * noShuffles)
                    print 'highest score so far: %1.3f' %np.nanmax(scores)
                    print 'elapsed time: %1.4f minutes' \
                                            %((time.time() - start) / 60)
                    print '-' * 80
                    #restart timer
                    start = time.time()
                elif index%50 == 0 and index == 0:
                    print '-' * 80
                    print '%s' %dataLabel
                    print 'time elapsed for loading: %1.4f minutes' \
                                            %((time.time() - start) / 60)
                    print 'starting training now...'

                NN = train_NN(X_train, Y_train, unitNo, batchSize)
                score = predict_NN(NN, X_test, Y_test)



                index += 1
                scores[shuffle, unitIndex, batchIndex] = score
                label = 'S%iU%iBS%i' %(shuffle, unitNo, batchSize)

    fileName = config.ROOT + '/DATA_structures/NN/' + task + '_' + dataLabel \
                                    + '_units_batch_size_optimization_scores.p'

    pickle.dump(scores, open(fileName, 'wb'))
    print 'processing time for %s-%s: %1.2f minutes' \
                    %(task, dataLabel, ((time.time() - start) / 60))
    return


if __name__ == '__main__':

    ROOT = config.ROOT
    pool = mp.Pool(processes=8)
    idx = pd.IndexSlice
    taskLabels = ['DSR','PSR']
    dataLabels = ['FirstTraining', 'MidTraining', \
                  'Saline', 'MPFC', 'OFC', \
                  'Ipsi', 'Contra', ]

    task = 'PSR'


    files = [ROOT + '/DATA_structures/' + task + '_TbyT_' \
                          + dataLabel + '.p' for dataLabel in dataLabels]
    hyperparameter_tuning(files[4])
    #s = pool.map(hyperparameter_tuning, files)
