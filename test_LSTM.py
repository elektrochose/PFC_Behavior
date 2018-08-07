
import itertools
import numpy as np
import multiprocessing as mp
import pandas as pd
import pickle
from behavioral_performance.utils import fileNames
from keras.models import load_model
from RNNmodule.SequenceClass import Sequences

ROOT = '/home/pablo/python/'
datatype = 'binaryOmni'
model_dir = ROOT + 'Models/LSTM/Pablo/' + datatype + '/'
sequence_dir = ROOT + 'DATA_structures/RNN_sequences/' + datatype + '/'
Layers = ['Single', 'Double', 'Triple']
hidden_dimensions = [2, 5, 10, 20, 50, 100]
idx = pd.IndexSlice





def propagate_models(seqs, fileName):
    scores = pd.DataFrame(np.zeros([14, 18]),
                          index = fileNames,
                          columns = pd.MultiIndex.from_product(
                                        [Layers, hidden_dimensions],
                                         names = ['Layer', 'HD']))
    for Layer, hd in itertools.product(Layers, hidden_dimensions):

        print '%s - %s - %i' %(fileName, Layer, hd)
        model_path = model_dir + Layer + '/' + fileName[:-2] + str(hd) + '.h5'
        model = load_model(model_path)
        val_scores = model.evaluate(seqs.X_validate, seqs.y_validate)
        scores.loc[fileName, idx[Layer, hd]] = val_scores[1]


    pickle.dump(scores,
                open(ROOT + \
                'Model_Evaluation/LSTM/' \
                + datatype +  '_classification_behavior'
                + fileName, 'wb'))

def glue_datasets_back():
    masterScores = pd.DataFrame(np.zeros([14, 18]),
                          index = fileNames,
                          columns = pd.MultiIndex.from_product(
                                        [Layers, hidden_dimensions],
                                         names = ['Layer', 'HD']))
    for fileName in fileNames:
        scores = pickle.load(open(ROOT + 'Model_Evaluation/LSTM/' \
                                    + datatype + '_classification_behavior'
                                    + fileName, 'rb'))
        masterScores.loc[fileName, :] = scores.loc[fileName, :]
    pickle.dump(masterScores, open(ROOT + 'Model_Evaluation/LSTM/' \
                        + datatype +  '_classification_behavior.p', 'wb'))


if __name__ == '__main__' :
    pool = mp.Pool(processes = 32)
    for fileName in fileNames:
        sequence_path = sequence_dir + fileName
        seqs = pickle.load(open(sequence_path, 'rb'))
        pool.apply_async(propagate_models, [seqs, fileName])
    pool.close()
    pool.join()
    glue_datasets_back()
