from __future__ import division, print_function

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


def create_model(sequence_length, feature_dim, RANDOM_STATE, hd,
                 cell_type = 'RNN',
                 dropout = 0.2):

    '''
    Arguments:
    sequence_length - how many trials back to look
    feature_dim -
    cell_type - whether to include simple RNN or LSTM cells
    drouput - 0 <= rate < 1

    '''

    noLayers = int(np.sum(hd != 0))

    if cell_type == 'LSTM':
        RNNobj = LSTM
    elif cell_type == 'RNN':
        RNNobj = SimpleRNN

    #create model
    model = Sequential()
    if noLayers == 1:
        model.add(RNNobj(input_shape = (sequence_length, feature_dim),
                         units = hd[0]))
        model.add(Dropout(dropout))
    elif noLayers == 2:
        model.add(RNNobj(return_sequences = True,
                       input_shape = (sequence_length, feature_dim),
                       units = hd[0]))
        model.add(Dropout(dropout))
        model.add(RNNobj(hd[1]))
        model.add(Dropout(dropout))
    elif noLayers == 3:
        model.add(RNNobj(return_sequences = True,
                       input_shape = (sequence_length, feature_dim),
                       units = hd[0]))
        model.add(Dropout(dropout))
        model.add(RNNobj(return_sequences = True, units = hd[1]))
        model.add(Dropout(dropout))
        model.add(RNNobj(hd[2]))
        model.add(Dropout(dropout))

    #give probabilities
    model.add(Dense(feature_dim, activation='sigmoid'))
    if feature_dim == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(loss = loss,
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    model.RANDOM_STATE = RANDOM_STATE
    return model
