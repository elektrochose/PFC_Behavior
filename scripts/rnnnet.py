from __future__ import division, print_function

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout
from keras import backend as K


def create_model(sequence_length,
                 feature_dim,
                 RANDOM_STATE,
                 UNITS,
                 cell_type = 'RNN',
                 dropout = 0.2):

    '''
    Arguments:
    sequence_length - how many trials back to look
    feature_dim - how big the vocabulary space is
    cell_type - whether to include simple RNN or LSTM cells
    drouput - 0 <= rate < 1
    UNITS - array specifying how many units per layer, if value is 0 then
    we don't want that as a layer: e.g. [10 10 10] is 3 layers with 10 units
    each. [20 50 0] is a 2-layer network with 20 units in the first layer and
    50 units in the second layer.

    output:
    compiled model, with ADAM optimizer with default learning rate

    '''

    noLayers = int(np.sum(UNITS != 0))

    if cell_type == 'LSTM':
        RNNobj = LSTM
    elif cell_type == 'RNN':
        RNNobj = SimpleRNN

    #create model
    model = Sequential()
    if noLayers == 1:
        model.add(RNNobj(input_shape = (sequence_length, feature_dim),
                         units = UNITS[0]))
        model.add(Dropout(dropout))
    elif noLayers == 2:
        model.add(RNNobj(return_sequences = True,
                       input_shape = (sequence_length, feature_dim),
                       units = UNITS[0]))
        model.add(Dropout(dropout))
        model.add(RNNobj(UNITS[1]))
        model.add(Dropout(dropout))
    elif noLayers == 3:
        model.add(RNNobj(return_sequences = True,
                       input_shape = (sequence_length, feature_dim),
                       units = UNITS[0]))
        model.add(Dropout(dropout))
        model.add(RNNobj(return_sequences = True, units = UNITS[1]))
        model.add(Dropout(dropout))
        model.add(RNNobj(UNITS[2]))
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
