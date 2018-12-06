#!/usr/bin/env python

from __future__ import division, print_function

import os
import logging


import pandas as pd
import numpy as np
import pickle
import multiprocessing as mp

import opts
from dataset import create_sequences
from rnnnet import create_model

from keras.callbacks import ModelCheckpoint
from keras import backend as K

ROOT = os.environ['HOME'] + '/python/'




def train():
    logging.basicConfig(level=logging.INFO)

    args = opts.parse_arguments()

    logging.info("Loading dataset...")

    SeqIterTrain, SeqIterVal, SeqIterTest = \
                        create_sequences(args.data_dir,
                         SEQ_LENGTH = args.SEQ_LENGTH,
                         RANDOM_STATE = args.RANDOM_STATE,
                         MIRROR = args.MIRROR,
                         SHUFFLE = args.SHUFFLE,
                         train_size = args.train_size,
                         validate_size = args.validate_size,
                         test_size = args.test_size)


    logging.info("Building model...")

    model = create_model(args.SEQ_LENGTH,
                         args.feature_dim,
                         args.RANDOM_STATE,
                         args.units,
                         cell_type = args.RNNobj,
                         dropout = args.dropout)


    callbacks = \
        [ModelCheckpoint(filepath = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5',
                         monitor = 'val_acc',
                         save_best_only = True)]

    logging.info("Begin training.")
    History = model.fit_generator(SeqIterTrain,
                                epochs = args.epochs,
                                steps_per_epoch = SeqIterTrain.steps_per_epoch,
                                validation_data = SeqIterVal,
                                validation_steps = SeqIterVal.steps_per_epoch,
                                callbacks = callbacks,
                                verbose = 0)
    pickle.dump(History.history, open('loss_acc_history.p','wb'))

    #let's clean up - delete all models except the best one
    contents = [w for w in os.listdir('.') if w.startswith('w')]
    scores = {w: float(w[w.find('-') + 1:-5]) for w in contents}
    scores = sorted(scores.iteritems(), key=lambda (k,v):(v,k), reverse=True)
    for i, (key,val) in enumerate(scores):
        if i > 0: os.remove(key)


if __name__ == '__main__':
    train()
