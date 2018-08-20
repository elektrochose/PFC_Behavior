'''
this app will go thru models folder in epsilon and systematically clean models
that we don't want. we are running out of memory over there ! because of stupid
partitions. not sure what to make of it.
'''
#BOILERPLATE _______________________
#MODULES ______________________
ROOT = '/Users/pablomartin/python/'
import os
import time
import pickle
import numpy as np
import pandas as pd
import itertools
import operator
import pysftp



def clean_model_dir(model_dir):
    model_files = connection.listdir(model_dir)
    '''check if it's a model directory as we have defined them so far - this is
    strict as we are deleting stuff. all files must start with w or l'''
    if model_files:
        is_dir_model = np.float(np.sum([1 for w in model_files
                if w.startswith('w') or w.startswith('l')])) / len(model_files)
        if is_dir_model == 1:
            #find best model
            scores = dict([(w, float(w[w.find('-') + 1: -5]))
                           for index, w in enumerate(model_files) if w.startswith('w')])
            best_model = max(scores, key=scores.get)
            for cFile in model_files:
                if cFile != best_model and cFile != 'loss_acc_history.p':
                    print 'deleting: %s ' %(model_dir + '/' + cFile)
                    connection.remove(model_dir + '/' + cFile)
def dummy(str):
    pass


connection = pysftp.Connection('10.81.104.156', username='pablo', password='pablo2014')
connection.walktree('/home/pablo/python/Models', dummy , clean_model_dir, dummy , recurse=True)
