import os
import numpy as np
import pandas as pd
from session_file_reader import session_text_extractor
from datetime import datetime as dt
ROOT = os.environ['HOME'] + '/python/'

class Session(object):

    def __init__(self, PATH = ROOT + '/cleanDATA/Cohort2/1/2_3_2017/'):
        if not os.path.isdir(PATH):
            print 'Session not found.'
            break
        header, info = session_text_extractor(PATH)
        self.add_block_info(info)
        self.header = header
        self.info = info
        self.reduced_df()
        #drug: saline = 0, muscimol = 1, something else = -1
        if self.header['notes'].find('saline') > 0 or self.header[
                        'notes'].find('muscimol') > 0:
            #actually a session we care about
            self.drug = int(not (self.header['notes'].find('saline')>0))
            self.drug = int(self.header['notes'].find('muscimol')>0)
        else:
            self.drug = -1


        self.regime = ''
        for ROI in ['PL','OFC','ipsi','x','mPFC']:

            if self.header['notes'].find('Naive') < 0:
                if self.header['notes'].find(ROI) >= 0 \
                and self.header['notes'].find('saline') < 0:
                    self.regime = ROI

                elif self.header['notes'].find(ROI) >= 0 \
                and self.header['notes'].find('saline') >= 0:
                    self.regime = ROI + '-saline'
            else:
                if self.header['notes'].find(ROI) >= 0 \
                and self.header['notes'].find('Saline') < 0:
                    self.regime = ROI
                elif self.header['notes'].find('Saline') >= 0:
                    self.regime = 'Saline'


        #datetime object for ease of sorting
        self.dateObj = dt.strptime(self.header['date'].replace('_','-'), "%d-%m-%Y")

        #give unique ratID to the session
        self.ratID = self.header['cohort'] + self.header['rat']




    def add_block_info(self, info):
        revPoints = 1 + np.nonzero(np.diff(info['GA']))[0]
        revPoints = [0] + [w for w in revPoints] + [len(info)]
        revPoints = [w - v for w,v in zip(revPoints[::-1], revPoints[-2::-1])][::-1]
        block_tuples =[((e+1),g) for e,w in enumerate(revPoints) for g in range(w)]
        my_index = pd.MultiIndex.from_tuples(block_tuples, names = ['block','trial'])
        info.index = my_index
        return

    def reduced_df(self):
        self.info['Choice1'] = (self.info['Choice1'] == 8).astype('int16')
        self.info['Choice2'] = (self.info['Choice2'] == 11).astype('int16')
        self.info['GA'] = (self.info['GA'] == 2).astype('int16')
        for field in ['SA','Correct','AR']:
            self.info[field] = self.info[field].astype('int16')
        self.info.drop(labels=['Error','Sensor','Phase'], axis=1, inplace=True)
        short_blocks = []
        for blabel, block in self.info.groupby(axis=0, level='block'):
            if len(block) < 9:
                short_blocks.append(blabel)
        if short_blocks:
            self.info.drop(labels=short_blocks, axis=0, inplace=True)
        return
