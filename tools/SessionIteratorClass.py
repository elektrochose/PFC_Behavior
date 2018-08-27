'''
SessionIterator object
given a list of filters, returns a list of session paths where the sessions in
that path meet the requirements of the filtered list. List must include only
one of 'Trained', or 'Naive'. I don't want to allow combining the datasets as
they belong to really different experiments.
'''
import sys
import os
from SessionClass import Session
ROOT = os.environ['HOME'] + '/python/'

class SessionIterator(object):

    def __init__(self, filters=None):


        self.DATA = ROOT + "cleanDATA/"
        self.filterList = {'task': ['deterministic','probabilistic'],
                         'drug': ['saline', 'muscimol','Saline','Musc'],
                         'ROI': ['PL','OFC','ipsi','x','mPFC','Saline'],
                         'training' : ['Naive','Training','Trained']}

        if filters is None:
            filters = []
        self.filters = filters
        self.sessionList = self.get_sessions()


    def get_sessions(self):
        #first step - sort what filters are wanted
        filterQuery = {'task':[], 'drug':[], 'ROI':[], 'training':[]}
        #search for terms that are there
        for cond in self.filters:
            tmp = [field for field in self.filterList\
                            if cond in self.filterList[field]]
            #if term was in the allowed list
            if len(tmp) > 0:
                filterQuery[tmp[0]].append(cond)
            else:
                print 'term not found: %s' %cond

        #can only look at Naive/Training/Trained datasets
        assert len(filterQuery['training']) == 1

        #if a condition is not present, include everything
        for field in ['task', 'drug', 'ROI']:
            if len(filterQuery[field]) == 0:
                filterQuery[field] = self.filterList[field]


        #second step - retrieve the appropriate sessions
        out = []
        if filterQuery['training'][0] == 'Naive':
            trialMinimum = 20
        elif filterQuery['training'][0] == 'Training':
            trialMinimum = 50
        elif filterQuery['training'][0] == 'Trained':
            trialMinimum = 50

        for dirName, subdirList, fileList in os.walk(self.DATA):
            if fileList:
                for fname in fileList:
                    if fname == 'log.txt':
                        sess = Session(PATH = dirName + '/')

                        sessPass = [0,0,0,0]
                        for entry in filterQuery['ROI']:
                            if sess.regime.find(entry) >= 0:
                                sessPass[0] += 1
                        for entry in filterQuery['task']:
                            if sess.header['task'].find(entry) >= 0:
                                sessPass[1] += 1
                        for entry in filterQuery['drug']:
                            if sess.header['notes'].find(entry) >= 0:
                                sessPass[2] += 1


                        if filterQuery['training'][0] == 'Naive':
                            if sess.header['notes'].find('Naive') >= 0:
                                sessPass[3] += 1
                        if filterQuery['training'][0] == 'Training':
                            if sess.header['notes']=='TRAINING':
                                sessPass[3] += 1
                        elif filterQuery['training'][0] == 'Trained':
                            if sess.header['notes'].find('Naive') < 0:
                                sessPass[3] += 1

                        #if all criteria were met
                        if filterQuery['training'][0] == 'Naive' \
                        or filterQuery['training'][0] == 'Trained':
                            if sum(sessPass) == len(sessPass) \
                            and len(sess.info) >= trialMinimum:
                                print sess.header
                                out.append(dirName + '/')
                        elif filterQuery['training'][0] == 'Training':
                            if (sessPass[1] + sessPass[3])  == 2 \
                            and len(sess.info) >= trialMinimum:
                                print sess.header
                                out.append(dirName + '/')

        return out
