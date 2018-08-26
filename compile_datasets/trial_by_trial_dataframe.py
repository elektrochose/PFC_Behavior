import config
import sys
import numpy as np
import pickle
from session_iterator import SessionIterator
import pandas as pd


ROOT = config.ROOT

#throws out the first 10 trials of each session and centers RTs
def prepare_dataframe(df):
    idx = pd.IndexSlice



    #centering continuous variables
    for cats in ['RT1','RT2']:
        for column in df[cats].groupby(axis = 1, level=0):
            df.loc[:,idx[cats, column[0]]] = \
                (column[1] - column[1].mean()) / column[1].std()

    df.dropna(axis = 0, inplace=True)
    df.sort_index(axis = 0, inplace = True)
    df.sort_index(axis = 1, inplace = True)

    return df



def session2dataframe(sess, sessNumber, T = 10):
    iterables = [['choice',
                  'reward',
                  'GA',
                  'SA',
                  'RT1',
                  'RT2',
                  'choice_AR',
                  'congruent',
                  'winStay',
                  'loseShift'],
                  list(range(T + 1))]

    column_index = \
    pd.MultiIndex.from_product(iterables, names=['type','trials_ago'])

    #creating the row index
    noBlocks = int(len(sess.reversalPoints)) + 1
    blockLabels = range(1, noBlocks + 1)
    noTrials = [np.shape(sess.get_block(w))[0] for w in range(1, noBlocks + 1)]
    blockLabels = \
    [blockLabels[i] for i in range(len(noTrials)) for v in range(noTrials[i])]
    noTrials = [w for blockTrials in noTrials for w in range(blockTrials)]
    sessionLabel = ['S' + str(sessNumber) for w in range(len(blockLabels))]
    arrays = [sessionLabel, blockLabels, noTrials]
    row_index = pd.MultiIndex.from_arrays(arrays,
                                        names=['session','block','trial'])

    df = pd.DataFrame(np.full([len(row_index),len(column_index)],np.NaN),
                      index = row_index,
                      columns=column_index)

    #now we fill in the dataframe
    for trial in range(sess.info.shape[0]):
        for t in range(T + 1)[::-1]:
            pastTrial = trial - t
            if pastTrial >= 0:
                df['choice', t].iat[trial] = \
                                    sess.info.loc[pastTrial, 'Choice 2'] - 10
                df['reward', t].iat[trial] = \
                                    sess.info.loc[pastTrial, 'AR']
                df['GA', t].iat[trial] = sess.info.loc[pastTrial, 'GA'] - 1
                df['SA', t].iat[trial] = sess.info.loc[pastTrial, 'SA']
                #initiate trial
                df['RT1', t].iat[trial] = \
                        np.sum(sess.info.loc[pastTrial,['t1','t2']])
                #journey RT
                df['RT2', t].iat[trial] = \
                        np.sum(sess.info.loc[pastTrial,['t3','t4']])

                if sess.info.loc[pastTrial, 'Choice 2'] == 10:

                    df['choice_AR', t].iat[trial] = \
                                            sess.info.loc[pastTrial, 'AR']

                elif sess.info.loc[pastTrial, 'Choice 2'] == 11:
                    df['choice_AR', t].iat[trial] = \
                                       -1 * sess.info.loc[pastTrial, 'AR']

                #this will be 1 for DSR always, will change with PSR
                df['congruent',t].iat[trial] = \
                                int(sess.info.loc[pastTrial,'Correct'] == \
                                                sess.info.loc[pastTrial,'AR'])

                #need to go an extra trial back in time for win-stay/lose-shift
                if pastTrial - 1 >= 0:


                    if sess.info.loc[pastTrial - 1, 'AR'] == 1:
                        df['winStay', t].iat[trial] = \
                            int(sess.info.loc[pastTrial, 'Choice 2'] == \
                                sess.info.loc[pastTrial - 1, 'Choice 2'])

                        df['loseShift', t].iat[trial] = 0

                    elif sess.info.loc[pastTrial - 1, 'AR'] == 0:
                        df['winStay', t].iat[trial] = 0
                        df['loseShift', t].iat[trial] = \
                            int(sess.info.loc[pastTrial, 'Choice 2'] != \
                                sess.info.loc[pastTrial - 1, 'Choice 2'])
                    else:
                        df['winStay', t].iat[trial] = np.NaN
                        df['loseShift', t].iat[trial] = np.NaN


    df.sort_index(axis = 0, inplace = True)
    df.sort_index(axis = 1, inplace = True)
    return df




if __name__ == '__main__':

    T = 10
    task = ['DSR','PSR']
    dataLabels = ['Saline', 'MPFC', 'OFC', \
                  'Ipsi', 'Contra', 'FirstTraining', 'MidTraining']

    #load datasets
    for regimeIndex, regime in enumerate(['deterministic','probabilistic']):

        print "loading %s sessions..." %regime
        saline = SessionIterator([regime, 'saline'])
        mpfc = SessionIterator([regime,'muscimol','PL'])
        ofc = SessionIterator([regime,'muscimol','OFC'])
        ipsi = SessionIterator([regime,'muscimol','ipsi'])
        contra = SessionIterator([regime,'muscimol','x'])
        print "finishing loading sessions"
        datasets = [saline, mpfc, ofc, ipsi, contra]


        for condIndex, condition in enumerate(datasets):
            frames = []
            for sessIndex, sess in enumerate(condition.sessionList):
                frames.append(session2dataframe(sess, sessIndex, T = T))

            result = pd.concat(frames, axis=0)
            print "created %s dataframe" %dataLabels[condIndex]
            #save the results
            pickle.dump(result,
            open('DATA_structures/' + task[regimeIndex] + '_TbyT_' \
            + dataLabels[condIndex] + '.p','wb'))

    training = pickle.load(open(ROOT + '/DATA_structures/trainingData.p','rb'))
    for regimeIndex, cohort in enumerate([['2','4'],['3','5']]):
        fsdf = []
        msdf = []
        animals = [w for w in training.keys() if w[0] in cohort]
        for animalIndex, animal in enumerate(animals):
            sessions = training[animal]
            #first session
            fsdf.append(session2dataframe(sessions[0], animalIndex, T = T))
            #mid session
            msdf.append(session2dataframe(\
            sessions[int(np.floor(len(sessions)/2))], animalIndex, T = T))
        FSDF = pd.concat(fsdf)
        pickle.dump(FSDF,
        open('DATA_structures/' + task[regimeIndex] + '_TbyT_' \
        + 'FirstTraining.p','wb'))
        pickle.dump(MSDF,
        open('DATA_structures/' + task[regimeIndex] + '_TbyT_' \
        + 'MidTraining.p','wb'))
