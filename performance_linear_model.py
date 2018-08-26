
import sys
import numpy as np
import pickle
from behavioral_performance.tools.session_iterator import SessionIterator 
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf






def plot_data_first(table):
    Y = table[:,0]
    saline = table[:,1]
    mpfc = table[:,2]
    ofc = table[:,3]
    datasets = [saline, mpfc, ofc]
    colors = ['b','r','g']
    labels = ['saline','PL','OFC']

    for dataIndex, data in enumerate(datasets):
        plt.scatter(data, Y, c=colors[dataIndex], label=labels[dataIndex])

    plt.title('Block 1 - PSR', FontSize= 14)
    plt.xlabel('Congruency')
    plt.ylabel('Percent Correct')
    plt.legend()
    plt.show()







if __name__ == '__main__':

    #load datasets
    for task in ['DSR','PSR']:
        table = pickle.load(open('DATA_structures/' + task
                                            + '_DATAFRAME.p', 'rb'))


        saline_table = table[table['Infusion_String']=='AASaline']
        model = smf.ols('Performance ~ Notes * Rev1', data=saline_table)
        print model.fit().summary()


        formulas = {'DSR':'Performance ~ C(GA) + Animal + Order \
                                       + Infusion_String * Rev1',
                    'PSR':'Performance ~ GA + Animal + Order \
                                       + Uncertainty * Infusion_String * Rev1'}

        model = smf.ols(formulas[task], data=table)
        side_bias = smf.ols('Performance ~ Animal * C(GA)',data=table)
        print side_bias.fit().summary()
        results = model.fit()
        #print results.summary()
