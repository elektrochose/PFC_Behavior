import os
import pickle
import numpy as np
import pandas as pd
from patsy.contrasts import Treatment
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

ROOT = os.environ['HOME'] + '/python/'


def plot_performance(Naive, save_figure=0):

    fig, ax = plt.subplots()
    colors = ['b','g','r']
    offset = [-0.1, 0, 0.1]
    for index, (key, subData) in enumerate(Naive.groupby(['Infusion_String'])):
        subData['Rev1'] = subData['Rev1'] + offset[index]
        subData.plot(ax=ax, x='Rev1',y='Perf',
                          kind='scatter',
                          c = colors[index],
                          label=key)

    plt.xlim([-0.5,1.5])
    plt.ylim([0.4,0.9])

    plt.xticks(range(2),['B1','Rev1'])
    plt.xlabel('')
    plt.legend(loc = 'best')
    plt.title('Performance in DSR Task - Naive Animals', FontSize=16)
    if save_figure == 1:
        plt.savefig('Results/Naive_Performance.jpg',dpi=400)
    elif save_figure == 0:
        plt.show()









Naive = pickle.load(open(ROOT + \
            '/DATA_structures/block_dataframe/Naive_BLOCK_DATAFRAME.p', 'rb'))

# TF_Naive = Naive[(Naive['Timing'] \
#             < np.mean(Naive['Timing']) + 2*np.nanstd(Naive['Timing']))]


plot_performance(Naive, save_figure=0)


timing = smf.ols('Timing ~ Animal + \
                           C(Infusion_String, Treatment(reference="Saline-Naive")) \
                           * C(Rev1)', data = Naive)

performance = smf.ols('Perf ~ C(Infusion_String, Treatment(reference="Saline-Naive"))\
                              * C(Rev1)',
                              data = Naive)

performance2 = smf.ols('Perf ~ C(Infusion_String, Treatment(reference="Saline-Naive"))\
                              * C(Rev1)\
                              * C(Order, Treatment(reference = 0))',
                              data = Naive)
performance3 = smf.ols('Perf ~ Animal + C(Infusion_String, Treatment(reference="Saline-Naive"))\
                              * C(Rev1)',
                              data = Naive)

performance4 = smf.ols('Perf ~ Animal + C(GA) + \
                              C(Infusion_String, Treatment(reference="Saline-Naive"))\
                              * C(Rev1)',
                              data = Naive)

performanceZ = smf.ols('Perf ~ Animal + C(GA) + \
                              C(Order, Treatment(reference = 0)) * \
                              C(Infusion_String, Treatment(reference="Saline-Naive"))\
                              * C(Rev1)',
                              data = Naive)

for data in [performance, performance2,performance3,performance4, performanceZ]:
    results = data.fit()
    print results.summary()
