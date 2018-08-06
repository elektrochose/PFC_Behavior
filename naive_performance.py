import config
import sys
import numpy as np
import pickle
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def plot_performance(Naive, save_figure=0):

    fig, ax = plt.subplots()
    colors = ['b','g','r']
    offset = [-0.1, 0, 0.1]
    for index, typeData in enumerate(Naive.groupby(['Infusion_String'])):
        key = typeData[0]
        subData = typeData[1]
        subData['Rev1'] = subData['Rev1'] + offset[index]
        subData.plot(ax=ax, x='Rev1',y='Performance',
                          kind='scatter',
                          c = colors[index],
                          label=key)

    plt.xlim([-0.5,1.5])
    plt.ylim([0.4,0.9])

    plt.xticks(range(2),['B1','Rev1'])
    plt.xlabel('')
    plt.legend(loc = 'best')
    plt.title('Performance in DSR Task - Naive Animals',FontSize=16)
    if save_figure == 1:
        plt.savefig('Results/Naive_Performance.jpg',dpi=400)
    elif save_figure == 0:
        plt.show()









Naive = pickle.load(open(config.ROOT
                                + '/DATA_structures/Naive_DATAFRAME.p', 'rb'))

TF_Naive = Naive[(Naive['Timing'] \
            < np.mean(Naive['Timing']) + 2*np.nanstd(Naive['Timing']))]

plot_performance(TF_Naive, save_figure=1)


timing = smf.ols('Timing ~ Animal + Infusion_String * Rev1', data = Naive)
performance = smf.ols('Performance ~ Animal + GA + \
                                    C(Order) * Infusion_String * Rev1',
                                    data = TF_Naive)

for data in [timing, performance]:
    results = data.fit()
    print results.summary()
