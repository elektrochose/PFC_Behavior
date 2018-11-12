import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal as sig

ROOT = os.environ['HOME'] + '/python/'
colors = ["#3F5D7D", "#559e83", "#7d3f5d", "#5d7d3f", "#ae5a41" ]

def fancy_line(avg, sem, ax, colorOne):
    ax.fill_between(range(len(avg)), avg - sem, avg + sem, color = colorOne)
    ax.plot(range(len(avg)), avg, color = "white", lw=2)

def prepare_line(array):
    avg = np.nanmean(array, axis = 0)
    sem = (np.nanstd(array, axis = 0) / np.sqrt(len(array)))
    return avg, sem

def training_plot(lines, legend_labels, title, large_plot = 1):
    '''
    lines must be a list of arrays, with dimensions:
    subjects x datapoints
    legend_labels must be corresponding labels for each array
    '''
    assert len(lines) == len(legend_labels)
    datapoints = lines[0].shape[1]

    if large_plot:
    	fig, ax = plt.subplots(figsize=(20,10))
    else:
    	fig, ax = plt.subplots(figsize=(10,6))

    ax.set_xlim([0, datapoints - 1])
    ax.set_ylim([0.45, 0.8])
    ax.set_xticks(range(datapoints))
    ax.set_xticklabels(['Naive','','','','Mid-Training','','','','Criteria'],
                        fontsize = 14)

    P = []
    for index, line in enumerate(lines):
        avg, sem = prepare_line(line)
        fancy_line(avg, sem, ax, colors[index])
        P.append(Rectangle((0, 0), 1, 1, fc = colors[index]))

    plt.legend(P, legend_labels, fontsize=14, loc = 'best')
    plt.title(title, fontsize = 20)
    plt.show()
