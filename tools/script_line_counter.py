from __future__ import print_function
import os

ROOT = os.environ['HOME'] + '/python/'

d = {'behavioral_performance': 0, 'Visualize' : 0,
     'Hunter' : 0, 'Vincent_Project' : 0, 'maze_scripts': 0}

for key, val in d.iteritems():
    for tuple in os.walk(ROOT + key):
        if tuple[2]:
            for files in [files for files in tuple[2]
                          if (files[-3:]=='.py'
                          or files[-4:] == '.ino'
                          or files[-2:] == '.m')]:
                f = open(tuple[0] + '/' + files,'rb')
                for line in f: val+=1
                f.close()
    d[key] = val
    print(key,':', val)
