import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
#import cPickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
    _iter[0] += 1

def plot(name, value):
    _since_last_flush[name][_iter[0]] = value
    #print(_since_last_flush)

def flush(log_dir):
    prints = []

    for name, vals in _since_last_flush.items():
        #prints.append("{}\t{}".format(name, np.mean(vals.values())))
        _since_beginning[name].update(vals)
        """
        print(name)
        print("#######################")
        print(_since_beginning[name])
        print("#######################")
        print(_since_beginning[name].keys())
        print("#######################")
        print(list(_since_beginning[name].keys()))
        print("#######################")
        """
        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        os.makedirs(os.path.join(log_dir), exist_ok=True)
        plt.savefig(os.path.join(log_dir, name + '.png'))
    """
    print "iter {}\t{}".format(_iter[0], "\t".join(prints))
    _since_last_flush.clear()
    
    with open('log.pkl', 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
    """
