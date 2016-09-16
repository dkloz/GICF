"""Simple io functions to save/load parameters theta """
import numpy as np


def save_theta(theta, name, acc='', best=False):
    if '_theta' not in name:
        name += '_theta'
    name += acc
    if best:
        name += '_best'
    np.savetxt(name, theta, delimiter=',')


def load_theta(name):  # returns a vector with parameter theta
    print 'loading theta'
    if 'theta' in name:
        return np.loadtxt(name, delimiter=',')
    return np.loadtxt(name + '_theta', delimiter=',')
