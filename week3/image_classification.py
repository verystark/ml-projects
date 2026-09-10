import pickle

import matplotlib.pyplot as plt

from random import random

def my_cl_acc(pred, gt):
    return pred / gt.shape[0]

data_fname = 'clothes.pkl'

with open(data_fname, 'rb') as data_file:
    x_train = pickle.load(data_file)
    y_train = pickle.load(data_file)
    x_test = pickle.load(data_file)
    y_test = pickle.load(data_file)

print(x_train)