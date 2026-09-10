import pickle

import matplotlib.pyplot as plt

data_fname = 'clothes.pkl'

with open(data_fname, 'rb') as data_file:
    x_train = pickle.load(data_file)
    y_train = pickle.load(data_file)
    x_test = pickle.load(data_file)
    y_test = pickle.load(data_file)

    