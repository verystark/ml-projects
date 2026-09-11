import pickle

import matplotlib.pyplot as plt

import random

def my_cl_acc(pred, gt):
    correct_class = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct_class += 1
    return correct_class / len(gt)

data_fname = 'clothes.pkl'

with open(data_fname, 'rb') as data_file:
    x_train = pickle.load(data_file)
    y_train = pickle.load(data_file)
    x_test = pickle.load(data_file)
    y_test = pickle.load(data_file)

# testing the function
pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
gt = []
for i in range(10):
    gt.append(random.randrange(0, 10))
    
print(my_cl_acc(pred, gt))

