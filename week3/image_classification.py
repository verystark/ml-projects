import pickle
import matplotlib.pyplot as plt
import random
import numpy as np

def my_1nn(x_train, y_train, x_test):
    labels_index = []
    for i in x_test:
        distance_all_pixels = []
        for j in x_train:
            distance = sum(np.sqrt((j-i)**2))
            distance_all_pixels.append(distance)
        labels_index.append(np.argmin(distance_all_pixels))

    labels_pred = []
    for i in labels_index:
        labels_pred.append(y_train[i])

    return labels_pred

    
def sanity_check():
    pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print('Do 10 sanity checks:')
    for i in range(10):
        gt = []
        for i in range(10):
            gt.append(random.randrange(0, 10))
        print(f'sanity check: {my_cl_acc(pred, gt)}')

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

sanity_check()

x_train = np.reshape(x_train, (60000, 784))
x_test = np.reshape(x_test, (10000, 784))

pred = my_1nn(x_train[:10000], y_train[:10000], x_test[:100])
print(my_cl_acc(pred, y_test[:100]))

