import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
from datetime import datetime

def my_1nn(x_train, y_train, x_test):

    # flatten training data to 60000x784 matrix
    x_train = np.reshape(x_train, (len(x_train), 784))
    x_test = np.reshape(x_test, (len(x_test), 784))

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

def main():

    start_time = datetime.now()
    data_fname = 'clothes.pkl'

    with open(data_fname, 'rb') as data_file:
        x_train = pickle.load(data_file)
        y_train = pickle.load(data_file)
        x_test = pickle.load(data_file)
        y_test = pickle.load(data_file)

    sanity_check()

    pred = my_1nn(x_train, y_train, x_test[:100])

    end_time = datetime.now()
    processing_time = end_time - start_time
    minutes = processing_time.seconds // 60
    seconds = processing_time.seconds % 60
    
    print(f'Processing time: {minutes}min {seconds}s for')
    print(f'1_NN classification accuracy is {my_cl_acc(pred, y_test[:100])}')


if __name__ == "__main__":
    main()