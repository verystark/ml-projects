import pickle
import random
import numpy as np
from datetime import datetime

# classify test data based on training data using 1NN
def my_1nn(x_train, y_train, x_test):

    # flatten training data to 60000x784 matrix
    x_train = np.reshape(x_train, (len(x_train), 784))
    x_test = np.reshape(x_test, (len(x_test), 784))

    labels_index = []
    for i in x_test:
        labels_index.append(np.argmin(np.sum((x_train-i)**2, axis=1)))

    return y_train[labels_index]

# calculating accuracy of classifier
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

    pred = my_1nn(x_train, y_train, x_test[:500])

    end_time = datetime.now()
    processing_time = end_time - start_time
    minutes = processing_time.seconds // 60
    seconds = processing_time.seconds % 60

    print('Classifying images...')
    print(f'Processing time: {minutes}min {seconds}s for')
    print(f'1_NN classification accuracy is {my_cl_acc(pred, y_test[:500])}')

if __name__ == "__main__":
    main()