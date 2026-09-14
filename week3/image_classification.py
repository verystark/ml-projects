import pickle
import numpy as np
from datetime import datetime

# classify test data based on training data using 1NN
def my_1nn(x_train, y_train, x_test):

    # flatten training data to 60000x784 matrix
    x_train = np.reshape(x_train, (len(x_train), 784)).astype(np.float32)
    x_test = np.reshape(x_test, (len(x_test), 784)).astype(np.float32)

    labels_index = []
    batch = 2000
    # go through test images in batches to reduce memory usage
    for i in range(0, len(x_test), batch):
        x_test_subset = x_test[i:i+batch]
        # solve euclidean distance via matrix multiplications
        labels_index.extend(np.argmin(-2 * x_test_subset @ x_train.T + np.sum(x_test_subset**2, axis=1)[:, np.newaxis] + np.sum(x_train**2, axis=1)[np.newaxis, :], axis=1))

    return y_train[labels_index]

# calculating accuracy of classifier
def my_cl_acc(pred, gt):
    correct_class = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct_class += 1
    return correct_class / len(gt)

def main():
    print('Classifying images...')

    start_time = datetime.now()
    data_fname = 'clothes.pkl'

    with open(data_fname, 'rb') as data_file:
        x_train = pickle.load(data_file)
        y_train = pickle.load(data_file)
        x_test = pickle.load(data_file)
        y_test = pickle.load(data_file)

    pred = my_1nn(x_train, y_train, x_test)

    end_time = datetime.now()
    processing_time = end_time - start_time
    minutes = processing_time.seconds // 60
    seconds = processing_time.seconds % 60

    print(f'Processing time: {minutes}min {seconds}s for')
    print(f'1_NN classification accuracy is {my_cl_acc(pred, y_test)}')

if __name__ == "__main__":
    main()