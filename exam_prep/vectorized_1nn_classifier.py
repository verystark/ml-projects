import numpy as np


def my_1nn(X_tr, y_tr, X_te):
    y_pred = []
    for x in X_te:
        dists = np.sum((X_tr - x) ** 2, axis=1)
        nearest_idx = np.argmin(dists)
        y_pred.append(y_tr[nearest_idx])
    
    return np.array(y_pred)

def my_acc(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    print("=== Testing Vectorized 1-NN Classifier ===")
    
    # 1. Create small dummy dataset (N=100 train images, M=20 test images, D=784 features)
    # Using 10 classes (digits 0-9)
    X_train = np.random.randn(100, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, size=100)

    X_test = np.random.randn(20, 784).astype(np.float32)
    y_test = np.random.randint(0, 10, size=20)

    # 2. Run 1-NN prediction
    y_pred = my_1nn(X_train, y_train, X_test)

    # 3. Compute accuracy
    accuracy = my_acc(y_test, y_pred)

    # 4. Print results
    print(f"Predictions array shape : {y_pred.shape}")
    print(f"Sample Predictions      : {y_pred[:5]}")
    print(f"Sample Ground Truth     : {y_test[:5]}")
    print(f"Classification Accuracy : {accuracy:.2f} ({accuracy * 100:.1f}%)")


if __name__ == "__main__":
    main()