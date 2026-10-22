import numpy as np

def soft_max(z):
   e = np.exp(z - np.max(z, axis=1, keepdims=True))
   return e / np.sum(e, axis=1, keepdims=True)

def nn_forward(X, W1, b1, W2, b2):
    h = np.maximum(0, (X @ W1 + b1))
    z = h @ W2 + b2
    out = soft_max(z)

    return np.argmax(out, axis=1)

def main():
    print("=== Testing Pure NumPy Forward Pass ===")

    # Initialize dummy shapes: Batch size N=5, Features D=784
    N = 5
    X = np.random.randn(N, 784).astype(np.float32)

    # Weight matrices and biases
    W1 = np.random.randn(784, 20) * 0.01
    b1 = np.zeros((1, 20))

    W2 = np.random.randn(20, 10) * 0.01
    b2 = np.zeros((1, 10))

    # Run forward pass
    predictions = nn_forward(X, W1, b1, W2, b2)

    print(f"Input batch shape  : {X.shape}")
    print(f"Predictions shape  : {predictions.shape} (Expected: ({N},))")
    print(f"Predicted Classes  : {predictions}")

if __name__ == "__main__":
    main()