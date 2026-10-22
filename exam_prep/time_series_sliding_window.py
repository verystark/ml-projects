import numpy as np

def make_sliding_window(data, M, K):
    X, Y = [], []
    for i in range(len(data) - M - K + 1):
        X.append(data[i : i + M])
        Y.append(data[i + M : i + M + K])

    return np.array(X), np.array(Y)

def main():
    print("=== Testing Time Series Sliding Window ===")

    # 1. Generate 200 sequential price points (0, 1, 2, ..., 199)
    N, M, K = 200, 100, 10
    data = np.arange(N)

    # 2. Extract windows
    X, Y = make_sliding_window(data, M, K)

    # Expected number of windows: S = 200 - 100 - 10 + 1 = 91
    print(f"Total data points (N) : {N}")
    print(f"History window (M)   : {M}")
    print(f"Target window (K)    : {K}")
    print(f"\nExtracted Shapes:")
    print(f"  X shape: {X.shape} (Expected: (91, 100))")
    print(f"  Y shape: {Y.shape}  (Expected: (91, 10))")

    # 3. Sanity check first and last windows
    print("\nWindow Alignment Checks:")
    print(f"  First X ends at: {X[0, -1]} | First Y starts at: {Y[0, 0]}")
    print(f"  Last Y ends at : {Y[-1, -1]} (Data max index: {N - 1})")


if __name__ == "__main__":
    main()