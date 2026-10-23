import numpy as np

def calculate_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def calculate_sse(y_true, y_pred):
    return float(np.sum((y_true - y_pred) ** 2))

def main():
    y_true = np.random.rand(1000)
    y_pred = np.random.rand(1000)

    print(f'SSE: {calculate_sse(y_true, y_pred)}')
    print(f'MSE: {calculate_mse(y_true, y_pred)}')

if __name__ == "__main__":
    main()