import numpy as np

def predict_trend(x, a, b):
    y_pred = a*x+b
    return y_pred

def main():
    x = np.random.rand(1000)
    a = 5
    b = 10

    print(predict_trend(x, a, b))

if __name__ == "__main__":
    main()