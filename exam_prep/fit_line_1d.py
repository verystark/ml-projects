import numpy as np


def fit_line_1d(x, y):
    a = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
    b = np.mean(y) - a * np.mean(x)
    return a, b


def main():
    print(fit_line_1d(np.random.rand(1000), np.random.rand(1000)))


if __name__ == "__main__":
    main()