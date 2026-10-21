import numpy as np

def fit_and_predict(X_tr, y_tr, X_te):
    X_b = np.column_stack((np.ones(len(X_tr)), X_tr))
    X_teb = np.column_stack((np.ones(len(X_te)), X_te))

    theta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y_tr)

    y_pred = X_teb @ theta

    return y_pred

def main():
    print("=== Testing Normal Equation Linear Regression ===")

    # Generate synthetic 1D data following the line: y = 2x + 5
    np.random.seed(42)
    X_tr = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_tr = 2.0 * X_tr.ravel() + 5.0  # true y = [7, 9, 11, 13, 15]

    # Test inputs
    X_te = np.array([[6.0], [10.0]])  # Expected predictions: 17.0 and 25.0

    # Fit and predict
    y_pred = fit_and_predict(X_tr, y_tr, X_te)

    print(f"Test Inputs (X_te)  : {X_te.ravel()}")
    print(f"Predictions (y_pred): {y_pred}")


if __name__ == "__main__":
    main()