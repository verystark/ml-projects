import numpy as np

def prep_images(X_train, X_test):
    X_train = np.reshape(X_train, (len(X_train), 784)).astype(np.float32)
    X_test = np.reshape(X_test, (len(X_test), 784)).astype(np.float32)

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)

    eps = 1e-15

    X_train_norm = (X_train - X_mean) / (X_std + eps)
    X_test_norm = (X_test - X_mean) / (X_std + eps)

    return X_train_norm, X_test_norm


def main():
    print("Generating dummy image data...")
    X_train = np.random.randint(0, 256, (60000, 28, 28), dtype=np.uint8)
    X_test = np.random.randint(0, 256, (10000, 28, 28), dtype=np.uint8)

    print("\nOriginal Shapes & Types:")
    print(f"  X_train: shape {X_train.shape}, dtype {X_train.dtype}")
    print(f"  X_test:  shape {X_test.shape}, dtype {X_test.dtype}")

    # Process data
    X_train_norm, X_test_norm = prep_images(X_train, X_test)

    print("\n" + "=" * 45)
    print("      DATA PREPROCESSING SUMMARY REPORT      ")
    print("=" * 45)

    print("\n1. Matrix Shapes & Data Types:")
    print(f"  X_train_norm -> Shape: {X_train_norm.shape} | Type: {X_train_norm.dtype}")
    print(f"  X_test_norm  -> Shape: {X_test_norm.shape}  | Type: {X_test_norm.dtype}")

    print("\n2. Training Set Statistics (Target: Mean ≈ 0.0, Std ≈ 1.0):")
    print(f"  Overall Mean : {X_train_norm.mean():.6f}")
    print(f"  Overall Std  : {X_train_norm.std():.6f}")
    print(f"  Min Value    : {X_train_norm.min():.4f}")
    print(f"  Max Value    : {X_train_norm.max():.4f}")

    print("\n3. Test Set Statistics (Normalized using Train stats):")
    print(f"  Overall Mean : {X_test_norm.mean():.6f}")
    print(f"  Overall Std  : {X_test_norm.std():.6f}")

    print("\n" + "=" * 45)
    print("Status: Preprocessing completed successfully!")
    print("=" * 45)

if __name__ == '__main__':
    main()