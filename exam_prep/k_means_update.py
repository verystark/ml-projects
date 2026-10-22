import numpy as np

def kmeans_step(X, centers):
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)

    new_centers = np.array([X[labels == j].mean(axis=0) for j in range(centers.shape[0])])
    return new_centers, labels

def main():
    print("=== Testing Single K-Means Iteration ===")

    # Create 6 2D points split into 2 visual clusters
    X = np.array([
        [1.0, 2.0], [1.5, 1.8], [1.2, 2.2],  # Group 1 (around x=1.2, y=2.0)
        [8.0, 9.0], [8.5, 8.8], [8.2, 9.2]   # Group 2 (around x=8.2, y=9.0)
    ])

    # Initial rough cluster centers
    initial_centers = np.array([
        [0.0, 0.0],
        [10.0, 10.0]
    ])

    print("Initial Centers:")
    print(initial_centers)

    # Run single K-Means update step
    new_centers, labels = kmeans_step(X, initial_centers)

    print("\nResults after 1 Step:")
    print(f"Assigned Labels : {labels}")
    print("Updated Centers:")
    print(new_centers)

if __name__ == "__main__":
    main()