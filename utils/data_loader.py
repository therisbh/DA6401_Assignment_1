"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


def load_data(dataset_name):
    """
    Load MNIST or Fashion-MNIST dataset using keras.

    Args:
        dataset_name: 'mnist' or 'fashion_mnist'

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        X values are normalized and flattened.
        y values are one-hot encoded.
    """
    '''if dataset_name == "mnist":
        from keras.datasets import mnist
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use mnist or fashion_mnist.")'''
    
    if dataset_name == "mnist":
        data = fetch_openml("mnist_784", version=1, as_frame=False)
        X = data.data
        y = data.target.astype(int)

    elif dataset_name == "fashion_mnist":
        data = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
        X = data.data
        y = data.target.astype(int)

    # split manually
    X_train_full, X_test = X[:60000], X[60000:]
    y_train_full, y_test = y[:60000], y[60000:]

    # flatten from (N, 28, 28) to (N, 784)
    X_train_full = X_train_full.reshape(-1, 784).astype(np.float64)
    X_test = X_test.reshape(-1, 784).astype(np.float64)

    # normalize pixel values to [0, 1]
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0

    # split training into train and validation (90/10 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    # one-hot encode labels
    num_classes = 10
    y_train_oh = one_hot_encode(y_train, num_classes)
    y_val_oh = one_hot_encode(y_val, num_classes)
    y_test_oh = one_hot_encode(y_test, num_classes)

    print(f"Loaded {dataset_name}:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, y_test


def one_hot_encode(y, num_classes):
    """Convert integer labels to one-hot encoding"""
    n = len(y)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1
    return one_hot
