"""
Data Loading and Preprocessing
"""
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(dataset_name, val_split=0.1, seed=42):
    if dataset_name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test  = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=seed, stratify=y_train
    )
    print(f"Loaded {dataset_name}: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def one_hot_encode(y, num_classes=10):
    n = len(y)
    oh = np.zeros((n, num_classes))
    oh[np.arange(n), y] = 1
    return oh