"""
Loss/Objective Functions and Their Derivatives
"""

import numpy as np

def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def _to_onehot(y, num_classes=10):
    """Convert integer labels or one-hot to one-hot."""
    if y.ndim == 2:
        return y
    n = len(y)
    oh = np.zeros((n, num_classes))
    oh[np.arange(n), y.astype(int)] = 1
    return oh

def cross_entropy_loss(y_true, logits):
    y_true = _to_onehot(y_true, logits.shape[1])
    probs = _softmax(logits)
    probs = np.clip(probs, 1e-12, 1.0)
    loss = -np.sum(y_true * np.log(probs)) / y_true.shape[0]
    return loss

def cross_entropy_grad(y_true, logits):
    y_true = _to_onehot(y_true, logits.shape[1])
    batch_size = y_true.shape[0]
    probs = _softmax(logits)
    return (probs - y_true) / batch_size

def mse_loss(y_true, logits):
    y_true = _to_onehot(y_true, logits.shape[1])
    probs = _softmax(logits)
    loss = np.mean(np.sum((probs - y_true) ** 2, axis=1))
    return loss

def mse_grad(y_true, logits):
    y_true = _to_onehot(y_true, logits.shape[1])
    batch_size = y_true.shape[0]
    probs = _softmax(logits)
    dl_dp = 2.0 * (probs - y_true) / batch_size
    dot = np.sum(dl_dp * probs, axis=1, keepdims=True)
    dz = probs * (dl_dp - dot)
    return dz

def get_loss(name):
    if name == "cross_entropy":
        return cross_entropy_loss, cross_entropy_grad
    elif name == "mse":
        return mse_loss, mse_grad
    else:
        raise ValueError("loss must be cross_entropy or mse")