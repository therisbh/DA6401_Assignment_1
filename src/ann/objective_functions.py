"""
Loss/Objective Functions and Their Derivatives
"""
import numpy as np
from ann.activations import softmax

def cross_entropy(logits, y_true):
    probs = softmax(logits)
    n = y_true.shape[0]
    # handle both integer labels and one-hot
    if y_true.ndim == 2:
        log_p = -np.sum(y_true * np.log(probs + 1e-9), axis=1)
    else:
        log_p = -np.log(probs[np.arange(n), y_true.astype(int)] + 1e-9)
    return np.mean(log_p)

def cross_entropy_grad(logits, y_true):
    probs = softmax(logits)
    n = y_true.shape[0]
    if y_true.ndim == 2:
        return (probs - y_true) / n
    else:
        probs[np.arange(n), y_true.astype(int)] -= 1
        return probs / n

def mse(logits, y_true):
    probs = softmax(logits)
    n, c = probs.shape
    if y_true.ndim == 2:
        one_hot = y_true
    else:
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), y_true.astype(int)] = 1
    return np.mean((probs - one_hot) ** 2)

def mse_grad(logits, y_true):
    probs = softmax(logits)
    n, c = probs.shape
    if y_true.ndim == 2:
        one_hot = y_true
    else:
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), y_true.astype(int)] = 1
    diff = probs - one_hot
    grad = np.zeros_like(probs)
    for k in range(c):
        dsm = probs * (np.eye(c)[k] - probs[:, k:k+1])
        grad[:, k] = np.sum((2.0 / c) * diff * dsm, axis=1)
    return grad / n

LOSS_FN = {"cross_entropy": cross_entropy, "mse": mse}
LOSS_GRAD = {"cross_entropy": cross_entropy_grad, "mse": mse_grad}

# legacy interface
def get_loss(name):
    return LOSS_FN[name], LOSS_GRAD[name]