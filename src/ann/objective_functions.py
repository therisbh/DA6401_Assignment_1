"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

# ---- Loss functions ----
# Both loss functions take y_true (one-hot) and logits (raw network output, no softmax)
# We apply softmax inside here because forward() returns raw logits now

def _softmax(z):
    # stable softmax - subtract max to avoid big exp values
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_true, logits):
    # first get probabilities from logits
    probs = _softmax(logits)
    probs = np.clip(probs, 1e-12, 1.0)  # avoid log(0)
    loss = -np.sum(y_true * np.log(probs)) / y_true.shape[0]
    return loss


def cross_entropy_grad(y_true, logits):
    # when softmax and cross entropy are combined, gradient simplifies to:
    # dL/dz = (softmax(z) - y_true) / batch_size
    batch_size = y_true.shape[0]
    probs = _softmax(logits)
    return (probs - y_true) / batch_size


def mse_loss(y_true, logits):
    probs = _softmax(logits)
    # mean squared error between predicted probs and one-hot labels
    loss = np.mean(np.sum((probs - y_true) ** 2, axis=1))
    return loss


def mse_grad(y_true, logits):
    # gradient of MSE w.r.t logits
    # we need the softmax jacobian here:
    # dL/dz_i = p_i * (dL/dp_i - sum_j(dL/dp_j * p_j))
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