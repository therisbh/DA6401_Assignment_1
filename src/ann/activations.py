"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

# ---- Activation functions and their derivatives ----
# Each function takes z (the pre-activation values) and returns the result

def sigmoid(z):
    # clip so we don't get overflow in exp
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    # 1 where z > 0, else 0
    return (z > 0).astype(float)

def softmax(z):
    # subtract max in each row to avoid overflow (numerically stable)
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
