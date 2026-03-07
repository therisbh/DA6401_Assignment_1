"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp
"""

import numpy as np


class SGD:
    def __init__(self, lr, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class Momentum:
    def __init__(self, lr, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_W = None
        self.v_b = None

    def update(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]


class NAG:
    def __init__(self, lr, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_W = None
        self.v_b = None

    def update(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]
        for i, layer in enumerate(layers):
            old_v_W = self.v_W[i].copy()
            old_v_b = self.v_b[i].copy()
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W -= self.lr * ((1 + self.beta) * self.v_W[i] - self.beta * old_v_W)
            layer.b -= self.lr * ((1 + self.beta) * self.v_b[i] - self.beta * old_v_b)


class RMSProp:
    def __init__(self, lr, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.sq_W = None
        self.sq_b = None

    def update(self, layers):
        if self.sq_W is None:
            self.sq_W = [np.zeros_like(l.W) for l in layers]
            self.sq_b = [np.zeros_like(l.b) for l in layers]
        for i, layer in enumerate(layers):
            self.sq_W[i] = self.beta * self.sq_W[i] + (1 - self.beta) * layer.grad_W ** 2
            self.sq_b[i] = self.beta * self.sq_b[i] + (1 - self.beta) * layer.grad_b ** 2
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.sq_W[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.sq_b[i]) + self.eps)


def get_optimizer(name, lr, weight_decay=0.0):
    if name == "sgd":
        return SGD(lr=lr, weight_decay=weight_decay)
    elif name == "momentum":
        return Momentum(lr=lr, weight_decay=weight_decay)
    elif name == "nag":
        return NAG(lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return RMSProp(lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer must be sgd, momentum, nag, or rmsprop")