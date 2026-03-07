"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from ann.activations import sigmoid, sigmoid_grad, tanh, tanh_grad, relu, relu_grad


class NeuralLayer:

    def __init__(self, input_size, output_size, activation, weight_init):

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
            self.b = np.zeros((1, output_size))
        elif weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
            self.b = np.zeros((1, output_size))
        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size))
            self.b = np.zeros((1, output_size))
        else:
            raise ValueError("weight_init must be random or xavier")

        self.z = None
        self.a_in = None
        self.a_out = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, a_in):
        self.a_in = a_in
        self.z = a_in @ self.W + self.b

        if self.activation == "relu":
            self.a_out = relu(self.z)
        elif self.activation == "sigmoid":
            self.a_out = sigmoid(self.z)
        elif self.activation == "tanh":
            self.a_out = tanh(self.z)
        elif self.activation == "linear":
            self.a_out = self.z.copy()
        else:
            raise ValueError("activation must be relu, sigmoid, tanh, or linear")

        return self.a_out

    def backward(self, delta, weight_decay=0.0):
        if self.activation == "relu":
            dz = delta * relu_grad(self.z)
        elif self.activation == "sigmoid":
            dz = delta * sigmoid_grad(self.z)
        elif self.activation == "tanh":
            dz = delta * tanh_grad(self.z)
        elif self.activation == "linear":
            dz = delta
        else:
            raise ValueError("unknown activation in backward")

        # pure gradient without L2 term — optimizer handles regularization
        self.grad_W = self.a_in.T @ dz
        self.grad_b = np.sum(dz, axis=0, keepdims=True)

        delta_prev = dz @ self.W.T
        return delta_prev