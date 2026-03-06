"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import sigmoid, sigmoid_grad, tanh, tanh_grad, relu, relu_grad


class NeuralLayer:
    # one layer of the neural network
    # stores weights, biases, and gradients after backprop

    def __init__(self, input_size, output_size, activation, weight_init):

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation   # string like relu, sigmoid, tanh, linear

        # initialize weights based on the chosen method
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
            self.b = np.zeros((1, output_size))
        elif weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
            self.b = np.zeros((1, output_size))
        elif weight_init == "zeros":
            # all zeros - used for the symmetry experiment in W&B report
            self.W = np.zeros((input_size, output_size))
            self.b = np.zeros((1, output_size))
        else:
            raise ValueError("weight_init must be random or xavier")

        # these get filled during forward and used in backward
        self.z = None      # pre-activation (linear combination)
        self.a_in = None   # input that came into this layer
        self.a_out = None  # output after activation

        # gradients - autograder checks these after backward()
        self.grad_W = None
        self.grad_b = None

    def forward(self, a_in):
        # save input so we can use it in backward
        self.a_in = a_in

        # linear step: z = input * W + b
        self.z = a_in @ self.W + self.b

        # apply activation
        if self.activation == "relu":
            self.a_out = relu(self.z)
        elif self.activation == "sigmoid":
            self.a_out = sigmoid(self.z)
        elif self.activation == "tanh":
            self.a_out = tanh(self.z)
        elif self.activation == "linear":
            # output layer has no activation - returns z directly (raw logits)
            self.a_out = self.z.copy()
        else:
            raise ValueError("activation must be relu, sigmoid, tanh, or linear")

        return self.a_out

    def backward(self, delta, weight_decay=0.0):
        # delta is dL/da coming from next layer
        # multiply by activation derivative to get dL/dz

        if self.activation == "relu":
            dz = delta * relu_grad(self.z)
        elif self.activation == "sigmoid":
            dz = delta * sigmoid_grad(self.z)
        elif self.activation == "tanh":
            dz = delta * tanh_grad(self.z)
        elif self.activation == "linear":
            # derivative of identity is 1
            dz = delta
        else:
            raise ValueError("unknown activation in backward")

        # gradient for W (plus L2 regularization term)
        self.grad_W = self.a_in.T @ dz + weight_decay * self.W

        # gradient for b: sum over the batch
        self.grad_b = np.sum(dz, axis=0, keepdims=True)

        # gradient to pass to previous layer
        delta_prev = dz @ self.W.T

        return delta_prev

    
