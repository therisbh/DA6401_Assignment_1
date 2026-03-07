"""
Neural Layer Implementation
"""
import numpy as np
from ann.activations import ACT_FN, ACT_GRAD

class NeuralLayer:
    def __init__(self, in_dim, out_dim, activation, weight_init):
        self.activation = activation
        self.grad_W=None
        self.grad_b= None
        self._input= None
        self._z =None

        if weight_init =="xavier":
            limit = np.sqrt(6.0/(in_dim + out_dim))
            self.W =np.random.uniform(-limit, limit, (in_dim, out_dim))
        else:
            self.W =np.random.randn(in_dim, out_dim) * 0.01
        self.b =np.zeros((1, out_dim))

    def forward(self, x):
        self._input= x
        self._z =x@self.W +self.b
        if self.activation is None:
            return self._z
        return ACT_FN[self.activation](self._z)

    def backward(self, delta):

        if self.activation is not None:
            delta = delta * ACT_GRAD[self.activation](self._z)


        self.grad_W = (self._input.T @ delta)
        self.grad_b = np.mean(delta, axis=0, keepdims=True)

        return delta @ self.W.T


Layer = NeuralLayer