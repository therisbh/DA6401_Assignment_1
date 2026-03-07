"""
Optimization Algorithms
"""
import numpy as np

class SGD:
    def __init__(self, lr, weight_decay=0.0, **kwargs):
        self.lr =lr
        self.wd =weight_decay

    def init_state(self, layers): pass

    def step(self, layers):
        for layer in layers:
            layer.W -=self.lr * (layer.grad_W + self.wd * layer.W)
            layer.b -=self.lr * layer.grad_b

    def update(self,layers): 
        self.step(layers)


class Momentum:
    def __init__(self, lr, weight_decay=0.0, beta=0.9, **kwargs):
        self.lr = lr; self.wd = weight_decay; self.beta = beta
        self.vW = []; self.vb= []

    def init_state(self, layers):
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):
        for i, layer in enumerate(layers):
            self.vW[i] =self.beta* self.vW[i]+self.lr* (layer.grad_W+ self.wd* layer.W)
            self.vb[i] = self.beta * self.vb[i] + self.lr* layer.grad_b
            layer.W -= self.vW[i]
            layer.b -=self.vb[i]

    def update(self, layers):
        if not self.vW: 
            self.init_state(layers)
        self.step(layers)


class NAG:
    def __init__(self, lr, weight_decay=0.0, beta=0.9, **kwargs):
        self.lr= lr 
        self.wd = weight_decay 
        self.beta= beta
        self.vW= [] 
        self.vb = []

    def init_state(self, layers):
        self.vW =[np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):
        for i, layer in enumerate(layers):
            vW_prev= self.vW[i].copy()
            vb_prev = self.vb[i].copy()
            self.vW[i] = self.beta * self.vW[i] + self.lr * (layer.grad_W + self.wd * layer.W)
            self.vb[i] = self.beta * self.vb[i] + self.lr * layer.grad_b
            layer.W -=(1 + self.beta) * self.vW[i] - self.beta * vW_prev
            layer.b -= (1 + self.beta) * self.vb[i] - self.beta * vb_prev

    def update(self, layers):
        if not self.vW: 
            self.init_state(layers)
        self.step(layers)


class RMSProp:
    def __init__(self, lr, weight_decay=0.0, beta=0.9, eps=1e-8, **kwargs):
        self.lr =lr 
        self.wd = weight_decay
        self.beta= beta
        self.eps = eps
        self.sW =[]
        self.sb =[]

    def init_state(self, layers):
        self.sW= [np.zeros_like(l.W) for l in layers]
        self.sb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):
        for i, layer in enumerate(layers):
            gW = layer.grad_W +self.wd*layer.W
            gb =layer.grad_b
            self.sW[i] =self.beta* self.sW[i]+ (1-self.beta)* gW**2
            self.sb[i] =self.beta *self.sb[i] +(1 - self.beta) *gb** 2
            layer.W -=self.lr* gW /(np.sqrt(self.sW[i]) +self.eps)
            layer.b -= self.lr* gb / (np.sqrt(self.sb[i])+ self.eps)

    def update(self, layers):
        if not self.sW: 
            self.init_state(layers)
        self.step(layers)

OPTIMIZERS ={"sgd": SGD, "momentum": Momentum, "nag": NAG, "rmsprop": RMSProp}

def get_optimizer(name, lr, weight_decay=0.0):
    return OPTIMIZERS[name](lr=lr, weight_decay=weight_decay)