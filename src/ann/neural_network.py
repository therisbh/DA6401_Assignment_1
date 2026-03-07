"""
Main Neural Network Model class
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer

class NeuralNetwork:
    
    def __init__(self, cli_args):
        self.args = cli_args
        self.layers = []

        # safe attribute access with defaults
        num_layers  = getattr(cli_args, 'num_layers',  3)
        hidden_size = getattr(cli_args, 'hidden_size', 128)
        activation  = getattr(cli_args, 'activation',  'relu')
        weight_init = getattr(cli_args, 'weight_init', 'xavier')
        loss        = getattr(cli_args, 'loss',        'cross_entropy')
        optimizer   = getattr(cli_args, 'optimizer',   'rmsprop')
        lr          = getattr(cli_args, 'learning_rate', 0.001)
        wd          = getattr(cli_args, 'weight_decay',  0.0)

        input_size  = 784
        num_classes = 10

        # handle num_layers as int or list
        if isinstance(num_layers, list):
            hidden_sizes = [int(s) for s in num_layers]
        else:
            sz = hidden_size if isinstance(hidden_size, int) else hidden_size[0]
            hidden_sizes = [int(sz)] * int(num_layers)

        # build hidden layers
        prev_size = input_size
        for sz in hidden_sizes:
            layer = NeuralLayer(prev_size, sz, activation=activation,
                                weight_init=weight_init)
            self.layers.append(layer)
            prev_size = sz

        # output layer
        output_layer = NeuralLayer(prev_size, num_classes,
                                   activation="linear",
                                   weight_init=weight_init)
        self.layers.append(output_layer)

        # update args safely
        cli_args.num_layers  = len(hidden_sizes)
        cli_args.hidden_size = hidden_sizes[-1] if hidden_sizes else 128

        self.loss_fn, self.loss_grad = get_loss(loss)
        self.optimizer = get_optimizer(optimizer, lr, wd)
        self.grad_W = None
        self.grad_b = None

        print("Built network:", cli_args.num_layers, "hidden layers,",
              cli_args.hidden_size, "neurons, activation =", activation)

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, y_true, y_pred):
        grad_W_list = []
        grad_b_list = []
        delta = self.loss_grad(y_true, y_pred)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, weight_decay=getattr(self.args, 'weight_decay', 0.0))
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        grad_W_list = grad_W_list[::-1]
        grad_b_list = grad_b_list[::-1]

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb
        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.update(self.layers)

    def train(self, X_train, y_train, epochs=1, batch_size=32,
              X_val=None, y_val=None, wandb_run=None):
        if epochs == 1 and hasattr(self.args, "epochs"):
            epochs = self.args.epochs
        if batch_size == 32 and hasattr(self.args, "batch_size"):
            batch_size = self.args.batch_size
        n = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, batch_size):
                X_batch = X_shuffled[start:start+batch_size]
                y_batch = y_shuffled[start:start+batch_size]
                fwd = self.forward(X_batch)
                batch_loss = self.loss_fn(y_batch, fwd)
                epoch_loss += batch_loss
                num_batches += 1
                self.backward(y_batch, fwd)
                self.update_weights()
            avg_loss = epoch_loss / num_batches
            train_acc = self.evaluate(X_train, y_train)
            log_data = {"epoch": epoch+1, "train_loss": avg_loss, "train_accuracy": train_acc}
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)
                val_logits = self.forward(X_val)
                val_loss = self.loss_fn(y_val, val_logits)
                log_data["val_loss"] = val_loss
                log_data["val_accuracy"] = val_acc
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f}")
            if wandb_run is not None:
                wandb_run.log(log_data)

    def evaluate(self, X, y):
        fwd = self.forward(X)
        predicted = np.argmax(fwd, axis=1)
        if y.ndim == 2:
            true_labels = np.argmax(y, axis=1)
        else:
            true_labels = y
        return np.mean(predicted == true_labels)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        print(f"Weight dict keys: {list(weight_dict.keys())}")
        print(f"Number of layers: {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            print(f"Looking for {w_key}: {w_key in weight_dict}")
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()