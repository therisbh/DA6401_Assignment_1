"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .neural_layer import NeuralLayer
from .objective_functions import get_loss
from .optimizers import get_optimizer

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.args = cli_args
        self.layers = []

        # input size is 784 (28x28 image flattened), output is 10 classes
        input_size = 784
        num_classes = 10

        # build hidden layers one by one
        prev_size = input_size
        for i in range(cli_args.num_layers):
            layer = NeuralLayer(prev_size, cli_args.hidden_size,
                                activation=cli_args.activation,
                                weight_init=cli_args.weight_init)
            self.layers.append(layer)
            prev_size = cli_args.hidden_size

        # output layer - uses linear activation so forward() gives raw logits
        output_layer = NeuralLayer(prev_size, num_classes,
                                   activation="linear",
                                   weight_init=cli_args.weight_init)
        self.layers.append(output_layer)

        # pick loss function and optimizer from CLI args
        self.loss_fn, self.loss_grad = get_loss(cli_args.loss)
        self.optimizer = get_optimizer(cli_args.optimizer,
                                       cli_args.learning_rate,
                                       cli_args.weight_decay)

        # these get filled during backward()
        self.grad_W = None
        self.grad_b = None

        print("Built network:", cli_args.num_layers, "hidden layers,",
              cli_args.hidden_size, "neurons, activation =", cli_args.activation)

    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a   # raw logits
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """

        """
        Backward propagation to compute gradients.
        Returns two numpy object arrays: grad_Ws, grad_bs.
        - grad_Ws[0] is gradient for the last (output) layer weights
        - grad_Ws[1] is gradient for the second-to-last layer, and so on.
        """
        grad_W_list = []
        grad_b_list = []

        # initial gradient from the loss w.r.t. logits (dL/dz for output layer)
        delta = self.loss_grad(y_true, y_pred)

        # backprop in reverse; output layer has linear activation so act_derivative=1
        for layer in reversed(self.layers):
            delta = layer.backward(delta, weight_decay=self.args.weight_decay)
            # collect grads - index 0 = last layer (output), index 1 = second-to-last, etc.
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # store as object arrays (avoids numpy trying to broadcast different shapes)
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update(self.layers)
    
    def train(self, X_train, y_train, epochs=1, batch_size=32, X_val=None, y_val=None, wandb_run=None):
        """
        Train the network for specified epochs.
        """
        # use passed params; fall back to self.args if not given
        if epochs == 1 and hasattr(self.args, "epochs"):
            epochs = self.args.epochs
        if batch_size == 32 and hasattr(self.args, "batch_size"):
            batch_size = self.args.batch_size
        n = X_train.shape[0]

        for epoch in range(epochs):
            # shuffle training data each epoch
            indices = np.random.permutation(n)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # forward -> loss -> backward -> update
                fwd = self.forward(X_batch)
                batch_loss = self.loss_fn(y_batch, fwd)
                epoch_loss += batch_loss
                num_batches += 1

                self.backward(y_batch, fwd)
                self.update_weights()

            avg_loss = epoch_loss / num_batches
            train_acc = self.evaluate(X_train, y_train)

            log_data = {"epoch": epoch + 1, "train_loss": avg_loss, "train_accuracy": train_acc}

            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)
                val_logits = self.forward(X_val)
                val_loss = self.loss_fn(y_val, val_logits)
                log_data["val_loss"] = val_loss
                log_data["val_accuracy"] = val_acc
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

            if wandb_run is not None:
                wandb_run.log(log_data)


    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        fwd = self.forward(X)
        predicted = np.argmax(fwd, axis=1)
        if y.ndim == 2:
            true_labels = np.argmax(y, axis=1)
        else:
            true_labels = y
        return np.mean(predicted == true_labels)


    def predict(self, X):
        """Return predicted class indices for input X"""
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def get_weights(self):
        """Return a dict of all layer weights. Keys: W0, b0, W1, b1, ..."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        """Load weights from a dict with keys W0, b0, W1, b1, ..."""
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
