"""
Main Neural Network Model class
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.activations import softmax
from ann.objective_functions import LOSS_FN, LOSS_GRAD, get_loss
from ann.optimizers import OPTIMIZERS, get_optimizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class NeuralNetwork:
    def __init__(self, cli_args):
        self.args= cli_args
        self.layers = []
        self._build()
        optimizer   = getattr(cli_args, 'optimizer', 'rmsprop')
        lr= getattr(cli_args, 'learning_rate', 0.001)
        wd=getattr(cli_args, 'weight_decay', 0.0)
        self.optimizer =OPTIMIZERS[optimizer](lr=lr, weight_decay=wd)
        self.optimizer.init_state(self.layers)
        self.grad_W =None
        self.grad_b= None

    def _build(self):
        a = self.args
        num_layers  =getattr(a, 'num_layers', 3)
        hidden_size= getattr(a, 'hidden_size', 128)
        activation= getattr(a, 'activation', 'relu')
        weight_init = getattr(a, 'weight_init', 'xavier')
        num_classes =getattr(a, 'num_classes', 10)
        input_size  = getattr(a, 'input_size', 784)

        #hiddensize can be int or list
        if isinstance(hidden_size, list):
            hidden_sizes = [int(s) for s in hidden_size]
        else:
            hidden_sizes = [int(hidden_size)] * int(num_layers)

        #pad/trim to num_layers
        # pad/trim hidden sizes to match num_layers
        if len(hidden_sizes) < num_layers:
            hidden_sizes += [hidden_sizes[-1]] * (num_layers - len(hidden_sizes))
        elif len(hidden_sizes) > num_layers:
            hidden_sizes = hidden_sizes[:num_layers]

        dims =[input_size]+ hidden_sizes +[num_classes]

        for i in range(len(dims) - 1):
            act = activation if i < len(dims) - 2 else None
            self.layers.append(NeuralLayer(dims[i], dims[i+1], act, weight_init))

        print("Built network:", num_layers, "hidden layers,",
              hidden_sizes[-1] if hidden_sizes else 128, "neurons, activation =", activation)

    def forward(self, X):
        out =X
        for layer in self.layers:
            out= layer.forward(out)
        return out

    def backward(self, y_true, y_pred):
        loss_name= getattr(self.args, 'loss', 'cross_entropy')
        delta =LOSS_GRAD[loss_name](y_pred, y_true)
        grads_w, grads_b = [], []
        for layer in reversed(self.layers):
            delta= layer.backward(delta)
            grads_w.insert(0, layer.grad_W)
            grads_b.insert(0, layer.grad_b)
        self.grad_W =np.empty(len(grads_w), dtype=object)
        self.grad_b =np.empty(len(grads_b), dtype=object)
        for i, (gw, gb) in enumerate(zip(grads_w, grads_b)):
            self.grad_W[i]= gw
            self.grad_b[i] = gb
        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    def train(self, X_train, y_train, epochs=1, batch_size=32,
              X_val=None, y_val=None, wandb_run=None):
        if epochs == 1 and hasattr(self.args, 'epochs'):
            epochs =self.args.epochs
        if batch_size == 32 and hasattr(self.args, 'batch_size'):
            batch_size = self.args.batch_size
        loss_name =getattr(self.args, 'loss', 'cross_entropy')
        n = X_train.shape[0]
        best_f1 = -1
        best_weights = None

        for epoch in range(epochs):
            idx =np.random.permutation(n)
            Xs, ys = X_train[idx], y_train[idx]
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                Xb =Xs[start:start+batch_size]
                yb =ys[start:start+batch_size]
                logits= self.forward(Xb)
                epoch_loss +=LOSS_FN[loss_name](logits, yb) * len(yb)
                self.backward(yb, logits)
                self.update_weights()
            epoch_loss /=n
            train_m =self.evaluate(X_train, y_train)
            log = {"epoch": epoch+1, "train_loss": epoch_loss,
                   "train_accuracy": train_m["accuracy"]}
            if X_val is not None:
                val_m = self.evaluate(X_val, y_val)
                log.update({"val_loss": val_m["loss"], "val_accuracy": val_m["accuracy"],
                            "val_f1": val_m["f1"]})
                if val_m["f1"] > best_f1:
                    best_f1= val_m["f1"]
                    best_weights = self.get_weights()
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | "
                      f"Train Acc: {train_m['accuracy']:.4f} | Val Acc: {val_m['accuracy']:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | "
                      f"Train Acc: {train_m['accuracy']:.4f}")
            if wandb_run is not None:
                wandb_run.log(log)
        return best_weights

    def evaluate(self, X, y):
        logits = self.forward(X)
        loss_name = getattr(self.args, 'loss', 'cross_entropy')
        loss = LOSS_FN[loss_name](logits, y)
        preds =np.argmax(logits, axis=1)
        if y.ndim == 2:
            true= np.argmax(y, axis=1)
        else:
            true = y.astype(int)
        acc= accuracy_score(true, preds)
        f1= f1_score(true, preds, average="macro", zero_division=0)
        prec = precision_score(true, preds, average="macro", zero_division=0)
        rec = recall_score(true, preds, average="macro", zero_division=0)
        return {"loss": loss, "accuracy": acc, "f1": f1,
                "precision": prec, "recall": rec, "logits": logits}

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def get_weights(self):
        return {f"W{i}": l.W.copy() for i, l in enumerate(self.layers)} | \
               {f"b{i}": l.b.copy() for i, l in enumerate(self.layers)}

    def set_weights(self, weights):
        
        for i, layer in enumerate(self.layers):
            
            if f"W{i}" in weights:
                layer.W =weights[f"W{i}"].copy()
                layer.b= weights[f"b{i}"].copy()