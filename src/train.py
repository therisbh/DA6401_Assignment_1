"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use: mnist or fashion_mnist')

    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005,
                        help='L2 regularization weight decay')

    # updated spec (27-02-2026): only sgd, momentum, nag, rmsprop required
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'],
                        help='Optimizer: sgd, momentum, nag, rmsprop')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'],
                        help='Loss function: cross_entropy or mse')

    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128,
                        help='Number of neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='Activation function for hidden layers')

    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'],
                        help='Weight initialization method')

    # wandb - -w_p short form added per updated spec
    parser.add_argument('-w_p', '--wandb_project', type=str, default='da6401_assignment1',
                        help='Weights and Biases Project ID')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights and Biases entity/username')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    # model saving - src/ folder as required by updated spec
    parser.add_argument('--model_save_path', type=str, default='best_model.npy',
                        help='Filename to save model weights (saved in src/ folder)')
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    np.random.seed(42)

    print("=== Training Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}, LR: {args.learning_rate}")
    print(f"Hidden layers: {args.num_layers}, Hidden size: {args.hidden_size}")
    print(f"Activation: {args.activation}, Loss: {args.loss}")
    print(f"Weight init: {args.weight_init}, Weight decay: {args.weight_decay}")
    print("=" * 30)

    X_train, X_val, X_test, y_train, y_val, y_test, y_test_raw = load_data(args.dataset)

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config={
                    "dataset": args.dataset,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "optimizer": args.optimizer,
                    "loss": args.loss,
                    "num_layers": args.num_layers,
                    "hidden_size": args.hidden_size,
                    "activation": args.activation,
                    "weight_init": args.weight_init,
                }
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing without wandb logging...")

    model = NeuralNetwork(args)
    model.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, X_val=X_val, y_val=y_val, wandb_run=wandb_run)

    test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # save model using get_weights() pattern as per updated spec
    # files go in src/ folder (same directory as this script)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(src_dir, args.model_save_path)
    best_weights = model.get_weights()
    np.save(save_path, best_weights)
    print(f"Model saved to {save_path}")

    # save config in src/ folder
    config_path = os.path.join(src_dir, "best_config.json")
    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "weight_init": args.weight_init,
        "test_accuracy": float(test_acc),
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    if wandb_run is not None:
        wandb_run.log({"test_accuracy": test_acc})
        wandb_run.finish()

    
    print("Training complete!")


if __name__ == '__main__':
    main()


