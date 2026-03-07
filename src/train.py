"""
Main Training Script
"""
import argparse
import json
import sys
import os
import numpy as np
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'])
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'])
    parser.add_argument('-w_p', '--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, default='best_model.npy')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_arguments()
    np.random.seed(42)

    # pad/trim hidden_size to match num_layers
    if isinstance(args.hidden_size, int):
        args.hidden_size = [args.hidden_size] * args.num_layers
    if len(args.hidden_size) < args.num_layers:
        args.hidden_size += [args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size))
    elif len(args.hidden_size) > args.num_layers:
        args.hidden_size = args.hidden_size[:args.num_layers]

    print("=== Training Configuration ===")
    print(f"Dataset: {args.dataset}, Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}, LR: {args.learning_rate}, WD: {args.weight_decay}")
    print(f"Layers: {args.num_layers}, Hidden: {args.hidden_size}")
    print(f"Activation: {args.activation}, Loss: {args.loss}, Init: {args.weight_init}")
    print("=" * 30)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    wandb_run = None
    if args.wandb_project is not None and not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                name=f"train_{args.dataset}_{args.optimizer}",
                reinit=True,
                id=str(uuid.uuid4())
            )
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")

    model = NeuralNetwork(args)
    best_weights = model.train(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val, y_val=y_val,
        wandb_run=wandb_run,
    )

    # load best weights from training
    if best_weights is not None:
        model.set_weights(best_weights)

    test_m = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_m['accuracy']:.4f} | F1: {test_m['f1']:.4f}")

    # save model in same dir as train.py (i.e. src/)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(src_dir, args.model_path)
    weights = model.get_weights()
    np.save(save_path, weights)
    print(f"Model saved to {save_path}")

    config = {
        "dataset": args.dataset, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "optimizer": args.optimizer, "loss": args.loss, "num_layers": args.num_layers,
        "hidden_size": args.hidden_size, "activation": args.activation,
        "weight_init": args.weight_init, "test_accuracy": float(test_m['accuracy']),
        "test_f1": float(test_m['f1']),
    }
    config_path = os.path.join(src_dir, "best_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    if wandb_run is not None:
        wandb_run.log({"test_accuracy": test_m['accuracy'], "test_f1": test_m['f1']})
        wandb_run.finish()

    print("Training complete!")


if __name__ == '__main__':
    main()