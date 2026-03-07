"""
Inference Script
"""
import argparse
import json
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
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
    parser.add_argument('--model_path', type=str, default='best_model.npy')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_arguments()

    # load config from best_config.json
    src_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(src_dir, 'best_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        for k, v in config.items():
            if k not in ('test_accuracy', 'test_f1', 'wandb_project', 'wandb_entity', 'no_wandb'):
                setattr(args, k, v)
        print("Loaded config from best_config.json")

    if isinstance(args.hidden_size, int):
        args.hidden_size = [args.hidden_size] * args.num_layers
    if len(args.hidden_size) < args.num_layers:
        args.hidden_size += [args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size))
    elif len(args.hidden_size) > args.num_layers:
        args.hidden_size = args.hidden_size[:args.num_layers]

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)
    model_path = os.path.join(src_dir, args.model_path)
    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)

    results = model.evaluate(X_test, y_test)
    print(f"\n=== Evaluation Results ===")
    print(f"Loss      : {results['loss']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1-score  : {results['f1']:.4f}")
    print("==========================")

    preds = np.argmax(results['logits'], axis=1)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(10)); plt.yticks(range(10))
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    wandb_run = None
    if args.wandb_project is not None:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                   name="inference", config=vars(args))
            wandb_run.log({
                "confusion_matrix": wandb.Image(plt),
                "test_accuracy": results['accuracy'],
                "test_f1": results['f1'],
            })
            wandb_run.finish()
        except Exception as e:
            print(f"wandb error: {e}")
    else:
        plt.savefig(os.path.join(src_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
        print("Saved confusion_matrix.png")

    return results


if __name__ == '__main__':
    main()