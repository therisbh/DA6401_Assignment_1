"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import sys
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


'''# Add src/ directory to path so ann/ and utils/ imports work
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
# Also add parent directory in case script is run from outside src/
_parent_dir = os.path.dirname(_src_dir)
if os.path.isdir(os.path.join(_parent_dir, 'src')):
    if os.path.join(_parent_dir, 'src') not in sys.path:
        sys.path.insert(0, os.path.join(_parent_dir, 'src'))'''

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ann.neural_network import NeuralNetwork
from ann.objective_functions import get_loss
from utils.data_loader import load_data



def parse_arguments():

    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=3)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005)
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'])
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, default=128)
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'])

    parser.add_argument('-w_p', '--wandb_project', type=str, default=None,
                        help='W&B project name (if not provided, wandb logging is skipped)')
    parser.add_argument('--wandb_entity', type=str, default=None)

    parser.add_argument('--model_path', type=str, default='best_model.npy')

    return parser.parse_args()


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):

    logits = model.forward(X_test)

    predicted = np.argmax(logits, axis=1)

    if y_test.ndim == 2:
        true_labels = np.argmax(y_test, axis=1)
    else:
        true_labels = y_test

    loss_fn, _ = get_loss(model.args.loss)
    loss = loss_fn(y_test, logits)

    acc = accuracy_score(true_labels, predicted)
    prec = precision_score(true_labels, predicted, average='macro', zero_division=0)
    rec = recall_score(true_labels, predicted, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted, average='macro', zero_division=0)

    return {
        "logits": logits,
        "predicted": predicted,
        "true_labels": true_labels,
        "loss": float(loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def main():

    args = parse_arguments()

    # initialize wandb only if project is explicitly provided
    wandb_run = None
    if args.wandb_project is not None:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name="q8_error_analysis",
                config=vars(args)
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing without wandb logging...")

    src_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(src_dir, args.model_path)

    config_path = os.path.join(src_dir, 'best_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        args.dataset = config.get('dataset', args.dataset)
        args.num_layers = config.get('num_layers', args.num_layers)
        args.hidden_size = config.get('hidden_size', args.hidden_size)
        args.activation = config.get('activation', args.activation)
        args.loss = config.get('loss', args.loss)
        args.optimizer = config.get('optimizer', args.optimizer)
        args.learning_rate = config.get('learning_rate', args.learning_rate)
        args.weight_decay = config.get('weight_decay', args.weight_decay)
        args.weight_init = config.get('weight_init', args.weight_init)

        print("Loaded config from best_config.json")

    print(f"Model : {model_path}")
    print(f"Dataset: {args.dataset}")

    X_train, X_val, X_test, y_train, y_val, y_test, y_test_raw = load_data(args.dataset)

    model = NeuralNetwork(args)

    weights = load_model(model_path)
    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)

    print("\n=== Evaluation Results ===")
    print(f"Loss      : {results['loss']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1-score  : {results['f1']:.4f}")
    print("==========================")

    # ---- Q2.8 ERROR ANALYSIS ----

    preds = results["predicted"]
    true_labels = results["true_labels"]

    cm = confusion_matrix(true_labels, preds)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    if wandb_run is not None:
        import wandb
        misclassified = []
        for i in range(len(true_labels)):
            if preds[i] != true_labels[i]:
                misclassified.append(
                    wandb.Image(
                        X_test[i].reshape(28, 28),
                        caption=f"True: {true_labels[i]} | Pred: {preds[i]}"
                    )
                )
        wandb_run.log({
            "confusion_matrix_image": wandb.Image(plt),
            "misclassified_examples": misclassified[:50],
            "test_accuracy": results["accuracy"]
        })
        wandb_run.finish()
        print("Logged misclassified images to W&B")
    else:
        plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
        print("Saved confusion_matrix.png locally (no W&B project provided)")

    return results


if __name__ == '__main__':
    main()