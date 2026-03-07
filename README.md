# DA6401 Assignment 1

**Name:** Rishabh Gupta  
**Roll Number:** DA25M024  

## W&B Reportlink:
https://api.wandb.ai/links/da25m024-iitm/81jbj901

## W&B Project link:
https://wandb.ai/da25m024-iitm/da6401_assignment1?nw=nwuserda25m024

## Github repo link:
https://github.com/therisbh/DA6401_Assignment_1

## Folder Structure

```
assignment_1/
│
│
├── notebook/
│ └── wandb_report.ipynb
│
├── src/
│ ├── ann/
│ │ ├── init.py
│ │ ├── activations.py
│ │ ├── neural_layer.py
│ │ ├── neural_network.py
│ │ ├── objective_functions.py
│ │ └── optimizers.py
│ │
│ ├── utils/
│ │ ├── init.py
│ │ └── data_loader.py
│ │
│ ├── best_config.json
│ ├── best_model.npy
│ ├── inference.py
│ └── train.py
│
└── requirements.txt

```
## Description

- Implemented a neural network from scratch for MNIST classification.
- Performed hyperparameter sweeps and experiment tracking using Weights & Biases.
- Analyzed effects of optimizers, activation functions, loss functions, and weight initialization.
- Conducted error analysis using confusion matrices and misclassified examples.
- Evaluated transfer performance of the trained model on the Fashion-MNIST dataset.