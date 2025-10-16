# loss_functions.py

import numpy as np

# -------------------------------
# Función de Pérdida: Softmax con Entropía Cruzada
# -------------------------------
def softmax_cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray) -> (float, np.ndarray):
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    epsilon = 1e-8
    softmax_probs = np.clip(softmax_probs, epsilon, 1. - epsilon)
    loss = -np.sum(labels * np.log(softmax_probs)) / logits.shape[0]
    return loss, softmax_probs-labels

def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> (float, float):
    "Cross entropy loss for one-hot encoded targets"
    return -np.sum(targets * np.log(predictions + 1e-8)), 


def mae(x1, x2) -> (float, np.ndarray):
    "Mean absolute err"
    return np.mean(np.abs(x1-x2)), np.sign(x1-x2)/x1.shape[0]

def mse(pred, label) -> (float, np.ndarray):
    "Mean absolute err"
    return np.mean((pred-label)**2), pred-label
