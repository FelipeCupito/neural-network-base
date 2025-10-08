"""
TP3 - Exercise 3: Multilayer Perceptron (MLP) for XOR, Parity Discrimination, and Digit Recognition

Three problems in one script:
1. XOR: Classic non-linearly separable logical function
2. Parity Discrimination: Classify digits (0-9) as odd or even using 7x5 binary patterns
3. Digit Discrimination: Recognize which digit (0-9) corresponds to the network's input
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_network.core.network import NeuralNetwork
from neural_network.core.trainer import Trainer
from neural_network.core.losses.mse import mse_loss
from neural_network.config import OptimizerConfig, ConfigLoader
import matplotlib.pyplot as plt


def create_xor_data():
    # Inputs in bipolar format {-1, 1}
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
    # Single output: 0 or 1 (normalized for sigmoid)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return X, y

def load_digit_patterns(file_path):
    """
    Load 7x5 digit patterns from file.
    Each digit is represented as 7 rows x 5 columns = 35 binary features.

    Returns:
        X: Array of shape (10, 35) - 10 digits, each with 35 features
        y: Array of shape (10, 1) - Parity labels (0=even, 1=odd)
        digit_labels: List of digit values [0, 1, 2, ..., 9]
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    X = np.array([list(map(int, " ".join(lines[i:i+7]).split())) for i in range(0, len(lines), 7)], dtype=np.float32)
    y = np.array([[i % 2] for i in range(len(X))], dtype=np.float32)
    return X, y, list(range(len(X)))


def load_digit_patterns_for_classification(file_path):
    """
    Load 7x5 digit patterns from file for digit classification (0-9).
    Each digit is represented as 7 rows x 5 columns = 35 binary features.

    Returns:
        X: Array of shape (10, 35) - 10 digits, each with 35 features
        y: Array of shape (10, 10) - One-hot encoded labels for digits 0-9
        digit_labels: List of digit values [0, 1, 2, ..., 9]
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    X = np.array([list(map(int, " ".join(lines[i:i+7]).split())) for i in range(0, len(lines), 7)], dtype=np.float32)

    # One-hot encode the labels for 10 classes
    num_digits = len(X)
    y = np.zeros((num_digits, 10), dtype=np.float32)
    for i in range(num_digits):
        y[i, i] = 1.0  # One-hot encoding

    digit_labels = list(range(num_digits))
    return X, y, digit_labels


def add_noise_to_patterns(X, noise_level=0.1, seed=None):
    """
    Add Gaussian noise to digit patterns.

    Args:
        X: Input patterns (N, 35) - binary patterns with values in {0, 1}
        noise_level: Standard deviation of Gaussian noise (0.0 to 1.0)
                    Recommended: 0.1-0.3 for moderate noise
        seed: Random seed for reproducibility

    Returns:
        Noisy patterns clipped to [0, 1] range
    """
    if seed is not None:
        np.random.seed(seed)

    X_noisy = X.copy()
    num_samples, num_features = X.shape

    # Add Gaussian noise with mean=0 and std=noise_level
    gaussian_noise = np.random.normal(0, noise_level, (num_samples, num_features))
    X_noisy = X_noisy + gaussian_noise

    # Clip values to [0, 1] range to keep them valid
    X_noisy = np.clip(X_noisy, 0, 1)

    return X_noisy.astype(np.float32)


def create_augmented_test_set(X, y, num_augmentations=10, noise_level=0.15, seed=None):
    """
    Create an augmented test set by generating multiple noisy versions of each digit.
    This creates a more robust test set with multiple samples per class.

    Args:
        X: Original clean patterns (N, 35)
        y: Original labels (N, 10) - one-hot encoded
        num_augmentations: Number of noisy versions to create per digit
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        X_test: Augmented test set (N * num_augmentations, 35)
        y_test: Repeated labels (N * num_augmentations, 10)
    """
    if seed is not None:
        np.random.seed(seed)

    X_test_list = []
    y_test_list = []

    for i in range(len(X)):
        for aug_idx in range(num_augmentations):
            # Create a noisy version with a different seed for each augmentation
            noise_seed = seed + i * num_augmentations + aug_idx if seed is not None else None
            X_noisy = add_noise_to_patterns(X[i:i+1], noise_level=noise_level, seed=noise_seed)
            X_test_list.append(X_noisy)
            y_test_list.append(y[i:i+1])

    X_test = np.vstack(X_test_list)
    y_test = np.vstack(y_test_list)

    return X_test, y_test


def print_digit(pattern, digit_label, rows=7, cols=5):
    """Print a digit pattern in a readable format."""
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            idx = i * cols + j
            row_str += "█" if pattern[idx] == 1 else " "
        print(f"  {row_str}")


def create_optimizer_config(optimizer_type: str, momentum: float = 0.9) -> OptimizerConfig:
    if optimizer_type == "sgd":
        return OptimizerConfig(type="sgd")
    elif optimizer_type == "sgd_momentum":
        return OptimizerConfig(type="sgd_momentum", momentum=momentum)
    elif optimizer_type == "adam":
        return OptimizerConfig(type="adam", beta1=0.9, beta2=0.999, epsilon=1e-8)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def train_mlp(problem_type: str, X, y, topology, learning_rate=0.5, epochs=10000,
              batch_size=None, optimizer_type="sgd_momentum", momentum=0.9,
              log_interval=1000, seed=42, verbose=True, extra_data=None):
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING MLP FOR {problem_type.upper()}")
        print(f"{'='*60}")
        print("\n  Training on CLEAN DATA ONLY")
        if extra_data and extra_data.get('X_noisy') is not None:
            noise_level = extra_data.get('noise_level', 0.0) * 100
            print(f"  (Noisy data with {noise_level:.1f}% noise will be used for testing)")

    if seed is not None:
        np.random.seed(seed)

    if batch_size is None:
        batch_size = len(X)

    network = NeuralNetwork(
        topology=topology,
        activation_type="SIGMOID",
        dropout_rate=0.0
    )

    optimizer_config = create_optimizer_config(optimizer_type, momentum)

    trainer = Trainer(
        learning_rate=learning_rate,
        epochs=epochs,
        network=network,
        loss_func=mse_loss,
        optimizer_config=optimizer_config
    )

    loss_history = []
    accuracy_history = []

    print("STARTING TRAINING")

    num_samples = X.shape[0]

    # Determine if this is multi-class classification
    is_multiclass = y.shape[1] > 1 if len(y.shape) > 1 else False

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        # Shuffle data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_inputs = X[indices]
        shuffled_labels = y[indices]

        # Mini-batch training
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_inputs = shuffled_inputs[start_idx:end_idx]
            batch_labels = shuffled_labels[start_idx:end_idx]

            # Forward pass
            predictions = network.forward(batch_inputs, training=True)

            if np.isnan(predictions).any() or np.isinf(predictions).any():
                if verbose:
                    print(f"Warning: NaN or Inf values detected at epoch {epoch}")
                continue

            batch_loss, _ = mse_loss(predictions, batch_labels)
            total_loss += batch_loss

            # Backward pass - calculate deltas
            deltas = []
            delta_output = (predictions - batch_labels) * network.layers[-1].get_activation_derivative()
            deltas.insert(0, delta_output)

            # Backpropagate through hidden layers
            for l in range(len(network.layers) - 2, -1, -1):
                current_layer = network.layers[l]
                next_layer = network.layers[l + 1]
                weights_next_layer = next_layer.weights[:-1, :]
                delta_next = deltas[0]
                delta = np.dot(delta_next, weights_next_layer.T) * current_layer.get_activation_derivative()
                deltas.insert(0, delta)

            # Calculate gradients
            all_gradients = []
            for l, layer in enumerate(network.layers):
                inputs_to_use = batch_inputs if l == 0 else network.layers[l - 1].outputs
                delta = deltas[l]
                grad_weights = np.dot(inputs_to_use.T, delta) / delta.shape[0]
                grad_bias = np.mean(delta, axis=0)
                gradients = np.vstack([grad_weights, grad_bias])
                layer_gradients = [gradients[:, i] for i in range(gradients.shape[1])]
                all_gradients.append(layer_gradients)

            # Update weights using optimizer
            trainer.optimizer.update_network(network, all_gradients, learning_rate)

        # Calculate accuracy
        predictions = network.forward(X, training=False)

        if is_multiclass:
            # Multi-class classification
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1)
            accuracy = np.mean(predicted_classes == true_classes) * 100
        else:
            # Binary classification
            predicted_classes = np.where(predictions > 0.5, 1, 0)
            accuracy = np.mean(predicted_classes == y) * 100

        loss_history.append(total_loss)
        accuracy_history.append(accuracy)

        if verbose and (epoch % log_interval == 0 or epoch == 1):
            print(f"Epoch {epoch:5d}/{epochs} - Loss: {total_loss:.6f} - Accuracy: {accuracy:.2f}%")

        if accuracy == 100.0 and epoch > 100:
            print(f"\n✓ Perfect accuracy achieved at epoch {epoch}!")
            break

    print("-"*60)
    print("TRAINING COMPLETE")
    print("-"*60)
    print("\n" + "="*60)
    print("POST-TRAINING EVALUATION")
    print("="*60)

    print("\n1. Evaluating on CLEAN training data...")
    predictions = network.forward(X, training=False)

    if is_multiclass:
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        final_accuracy = np.mean(predicted_classes == true_classes) * 100
    else:
        predicted_classes = np.where(predictions > 0.5, 1, 0)
        true_classes = y
        final_accuracy = np.mean(predicted_classes == y) * 100

    evaluation_metrics = {
        'clean_accuracy': final_accuracy
    }
    print(f" ✓ Clean Data Accuracy: {final_accuracy:.2f}%")

    # Optional evaluation on noisy data (same labels as clean digits)
    if extra_data and extra_data.get('X_noisy') is not None:
        print("\n2. Evaluating on NOISY test data...")
        X_noisy = extra_data['X_noisy']
        predictions_noisy = network.forward(X_noisy, training=False)

        if is_multiclass:
            predicted_classes_noisy = np.argmax(predictions_noisy, axis=1)
            noisy_accuracy = np.mean(predicted_classes_noisy == true_classes) * 100
        else:
            predicted_classes_noisy = np.where(predictions_noisy > 0.5, 1, 0)
            noisy_accuracy = np.mean(predicted_classes_noisy == y) * 100

        evaluation_metrics['noisy_accuracy'] = noisy_accuracy
        if extra_data.get('noise_level') is not None:
            evaluation_metrics['noise_level'] = extra_data['noise_level']

        noise_level = extra_data.get('noise_level', 0.0) * 100
        print(f" ✓ Noisy Data Accuracy ({noise_level:.1f}% noise): {noisy_accuracy:.2f}%")

        # Calculate accuracy drop
        accuracy_drop = final_accuracy - noisy_accuracy
        print(f"\n    Accuracy Drop due to Noise: {accuracy_drop:.2f}%")
        print(f"    Robustness: {(noisy_accuracy/final_accuracy)*100:.1f}% of clean performance")
    print("="*60)

    if verbose:
        print(f"\nEvaluation Complete")

    return network, loss_history, accuracy_history, final_accuracy, evaluation_metrics


def print_xor_results(network, X, y):
    """Print XOR test results in a readable format."""
    print("\n" + "="*60)
    print("XOR NETWORK PREDICTIONS")
    print("="*60)

    predictions = network.forward(X, training=False)
    predicted_classes = np.where(predictions > 0.5, 1, 0)

    # Convert to bipolar for display
    bipolar_map = {0: -1, 1: 1}

    print(f"\n{'Input':<15}  {'Expected':<10}  {'Raw Output':<12}  {'Predicted':<12}  {'Correct'}")
    print("-" * 85)

    for i in range(len(X)):
        input_str = f"[{X[i, 0]:2.0f}, {X[i, 1]:2.0f}]"
        expected_bipolar = bipolar_map[int(y[i, 0])]
        raw_output = predictions[i, 0]
        predicted_bipolar = bipolar_map[int(predicted_classes[i, 0])]
        correct = "✓" if predicted_classes[i, 0] == y[i, 0] else "✗"

        print(f"{input_str:<15}  {expected_bipolar:^10}  {raw_output:^12.4f}  {predicted_bipolar:^12}  {correct}")

    accuracy = np.mean(predicted_classes == y) * 100
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("="*60)


def print_parity_results(network, X, y, digit_labels):
    """Print parity discrimination results in a readable format."""
    print("\n" + "="*60)
    print("PARITY DISCRIMINATION PREDICTIONS")
    print("="*60)

    predictions = network.forward(X, training=False)
    predicted_classes = np.where(predictions > 0.5, 1, 0)

    # Show each digit pattern with its prediction
    for i, digit in enumerate(digit_labels):
        print(f"\n{'─' * 40}")
        print(f"Analyzing Digit: {digit}")
        print(f"{'─' * 40}")
        print_digit(X[i], digit)

        # Show prediction
        expected_parity = "ODD" if y[i, 0] == 1 else "EVEN"
        predicted_parity = "ODD" if predicted_classes[i, 0] == 1 else "EVEN"
        raw_output = predictions[i, 0]
        correct = "✓" if predicted_classes[i, 0] == y[i, 0] else "✗"

        print(f"\n  Expected:  {expected_parity}")
        print(f"  Raw Output: {raw_output:.4f}")
        print(f"  Predicted: {predicted_parity}")
        print(f"  Result:    {correct} {'CORRECT' if correct == '✓' else 'INCORRECT'}")

    print(f"\n{'─' * 40}")
    accuracy = np.mean(predicted_classes == y) * 100
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("="*60)


def print_digit_classification_results(network, X, y, digit_labels, X_noisy=None, noise_level=0.0):
    """Print digit classification results in a readable format."""
    print("\n" + "="*60)
    print("DIGIT CLASSIFICATION PREDICTIONS")
    print("="*60)

    # Test on clean data
    predictions = network.forward(X, training=False)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y, axis=1)

    print("\n" + "-"*60)
    print("CLEAN DATA RESULTS")
    print("-"*60)

    # Show each digit pattern with its prediction
    for i, digit in enumerate(digit_labels):
        print(f"\n{'─' * 40}")
        print(f"Digit: {digit}")
        print(f"{'─' * 40}")
        print_digit(X[i], digit)

        # Show prediction
        predicted_digit = predicted_classes[i]
        confidence = predictions[i, predicted_digit] * 100
        correct = "✓" if predicted_digit == true_classes[i] else "✗"

        print(f"\n  Expected:   {true_classes[i]}")
        print(f"  Predicted:  {predicted_digit}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Result:     {correct} {'CORRECT' if correct == '✓' else 'INCORRECT'}")

        # Show top-3 predictions
        top3_indices = np.argsort(predictions[i])[::-1][:3]
        print(f"\n  Top 3 Predictions:")
        for rank, idx in enumerate(top3_indices, 1):
            print(f"    {rank}. Digit {idx}: {predictions[i, idx]*100:.2f}%")

    accuracy = np.mean(predicted_classes == true_classes) * 100
    print(f"\n{'─' * 40}")
    print(f"Overall Accuracy (Clean): {accuracy:.2f}%")
    print("="*60)

    # Test on noisy data if provided
    if X_noisy is not None:
        print("\n" + "-"*60)
        print(f"NOISY DATA RESULTS (Noise Level: {noise_level*100:.1f}%)")
        print("-"*60)

        predictions_noisy = network.forward(X_noisy, training=False)
        predicted_classes_noisy = np.argmax(predictions_noisy, axis=1)

        for i, digit in enumerate(digit_labels):
            print(f"\n{'─' * 40}")
            print(f"Digit: {digit} (with noise)")
            print(f"{'─' * 40}")
            print_digit(X_noisy[i], digit)

            # Show prediction
            predicted_digit = predicted_classes_noisy[i]
            confidence = predictions_noisy[i, predicted_digit] * 100
            correct = "✓" if predicted_digit == true_classes[i] else "✗"

            print(f"\n  Expected:   {true_classes[i]}")
            print(f"  Predicted:  {predicted_digit}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Result:     {correct} {'CORRECT' if correct == '✓' else 'INCORRECT'}")

        accuracy_noisy = np.mean(predicted_classes_noisy == true_classes) * 100
        print(f"\n{'─' * 40}")
        print(f"Overall Accuracy (Noisy): {accuracy_noisy:.2f}%")
        print(f"Accuracy Drop: {accuracy - accuracy_noisy:.2f}%")
        print("="*60)


def plot_training_comparison(xor_history, parity_history, output_dir):
    """Plot training history comparison between XOR and Parity problems."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # XOR Loss
    ax1.plot(range(1, len(xor_history['loss']) + 1), xor_history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('XOR - Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # XOR Accuracy
    ax2.plot(range(1, len(xor_history['accuracy']) + 1), xor_history['accuracy'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('XOR - Training Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # Parity Loss
    ax3.plot(range(1, len(parity_history['loss']) + 1), parity_history['loss'], 'r-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss (MSE)', fontsize=12)
    ax3.set_title('Parity Discrimination - Training Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Parity Accuracy
    ax4.plot(range(1, len(parity_history['accuracy']) + 1), parity_history['accuracy'], 'orange', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Parity Discrimination - Training Accuracy', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining comparison plots saved to: {plot_path}")
    plt.close()


def plot_decision_boundary(network, X, y, output_dir, filename='xor_decision_boundary.png'):
    """Plot the decision boundary learned by the network (for 2D inputs only)."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on the mesh
    mesh_inputs = np.c_[xx.ravel(), yy.ravel()]
    Z = network.forward(mesh_inputs, training=False)
    Z = np.where(Z > 0.5, 1, -1)  # Convert to bipolar for visualization
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=[-1, 0, 1])
    plt.colorbar(label='Class', ticks=[-1, 1])

    # Plot training points
    bipolar_map = {0: -1, 1: 1}
    colors = ['red' if y[i, 0] == 1 else 'blue' for i in range(len(y))]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)

    # Add labels for each point
    for i, (x, y_val) in enumerate(X):
        label = str(bipolar_map[int(y[i, 0])])
        plt.annotate(label, (x, y_val), fontsize=14, ha='center', va='center', fontweight='bold', color='white')

    plt.xlabel('Input 1', fontsize=12)
    plt.ylabel('Input 2', fontsize=12)
    plt.title('XOR Decision Boundary', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Decision boundary plot saved to: {plot_path}")
    plt.close()


def plot_training_history(loss_history, accuracy_history, output_dir, problem_name):
    """Plot training loss and accuracy over epochs for a single problem."""
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(loss_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(epochs, loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title(f'{problem_name} - Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, accuracy_history, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{problem_name} - Training Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    safe_name = problem_name.lower().replace(' ', '_')
    plot_path = os.path.join(output_dir, f'{safe_name}_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved to: {plot_path}")
    plt.close()


def plot_confusion_matrix(network, X, y, digit_labels, output_dir, filename='confusion_matrix.png', title_suffix=''):
    """Plot confusion matrix for digit classification."""
    os.makedirs(output_dir, exist_ok=True)

    predictions = network.forward(X, training=False)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y, axis=1)

    # Create confusion matrix
    num_classes = len(digit_labels)
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(true_classes, predicted_classes):
        confusion[true_label, pred_label] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set ticks
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(digit_labels)
    ax.set_yticklabels(digit_labels)

    # Add labels
    ax.set_xlabel('Predicted Digit', fontsize=12)
    ax.set_ylabel('True Digit', fontsize=12)
    title = f'Digit Classification - Confusion Matrix{title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, confusion[i, j],
                          ha="center", va="center", color="black" if confusion[i, j] < confusion.max()/2 else "white",
                          fontsize=12, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {plot_path}")
    plt.close()


def run_experiment_from_config(config_path: str):
    config = ConfigLoader.load_config(config_path)

    print("\n" + "="*60)
    print(f"EXPERIMENT: {config.name.upper()}")
    print("="*60)
    print(f"\nDescription: {config.description}")

    problem_type = config.problem.type
    dataset_path = config.problem.dataset_path
    if dataset_path:
        dataset_path = Path(__file__).parent.parent.parent / dataset_path

    if config.seed is not None:
        np.random.seed(config.seed)

    # Load data based on problem type
    if problem_type == "xor":
        print("\nProblem Type: XOR")
        X, y = create_xor_data()
        extra_data = None
    elif problem_type == "parity":
        print("\nProblem Type: Parity Discrimination")
        if not dataset_path:
            raise ValueError("dataset_path is required for parity problem")
        X, y, digit_labels = load_digit_patterns(str(dataset_path))
        extra_data = {'digit_labels': digit_labels}
        print(f"\nDataset loaded: {len(X)} digits")
        print(f"Each digit: 7x5 = 35 binary features")
    elif problem_type == "digit_classification":
        print("\nProblem Type: Digit Classification (0-9)")
        if not dataset_path:
            raise ValueError("dataset_path is required for digit classification problem")
        X, y, digit_labels = load_digit_patterns_for_classification(str(dataset_path))

        # Get noise level from config if available
        noise_level = getattr(config.problem, 'noise_level', 0.0)
        X_noisy = None
        X_test_augmented = None
        y_test_augmented = None

        if noise_level > 0:
            # Create single noisy version for visual inspection
            X_noisy = add_noise_to_patterns(X, noise_level=noise_level, seed=config.seed)
            # Create augmented test set with multiple noisy samples per digit
            X_test_augmented, y_test_augmented = create_augmented_test_set(
                X, y, num_augmentations=100, noise_level=noise_level, seed=config.seed
            )
            print(f"  Augmented test set size: {len(X_test_augmented)} samples ({len(X_test_augmented)//10} per digit)")

        extra_data = {
            'digit_labels': digit_labels,
            'X_noisy': X_noisy,
            'X_test_augmented': X_test_augmented,
            'y_test_augmented': y_test_augmented,
            'noise_level': noise_level
        }
        print(f"\nDataset loaded: {len(X)} digits (0-9)")
        print(f"Each digit: 7x5 = 35 binary features")
        print(f"Output neurons: 10 (one-hot encoded)")
        if noise_level > 0:
            print(f"Noise level for testing: {noise_level*100:.1f}%")
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


    network, loss_history, accuracy_history, final_accuracy, evaluation_metrics = train_mlp(
        problem_type=problem_type,
        X=X,
        y=y,
        topology=config.network.architecture.topology,
        learning_rate=config.training.learning_rate,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        optimizer_type=config.training.optimizer.type,
        momentum=config.training.optimizer.momentum,
        log_interval=config.training.log_interval,
        seed=config.seed,
        verbose=True,
        extra_data=extra_data
    )

    # Print results
    if problem_type == "xor":
        print_xor_results(network, X, y)
    elif problem_type == "parity":
        print_parity_results(network, X, y, extra_data['digit_labels'])
    elif problem_type == "digit_classification":
        print_digit_classification_results(
            network, X, y, extra_data['digit_labels'],
            X_noisy=extra_data.get('X_noisy'),
            noise_level=extra_data.get('noise_level', 0.0)
        )

    if config.metrics.save_plots:
        plots_dir = Path(config.metrics.plots_path)
        plot_training_history(loss_history, accuracy_history, str(plots_dir), config.name)

        # Decision boundary for XOR
        if problem_type == "xor":
            safe_name = config.name.lower().replace(' ', '_')
            plot_decision_boundary(network, X, y, str(plots_dir), filename=f'{safe_name}_decision_boundary.png')

        # Confusion matrix for digit classification
        if problem_type == "digit_classification":
            safe_name = config.name.lower().replace(' ', '_')

            # Generate confusion matrix for clean training data
            plot_confusion_matrix(network, X, y, extra_data['digit_labels'], str(plots_dir),
                                filename=f'{safe_name}_confusion_matrix_clean.png',
                                title_suffix=' (Clean Training Data - 10 samples)')

            # Generate confusion matrix for augmented noisy test
            if extra_data.get('X_test_augmented') is not None:
                plot_confusion_matrix(network, extra_data['X_test_augmented'], extra_data['y_test_augmented'],
                                    extra_data['digit_labels'], str(plots_dir),
                                    filename=f'{safe_name}_confusion_matrix_test.png',
                                    title_suffix=f' (Noisy Test Set - {len(extra_data["X_test_augmented"])} samples, {extra_data.get("noise_level", 0.0)*100:.0f}% noise)')

    # Save weights
    if config.training.save_weights:
        weights_dir = Path(config.training.weights_path)
        weights_dir.mkdir(parents=True, exist_ok=True)
        safe_name = config.name.lower().replace(' ', '_')
        weights_file = weights_dir / f"{safe_name}_weights.npz"
        network.save_weights(str(weights_file))

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY:")
    print("="*60)
    print(f"\n Problem: {problem_type.upper()}")
    print(f" Topology: {config.network.architecture.topology}")
    print(f" Optimizer: {config.training.optimizer.type}")
    print(f" Learning Rate: {config.training.learning_rate}")
    print(f" Epochs trained: {len(loss_history)}")

    print(f"\n PERFORMANCE METRICS:")
    print(f" ├─ Clean Data Accuracy:  {evaluation_metrics['clean_accuracy']:.2f}%")
    if 'noisy_accuracy' in evaluation_metrics:
        noise_level = evaluation_metrics.get('noise_level', 0.0) * 100
        print(f" ├─ Noisy Data Accuracy:  {evaluation_metrics['noisy_accuracy']:.2f}% ({noise_level:.1f}% noise)")
        accuracy_drop = evaluation_metrics['clean_accuracy'] - evaluation_metrics['noisy_accuracy']
        robustness = (evaluation_metrics['noisy_accuracy'] / evaluation_metrics['clean_accuracy']) * 100
        print(f" ├─ Accuracy Drop:        {accuracy_drop:.2f}%")
        print(f" └─ Robustness Score:     {robustness:.1f}% of clean performance")

    print("\nEXPERIMENT COMPLETE!")
    print("="*60 + "\n")

    return network, loss_history, accuracy_history, final_accuracy


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="TP3 - Exercise 3: MLP for XOR, Parity Discrimination, and Digit Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_tp3.py xor_config.yaml
  python train_tp3.py parity_config.yaml
  python train_tp3.py digit_classification_config.yaml
  python train_tp3.py experiments/exercise_3/xor_config.yaml
        """
    )

    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default=None,
        help='Path to configuration file (YAML). If not provided, runs both problems.'
    )

    args = parser.parse_args()

    if args.config:
        # Single experiment mode
        config_path = Path(args.config)

        # If relative path, try to find it in the exercise_3 directory
        if not config_path.is_absolute():
            if not config_path.exists():
                exercise_dir = Path(__file__).parent
                config_path = exercise_dir / args.config

        if not config_path.exists():
            print(f"❌ Error: Configuration file not found: {config_path}")
            print("\nAvailable configs in exercise_3 directory:")
            exercise_dir = Path(__file__).parent
            for cfg in exercise_dir.glob("*.yaml"):
                print(f"  - {cfg.name}")
            sys.exit(1)

        print(f"\nLoading config: {config_path}")
        run_experiment_from_config(str(config_path))

    else:
        # Legacy mode: Run both problems sequentially
        print("\n" + "="*60)
        print("RUNNING ALL EXPERIMENTS (LEGACY MODE)")
        print("="*60)
        print("\nTip: Run individual experiments with:")
        print("  python train_tp3.py xor_config.yaml")
        print("  python train_tp3.py parity_config.yaml")
        print("  python train_tp3.py digit_classification_config.yaml")

        exercise_dir = Path(__file__).parent

        # Run XOR
        xor_config = exercise_dir / "xor_config.yaml"
        if xor_config.exists():
            run_experiment_from_config(str(xor_config))
        else:
            print("\n⚠️  Warning: xor_config.yaml not found, skipping XOR experiment")

        # Run Parity
        parity_config = exercise_dir / "parity_config.yaml"
        if parity_config.exists():
            run_experiment_from_config(str(parity_config))
        else:
            print("\n⚠️  Warning: parity_config.yaml not found, skipping Parity experiment")

        # Run Digit Classification
        digit_config = exercise_dir / "digit_classification_config.yaml"
        if digit_config.exists():
            run_experiment_from_config(str(digit_config))
        else:
            print("\n⚠️  Warning: digit_classification_config.yaml not found, skipping Digit Classification experiment")


if __name__ == "__main__":
    print("="*60)
    print("TP3 - EJERCICIO 3: MULTILAYER PERCEPTRON (MLP)")
    print("Implementación con NumPy y backpropagation manual")
    print("="*60)

    main()
