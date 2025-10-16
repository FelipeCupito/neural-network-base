"""
Optimizer Comparison Experiment for Digit Classification

Compares SGD, SGD with Momentum, and Adam optimizers across different learning rates.

Metrics:
- Epochs to reach 100% accuracy on clean data
- Accuracy on noisy data (15% Gaussian noise)
- Calibration score (average maximum probability)
"""

import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_network.core.network import NeuralNetwork
from neural_network.core.trainer import Trainer
from neural_network.core.losses.mse import mse_loss
from neural_network.config import OptimizerConfig


def load_digit_patterns_for_classification(file_path):
    """Load 7x5 digit patterns for classification."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    X = np.array([list(map(int, " ".join(lines[i:i+7]).split()))
                  for i in range(0, len(lines), 7)], dtype=np.float32)

    # One-hot encode the labels for 10 classes
    num_digits = len(X)
    y = np.zeros((num_digits, 10), dtype=np.float32)
    for i in range(num_digits):
        y[i, i] = 1.0

    return X, y


def add_gaussian_noise(X, noise_level=0.15, seed=None):
    """Add Gaussian noise to patterns."""
    if seed is not None:
        np.random.seed(seed)

    X_noisy = X.copy()
    num_samples, num_features = X.shape

    gaussian_noise = np.random.normal(0, noise_level, (num_samples, num_features))
    X_noisy = X_noisy + gaussian_noise
    X_noisy = np.clip(X_noisy, 0, 1)

    return X_noisy.astype(np.float32)


def create_optimizer_config(optimizer_type: str, momentum: float = 0.9,
                           beta1: float = 0.9, beta2: float = 0.999) -> OptimizerConfig:
    """Create optimizer configuration."""
    if optimizer_type == "sgd":
        return OptimizerConfig(type="sgd")
    elif optimizer_type == "sgd_momentum":
        return OptimizerConfig(type="sgd_momentum", momentum=momentum)
    elif optimizer_type == "adam":
        return OptimizerConfig(type="adam", beta1=beta1, beta2=beta2, epsilon=1e-8)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def train_and_evaluate(X, y, X_noisy, topology, learning_rate, optimizer_type,
                      momentum=0.9, max_epochs=20000, seed=42, verbose=False):
    np.random.seed(seed)

    network = NeuralNetwork(
        topology=topology,
        activation_type="SIGMOID",
        dropout_rate=0.0
    )

    optimizer_config = create_optimizer_config(optimizer_type, momentum=momentum)

    trainer = Trainer(
        learning_rate=learning_rate,
        epochs=max_epochs,
        network=network,
        loss_func=mse_loss,
        optimizer_config=optimizer_config
    )

    num_samples = X.shape[0]
    batch_size = num_samples  # Full batch

    #Metrics
    epochs_to_100 = None
    acc_arr = np.empty(max_epochs)
    loss_arr = np.empty(max_epochs)

    for epoch in range(1, max_epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_inputs = X[indices]
        shuffled_labels = y[indices]

        predictions = network.forward(shuffled_inputs, training=True)

        if np.isnan(predictions).any() or np.isinf(predictions).any():
            if verbose:
                print(f"Warning: NaN/Inf at epoch {epoch}")
            break

        batch_loss, _ = mse_loss(predictions, shuffled_labels)
        loss_arr[epoch-1] = batch_loss

        # Backward pass
        deltas = []
        delta_output = (predictions - shuffled_labels) * network.layers[-1].get_activation_derivative()
        deltas.insert(0, delta_output)

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
            inputs_to_use = shuffled_inputs if l == 0 else network.layers[l - 1].outputs
            delta = deltas[l]
            grad_weights = np.dot(inputs_to_use.T, delta) / delta.shape[0]
            grad_bias = np.mean(delta, axis=0)
            gradients = np.vstack([grad_weights, grad_bias])
            layer_gradients = [gradients[:, i] for i in range(gradients.shape[1])]
            all_gradients.append(layer_gradients)

        # Update weights
        trainer.optimizer.update_network(network, all_gradients, learning_rate)

        # Check accuracy
        predictions = network.forward(X, training=False)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(predicted_classes == true_classes) * 100
        #Save metric
        acc_arr[epoch-1] = accuracy

        if accuracy == 100.0 and epochs_to_100 is None:
            epochs_to_100 = epoch
            if verbose:
                print(f"  Reached 100% at epoch {epoch}")
            acc_arr[epoch-1:] = accuracy
            loss_arr[epoch-1:] = batch_loss
            break

    if epochs_to_100 is None:
        epochs_to_100 = max_epochs
        if verbose:
            print(f"  Did not reach 100% (final: {accuracy:.2f}%)")

    # Evaluate on noisy data
    predictions_noisy = network.forward(X_noisy, training=False)
    predicted_classes_noisy = np.argmax(predictions_noisy, axis=1)
    noisy_accuracy = np.mean(predicted_classes_noisy == true_classes) * 100

    # Calculate calibration score (average maximum probability)
    # Higher values indicate the model is more confident in its predictions
    max_probs = np.max(predictions_noisy, axis=1)
    calibration_score = np.mean(max_probs) * 100

    return {
        'epochs_to_100': epochs_to_100,
        'noisy_accuracy': noisy_accuracy,
        'calibration_score': calibration_score,
        'loss_arr': loss_arr,
        'acc_arr': acc_arr
    }


def run_optimizer_comparison(dataset_path, noise_level=0.15, seed=42):
    """Run complete optimizer comparison experiment."""

    print("="*70)
    print("OPTIMIZER COMPARISON EXPERIMENT")
    print("="*70)
    print(f"\nDataset: {dataset_path}")
    print(f"Noise level: {noise_level*100:.1f}%")
    print(f"Topology: [35, 25, 15, 10]")
    print(f"Max epochs: 20000")
    print(f"Seed: {seed}")

    X, y = load_digit_patterns_for_classification(dataset_path)
    X_noisy = add_gaussian_noise(X, noise_level=noise_level, seed=seed)

    experiments = [
        # SGD (no momentum)
        {'optimizer': 'sgd', 'lr': 0.001, 'momentum': 0.0},
        {'optimizer': 'sgd', 'lr': 0.01, 'momentum': 0.0},
        {'optimizer': 'sgd', 'lr': 0.1, 'momentum': 0.0},

        # SGD with Momentum
        {'optimizer': 'sgd_momentum', 'lr': 0.001, 'momentum': 0.9},
        {'optimizer': 'sgd_momentum', 'lr': 0.01, 'momentum': 0.9},
        {'optimizer': 'sgd_momentum', 'lr': 0.1, 'momentum': 0.9},

        # Adam
        {'optimizer': 'adam', 'lr': 0.001, 'momentum': 0.9},
        {'optimizer': 'adam', 'lr': 0.01, 'momentum': 0.9},
        {'optimizer': 'adam', 'lr': 0.1, 'momentum': 0.9},
    ]

    topology = [35, 25, 15, 10]
    results = []

    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)

    for i, exp in enumerate(experiments, 1):
        optimizer_name = exp['optimizer'].upper().replace('_', ' ')
        print(f"\n[{i}/{len(experiments)}] {optimizer_name} - LR={exp['lr']}")

        metrics = train_and_evaluate(
            X=X,
            y=y,
            X_noisy=X_noisy,
            topology=topology,
            learning_rate=exp['lr'],
            optimizer_type=exp['optimizer'],
            momentum=exp['momentum'],
            max_epochs=20000,
            seed=seed,
            verbose=True
        )

        result = {
            'optimizer': optimizer_name,
            'lr': exp['lr'],
            'epochs_to_100': metrics['epochs_to_100'],
            'noisy_accuracy': metrics['noisy_accuracy'],
            'calibration': metrics['calibration_score'],
            'loss_arr': metrics['loss_arr'],
            'acc_arr': metrics['acc_arr']

        }
        results.append(result)

        print(f"  ✓ Epochs to 100%: {metrics['epochs_to_100']}")
        print(f"  ✓ Noisy Accuracy: {metrics['noisy_accuracy']:.2f}%")
        print(f"  ✓ Calibration: {metrics['calibration_score']:.2f}%")

    return results


def plot_comparison_results(results, output_dir):
    """Create visualization plots for the comparison."""
    os.makedirs(output_dir, exist_ok=True)

    # Change from 2x2 to 3x2 grid to accommodate new plots
    plt.rcParams['figure.figsize'] = (20, 16)  # Increased height for additional plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 16))

    # Color palette for optimizers
    optimizer_colors = {
        'SGD': '#e74c3c',
        'SGD MOMENTUM': '#3498db',
        'ADAM': '#2ecc71'
    }
    linetypes = {
        0.001: '-',
        0.01: ':',
        0.1: '-.'
    }

    labels = [f"{r['optimizer']}\nLR={r['lr']}" for r in results]
    colors = [optimizer_colors[r['optimizer']] for r in results]
    epochs = [r['epochs_to_100'] for r in results]
    noisy_acc = [r['noisy_accuracy'] for r in results]
    calibration = [r['calibration'] for r in results]

    # 1. Epochs to 100% - Bar plot
    ax1 = axes[0, 0]
    x_pos = np.arange(len(results))
    bars = ax1.bar(x_pos, epochs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, epochs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Epochs to 100% Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Training Speed Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Noisy Accuracy - Bar plot
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, noisy_acc, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, noisy_acc):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy on Noisy Data (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Robustness to Noise (15% Gaussian)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.grid(axis='y', alpha=0.3)

    # 3. Calibration Score - Bar plot
    ax3 = axes[1, 0]
    bars = ax3.bar(x_pos, calibration, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, calibration):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax3.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Avg. Maximum Probability (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Model Calibration (Confidence)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.set_ylim([0, 105])
    ax3.grid(axis='y', alpha=0.3)

    # 4. Combined comparison - Line plot by optimizer
    ax4 = axes[1, 1]

    # Group by optimizer
    optimizers = {}
    for r in results:
        opt = r['optimizer']
        if opt not in optimizers:
            optimizers[opt] = {'lr': [], 'acc': []}
        optimizers[opt]['lr'].append(r['lr'])
        optimizers[opt]['acc'].append(r['noisy_accuracy'])

    for opt, data in optimizers.items():
        sorted_indices = np.argsort(data['lr'])
        sorted_lr = np.array(data['lr'])[sorted_indices]
        sorted_acc = np.array(data['acc'])[sorted_indices]
        ax4.plot(sorted_lr, sorted_acc,
                marker='o', linewidth=2.5, markersize=10, label=opt,
                color=optimizer_colors[opt], alpha=0.8)

    ax4.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Noisy Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy vs Learning Rate by Optimizer', fontsize=14, fontweight='bold')
    ax4.set_xscale('log')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_ylim([0, 105])

    # 5: Loss vs Epochs for each optimizer
    ax5 = axes[0, 2]
    
    for r in results:
        optimizer = r['optimizer']
        lr = r['lr']
        loss_arr = r['loss_arr']
        epochs_range = range(1, len(loss_arr) + 1)
        
        label = f"{optimizer} LR={lr}"
        ax5.plot(epochs_range, loss_arr, 
                color=optimizer_colors[optimizer], 
                alpha=0.5, 
                linewidth=3,
                linestyle=linetypes[lr],
                label=label)

    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax5.set_title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=9, loc='best', handlelength=4)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(bottom=0)  # Often useful for loss plots

    # 6: Accuracy vs Epochs for each optimizer
    ax6 = axes[1, 2]
    
    for r in results:
        optimizer = r['optimizer']
        lr = r['lr']
        acc_arr = r['acc_arr']
        epochs_range = range(1, len(acc_arr) + 1)
        
        label = f"{optimizer} LR={lr}"
        ax6.plot(epochs_range, acc_arr, 
                color=optimizer_colors[optimizer], 
                alpha=0.5, 
                linewidth=3,
                linestyle=linetypes[lr],
                label=label)

    ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Training Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=9, loc='best', handlelength=4)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 105])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'optimizer_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plots saved to: {plot_path}")
    plt.close()


def print_summary_table(results):
    """Print a formatted summary table."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Optimizer':<15} {'LR':<8} {'Epochs':<10} {'Noisy Acc':<12} {'Calibration':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r['optimizer']:<15} {r['lr']:<8.3f} {r['epochs_to_100']:<10} "
              f"{r['noisy_accuracy']:<12.2f} {r['calibration']:<12.2f}")

    print("\n" + "="*70)
    print("BEST PERFORMERS")
    print("="*70)

    # Best by each metric
    best_speed = min(results, key=lambda x: x['epochs_to_100'])
    best_noisy = max(results, key=lambda x: x['noisy_accuracy'])
    best_calibration = max(results, key=lambda x: x['calibration'])

    print(f"\n🏆 Fastest Training (Epochs to 100%):")
    print(f"   {best_speed['optimizer']} (LR={best_speed['lr']})")
    print(f"   Epochs: {int(best_speed['epochs_to_100'])}")

    print(f"\n🛡️  Best Robustness (Noisy Accuracy):")
    print(f"   {best_noisy['optimizer']} (LR={best_noisy['lr']})")
    print(f"   Accuracy: {best_noisy['noisy_accuracy']:.2f}%")

    print(f"\n🎯 Best Calibration (Confidence):")
    print(f"   {best_calibration['optimizer']} (LR={best_calibration['lr']})")
    print(f"   Score: {best_calibration['calibration']:.2f}%")

    # Overall recommendation (weighted score)
    max_epochs = max(r['epochs_to_100'] for r in results)
    for r in results:
        r['overall_score'] = (
            (1 - r['epochs_to_100'] / max_epochs) * 0.3 +  # 30% weight on speed
            (r['noisy_accuracy'] / 100) * 0.5 +  # 50% weight on accuracy
            (r['calibration'] / 100) * 0.2   # 20% weight on calibration
        )

    best_overall = max(results, key=lambda x: x['overall_score'])
    print(f"\n⭐ Overall Best Configuration:")
    print(f"   {best_overall['optimizer']} (LR={best_overall['lr']})")
    print(f"   Epochs: {int(best_overall['epochs_to_100'])}")
    print(f"   Noisy Accuracy: {best_overall['noisy_accuracy']:.2f}%")
    print(f"   Calibration: {best_overall['calibration']:.2f}%")

    print("\n" + "="*70)


def save_results_csv(results, output_dir):
    """Save results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'optimizer_comparison_results.csv')

    with open(csv_path, 'w') as f:
        f.write("Optimizer,Learning Rate,Epochs to 100%,Noisy Accuracy (%),Calibration Score (%)\n")
        for r in results:
            f.write(f"{r['optimizer']},{r['lr']},{r['epochs_to_100']},"
                   f"{r['noisy_accuracy']:.2f},{r['calibration']:.2f}\n")

    print(f"\n💾 Results saved to: {csv_path}")


def main():
    dataset_path = Path(__file__).parent.parent.parent / "resources/datasets/TP3-ej3-digitos.txt"
    output_dir = Path(__file__).parent / "outputs" / "optimizer_comparison"

    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return

    results = run_optimizer_comparison(
        dataset_path=str(dataset_path),
        noise_level=0.15,
        seed=42
    )

    print_summary_table(results)
    save_results_csv(results, str(output_dir))
    plot_comparison_results(results, str(output_dir))
    print("\n✅ EXPERIMENT COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
