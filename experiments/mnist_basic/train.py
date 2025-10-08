# main.py
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys
import copy
from pathlib import Path
# Agregar el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.core.network import NeuralNetwork
from neural_network.config import OptimizerConfig
from neural_network.core.losses.functions import softmax_cross_entropy_with_logits
from neural_network.core.trainer import Trainer



def load_mnist():
    """Carga el conjunto de datos MNIST utilizando tensorflow.keras."""
    from tensorflow.keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizar las imágenes y aplanarlas
    X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def preprocess_data(X, y):
    """Convierte las etiquetas a formato one-hot."""
    # Las imágenes ya están normalizadas
    # Convertir las etiquetas a formato one-hot
    num_classes = 10
    y_one_hot = np.zeros((y.size, num_classes), dtype=np.float32)
    y_one_hot[np.arange(y.size), y] = 1.0

    return X, y_one_hot


def main():
    # Paso 1: Cargar y preprocesar los datos
    print("Cargando datos de MNIST...")
    X_train, y_train_labels, X_test, y_test_labels = load_mnist()
    X_train, y_train = preprocess_data(X_train, y_train_labels)
    X_test, y_test = preprocess_data(X_test, y_test_labels)

    # Paso 2: Crear e inicializar la red neuronal
    topology = [784, 128, 64, 10]
    activation_type = "RELU"
    learning_rate = 0.001
    epochs = 50
    batch_size = 32
    dropout_rate = 0.0

    print("Inicializando la red neuronal...")
    nn = NeuralNetwork(
        topology=topology,
        activation_type=activation_type,
        dropout_rate=dropout_rate
    )

    tr = Trainer( learning_rate, epochs, nn,
        softmax_cross_entropy_with_logits, optimizer_config=OptimizerConfig("SGD") )

    # Paso 3a Entrenar la red
    print("Entrenando la red neuronal...")
    tr.train(X_train, y_train, batch_size)

    # Crear conjunto de validación (10% de los datos)
    val_split = 0.1
    split_idx = int((1 - val_split) * len(X_train))
    x_train_main, x = X_train[:split_idx], X_train[split_idx:]
    y_train_main, y = y_train[:split_idx], y_train[split_idx:]

    # Entrenar con validación
    train_losses, val_losses = tr.train(
        x_train_main, y_train_main,
        batch_size=batch_size,
        verbose=True,
        x_val=x,
        y_val=y,
        patience=10
    )

    #Paso 3b: graph loss and validation
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Early Stopping - Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "mnist_early_stopping_loss.png", dpi=300, bbox_inches="tight")

    # Paso 4: Evaluar la red en el conjunto de prueba
    print("Evaluando la red en el conjunto de prueba...")
    accuracy_before_saving = nn.evaluate(X_test, y_test)
    print(f"Precisión antes de guardar los pesos: {accuracy_before_saving * 100:.2f}%")

    # predictions on the test set
    y_pred_probs = nn.forward(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # manually create a confusion matrix
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true_labels, y_pred_labels):
        cm[true, pred] += 1

    # graph
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("MNIST Confusion Matrix (Manual)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    # add numbers above the cells
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "mnist_confusion_matrix_manual.png", dpi=300, bbox_inches="tight")
    plt.show()

    # global percision
    accuracy = np.trace(cm) / np.sum(cm) * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")

    #per class accuracy
    print("\nPercision per digit:")
    class_accuracies = []

    for i in range(num_classes):
        true_positives = cm[i, i]
        total_samples = np.sum(cm[i, :])
        acc = (true_positives / total_samples * 100) if total_samples > 0 else 0.0
        class_accuracies.append(acc)
        print(f"  Dígito {i}: {acc:.2f}%")

    # Total mean
    mean_class_acc = np.mean(class_accuracies)
    print(f"\nPrecisión promedio por clase: {mean_class_acc:.2f}%")

    # visualization of the percision per class
    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(num_classes), class_accuracies, color="skyblue", edgecolor="black")
    plt.xlabel("Dígito")
    plt.ylabel("Precisión (%)")
    plt.title("Precisión por dígito en el conjunto de prueba")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / "mnist_per_class_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()

    # comparison of optimizers
    print("\n=== Comparando optimizadores ===")

    optimizers_to_test = [
        {"name": "SGD", "config": OptimizerConfig("SGD")},
        {"name": "SGD + Momentum", "config": OptimizerConfig("SGD_MOMENTUM", momentum=0.9)},
        {"name": "Adam", "config": OptimizerConfig("ADAM")},
    ]

    results = []

    for opt in optimizers_to_test:
        print(f"\nEntrenando con {opt['name']}...")
        nn_opt = NeuralNetwork(topology=topology, activation_type=activation_type, dropout_rate=dropout_rate)
        tr_opt = Trainer(
            learning_rate=learning_rate,
            epochs=epochs,
            network=nn_opt,
            loss_func=softmax_cross_entropy_with_logits,
            optimizer_config=opt["config"]
        )

        start_time = time.perf_counter()

        train_losses, val_losses = tr_opt.train(
            X_train_main, y_train_main,
            batch_size=batch_size,
            verbose=False,
            x=X_val,
            y=y_val,
            patience=10
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        acc = nn_opt.evaluate(X_test, y_test) * 100
        print(f"{opt['name']} → Precisión: {acc:.2f}%")
        results.append({"optimizer": opt["name"], "val_loss": val_losses[-1] if val_losses else None, "test_acc": acc})

        # Comparison plot
        plt.figure(figsize=(7, 5))
        names = [r["optimizer"] for r in results]
        accs = [r["test_acc"] for r in results]
        plt.bar(names, accs, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black")
        plt.ylabel("Precisión en test (%)")
        plt.title("Comparación de optimizadores en MNIST")
        plt.ylim(0, 100)
        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plots_dir / "mnist_optimizer_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("\n=== Resumen de optimizadores ===")
        print(f"{'Optimizador':<20} {'Precisión (%)':<15} {'Tiempo (s)':<10}")
        print("-" * 45)
        for r in results:
            print(f"{r['optimizer']:<20} {r['test_acc']:<15.2f} {r['time_sec']:<10.2f}")

    # Carpeta de salida para resultados
    results_dir = Path("./outputs/")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Nombre del archivo con marca de tiempo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"mnist_optimizer_results_{timestamp}.csv"

    # Guardar resultados
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "test_acc", "val_loss", "time_sec"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResultados guardados en: {csv_path}")

    # Paso 5: Guardar los pesos entrenados (incluyendo la topología y activación)
    weight_folder = 'weights'
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    weight_file = os.path.join(weight_folder, 'mnist_weights.npz')
    nn.save_weights(weight_file)
    print(f"Pesos y configuración guardados en '{weight_file}'.")

    # Paso 6: Crear una nueva instancia de la red usando from_weights_file
    print("Creando una nueva instancia de la red neuronal desde el archivo de pesos...")
    nn_loaded = NeuralNetwork.from_weights_file(weight_file)

    # Paso 7: Evaluar la nueva instancia en el conjunto de prueba
    print("Evaluando la nueva instancia de la red en el conjunto de prueba...")
    accuracy_after_loading = nn_loaded.evaluate(X_test, y_test)
    print(f"Precisión después de cargar los pesos: {accuracy_after_loading * 100:.2f}%")

    # Verificar si las precisiones son iguales
    if np.isclose(accuracy_before_saving, accuracy_after_loading, atol=1e-6):
        print("La precisión es la misma antes y después de cargar los pesos.")
    else:
        print("La precisión difiere después de cargar los pesos.")

    # Verificar que las predicciones son iguales
    print("Verificando que las predicciones son iguales...")
    predictions_original = nn.predict(X_test[:10])
    predictions_loaded = nn_loaded.predict(X_test[:10])

    if np.allclose(predictions_original, predictions_loaded, atol=1e-6):
        print("Las predicciones son iguales en ambas redes.")
    else:
        print("Las predicciones difieren entre las redes.")

    # Opcional: Eliminar el archivo de pesos si no se necesita
    # os.remove(weight_file)


if __name__ == '__main__':
    main()
