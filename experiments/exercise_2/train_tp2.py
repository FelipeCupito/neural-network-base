import numpy as np
from pandas import read_csv, DataFrame
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from pickle import dump

sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.core.perceptron import Perceptron
from neural_network.core.trainer import k_fold_cross_validate, Trainer
from neural_network.core.network import NeuralNetwork
from neural_network.config import OptimizerConfig, WeightInitConfig
from neural_network.core.losses.functions import mae, mse

DATASET_PATH = str(Path(__file__).parent.parent.parent / "resources" / "datasets" / "TP3-ej2-conjunto.csv")


def normalize(y, y_min, y_max, feature_range=(0, 1)):
    min_range, max_range = feature_range
    return min_range + (y - y_min) * (max_range - min_range) / (y_max - y_min)


def denormalize(y_normalized, y_min, y_max, feature_range=(0, 1)):
    min_range, max_range = feature_range
    return y_min + (y_normalized - min_range) * (y_max - y_min) / (max_range - min_range)


def train_perceptron(df: DataFrame, perceptron: Perceptron, learning_rate: float = 0.1, max_epochs: int = 100,
                     seed: int = 42, y_min: float = None, y_max: float = None):
    # Configurar semilla
    if seed is not None:
        np.random.seed(seed)

    # Crear datos según el problema
    y_original = df['y'].values
    X = df.drop(columns=['y']).values

    # Todas las funciones usan el mismo rango para consistencia en la comparación
    feature_range = (0, 1)
    activation_name = perceptron.activation.name.upper()

    print(f"\n{'=' * 50}")
    print(f"ENTRENANDO PERCEPTRÓN")
    print(f"{'=' * 50}")
    if y_min is not None and y_max is not None:
        y = normalize(y_original, y_min, y_max, feature_range)
        print(f"Datos y normalizados al rango {feature_range} para activación {activation_name}")
    else:
        y = y_original
        print(f"Datos y sin normalizar (activación {activation_name})")

    print(f"\nParámetros:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Seed: {seed}")
    print(
        f"  - Pesos iniciales: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}, {perceptron.weights[2]:.4f}]")
    print(f"  - Bias inicial: {perceptron.weights[-1]:.4f}")

    print(f"\n--- INICIANDO ENTRENAMIENTO ---")

    l2_losses = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        epoch_updates = 0
        for i, (inputs, target) in enumerate(zip(X, y)):
            x_input = inputs.reshape(1, -1)
            prediction = perceptron.calculate_output(x_input)#[0]
            error = (target - prediction[0])
            # if activation_name == 'SIGMOID':
            #     print(f'h={prediction[0]}')
            #     print(f'Sigmoid grad = {perceptron.get_activation_derivative()[0]}')
            # w = w + lr * error * x
            perceptron.weights[:-1] += learning_rate*error*inputs*(perceptron.get_activation_derivative()[0])
            # b = b + lr * error
            perceptron.weights[-1] += learning_rate * error*perceptron.get_activation_derivative()[0]
            epoch_updates += 1

        preds = np.zeros(len(y))
        for i, inputs in enumerate(X):
            x_input = inputs.reshape(1, -1)
            preds[i]= perceptron.calculate_output(x_input)[0]

        # Calcular L1 loss en escala original
        l2_losses[epoch] = (y_max-y_min)*np.mean(np.abs(preds - y)**2)
        # print(f"Época {epoch+1:2d}: Mean L1 error={l1_losses[epoch]:.3f}, Updates={epoch_updates}")

    print(f"Pesos finales: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}, {perceptron.weights[2]:.4f}]")
    print(f"Bias final: {perceptron.weights[-1]:.4f}")

    return l2_losses, perceptron


if __name__ == "__main__":
    print("TP3 - EJERCICIO 2: PERCEPTRÓN SIMPLE")
    print("-" * 50)
    max_epochs = 2000
    lr = 2e-2
    # Load dataset
    df = read_csv(DATASET_PATH)
    print(f"Dataset cargado desde '{DATASET_PATH}'")
    print(f"Primeras filas del dataset:\n{df.head()}")
    print(df.info())

    # Calcular y_min e y_max para normalización
    y_min = df.y.min()
    y_max = df.y.max()
    print(f'Max y value = {y_max}')
    print(f'Min y value = {y_min}')

    # Configuración de pesos inicial común para todos los modelos
    weight_config = WeightInitConfig(seed=42)

    # Entrenar y evaluar funcionamiento con perceptron lineal
    l_mdl = Perceptron(num_inputs=3, activation_type="LINEAR", weight_init_config=weight_config)
    l1_linear, trained_mdl = train_perceptron(df, l_mdl, learning_rate=lr, max_epochs=max_epochs, seed=42, y_min=y_min,
                                              y_max=y_max)
    print(f'Final L1 loss (LINEAR): {l1_linear[-1]:.3f}')

    # Entrenar y evaluar funcionamiento con perceptron sigmoid
    non_l_mdl = Perceptron(num_inputs=3, activation_type="SIGMOID", weight_init_config=weight_config)
    l1_sigmoid, trained_mdl = train_perceptron(df, non_l_mdl, learning_rate=lr, max_epochs=max_epochs, seed=42,
                                               y_min=y_min, y_max=y_max)
    print(f'Final L1 loss (SIGMOID): {l1_sigmoid[-1]:.3f}')

    # Entrenar y evaluar funcionamiento con perceptron Relu (rango -1, 1)
    bipolar_mdl = Perceptron(num_inputs=3, activation_type="RELU", weight_init_config=weight_config)
    l1_bipolar, trained_mdl = train_perceptron(df, bipolar_mdl, learning_rate=lr, max_epochs=max_epochs, seed=42,
                                               y_min=y_min, y_max=y_max)
    print(f'Final L1 loss (RELU): {l1_bipolar[-1]:.3f}')

    # Mean L1 error comparison
    plt.figure(figsize=(10, 8))
    plt.rcParams['axes.labelsize'] = 16  # x and y labels
    plt.rcParams['xtick.labelsize'] = 14  # x-axis ticks
    plt.rcParams['ytick.labelsize'] = 14  # y-axis ticks
    plt.rcParams['axes.titlesize'] = 20  # title
    plt.title("MAE vs epochs")
    plt.plot(range(1, max_epochs + 1), l1_linear, label='Linear activation', lw=4)
    plt.plot(range(1, max_epochs + 1), l1_sigmoid, label='Sigmoid activation', lw=4)
    plt.plot(range(1, max_epochs + 1), l1_bipolar, label='ReLU activation', lw=4)
    plt.ylabel('Mean(|y-ŷ|)')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(fontsize=16)
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(str(output_dir / "ex2.png"))
    plt.show()

    # Análisis del efecto de la tasa de aprendizaje
    print(f"\n{'=' * 60}")
    print(f"ANÁLISIS DE TASA DE APRENDIZAJE")
    print(f"{'=' * 60}")

    # Seleccionar el mejor modelo y diferentes learning rates
    learning_rates = [1e0, 1e-1,1e-2, 1e-3, 1e-4]
    lr_losses = {}

    print(f"Entrenando modelo RELU con diferentes tasas de aprendizaje...")
    for lr_test in learning_rates:
        print(f"\n--- Learning Rate: {lr_test} ---")
        relu_lr_mdl = Perceptron(num_inputs=3, activation_type="Sigmoid", weight_init_config=weight_config)
        l1_lr, _ = train_perceptron(df, relu_lr_mdl, learning_rate=lr_test, max_epochs=max_epochs, seed=42, y_min=y_min,
                                    y_max=y_max)
        lr_losses[lr_test] = l1_lr
        print(f'Final L1 loss (Sigmoid, lr={lr_test}): {l1_lr[-1]:.3f}')

    # Gráfico de comparación de learning rates
    plt.figure(figsize=(10, 8))
    plt.rcParams['axes.labelsize'] = 16  # x and y labels
    plt.rcParams['xtick.labelsize'] = 14  # x-axis ticks
    plt.rcParams['ytick.labelsize'] = 14  # y-axis ticks
    plt.rcParams['axes.titlesize'] = 20  # title
    plt.title("MAE vs epochs - Learning Rate Comparison (Sigmoid)")

    colors = ['red', 'blue', 'green', 'orange', 'black']
    for i, lr_test in enumerate(learning_rates):
        plt.plot(range(1, max_epochs + 1), lr_losses[lr_test],
                 label=f'LR = {lr_test}', lw=4, color=colors[i])

    plt.ylabel('Mean(|y-ŷ|)')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.savefig(str(output_dir / "ex2_learning_rates.png"))
    plt.show()

    # Analisis del poder de generalizacion del peceptron con función de activación ReLU
    print(f"\n{'=' * 60}")
    print(f"ANÁLISIS DE GENERALIZACIÓN (K-FOLD)")
    print(f"{'=' * 60}")

    # Usar datos normalizados para consistencia con el experimento principal
    y_norm = normalize(df['y'].values, y_min, y_max, (0, 1))
    X = df.drop(columns=['y']).values

    topology = [3, 1]
    mdl = NeuralNetwork(topology, activation_type='SIGMOID')
    b_size = 1
    opt_cfg = OptimizerConfig(type='SGD')
    loss = mse
    k = 5
    scoring = 'mae'

    # tr = Trainer(learning_rate=lr, epochs=max_epochs, optimizer_config=opt_cfg,
    #                 loss_func=loss, network=mdl)
    # tr_losses, _ = tr.train(X, y_norm, b_size)

    # plt.figure()
    # plt.plot(np.array(tr_losses)*(y_max-y_min))
    # plt.show()

    results = k_fold_cross_validate(X, y_norm, mdl, opt_cfg, loss, lr,
                                    b_size, max_epochs, k, scoring,
                                    verbose=True, patience=max_epochs)
    with open(str(output_dir / 'ex2_dict.pkl'), 'wb') as file:
        dump(results, file)
    print(results.keys())
    for i, score in enumerate(results['fold_scores']):
        results['fold_scores'][i] = score*(y_max-y_min)
    fold_details = results['fold_details']
    print(f'Mean {scoring} over {results["n_folds"]} folds: {results["mean_score"]*(y_max-y_min):.4f} ± {results["std_score"]*(y_max-y_min):.4f}')
    plt.figure(figsize=(10, 8))
    plt.rcParams['axes.labelsize'] = 16  # x and y labels
    plt.rcParams['xtick.labelsize'] = 14  # x-axis ticks
    plt.rcParams['ytick.labelsize'] = 14  # y-axis ticks
    plt.rcParams['axes.titlesize'] = 20  # title
    plt.title(f'{scoring.upper()} per fold')
    plt.bar(range(1, results['n_folds'] + 1), results['fold_scores'], color='lightgreen')
    plt.ylabel(scoring.upper())
    plt.xlabel('Fold')
    plt.xticks(range(1, results['n_folds'] + 1))
    plt.grid(True)
    plt.savefig(str(output_dir / "ex2_barplot.png"))
    plt.show()


    # # Check linearity of data
    # # Simple linearity test using correlations
    # def simple_linearity_test(coords, outputs):
    #     # Calculate correlation coefficients between each coordinate and output
    #     correlations = np.array([np.corrcoef(coords[:, i], outputs)[0, 1]
    #                              for i in range(coords.shape[1])])

    #     # Calculate overall correlation (you could also use multiple R)
    #     X_with_ones = np.column_stack([np.ones(len(coords)), coords])
    #     beta = np.linalg.lstsq(X_with_ones, outputs, rcond=None)[0]
    #     predictions = X_with_ones @ beta
    #     r_squared = np.corrcoef(outputs, predictions)[0, 1] ** 2

    #     return correlations, r_squared


    # correlations, r_squared = simple_linearity_test(X, y_norm)
    # print(f"Correlations with output: {correlations}")
    # print(f"R-squared: {r_squared:.4f}")
