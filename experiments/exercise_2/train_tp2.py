import numpy as np
from pandas import read_csv, DataFrame
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from pickle import dump
# Agregar el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.core.perceptron import Perceptron
from neural_network.core.trainer import k_fold_cross_validate
from neural_network.core.network import NeuralNetwork
from neural_network.config import OptimizerConfig
from neural_network.core.losses.functions import mae

DATASET_PATH = "./resources/datasets/TP3-ej2-conjunto.csv"


def train_perceptron(df: DataFrame, perceptron: Perceptron, learning_rate: float = 0.1, max_epochs: int = 100, seed: int = 42):
    """
    Entrenar un perceptrón simple para el problema especificado.
    
    Args:
        problem_type: "and" o "xor"
        learning_rate: Tasa de aprendizaje
        max_epochs: Número máximo de épocas
        seed: Semilla para reproducibilidad
    """
    print(f"\n{'='*50}")
    print(f"ENTRENANDO PERCEPTRÓN")
    print(f"{'='*50}")
    
    # Configurar semilla
    if seed is not None:
        np.random.seed(seed)
    
    # Crear datos según el problema
    y = df['y'].values
    X = df.drop(columns=['y']).values
    
    print(f"\nDatos de entrenamiento:")
    print("Entrada  | Salida")
    print("-" * 17)
    
    # Crear perceptrón usando la clase existente
    print(f"\nParámetros:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Seed: {seed}")
    print(f"  - Pesos iniciales: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}]")
    print(f"  - Bias inicial: {perceptron.weights[-1]:.4f}")
    
    # Algoritmo de entrenamiento clásico del perceptrón
    print(f"\n--- INICIANDO ENTRENAMIENTO ---")
    
    l1_losses = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        epoch_updates = 0   
        # Entrenar con cada muestra individualmente
        for i, (inputs, target) in enumerate(zip(X, y)):
            # Preparar entrada como matriz (1 sample)
            x_input = inputs.reshape(1, -1)      
            # Forward pass
            prediction = perceptron.calculate_output(x_input)[0]          
            # Calcular error
            error = target - prediction  
            # Actualización manual de pesos (algoritmo clásico de perceptrón)
            # w = w + lr * error * x
            perceptron.weights[:-1] += learning_rate * error * inputs
            # b = b + lr * error
            perceptron.weights[-1] += learning_rate * error
            # print(f"  Época {epoch+1:2d}, Muestra {i+1}: Error={error:2}, "
            #       f"Pesos=[{perceptron.weights[0]:6.3f}, {perceptron.weights[1]:6.3f}, {perceptron.weights[2]:6.3f}], "
            #       f"Bias={perceptron.weights[-1]:6.3f}")
            epoch_updates += 1  
        predictions = []
        for inputs in X:
            x_input = inputs.reshape(1, -1)
            pred = perceptron.calculate_output(x_input)[0]
            predictions.append(pred)
        
        l1_losses[epoch] = np.mean(np.abs(np.array(predictions) - y))
        print(f"Época {epoch+1:2d}: Mean L1 error={l1_losses[epoch]:.3f}, Updates={epoch_updates}")
        
    print(f"Pesos finales: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}, {perceptron.weights[2]:.4f}]")
    print(f"Bias final: {perceptron.weights[-1]:.4f}")
    
    return l1_losses, perceptron


if __name__ == "__main__":
    print("TP3 - EJERCICIO 2: PERCEPTRÓN SIMPLE")
    print("-" * 50)
    max_epochs = 200
    lr = 1e-3
    #Load dataset
    df = read_csv(DATASET_PATH)
    print(f"Dataset cargado desde '{DATASET_PATH}'")
    print(f"Primeras filas del dataset:\n{df.head()}")
    print(df.info())
    print(f'Max y value = {df.y.max()}')
    print(f'Min y value = {df.y.min()}')
    # Entrenar y evaluar funcionamiento con perceptron lineal
    l_mdl = Perceptron(num_inputs=3, activation_type="LINEAR")
    l1_linear, trained_mdl = train_perceptron(df, l_mdl, learning_rate=lr, max_epochs=max_epochs, seed=42)
    print(f'Final L1 loss (LINEAR): {l1_linear[-1]:.3f}')
    # Entrenar y evaluar funcionamiento con perceptron no-lineal
    non_l_mdl = Perceptron(num_inputs=3, activation_type="SIGMOID")
    l1_nonlinear, trained_mdl = train_perceptron(df, non_l_mdl, learning_rate=lr, max_epochs=max_epochs, seed=42)
    print(f'Final L1 loss (SIGMOID): {l1_nonlinear[-1]:.3f}')
    #Mean L1 error comparison
    plt.figure(figsize=(10,8))
    plt.rcParams['axes.labelsize'] = 16    # x and y labels
    plt.rcParams['xtick.labelsize'] = 14   # x-axis ticks
    plt.rcParams['ytick.labelsize'] = 14   # y-axis ticks
    plt.rcParams['axes.titlesize'] = 20    # title
    plt.title("MAE vs epochs")
    plt.plot(range(1, max_epochs+1),l1_linear, label='Linear activation', lw=4)
    plt.plot(range(1, max_epochs+1),l1_nonlinear, label='Sigmoid activation', lw=4)
    plt.ylabel('Mean(|y-ŷ|)')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.savefig("./outputs/ex2.png")
    plt.show()


    #Analisis del poder de generalizacion del peceptron con función de activación Relu
    y = df['y'].values
    X = df.drop(columns=['y']).values
    topology = [3,1]
    mdl = NeuralNetwork(topology, activation_type='LINEAR', )
    b_size = 1
    opt_cfg = OptimizerConfig(type = 'SGD')
    loss = mae
    k = 5
    scoring = 'mae'

    results = k_fold_cross_validate(X, y, mdl, opt_cfg, loss, lr, b_size, max_epochs, k, scoring)
    with open('./outputs/ex2_dict.pkl', 'wb') as file:
        dump(results, file)
    print(results.keys())
    fold_details = results['fold_details']
    print(f'Mean {scoring} over {results["n_folds"]} folds: {results["mean_score"]:.4f} ± {results["std_score"]:.4f}')
    plt.figure(figsize=(10,8))
    plt.rcParams['axes.labelsize'] = 16    # x and y labels
    plt.rcParams['xtick.labelsize'] = 14   # x-axis ticks
    plt.rcParams['ytick.labelsize'] = 14   # y-axis ticks
    plt.rcParams['axes.titlesize'] = 20    # title
    plt.title(f'{scoring.upper()} per fold')
    plt.bar(range(1, results['n_folds']+1), results['fold_scores'], color='lightgreen')
    plt.ylabel(scoring.upper())
    plt.xlabel('Fold')
    plt.xticks(range(1, results['n_folds']+1))
    plt.grid(True)
    plt.savefig("./outputs/ex2_barplot.png")
    plt.show()

    #Check linearity of data
    # Simple linearity test using correlations
    def simple_linearity_test(coords, outputs):
        # Calculate correlation coefficients between each coordinate and output
        correlations = np.array([np.corrcoef(coords[:, i], outputs)[0, 1] 
                            for i in range(coords.shape[1])])
        
        # Calculate overall correlation (you could also use multiple R)
        X_with_ones = np.column_stack([np.ones(len(coords)), coords])
        beta = np.linalg.lstsq(X_with_ones, outputs, rcond=None)[0]
        predictions = X_with_ones @ beta
        r_squared = np.corrcoef(outputs, predictions)[0, 1] ** 2
        
        return correlations, r_squared

    correlations, r_squared = simple_linearity_test(X, y)
    print(f"Correlations with output: {correlations}")
    print(f"R-squared: {r_squared:.4f}")

    # #3D scatter plot of the dataset
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # scatter = ax.scatter(df['x1'], df['x2'], df['x3'], 
    #                     c=df['y'], cmap='gist_heat', 
    #                     s=50, alpha=0.8)

    # plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='y values')

    # ax.set_xlabel('X1')
    # ax.set_ylabel('X2')
    # ax.set_zlabel('X3')
    # ax.set_title('3D Scatter Plot')
    # plt.show()

    #Plot fold 5
    # fold4_details = fold_details[4]
    # train_indices = fold4_details['train_indices']
    # test_indices = fold4_details['test_indices']
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # scatter = ax.scatter(X[train_indices,0], X[train_indices,1], X[train_indices,2], 
    #                     c='red', label='Training', 
    #                     s=50, alpha=0.8)
    # scatter = ax.scatter(X[test_indices,0], X[test_indices,1], X[test_indices,2], 
    #                     c='blue', label='Training', 
    #                     s=50, alpha=0.8)

    # ax.set_xlabel('X1')
    # ax.set_ylabel('X2')
    # ax.set_zlabel('X3')
    # ax.set_title('3D Scatter Plot')
    # plt.show()