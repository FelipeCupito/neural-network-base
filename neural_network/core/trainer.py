import numpy as np
from typing import List, Optional, Union, Any, Dict
from .network import NeuralNetwork
from .optimizers import OptimizerFunction, OptimizerFunctionFactory
from ..config import OptimizerConfig
from ..utils.data_utils import calculate_score, k_fold_split, stratified_k_fold_split

class Trainer:
    def __init__(self, learning_rate, epochs, network: NeuralNetwork, loss_func,
                optimizer_config: Optional[OptimizerConfig] = None,):
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.network = network
        self.loss_f = loss_func
        if optimizer_config is None:
            optimizer_config = OptimizerConfig()
        self.optimizer: OptimizerFunction = self._create_optimizer(optimizer_config)

    def train(self, training_inputs: np.ndarray, training_labels: np.ndarray, batch_size: int = 32, verbose=False,
              x_val = None, y_val = None, patience = 10) -> None:
        if training_inputs.shape[0] != training_labels.shape[0]:
            raise ValueError("El número de muestras en 'training_inputs' y 'training_labels' debe ser el mismo.")

        num_samples = training_inputs.shape[0]
        train_losses, val_losses = [], []
        best_val_loss = np.inf
        best_weights = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            shuffled_inputs = training_inputs[indices]
            shuffled_labels = training_labels[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_inputs = shuffled_inputs[start_idx:end_idx]
                batch_labels = shuffled_labels[start_idx:end_idx]

                y = self.network.forward(batch_inputs, training=True)

                if np.isnan(y).any() or np.isinf(y).any():
                    print("Advertencia: y contiene valores NaN o Inf en la época", epoch)
                    continue

                batch_loss, loss_grad = self.loss_f(y, batch_labels)
                total_loss += batch_loss

                # FIX 1: Ensure batch_loss has correct shape for backpropagation
                delta_output = loss_grad
                
                deltas = [delta_output]  # Start with output layer delta

                # Backpropagation through layers
                for l in range(len(self.network.layers) - 2, -1, -1):
                    current_layer = self.network.layers[l]
                    next_layer = self.network.layers[l + 1]
                    
                    # FIX 2: Handle weights extraction properly
                    # Assuming each layer has .weights attribute of shape (input_dim+1, num_neurons)
                    # where last row is bias weights
                    weights_next_layer = next_layer.weights[:-1, :]  # Exclude bias, shape: (next_layer_input_dim, num_neurons_next)
                    
                    delta_next = deltas[0]  # Current delta from next layer
                    
                    # FIX 3: Proper matrix multiplication for backpropagation
                    # delta_next shape: (batch_size, num_neurons_next)
                    # weights_next_layer shape: (next_layer_input_dim, num_neurons_next)
                    # We want: delta_next dot weights_next_layer.T
                    if delta_next.ndim == 1:
                        delta_next = delta_next.reshape(1, -1)  # Ensure 2D
                    
                    # Correct backpropagation: δ_l = (δ_{l+1} · W_{l+1}^T) ⊙ f'(z_l)
                    delta_current = np.dot(delta_next, weights_next_layer.T) * current_layer.get_activation_derivative()
                    
                    deltas.insert(0, delta_current)

                # FIX 4: Calculate gradients with proper dimension handling
                all_gradients = []
                for l, layer in enumerate(self.network.layers):
                    # Get inputs to this layer
                    if l == 0:
                        inputs_to_use = batch_inputs  # Shape: (batch_size, input_dim)
                    else:
                        inputs_to_use = self.network.layers[l - 1].outputs  # Shape: (batch_size, prev_layer_neurons)
                    
                    delta = deltas[l]  # Shape: (batch_size, current_layer_neurons)
                    
                    # FIX 5: Ensure inputs have bias term for gradient calculation
                    # Add bias unit to inputs: (batch_size, input_dim + 1)
                    inputs_with_bias = np.column_stack([inputs_to_use, np.ones(inputs_to_use.shape[0])])
                    
                    # FIX 6: Proper gradient calculation
                    # ∇W = (a_{l-1}^T · δ_l) / batch_size
                    if delta.ndim == 1:
                        delta = delta.reshape(-1, 1)  # Ensure 2D
                    
                    # Gradient for all weights (including bias)
                    gradients = np.dot(inputs_with_bias.T, delta) / batch_inputs.shape[0]
                    
                    # Split gradients for each perceptron
                    layer_gradients = [gradients[:, i] for i in range(gradients.shape[1])]
                    all_gradients.append(layer_gradients)
                self.optimizer.update_network(self.network, all_gradients, self.learning_rate)
                    
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Época {epoch}/{self.epochs} - Pérdida: {total_loss}")

            avg_train_loss = total_loss/ (num_samples // batch_size)
            train_losses.append(avg_train_loss)

            #validation split
            if x_val is not None:
                y_pred = self.network.forward(x_val, training = False)
                val_loss, _ = self.loss_f(y_pred, y_val)
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = [layer.weights.copy() for layer in self.network.layers]
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose:
                    print(f"Epoch {epoch}/{self.epochs} - Loss : {avg_train_loss:.4f} - Val: {val_loss:.4f}")

                if patience_counter >= patience:
                    print(f"Early stopping in epoch {epoch} (without patience {patience} imrpoving through epochs")
                    if best_weights is not None:
                        for layer, w in zip(self.network.layers, best_weights):
                            layer.weights = w
                    break
            else:
                if verbose:
                    print(f"epoch {epoch}/{self.epochs} - Loss: {avg_train_loss:.4f}")

        return train_losses, val_losses

    def _create_optimizer(self, optimizer_config: OptimizerConfig) -> OptimizerFunction:
        """Create optimizer instance 
        ased on configuration."""
        # Pass all optimizer config parameters, factory will filter what each optimizer needs
        return OptimizerFunctionFactory.create(
            optimizer_config.type,
            beta1=optimizer_config.beta1,
            beta2=optimizer_config.beta2,
            epsilon=optimizer_config.epsilon,
            momentum=optimizer_config.momentum
        )

def k_fold_cross_validate(
    X: np.ndarray, 
    y: np.ndarray, 
    model: Any,
    opt_cfg: OptimizerConfig,
    loss_func,
    lr: float = 1e-3,
    batch_size: int = 4,
    epochs: int = 1000,
    k: int = 5,
    scoring: str = 'accuracy',
    shuffle: bool = True,
    random_state: int = None,
    return_models: bool = False,
    stratified: bool = False
) -> Dict[str, Union[float, List]]:
    """
    Perform k-fold cross-validation on a given dataset.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    model : object
        Model object with fit() and predict() methods
    k : int
        Number of folds
    scoring : str
        Scoring metric
    shuffle : bool
        Whether to shuffle the data
    random_state : int
        Random seed for reproducibility
    return_models : bool
        Whether to return trained models for each fold
    stratified : bool
        Whether to use stratified k-fold (for classification)
    
    Returns:
    --------
    results : Dict
        Dictionary containing cross-validation results
    """
    
    # Input validation
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    if k <= 1 or k > len(X):
        raise ValueError("k must be between 2 and number of samples")
    
    n_samples = len(X)
    
    # Generate folds
    if stratified:
        folds = stratified_k_fold_split(y, k, shuffle, random_state)
    else:
        folds = k_fold_split(n_samples, k, shuffle, random_state)
    
    # Initialize lists to store results
    scores = []
    models = [] if return_models else None
    fold_details = []

    #Save original weights
    og_w = model.save_weights('tmp_og_weights')
    
    for fold, (train_idx, test_idx) in enumerate(folds):
        fold_model = model
        fold_model.load_weights('tmp_og_weights.npz')
        tr = Trainer(learning_rate=lr, epochs=epochs, optimizer_config=opt_cfg,
                    loss_func=loss_func, network=fold_model)
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        tr.train(X_train, y_train, batch_size)
        
        # Make predictions
        y_pred = fold_model.forward(X_test)
        
        # Calculate score
        score = calculate_score(y_test, y_pred, scoring)
        scores.append(score)
        
        # Store model if requested
        if return_models:
            models.append(fold_model)
        
        # Store fold details
        fold_details.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': score,
            'train_indices': train_idx,
            'test_indices': test_idx
        })
    
    # Calculate statistics
    results = {
        'fold_scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'fold_details': fold_details,
        'n_folds': k,
        'scoring': scoring
    }
    
    if return_models:
        results['models'] = models
    
    return results
