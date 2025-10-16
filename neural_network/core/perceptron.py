# perceptron.py

import numpy as np
from typing import Optional
from .activations import ActivationFunction, ActivationFunctionFactory
from ..config import WeightInitConfig
import os

class Perceptron:
    def __init__(self, num_inputs: int, activation_type: str,
                 weight_init_config: Optional[WeightInitConfig] = None) -> None:
        if num_inputs <= 0:
            raise ValueError("El número de entradas 'num_inputs' debe ser mayor que cero.")

        # Use default configs if not provided
        if weight_init_config is None:
            weight_init_config = WeightInitConfig()

        self.weights: np.ndarray = self._initialize_weights(num_inputs+1, weight_init_config)
        self.activation: ActivationFunction = ActivationFunctionFactory.create(activation_type)
        self.last_input: np.ndarray = np.array([])
        self.last_total: np.ndarray = np.array([])
        self.last_output: np.ndarray = np.array([])

    def calculate_output(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape[1] != self.weights.shape[0] - 1:
            raise ValueError(f"El número de características en 'inputs' ({inputs.shape[1]}) debe ser {self.weights.shape[0] - 1}.")
        
        bias_column = np.ones((inputs.shape[0], 1), dtype=np.float32)
        inputs_with_bias = np.hstack([inputs, bias_column])
        
        self.last_input = inputs_with_bias
        self.last_total = np.dot(inputs_with_bias, self.weights)
        self.last_output = self.activation.activate(self.last_total)
        return self.last_output


    def get_activation_derivative(self) -> np.ndarray:
        """Get derivative of the activation function at the last output."""
        if self.last_total.size == 0:
            raise ValueError("calculate_output debe ser llamado antes de get_activation_derivative.")
        return self.activation.derivative(self.last_total)
    
    def get_bias(self) -> float:
        """Get the bias term (last weight)."""
        return self.weights[-1]

    def save_weights(self, file_path: str) -> None:
        data = {}
        data['weights'] = self.weights
        np.savez(file_path, **data)
        print(f"Pesos guardados en '{file_path}'.")

    def load_weights(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
        data = np.load(file_path, allow_pickle=True)
        self.weights = data['weights']
        print(f"Pesos cargados desde '{file_path}'.")

    def _initialize_weights(self, num_inputs: int, weight_config: WeightInitConfig) -> np.ndarray:
        """Initialize weights based on configuration."""
        if weight_config.seed is not None:
            np.random.seed(weight_config.seed)
        
        init_type = weight_config.init_type.lower()
        
        if init_type == "zeros":
            return np.zeros(num_inputs, dtype=np.float32)
        elif init_type == "ones":
            return np.ones(num_inputs, dtype=np.float32)
        elif init_type == "random":
            return (np.random.randn(num_inputs) * 0.01).astype(np.float32)
        else:
            raise ValueError(f"Tipo de inicialización de pesos no soportado: {init_type}")
