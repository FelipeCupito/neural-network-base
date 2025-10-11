import numpy as np
from typing import Optional, Tuple


class MinMaxScaler:
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        X = np.asarray(X)
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Evitar división por cero para características constantes
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        
        # Calcular parámetros de escalado
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("El escalador debe ser ajustado antes de transformar los datos.")
        
        X = np.asarray(X)
        return X * self.scale_ + self.min_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("El escalador debe ser ajustado antes de aplicar la transformación inversa.")
        
        X = np.asarray(X)
        return (X - self.min_) / self.scale_
    
    def get_params(self) -> dict:
        if not self.is_fitted_:
            return {}
        
        return {
            'feature_range': self.feature_range,
            'data_min_': self.data_min_,
            'data_max_': self.data_max_,
            'data_range_': self.data_range_,
            'scale_': self.scale_,
            'min_': self.min_,
            'is_fitted_': self.is_fitted_
        }
    
    def set_params(self, params: dict) -> 'MinMaxScaler':
        if not params:
            return self
        
        self.feature_range = params['feature_range']
        self.data_min_ = params['data_min_']
        self.data_max_ = params['data_max_']
        self.data_range_ = params['data_range_']
        self.scale_ = params['scale_']
        self.min_ = params['min_']
        self.is_fitted_ = params['is_fitted_']
        
        return self
    
    def __str__(self) -> str:
        if self.is_fitted_:
            return f"MinMaxScaler(feature_range={self.feature_range}, fitted=True)"
        else:
            return f"MinMaxScaler(feature_range={self.feature_range}, fitted=False)"