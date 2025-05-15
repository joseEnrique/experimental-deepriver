from typing import Callable, Type, Union

import pandas as pd
import torch
from river import base
from torch import optim

from deep_river.base import RollingDeepEstimatorInitialized
from deep_river.regression import RegressorInitialized
from deep_river.utils.tensor_conversion import deque2rolling_tensor, float2tensor

class DualGPURegressorInitialized(RollingDeepEstimatorInitialized, RegressorInitialized):
    """
    Clase para regresión que utiliza GPU separadas para entrenamiento y predicción.
    
    Esta clase está diseñada para entrenar modelos en cuda:0 y hacer predicciones
    en cuda:1, maximizando el uso de recursos disponibles en sistemas multi-GPU.
    
    Attributes
    ----------
    module : torch.nn.Module
        Módulo PyTorch pre-inicializado para regresión.
    loss_fn : Union[str, Callable]
        Función de pérdida para el entrenamiento.
    optimizer_fn : Union[str, Type[optim.Optimizer]]
        Optimizador para el entrenamiento.
    lr : float
        Tasa de aprendizaje.
    is_feature_incremental : bool
        Si el modelo debe adaptarse a nuevas características.
    train_device : str
        Dispositivo para entrenamiento (por defecto "cuda:0").
    inference_device : str
        Dispositivo para inferencia (por defecto "cuda:1").
    seed : int
        Semilla para reproducibilidad.
    window_size : int
        Tamaño de la ventana deslizante.
    append_predict : bool
        Si se deben agregar las predicciones a la ventana.
    """
    
    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        train_device: str = "cuda:0",
        inference_device: str = "cuda:1",
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **kwargs
    ):

        self.train_device = train_device
        self.inference_device = inference_device
        
        # Verificar que los dispositivos son válidos
        if not torch.cuda.is_available():
            print("CUDA no está disponible. Usando CPU para entrenamiento e inferencia.")
            self.train_device = "cpu"
            self.inference_device = "cpu"
        else:
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2:
                print(f"Solo {num_gpus} GPU disponible(s). Es posible que entrenamiento e inferencia compartan el mismo dispositivo.")
            
            # Validar que los dispositivos existen
            try:
                if "cuda" in train_device:
                    torch.cuda.get_device_properties(torch.device(train_device))
                if "cuda" in inference_device:
                    torch.cuda.get_device_properties(torch.device(inference_device))
            except Exception as e:
                print(f"Error al verificar dispositivos: {str(e)}")
                print("Usando dispositivos predeterminados: cuda:0 para entrenamiento y cpu para inferencia.")
                self.train_device = "cuda:0"
                self.inference_device = "cpu"
                
        # Inicializar clase base (con dispositivo de entrenamiento)
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=self.train_device,
            seed=seed,
            window_size=window_size,
            append_predict=append_predict,
            **kwargs
        )
        
        # Configurar CUDA streams para operaciones asíncronas
        if "cuda" in self.train_device:
            self._train_stream = torch.cuda.Stream(device=torch.device(self.train_device))
        
        if "cuda" in self.inference_device and self.inference_device != self.train_device:
            self._inference_stream = torch.cuda.Stream(device=torch.device(self.inference_device))
            
        print(f"DualGPURegressorInitialized iniciado. Entrenamiento: {self.train_device}, Inferencia: {self.inference_device}")
            
    def learn_one(self, x: dict, y: base.typing.RegTarget, **kwargs) -> None:
        """
        Entrena con un ejemplo en el dispositivo de entrenamiento.
        
        Parameters
        ----------
        x : dict
            Ejemplo de entrada.
        y : base.typing.RegTarget
            Valor objetivo.
        """
        # Mover módulo al dispositivo de entrenamiento
        self.module.to(self.train_device)
        self.device = self.train_device
        
        # Actualizar características observadas y ventana
        self._update_observed_features(x)
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])
        
        if len(self._x_window) == self.window_size:
            # Usar CUDA stream para entrenamiento si está disponible
            if "cuda" in self.train_device:
                with torch.cuda.stream(self._train_stream):
                    x_t = self._deque2rolling_tensor(self._x_window)
                    y_t = torch.tensor([y], dtype=torch.float32, device=self.train_device).view(-1, 1)
                    self._learn(x=x_t, y=y_t)
                # Sincronizar para asegurar que el entrenamiento se complete
                self._train_stream.synchronize()
            else:
                x_t = self._deque2rolling_tensor(self._x_window)
                y_t = torch.tensor([y], dtype=torch.float32, device=self.train_device).view(-1, 1)
                self._learn(x=x_t, y=y_t)
        
        return self
        
    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """
        Predice un valor utilizando el dispositivo de inferencia.
        
        Parameters
        ----------
        x : dict
            Ejemplo de entrada.
            
        Returns
        -------
        base.typing.RegTarget
            Valor predicho.
        """
        # Mover módulo al dispositivo de inferencia
        self.module.to(self.inference_device)
        self.device = self.inference_device
        
        self._update_observed_features(x)
        
        # Preparar ventana para predicción
        x_win = self._x_window.copy()
        x_win.append([x.get(feature, 0) for feature in self.observed_features])
        if self.append_predict:
            self._x_window = x_win
            
        # Modo evaluación
        self.module.eval()
        
        # Usar CUDA stream para inferencia si está disponible
        if "cuda" in self.inference_device:
            with torch.cuda.stream(getattr(self, '_inference_stream', torch.cuda.default_stream())):
                with torch.inference_mode():
                    x_t = self._deque2rolling_tensor(x_win)
                    res = self.module(x_t)
                # Sincronizar si es necesario
                if hasattr(self, '_inference_stream'):
                    self._inference_stream.synchronize()
                # Mover resultado a CPU y convertir a numpy
                res = res.cpu().numpy(force=True).item()
        else:
            with torch.inference_mode():
                x_t = self._deque2rolling_tensor(x_win)
                res = self.module(x_t).numpy(force=True).item()
                
        return res
        
    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Entrena con múltiples ejemplos en el dispositivo de entrenamiento.
        
        Parameters
        ----------
        X : pd.DataFrame
            Ejemplos de entrada.
        y : pd.Series
            Valores objetivo.
        """
        # Mover módulo al dispositivo de entrenamiento
        self.module.to(self.train_device)
        self.device = self.train_device
        
        self._update_observed_features(X)
        
        # Preparar datos para entrenamiento
        X_filtered = X[list(self.observed_features)]
        self._x_window.extend(X_filtered.values.tolist())
        
        if len(self._x_window) == self.window_size:
            # Usar CUDA stream para entrenamiento si está disponible
            if "cuda" in self.train_device:
                with torch.cuda.stream(self._train_stream):
                    X_t = self._deque2rolling_tensor(self._x_window)
                    y_t = torch.tensor(y.values, dtype=torch.float32, device=self.train_device).view(-1, 1)
                    self._learn(x=X_t, y=y_t)
                # Sincronizar para asegurar que el entrenamiento se complete
                self._train_stream.synchronize()
            else:
                X_t = self._deque2rolling_tensor(self._x_window)
                y_t = torch.tensor(y.values, dtype=torch.float32, device=self.train_device).view(-1, 1)
                self._learn(x=X_t, y=y_t)
                
        return self
        
    def predict_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predice valores para múltiples ejemplos utilizando el dispositivo de inferencia.
        
        Parameters
        ----------
        X : pd.DataFrame
            Ejemplos de entrada.
            
        Returns
        -------
        pd.DataFrame
            Valores predichos.
        """
        # Mover módulo al dispositivo de inferencia
        self.module.to(self.inference_device)
        self.device = self.inference_device
        
        self._update_observed_features(X)
        
        # Preparar datos para predicción
        X_filtered = X[list(self.observed_features)]
        x_win = self._x_window.copy()
        x_win.extend(X_filtered.values.tolist())
        if self.append_predict:
            self._x_window = x_win
            
        # Modo evaluación
        self.module.eval()
        
        # Usar CUDA stream para inferencia si está disponible
        if "cuda" in self.inference_device:
            with torch.cuda.stream(getattr(self, '_inference_stream', torch.cuda.default_stream())):
                with torch.inference_mode():
                    x_t = self._deque2rolling_tensor(x_win)
                    y_preds = self.module(x_t)
                # Sincronizar si es necesario
                if hasattr(self, '_inference_stream'):
                    self._inference_stream.synchronize()
                # Convertir a numpy
                y_preds_numpy = y_preds.cpu().detach().numpy()
        else:
            with torch.inference_mode():
                x_t = self._deque2rolling_tensor(x_win)
                y_preds = self.module(x_t)
                y_preds_numpy = y_preds.detach().numpy()
                
        # Convertir a DataFrame con formato adecuado
        if len(y_preds_numpy.shape) > 1:
            return pd.DataFrame(y_preds_numpy)
        else:
            return pd.DataFrame(y_preds_numpy.reshape(-1, 1))
            
    def _deque2rolling_tensor(self, x_win):
        """
        Convierte una cola en un tensor, usando el dispositivo actual.
        """
        return deque2rolling_tensor(x_win, device=self.device) 