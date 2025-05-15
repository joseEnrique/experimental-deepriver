from typing import Callable, Type, Union, Dict, List, Optional

import pandas as pd
import torch
from river import base
from torch import optim
import concurrent.futures
import warnings
import numpy as np
import threading
import os
import sys
import subprocess
import pickle
import tempfile
import uuid
import json
import time

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

class MultiWorkerGPURegressor(RollingDeepEstimatorInitialized, RegressorInitialized):
    """
    Clase para regresión que utiliza múltiples workers en cada GPU para 
    maximizar el uso de los recursos en entrenamiento e inferencia.
    
    Esta clase está diseñada para entrenar en cuda:0 y predecir en cuda:1, 
    usando múltiples workers en paralelo para cada operación, maximizando
    el uso de GPU.
    
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
    train_workers : int
        Número de workers para tareas de entrenamiento.
    inference_workers : int
        Número de workers para tareas de inferencia.
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
        train_workers: int = 12,  # Aumentar el número de workers por defecto
        inference_workers: int = 12,
        max_batch_size: int = 1024,  # Tamaño máximo de batch para evitar OOM
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **kwargs
    ):
        # Verificar disponibilidad de CUDA
        if not torch.cuda.is_available():
            warnings.warn("CUDA no está disponible. Usando CPU para entrenamiento e inferencia.")
            train_device = "cpu"
            inference_device = "cpu"
        
        # Configurar dispositivos y workers
        self.train_device = train_device
        self.inference_device = inference_device
        
        # Usar más workers para mejor utilización de GPU
        self.train_workers = min(train_workers, 32)
        self.inference_workers = min(inference_workers, 32)
        self.max_batch_size = max_batch_size
        
        # Inicializar la clase base
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
        
        # Configuración para maximizar rendimiento CUDA
        if "cuda" in self.train_device or "cuda" in self.inference_device:
            # Activar benchmark para mejor rendimiento después de warmup
            torch.backends.cudnn.benchmark = True
            # Usar algoritmos determinísticos para threading seguro
            torch.backends.cudnn.deterministic = True
            # Asegurar que CUDA esté activado
            torch.backends.cudnn.enabled = True
        
        # Crear locks para sincronización
        self._train_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._executor_lock = threading.RLock()
        
        # Crear executors con thread pools (más seguro que multiprocessing)
        self._train_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.train_workers, 
            thread_name_prefix="train_worker"
        )
        self._inference_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.inference_workers, 
            thread_name_prefix="inference_worker"
        )
        
        # Crear streams CUDA para cada worker
        self._train_streams = []
        self._inference_streams = []
        
        if "cuda" in self.train_device:
            # Crear múltiples streams para paralelismo
            for _ in range(min(8, self.train_workers)):
                self._train_streams.append(
                    torch.cuda.Stream(device=torch.device(self.train_device))
                )
        
        if "cuda" in self.inference_device:
            # Crear múltiples streams para paralelismo
            for _ in range(min(8, self.inference_workers)):
                self._inference_streams.append(
                    torch.cuda.Stream(device=torch.device(self.inference_device))
                )
        
        # Hacer warmup del modelo para optimizar CUDA
        self._warmup_model()
        
        print(f"MultiWorkerGPURegressor iniciado para máxima utilización de GPU:")
        print(f"- Entrenamiento: {self.train_device} con {self.train_workers} workers")
        print(f"- Inferencia: {self.inference_device} con {self.inference_workers} workers")
        print(f"- Modo: ThreadPool para máxima compatibilidad")
    
    def _warmup_model(self):
        """Hacer un warmup del modelo para optimizar compilación JIT y caches CUDA"""
        try:
            # Crear un tensor pequeño para warmup
            n_features = len(self.observed_features) if self.observed_features else 1
            dummy_input = torch.zeros((2, n_features), device=self.train_device)
            dummy_target = torch.zeros((2, 1), device=self.train_device)
            
            # Ejecutar forward/backward en el dispositivo de entrenamiento
            self.module.to(self.train_device)
            with torch.cuda.stream(self._train_streams[0] if self._train_streams else torch.cuda.default_stream()):
                # Forward pass
                output = self.module(dummy_input)
                loss = self.loss_func(output, dummy_target)
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Ejecutar forward en el dispositivo de inferencia si es distinto
            if self.inference_device != self.train_device:
                self.module.to(self.inference_device)
                with torch.cuda.stream(self._inference_streams[0] if self._inference_streams else torch.cuda.default_stream()):
                    self.module.eval()
                    with torch.inference_mode():
                        _ = self.module(dummy_input.to(self.inference_device))
            
            # Sincronizar streams
            torch.cuda.synchronize(self.train_device)
            if self.inference_device != self.train_device:
                torch.cuda.synchronize(self.inference_device)
                
            print("Warmup completado - caches CUDA optimizados")
            
        except Exception as e:
            # No interrumpir la inicialización si el warmup falla
            print(f"Advertencia: No se pudo realizar warmup - {str(e)}")
    
    def __del__(self):
        """Limpieza de recursos cuando se destruye el objeto"""
        # Limpiar executors
        try:
            if hasattr(self, '_train_executor'):
                self._train_executor.shutdown(wait=False)
                
            if hasattr(self, '_inference_executor'):
                self._inference_executor.shutdown(wait=False)
            
            # Liberar memoria CUDA
            if hasattr(self, 'module') and self.module is not None:
                self.module = self.module.cpu()
            
            # Limpiar streams explícitamente
            if hasattr(self, '_train_streams'):
                self._train_streams = []
            
            if hasattr(self, '_inference_streams'):
                self._inference_streams = []
            
            # Forzar liberación de memoria
            torch.cuda.empty_cache()
        except Exception:
            # Ignorar errores durante la limpieza
            pass
    
    def _worker_learn(self, batch_id, x_tensor, y_tensor, worker_id=0):
        """
        Función de entrenamiento ejecutada por cada worker
        
        Parameters
        ----------
        batch_id : int
            ID del lote
        x_tensor : torch.Tensor
            Tensor de entrada
        y_tensor : torch.Tensor
            Tensor de objetivo
        worker_id : int
            ID del worker para seleccionar el stream
            
        Returns
        -------
        int
            ID del lote
        """
        # Obtener stream adecuado para este worker
        stream_idx = worker_id % len(self._train_streams) if self._train_streams else -1
        
        try:
            with self._train_lock:
                # Asegurar que el modelo esté en modo entrenamiento
                self.module.train()
                
                # Usar CUDA stream si está disponible
                if stream_idx >= 0:
                    with torch.cuda.stream(self._train_streams[stream_idx]):
                        self._learn(x=x_tensor, y=y_tensor)
                    # Sincronizar stream
                    self._train_streams[stream_idx].synchronize()
                else:
                    # Fallback a default stream
                    self._learn(x=x_tensor, y=y_tensor)
                    
                # Liberar memoria CUDA después de cada lote
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error en worker {worker_id}, lote {batch_id}: {str(e)}")
        
        return batch_id
    
    def _worker_predict(self, batch_id, x_tensor, worker_id=0):
        """
        Función de predicción ejecutada por cada worker
        
        Parameters
        ----------
        batch_id : int
            ID del lote
        x_tensor : torch.Tensor
            Tensor de entrada
        worker_id : int
            ID del worker para seleccionar el stream
            
        Returns
        -------
        tuple
            (batch_id, predicciones)
        """
        # Obtener stream adecuado para este worker
        stream_idx = worker_id % len(self._inference_streams) if self._inference_streams else -1
        
        try:
            with self._inference_lock:
                # Asegurar que el modelo esté en modo evaluación
                self.module.eval()
                
                # Usar CUDA stream si está disponible
                if stream_idx >= 0:
                    with torch.cuda.stream(self._inference_streams[stream_idx]):
                        with torch.inference_mode():
                            pred = self.module(x_tensor)
                            # Clonar resultado para evitar problemas de sincronización
                            pred_cpu = pred.cpu().clone() if "cuda" in self.inference_device else pred.clone()
                    # Sincronizar stream
                    self._inference_streams[stream_idx].synchronize()
                else:
                    # Fallback a default stream
                    with torch.inference_mode():
                        pred = self.module(x_tensor)
                        pred_cpu = pred.cpu().clone() if "cuda" in self.inference_device else pred.clone()
                
                # Liberar memoria CUDA después de cada lote
                torch.cuda.empty_cache()
                
                return batch_id, pred_cpu
                
        except Exception as e:
            print(f"Error en worker {worker_id}, lote {batch_id}: {str(e)}")
            return batch_id, None
    
    def _parallel_learn(self, x_batches, y_batches):
        """
        Entrenamiento en paralelo a través de múltiples workers
        
        Parameters
        ----------
        x_batches : List[torch.Tensor]
            Lista de lotes de tensores de entrada
        y_batches : List[torch.Tensor]
            Lista de lotes de tensores objetivo
        """
        # Preparar modelo para entrenamiento
        self.module.train()
        
        # Versión con threading (segura para funciones lambda)
        with self._executor_lock:
            futures = []
            for i, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
                # Asegurarse de que los datos estén en el dispositivo correcto
                x_device = x_batch.to(self.train_device, non_blocking=True)
                y_device = y_batch.to(self.train_device, non_blocking=True)
                
                futures.append(
                    self._train_executor.submit(
                        self._worker_learn,
                        i, x_device, y_device, i % self.train_workers
                    )
                )
            
            # Esperar a que terminen todos los futures
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Para capturar excepciones
                except Exception as e:
                    print(f"Error en future de entrenamiento: {str(e)}")
    
    def _parallel_predict(self, x_batches):
        """
        Predicción en paralelo a través de múltiples workers
        
        Parameters
        ----------
        x_batches : List[torch.Tensor]
            Lista de lotes de tensores de entrada
            
        Returns
        -------
        torch.Tensor
            Tensor combinado con todas las predicciones
        """
        # Preparar modelo para inferencia
        self.module.eval()
        
        # Almacenar resultados ordenados por batch_id
        batch_results = {}
        
        # Versión con threading (segura para funciones lambda)
        with self._executor_lock:
            futures = []
            for i, x_batch in enumerate(x_batches):
                # Asegurarse de que los datos estén en el dispositivo correcto
                x_device = x_batch.to(self.inference_device, non_blocking=True)
                
                futures.append(
                    self._inference_executor.submit(
                        self._worker_predict,
                        i, x_device, i % self.inference_workers
                    )
                )
            
            # Recolectar resultados
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_id, pred = future.result()
                    if pred is not None:
                        batch_results[batch_id] = pred
                except Exception as e:
                    print(f"Error en future de predicción: {str(e)}")
        
        # Ordenar resultados por batch_id
        ordered_results = []
        for i in range(len(x_batches)):
            if i in batch_results:
                ordered_results.append(batch_results[i])
        
        # Combinar resultados
        if ordered_results:
            try:
                return torch.cat(ordered_results, dim=0)
            except Exception as e:
                print(f"Error al combinar resultados: {str(e)}")
                # Retornar el primer resultado en caso de error
                return ordered_results[0] if ordered_results else torch.tensor([], device="cpu")
        else:
            # No hay resultados válidos
            return torch.tensor([], device="cpu")
    
    def _split_batch_for_gpu(self, tensor, num_splits):
        """
        Divide un tensor en múltiples lotes adaptados al tamaño de GPU
        
        Parameters
        ----------
        tensor : torch.Tensor
            Tensor a dividir
        num_splits : int
            Número de divisiones deseadas
            
        Returns
        -------
        List[torch.Tensor]
            Lista de tensores divididos
        """
        if tensor.shape[0] <= 1:
            return [tensor]  # No dividir tensores muy pequeños
        
        # Calcular tamaño de batch balanceando entre workers y memoria disponible
        batch_size = min(
            max(1, tensor.shape[0] // num_splits),  # División por workers
            self.max_batch_size  # Límite máximo para evitar OOM
        )
        
        return list(torch.split(tensor, batch_size))
    
    def _prepare_training_data(self, x, y):
        """
        Prepara datos para entrenamiento distribuyéndolos en múltiples batches
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada completo
        y : torch.Tensor
            Tensor objetivo completo
            
        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor]]
            Listas de batches de x e y
        """
        # Determinar número óptimo de batches basado en tamaño de datos
        optimal_splits = min(
            self.train_workers,  # No más batches que workers
            max(1, x.shape[0] // 32)  # Al menos 32 ejemplos por batch
        )
        
        # Dividir datos en batches
        x_batches = self._split_batch_for_gpu(x, optimal_splits)
        y_batches = self._split_batch_for_gpu(y, optimal_splits)
        
        return x_batches, y_batches
    
    def _prepare_inference_data(self, x):
        """
        Prepara datos para inferencia distribuyéndolos en múltiples batches
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada completo
            
        Returns
        -------
        List[torch.Tensor]
            Lista de batches de x
        """
        # Determinar número óptimo de batches basado en tamaño de datos
        optimal_splits = min(
            self.inference_workers,  # No más batches que workers
            max(1, x.shape[0] // 32)  # Al menos 32 ejemplos por batch
        )
        
        # Dividir datos en batches
        return self._split_batch_for_gpu(x, optimal_splits)
        
    def learn_one(self, x: dict, y: base.typing.RegTarget, **kwargs) -> "MultiWorkerGPURegressor":
        """
        Entrena con un ejemplo
        
        Parameters
        ----------
        x : dict
            Ejemplo de entrada
        y : base.typing.RegTarget
            Valor objetivo
            
        Returns
        -------
        MultiWorkerGPURegressor
            El estimador
        """
        # Mover módulo al dispositivo de entrenamiento
        self.module.to(self.train_device)
        self.device = self.train_device
        
        # Actualizar características observadas
        self._update_observed_features(x)
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])
        
        if len(self._x_window) == self.window_size:
            # Convertir a tensor
            x_t = self._deque2rolling_tensor(self._x_window)
            y_t = torch.tensor([y], dtype=torch.float32, device=self.train_device).view(-1, 1)
            
            # Para un solo ejemplo no usar paralelismo
            with self._train_lock:
                self._learn(x=x_t, y=y_t)
                
            # Limpiar memoria
            torch.cuda.empty_cache()
        
        return self
    
    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> "MultiWorkerGPURegressor":
        """
        Entrena con múltiples ejemplos usando paralelismo
        
        Parameters
        ----------
        X : pd.DataFrame
            Ejemplos de entrada
        y : pd.Series
            Valores objetivo
            
        Returns
        -------
        MultiWorkerGPURegressor
            El estimador
        """
        # Mover módulo al dispositivo de entrenamiento
        self.module.to(self.train_device)
        self.device = self.train_device
        
        # Actualizar características observadas
        self._update_observed_features(X)
        X_filtered = X[list(self.observed_features)]
        self._x_window.extend(X_filtered.values.tolist())
        
        if len(self._x_window) == self.window_size:
            # Convertir a tensores
            x_t = self._deque2rolling_tensor(self._x_window)
            y_t = torch.tensor(y.values, dtype=torch.float32, device=self.train_device).view(-1, 1)
            
            # Usar paralelismo solo si hay suficientes datos
            if x_t.shape[0] > 1:
                # Preparar batches óptimos para GPU
                x_batches, y_batches = self._prepare_training_data(x_t, y_t)
                
                # Entrenar en paralelo
                if len(x_batches) > 1:
                    self._parallel_learn(x_batches, y_batches)
                else:
                    # Un solo batch, no paralelizar
                    with self._train_lock:
                        self._learn(x=x_t, y=y_t)
            else:
                # Dataset muy pequeño
                with self._train_lock:
                    self._learn(x=x_t, y=y_t)
                    
            # Limpiar memoria
            torch.cuda.empty_cache()
        
        return self
    
    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """
        Predice para un ejemplo
        
        Parameters
        ----------
        x : dict
            Ejemplo de entrada
            
        Returns
        -------
        base.typing.RegTarget
            Valor predicho
        """
        # Mover módulo al dispositivo de inferencia
        self.module.to(self.inference_device)
        self.device = self.inference_device
        
        # Actualizar características observadas
        self._update_observed_features(x)
        
        # Preparar ventana de datos
        x_win = self._x_window.copy()
        x_win.append([x.get(feature, 0) for feature in self.observed_features])
        if self.append_predict:
            self._x_window = x_win
        
        # Convertir a tensor para predicción
        x_t = self._deque2rolling_tensor(x_win)
        
        # Para un solo ejemplo no usar paralelismo
        with self._inference_lock:
            self.module.eval()
            with torch.inference_mode():
                pred = self.module(x_t)
                # Convertir a valor escalar
                if "cuda" in self.inference_device:
                    result = pred.cpu().numpy(force=True).item()
                else:
                    result = pred.numpy(force=True).item()
                
        # Limpiar memoria
        torch.cuda.empty_cache()
                
        return result
    
    def predict_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predice para múltiples ejemplos usando paralelismo
        
        Parameters
        ----------
        X : pd.DataFrame
            Ejemplos de entrada
            
        Returns
        -------
        pd.DataFrame
            Dataframe con los valores predichos
        """
        # Mover módulo al dispositivo de inferencia
        self.module.to(self.inference_device)
        self.device = self.inference_device
        
        # Actualizar características observadas
        self._update_observed_features(X)
        
        # Preparar ventana de datos
        X_filtered = X[list(self.observed_features)]
        x_win = self._x_window.copy()
        x_win.extend(X_filtered.values.tolist())
        if self.append_predict:
            self._x_window = x_win
        
        # Convertir a tensor
        x_t = self._deque2rolling_tensor(x_win)
        
        # Usar paralelismo solo si hay suficientes datos
        if x_t.shape[0] > 1:
            # Preparar batches óptimos para GPU
            x_batches = self._prepare_inference_data(x_t)
            
            # Realizar predicción en paralelo
            if len(x_batches) > 1:
                predictions = self._parallel_predict(x_batches)
            else:
                # Un solo batch, no paralelizar
                with self._inference_lock:
                    self.module.eval()
                    with torch.inference_mode():
                        predictions = self.module(x_t)
                        if "cuda" in self.inference_device:
                            predictions = predictions.cpu()
        else:
            # Dataset muy pequeño
            with self._inference_lock:
                self.module.eval()
                with torch.inference_mode():
                    predictions = self.module(x_t)
                    if "cuda" in self.inference_device:
                        predictions = predictions.cpu()
        
        # Limpiar memoria
        torch.cuda.empty_cache()
        
        # Convertir a DataFrame
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.detach().numpy()
            
            # Asegurar formato correcto para DataFrame
            if len(predictions_np.shape) > 1:
                return pd.DataFrame(predictions_np)
            else:
                return pd.DataFrame(predictions_np.reshape(-1, 1))
        else:
            # En caso de error, devolver DataFrame vacío
            return pd.DataFrame()
        
    def _deque2rolling_tensor(self, x_win):
        """
        Convierte una cola en un tensor utilizando el dispositivo actual
        """
        return deque2rolling_tensor(x_win, device=self.device) 

class MultiProcessGPURegressor(RollingDeepEstimatorInitialized, RegressorInitialized):
    """
    Clase para regresión que usa múltiples procesos independientes para maximizar
    la utilización de la GPU con múltiples PIDs visibles en nvidia-smi.
    
    Esta clase lanza procesos Python independientes para las operaciones CUDA, cada
    uno con su propio PID visible, para maximizar el paralelismo en GPUs.
    
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
    train_device : str
        Dispositivo para entrenamiento (por defecto "cuda:0").
    inference_device : str
        Dispositivo para inferencia (por defecto "cuda:1").
    num_processes : int
        Número de procesos a ejecutar en paralelo.
    worker_script : str
        Ruta al script que ejecuta los workers. Si es None, se usa un script incorporado.
    window_size : int
        Tamaño de la ventana deslizante.
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
        num_processes: int = 4,
        worker_script: str = None,
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        worker_timeout: int = 60,
        **kwargs
    ):
        # Verificar disponibilidad de CUDA
        if not torch.cuda.is_available():
            warnings.warn("CUDA no está disponible. Usando CPU y solo 1 proceso.")
            train_device = "cpu"
            inference_device = "cpu"
            num_processes = 1
        
        # Configurar dispositivos y procesos
        self.train_device = train_device
        self.inference_device = inference_device
        self.num_processes = num_processes
        self.worker_script = worker_script
        self.worker_timeout = worker_timeout
        self.temp_dir = tempfile.mkdtemp(prefix="mpgpu_")
        
        # Inicializar clase base
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
        
        # Preparar script worker si no se proporciona uno
        if self.worker_script is None:
            self.worker_script = self._create_worker_script()
        
        # Inicializar procesos de worker
        self._init_worker_processes()
        
        print(f"MultiProcessGPURegressor iniciado para mostrar múltiples PIDs en nvidia-smi:")
        print(f"- Entrenamiento: {self.train_device}")
        print(f"- Inferencia: {self.inference_device}")
        print(f"- Procesos: {self.num_processes}")
        print("Cada proceso ejecuta operaciones CUDA con su propio PID")
    
    def _create_worker_script(self):
        """Crea un script temporal para los workers"""
        script_content = """
import os
import sys
import time
import torch
import pickle
import json
import numpy as np

def run_worker_task(task_info, device_id):
    # Configurar device
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    # Cargar información de la tarea
    task_type = task_info.get('task_type')
    input_file = task_info.get('input_file')
    output_file = task_info.get('output_file')
    worker_id = task_info.get('worker_id', 0)
    
    # Cargar datos de entrada
    with open(input_file, 'rb') as f:
        input_data = pickle.load(f)
    
    # Ejecutar tarea según el tipo
    if task_type == 'init':
        # Solo mostrar información e inicializar CUDA
        torch.cuda.init()
        result = {
            'status': 'success',
            'worker_id': worker_id,
            'pid': os.getpid(),
            'device': device,
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    elif task_type == 'learn':
        # Desempaquetar datos para entrenamiento
        module_state = input_data.get('module_state')
        x_tensor = input_data.get('x_tensor')
        y_tensor = input_data.get('y_tensor')
        optimizer_state = input_data.get('optimizer_state')
        loss_fn_name = input_data.get('loss_fn')
        
        # Cargar el modelo
        module = input_data.get('module_class')()
        module.load_state_dict(module_state)
        module.to(device)
        module.train()
        
        # Configurar optimizador
        optimizer = torch.optim.Adam(module.parameters())
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        
        # Configurar función de pérdida
        if loss_fn_name == 'mse':
            loss_fn = torch.nn.MSELoss()
        elif loss_fn_name == 'l1':
            loss_fn = torch.nn.L1Loss()
        else:
            loss_fn = torch.nn.MSELoss()  # Default
        
        # Mover datos a dispositivo
        x = torch.tensor(x_tensor, dtype=torch.float32, device=device)
        y = torch.tensor(y_tensor, dtype=torch.float32, device=device)
        
        # Entrenar
        for epoch in range(input_data.get('epochs', 1)):
            optimizer.zero_grad()
            output = module(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
        
        # Guardar resultados
        result = {
            'status': 'success',
            'worker_id': worker_id,
            'pid': os.getpid(),
            'device': device,
            'loss': loss.item(),
            'module_state': module.cpu().state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
    
    elif task_type == 'predict':
        # Desempaquetar datos para predicción
        module_state = input_data.get('module_state')
        x_tensor = input_data.get('x_tensor')
        
        # Cargar el modelo
        module = input_data.get('module_class')()
        module.load_state_dict(module_state)
        module.to(device)
        module.eval()
        
        # Mover datos a dispositivo
        x = torch.tensor(x_tensor, dtype=torch.float32, device=device)
        
        # Realizar predicción
        with torch.inference_mode():
            predictions = module(x)
            predictions_np = predictions.cpu().detach().numpy()
        
        # Guardar resultados
        result = {
            'status': 'success',
            'worker_id': worker_id,
            'pid': os.getpid(),
            'device': device,
            'predictions': predictions_np
        }
    
    else:
        result = {
            'status': 'error',
            'worker_id': worker_id,
            'pid': os.getpid(),
            'error': f'Unknown task type: {task_type}'
        }
    
    # Guardar resultado
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    # Escribir archivo de señalización
    signal_file = output_file + '.done'
    with open(signal_file, 'w') as f:
        f.write('done')
    
    return result

if __name__ == '__main__':
    # Recibir argumentos del worker
    task_file = sys.argv[1]
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    # Cargar tarea
    with open(task_file, 'r') as f:
        task_info = json.load(f)
    
    # Ejecutar tarea
    run_worker_task(task_info, device_id)
    
    # Mantener proceso vivo para mostrar en nvidia-smi
    keep_alive = os.environ.get('MPGPU_KEEP_ALIVE', 'false').lower() == 'true'
    if keep_alive:
        time.sleep(int(os.environ.get('MPGPU_KEEP_ALIVE_TIME', '300')))
"""
        # Escribir script a un archivo temporal
        script_path = os.path.join(self.temp_dir, "mpgpu_worker.py")
        with open(script_path, "w") as f:
            f.write(script_content)
        return script_path
    
    def _init_worker_processes(self):
        """Inicializa los procesos de worker"""
        self.worker_processes = []
        
        # Determinar cuántos workers usar para cada dispositivo
        train_workers = max(1, self.num_processes // 2)
        inference_workers = max(1, self.num_processes - train_workers)
        
        # Dispositivo de entrenamiento
        train_device_id = int(self.train_device.split(":")[-1]) if "cuda:" in self.train_device else 0
        
        # Dispositivo de inferencia
        inference_device_id = int(self.inference_device.split(":")[-1]) if "cuda:" in self.inference_device else 0
        
        # Iniciar workers de entrenamiento
        for i in range(train_workers):
            worker_id = f"train_{i}"
            self._start_worker(worker_id, train_device_id, keep_alive=True)
        
        # Iniciar workers de inferencia
        for i in range(inference_workers):
            worker_id = f"infer_{i}"
            self._start_worker(worker_id, inference_device_id, keep_alive=True)
        
        # Esperar a que los workers estén listos
        time.sleep(2)
    
    def _start_worker(self, worker_id, device_id, keep_alive=False):
        """Inicia un proceso de worker"""
        # Preparar archivos para comunicación
        task_file = os.path.join(self.temp_dir, f"task_{worker_id}.json")
        input_file = os.path.join(self.temp_dir, f"input_{worker_id}.pkl")
        output_file = os.path.join(self.temp_dir, f"output_{worker_id}.pkl")
        
        # Preparar tarea de inicialización
        task_info = {
            "task_type": "init",
            "worker_id": worker_id,
            "input_file": input_file,
            "output_file": output_file
        }
        
        # Guardar tarea
        with open(task_file, "w") as f:
            json.dump(task_info, f)
        
        # Guardar datos vacíos
        with open(input_file, "wb") as f:
            pickle.dump({}, f)
        
        # Configurar entorno
        env = os.environ.copy()
        if keep_alive:
            env["MPGPU_KEEP_ALIVE"] = "true"
            env["MPGPU_KEEP_ALIVE_TIME"] = str(3600)  # 1 hora
        
        # Iniciar proceso
        cmd = [sys.executable, self.worker_script, task_file, str(device_id)]
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Guardar información del proceso
        self.worker_processes.append({
            "worker_id": worker_id,
            "process": process,
            "device_id": device_id,
            "task_file": task_file,
            "input_file": input_file,
            "output_file": output_file,
            "pid": process.pid
        })
        
        return process.pid
    
    def _run_worker_task(self, task_type, worker_type, data, timeout=None):
        """Ejecuta una tarea en un worker"""
        # Seleccionar worker según el tipo
        available_workers = [w for w in self.worker_processes if worker_type in w["worker_id"]]
        if not available_workers:
            raise ValueError(f"No hay workers disponibles de tipo {worker_type}")
        
        # Seleccionar worker de forma round-robin
        worker_idx = hash(str(time.time())) % len(available_workers)
        worker = available_workers[worker_idx]
        
        # Preparar archivos para comunicación
        task_file = worker["task_file"]
        input_file = worker["input_file"]
        output_file = worker["output_file"]
        done_file = output_file + ".done"
        
        # Eliminar archivos antiguos
        if os.path.exists(done_file):
            os.remove(done_file)
        
        # Preparar tarea
        task_info = {
            "task_type": task_type,
            "worker_id": worker["worker_id"],
            "input_file": input_file,
            "output_file": output_file
        }
        
        # Guardar tarea
        with open(task_file, "w") as f:
            json.dump(task_info, f)
        
        # Guardar datos
        with open(input_file, "wb") as f:
            pickle.dump(data, f)
        
        # Reiniciar worker si está caído
        if worker["process"].poll() is not None:
            print(f"Worker {worker['worker_id']} caído, reiniciando...")
            pid = self._start_worker(worker["worker_id"], worker["device_id"])
            worker["process"] = next(w["process"] for w in self.worker_processes if w["pid"] == pid)
        
        # Esperar resultado
        timeout = timeout or self.worker_timeout
        start_time = time.time()
        while not os.path.exists(done_file) and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not os.path.exists(done_file):
            raise TimeoutError(f"Timeout esperando respuesta del worker {worker['worker_id']}")
        
        # Cargar resultado
        with open(output_file, "rb") as f:
            result = pickle.load(f)
        
        return result
    
    def __del__(self):
        """Limpieza de recursos al destruir el objeto"""
        try:
            # Terminar procesos
            for worker in self.worker_processes:
                if worker["process"].poll() is None:  # Proceso aún activo
                    worker["process"].terminate()
            
            # Esperar a que terminen
            time.sleep(1)
            
            # Forzar terminación si es necesario
            for worker in self.worker_processes:
                if worker["process"].poll() is None:  # Proceso aún activo
                    worker["process"].kill()
            
            # Limpiar archivos temporales
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error al eliminar {file_path}: {e}")
            
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
            
            # Liberar modelo
            if hasattr(self, 'module'):
                self.module = None
                
            # Vaciar caché CUDA
            torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error al limpiar recursos: {e}")
    
    def _prepare_module_data(self):
        """Prepara los datos del módulo para transferencia a través de pickle"""
        # Asegurarse de que el módulo esté en CPU para serialización
        self.module.cpu()
        
        # Extraer la clase del módulo para recreación
        module_class = self.module.__class__
        
        # Obtener estado del módulo
        module_state = self.module.state_dict()
        
        # Obtener estado del optimizador
        optimizer_state = self.optimizer.state_dict() if hasattr(self, 'optimizer') else None
        
        return {
            'module_class': module_class,
            'module_state': module_state,
            'optimizer_state': optimizer_state,
            'loss_fn': self.loss_fn if isinstance(self.loss_fn, str) else 'mse'
        }
    
    def _update_module_from_result(self, result):
        """Actualiza el módulo con el resultado del worker"""
        if 'module_state' in result:
            self.module.load_state_dict(result['module_state'])
        
        if 'optimizer_state' in result and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(result['optimizer_state'])
    
    def learn_one(self, x: dict, y: base.typing.RegTarget, **kwargs) -> "MultiProcessGPURegressor":
        """
        Entrena con un ejemplo usando un proceso independiente.
        
        Parameters
        ----------
        x : dict
            Ejemplo de entrada
        y : base.typing.RegTarget
            Valor objetivo
            
        Returns
        -------
        MultiProcessGPURegressor
            El estimador
        """
        # Actualizar características observadas
        self._update_observed_features(x)
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])
        
        if len(self._x_window) == self.window_size:
            # Convertir a tensor en CPU
            x_t = self._deque2rolling_tensor(self._x_window).cpu().numpy()
            y_t = np.array([[y]], dtype=np.float32)
            
            # Preparar datos del módulo
            module_data = self._prepare_module_data()
            
            # Datos completos para la tarea
            task_data = {
                **module_data,
                'x_tensor': x_t,
                'y_tensor': y_t,
                'epochs': 1
            }
            
            # Ejecutar tarea en worker
            result = self._run_worker_task('learn', 'train', task_data)
            
            # Actualizar módulo con resultado
            self._update_module_from_result(result)
            
            print(f"Entrenamiento realizado por worker PID: {result.get('pid')}")
        
        return self
    
    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> "MultiProcessGPURegressor":
        """
        Entrena con múltiples ejemplos usando procesos independientes.
        
        Parameters
        ----------
        X : pd.DataFrame
            Ejemplos de entrada
        y : pd.Series
            Valores objetivo
            
        Returns
        -------
        MultiProcessGPURegressor
            El estimador
        """
        # Actualizar características observadas
        self._update_observed_features(X)
        X_filtered = X[list(self.observed_features)]
        self._x_window.extend(X_filtered.values.tolist())
        
        if len(self._x_window) == self.window_size:
            # Convertir a tensores en CPU
            x_t = self._deque2rolling_tensor(self._x_window).cpu().numpy()
            y_t = y.values.reshape(-1, 1).astype(np.float32)
            
            # Dividir datos para procesos en paralelo
            batch_size = max(1, len(y_t) // (self.num_processes // 2))
            results = []
            
            # Ejecutar tareas en múltiples workers en paralelo
            for i in range(0, len(y_t), batch_size):
                # Obtener batch
                x_batch = x_t[i:i+batch_size]
                y_batch = y_t[i:i+batch_size]
                
                # Preparar datos del módulo
                module_data = self._prepare_module_data()
                
                # Datos completos para la tarea
                task_data = {
                    **module_data,
                    'x_tensor': x_batch,
                    'y_tensor': y_batch,
                    'epochs': 1
                }
                
                # Ejecutar tarea en worker
                result = self._run_worker_task('learn', 'train', task_data)
                results.append(result)
                
                # Actualizar módulo con el último resultado
                self._update_module_from_result(result)
            
            pids = [r.get('pid') for r in results]
            print(f"Entrenamiento realizado por workers PIDs: {pids}")
        
        return self
    
    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """
        Predice para un ejemplo usando un proceso independiente.
        
        Parameters
        ----------
        x : dict
            Ejemplo de entrada
            
        Returns
        -------
        base.typing.RegTarget
            Valor predicho
        """
        # Actualizar características observadas
        self._update_observed_features(x)
        
        # Preparar ventana de datos
        x_win = self._x_window.copy()
        x_win.append([x.get(feature, 0) for feature in self.observed_features])
        if self.append_predict:
            self._x_window = x_win
        
        # Convertir a tensor en CPU
        x_t = self._deque2rolling_tensor(x_win).cpu().numpy()
        
        # Preparar datos del módulo
        module_data = self._prepare_module_data()
        
        # Datos completos para la tarea
        task_data = {
            **module_data,
            'x_tensor': x_t
        }
        
        # Ejecutar tarea en worker
        result = self._run_worker_task('predict', 'infer', task_data)
        
        # Extraer predicción
        pred = result.get('predictions', [[0.0]])[0][0]
        
        print(f"Predicción realizada por worker PID: {result.get('pid')}")
        
        return pred
    
    def predict_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predice para múltiples ejemplos usando procesos independientes.
        
        Parameters
        ----------
        X : pd.DataFrame
            Ejemplos de entrada
            
        Returns
        -------
        pd.DataFrame
            Dataframe con los valores predichos
        """
        # Actualizar características observadas
        self._update_observed_features(X)
        
        # Preparar ventana de datos
        X_filtered = X[list(self.observed_features)]
        x_win = self._x_window.copy()
        x_win.extend(X_filtered.values.tolist())
        if self.append_predict:
            self._x_window = x_win
        
        # Convertir a tensor en CPU
        x_t = self._deque2rolling_tensor(x_win).cpu().numpy()
        
        # Dividir datos para procesos en paralelo
        batch_size = max(1, len(x_t) // (self.num_processes // 2))
        results = []
        pids = []
        
        # Ejecutar tareas en múltiples workers en paralelo
        for i in range(0, len(x_t), batch_size):
            # Obtener batch
            x_batch = x_t[i:i+batch_size]
            
            # Preparar datos del módulo
            module_data = self._prepare_module_data()
            
            # Datos completos para la tarea
            task_data = {
                **module_data,
                'x_tensor': x_batch
            }
            
            # Ejecutar tarea en worker
            result = self._run_worker_task('predict', 'infer', task_data)
            results.append(result.get('predictions', []))
            pids.append(result.get('pid'))
        
        # Combinar resultados
        all_predictions = np.vstack(results) if results else np.array([[]])
        
        print(f"Predicción realizada por workers PIDs: {pids}")
        
        # Convertir a DataFrame
        if all_predictions.size > 0:
            if len(all_predictions.shape) > 1:
                return pd.DataFrame(all_predictions)
            else:
                return pd.DataFrame(all_predictions.reshape(-1, 1))
        else:
            return pd.DataFrame()
    
    def _deque2rolling_tensor(self, x_win):
        """
        Convierte una cola en un tensor utilizando CPU para mayor compatibilidad
        """
        return deque2rolling_tensor(x_win, device="cpu") 