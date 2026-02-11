"""
Sistema de Estimação de Distância para Drones sobre Superfícies Aquáticas
Implementação Final para Produção
Author: Water Surface Distance Estimation Team
Version: 1.0.0
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, 
                                     Flatten, Input, concatenate, Lambda)
import numpy as np
import pandas as pd
import cv2
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy.interpolate import UnivariateSpline
import warnings

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÕES E PARÂMETROS
# ============================================================================

@dataclass
class ModelConfig:
    """Configurações do modelo e treinamento"""
    # Dimensões
    image_size: Tuple[int, int] = (224, 224)
    min_distance: float = 50.0
    max_distance: float = 800.0
    
    # Arquitetura
    conv_filters: List[int] = None
    dense_units: List[int] = None
    dropout_rate: float = 0.3
    
    # Treinamento
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 10
    validation_split: float = 0.2
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    model_save_path: str = "./models/final_model.h5"
    
    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [32, 64, 128, 256]
        if self.dense_units is None:
            self.dense_units = [512, 256, 128]

# ============================================================================
# PROCESSAMENTO DE IMAGENS E EXTRAÇÃO DE CARACTERÍSTICAS
# ============================================================================

class HSVProcessor:
    """Processador para extração do canal V do HSV"""
    
    @staticmethod
    def extract_value_channel(image: np.ndarray) -> np.ndarray:
        """
        Extrai o canal Value do espaço HSV
        
        Args:
            image: Imagem RGB
            
        Returns:
            Canal V normalizado
        """
        if len(image.shape) == 2:
            return image
        
        # Converter para HSV
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        value_channel = hsv[:, :, 2].astype(np.float32) / 255.0
        
        return value_channel

class WaterFeatureExtractor:
    """Extrator de características específicas para superfícies aquáticas"""
    
    @staticmethod
    def extract_features(image_v: np.ndarray) -> Dict[str, float]:
        """
        Extrai características relevantes da superfície da água
        
        Args:
            image_v: Canal V da imagem
            
        Returns:
            Dicionário com características extraídas
        """
        features = {}
        
        # 1. Análise de Frequência
        fft = np.fft.fft2(image_v)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Energia de alta frequência
        threshold = np.percentile(magnitude, 90)
        features['high_freq_energy'] = np.sum(magnitude[magnitude > threshold])
        
        # 2. Detecção de picos (reflexões)
        img_uint8 = (image_v * 255).astype(np.uint8)
        corners = cv2.goodFeaturesToTrack(img_uint8, 
                                          maxCorners=100,
                                          qualityLevel=0.01,
                                          minDistance=10)
        features['num_peaks'] = len(corners) if corners is not None else 0
        
        # 3. Análise de gradientes
        sobelx = cv2.Sobel(image_v, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image_v, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        # 4. Estatísticas de textura
        features['std_dev'] = np.std(image_v)
        features['entropy'] = -np.sum(image_v * np.log2(image_v + 1e-10))
        
        # 5. Estatísticas locais (janelas 16x16)
        h, w = image_v.shape
        window_size = 16
        local_stds = []
        
        for i in range(0, h - window_size, window_size):
            for j in range(0, w - window_size, window_size):
                window = image_v[i:i+window_size, j:j+window_size]
                local_stds.append(np.std(window))
        
        features['local_std_mean'] = np.mean(local_stds)
        features['local_std_var'] = np.var(local_stds)
        
        return features

# ============================================================================
# AUGMENTAÇÃO DE DADOS
# ============================================================================

class WaterAugmentation:
    """Augmentação específica para imagens de superfícies aquáticas"""
    
    def __init__(self, 
                 gamma_range: Tuple[float, float] = (0.7, 1.3),
                 noise_std: float = 0.02,
                 blur_prob: float = 0.3):
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.blur_prob = blur_prob
    
    def augment(self, image: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Aplica augmentações específicas para água
        
        Args:
            image: Imagem de entrada
            training: Se True, aplica augmentações estocásticas
            
        Returns:
            Imagem augmentada
        """
        if not training:
            return image
        
        augmented = image.copy()
        
        # Variação de iluminação (gamma correction)
        if np.random.random() > 0.5:
            gamma = np.random.uniform(*self.gamma_range)
            augmented = np.power(augmented, gamma)
        
        # Simular reflexões e ruído
        if np.random.random() > 0.5:
            noise = np.random.normal(0, self.noise_std, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)
        
        # Simular movimento/ondas com blur
        if np.random.random() < self.blur_prob:
            kernel_size = np.random.choice([3, 5])
            augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
        
        return augmented

# ============================================================================
# GERADORES DE DADOS CUSTOMIZADOS
# ============================================================================

class WaterDistanceDataGenerator(keras.utils.Sequence):
    """Gerador de dados customizado para o modelo multi-input"""
    
    def __init__(self,
                 dataframe: pd.DataFrame,
                 directory: str,
                 config: ModelConfig,
                 augmentation: Optional[WaterAugmentation] = None,
                 training: bool = True,
                 shuffle: bool = True):
        
        self.df = dataframe.reset_index(drop=True)
        self.directory = directory
        self.config = config
        self.augmentation = augmentation
        self.training = training
        self.shuffle = shuffle
        
        self.hsv_processor = HSVProcessor()
        self.feature_extractor = WaterFeatureExtractor()
        
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.config.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index * self.config.batch_size:(index + 1) * self.config.batch_size]
        
        batch_images = []
        batch_features = []
        batch_distances = []
        
        for idx in indices:
            # Carregar imagem
            img_path = Path(self.directory) / self.df.loc[idx, 'nome']
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.config.image_size)
            
            # Extrair canal V
            value_channel = self.hsv_processor.extract_value_channel(image)
            
            # Aplicar augmentação
            if self.augmentation:
                value_channel = self.augmentation.augment(value_channel, self.training)
            
            # Extrair características
            features = self.feature_extractor.extract_features(value_channel)
            feature_vector = np.array(list(features.values()))
            
            # Adicionar ao batch
            batch_images.append(value_channel[..., np.newaxis])
            batch_features.append(feature_vector)
            batch_distances.append(self.df.loc[idx, 'distancia'])
        
        X = {
            'image_input': np.array(batch_images),
            'feature_input': np.array(batch_features)
        }
        y = np.array(batch_distances)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ============================================================================
# CAMADAS CUSTOMIZADAS
# ============================================================================

class PhysicalConstraintLayer(layers.Layer):
    """Camada para impor restrições físicas nas predições"""
    
    def __init__(self, min_distance: float, max_distance: float, **kwargs):
        super().__init__(**kwargs)
        self.min_distance = min_distance
        self.max_distance = max_distance
    
    def call(self, inputs):
        # Converter para range normalizado [0, 1]
        min_norm = self.min_distance / self.max_distance
        max_norm = 1.0
        
        return tf.clip_by_value(inputs, min_norm, max_norm)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'min_distance': self.min_distance,
            'max_distance': self.max_distance
        })
        return config

# ============================================================================
# CONSTRUÇÃO DO MODELO
# ============================================================================

class WaterDistanceModel:
    """Modelo completo para estimação de distância"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.calibrator = None
        
    def build_model(self, num_features: int = 8) -> keras.Model:
        """
        Constrói a arquitetura multi-input do modelo
        
        Args:
            num_features: Número de características extraídas
            
        Returns:
            Modelo compilado
        """
        # Branch de imagem (CNN)
        image_input = Input(shape=(*self.config.image_size, 1), name='image_input')
        
        x = image_input
        for i, filters in enumerate(self.config.conv_filters):
            x = Conv2D(filters, (5, 5), activation='relu', name=f'conv_{i}')(x)
            x = MaxPooling2D(name=f'pool_{i}')(x)
            
            if i >= 2:  # Dropout nas camadas mais profundas
                x = Dropout(self.config.dropout_rate, name=f'dropout_conv_{i}')(x)
        
        x = Flatten(name='flatten')(x)
        
        # Branch de características
        feature_input = Input(shape=(num_features,), name='feature_input')
        f = Dense(64, activation='relu', name='feature_dense_1')(feature_input)
        f = Dropout(self.config.dropout_rate, name='dropout_feat')(f)
        f = Dense(32, activation='relu', name='feature_dense_2')(f)
        
        # Fusão
        combined = concatenate([x, f], name='fusion')
        
        # Camadas densas finais
        z = combined
        for i, units in enumerate(self.config.dense_units):
            z = Dense(units, activation='relu', name=f'dense_{i}')(z)
            z = Dropout(self.config.dropout_rate, name=f'dropout_dense_{i}')(z)
        
        # Saída com restrições físicas
        output = Dense(1, activation='linear', name='output')(z)
        output = PhysicalConstraintLayer(
            self.config.min_distance, 
            self.config.max_distance,
            name='physical_constraint'
        )(output)
        
        # Criar modelo
        model = keras.Model(inputs=[image_input, feature_input], outputs=output)
        
        self.model = model
        return model
    
    def compile_model(self, 
                     optimizer: Optional[keras.optimizers.Optimizer] = None,
                     loss: Optional[str] = None,
                     metrics: Optional[List] = None):
        """
        Compila o modelo com configurações otimizadas
        
        Args:
            optimizer: Otimizador (padrão: Adam com learning rate schedule)
            loss: Função de perda (padrão: Huber loss para robustez)
            metrics: Métricas para monitoramento
        """
        if self.model is None:
            raise ValueError("Modelo deve ser construído antes da compilação")
        
        # Learning rate schedule
        if optimizer is None:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=1000,
                decay_rate=0.9,
                staircase=True
            )
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Huber loss é mais robusta a outliers que MSE
        if loss is None:
            loss = keras.losses.Huber(delta=0.1)
        
        # Métricas padrão
        if metrics is None:
            metrics = [
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse'),
                keras.metrics.MeanAbsolutePercentageError(name='mape')
            ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info("Modelo compilado com sucesso")
        logger.info(f"Otimizador: {optimizer.__class__.__name__}")
        logger.info(f"Loss: {loss.__class__.__name__}")
    
    def create_callbacks(self) -> List[callbacks.Callback]:
        """
        Cria callbacks para o treinamento
        
        Returns:
            Lista de callbacks
        """
        callback_list = []
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        # Model checkpoint
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint = callbacks.ModelCheckpoint(
            filepath=f"{self.config.checkpoint_dir}/model_{{epoch:03d}}_{{val_loss:.4f}}.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callback_list.append(checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # TensorBoard
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        tensorboard = callbacks.TensorBoard(
            log_dir=self.config.log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callback_list.append(tensorboard)
        
        # Custom callback para logging
        class MetricsLogger(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    logger.info(f"Epoch {epoch + 1} - " + 
                              " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()]))
        
        callback_list.append(MetricsLogger())
        
        return callback_list
    
    def fit(self,
            train_generator: WaterDistanceDataGenerator,
            val_generator: Optional[WaterDistanceDataGenerator] = None,
            epochs: Optional[int] = None,
            callbacks: Optional[List] = None) -> keras.callbacks.History:
        """
        Treina o modelo
        
        Args:
            train_generator: Gerador de dados de treino
            val_generator: Gerador de dados de validação
            epochs: Número de épocas
            callbacks: Lista de callbacks customizados
            
        Returns:
            Histórico de treinamento
        """
        if self.model is None:
            raise ValueError("Modelo deve ser compilado antes do treinamento")
        
        epochs = epochs or self.config.epochs
        
        if callbacks is None:
            callbacks = self.create_callbacks()
        
        logger.info(f"Iniciando treinamento por {epochs} épocas")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Amostras de treino: {len(train_generator.df)}")
        if val_generator:
            logger.info(f"Amostras de validação: {len(val_generator.df)}")
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Treinamento concluído")
        
        return history

# ============================================================================
# CALIBRAÇÃO PÓS-TREINAMENTO
# ============================================================================

class BiasCalibrator:
    """Sistema de calibração para correção de bias sistemático"""
    
    def __init__(self):
        self.calibration_curve = None
        self.calibration_data = None
    
    def calibrate(self, 
                  predictions: np.ndarray, 
                  ground_truth: np.ndarray,
                  smoothing: float = 0.1) -> None:
        """
        Aprende curva de calibração
        
        Args:
            predictions: Predições do modelo
            ground_truth: Valores verdadeiros
            smoothing: Parâmetro de suavização para spline
        """
        # Ordenar por predições para criar curva monotônica
        sorted_indices = np.argsort(predictions.flatten())
        pred_sorted = predictions[sorted_indices].flatten()
        truth_sorted = ground_truth[sorted_indices].flatten()
        
        # Criar spline de calibração
        self.calibration_curve = UnivariateSpline(
            pred_sorted, 
            truth_sorted,
            s=smoothing * len(predictions),
            k=3  # Spline cúbica
        )
        
        # Armazenar dados de calibração para análise
        self.calibration_data = {
            'predictions': pred_sorted,
            'ground_truth': truth_sorted
        }
        
        logger.info("Calibração concluída")
    
    def correct(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aplica correção de calibração
        
        Args:
            predictions: Predições para corrigir
            
        Returns:
            Predições corrigidas
        """
        if self.calibration_curve is None:
            warnings.warn("Calibrador não foi treinado, retornando predições originais")
            return predictions
        
        return self.calibration_curve(predictions)
    
    def save(self, filepath: str) -> None:
        """Salva calibração em arquivo"""
        if self.calibration_data is None:
            raise ValueError("Calibrador não foi treinado")
        
        np.savez(filepath, **self.calibration_data)
        logger.info(f"Calibração salva em {filepath}")
    
    def load(self, filepath: str) -> None:
        """Carrega calibração de arquivo"""
        data = np.load(filepath)
        self.calibration_data = dict(data)
        
        self.calibration_curve = UnivariateSpline(
            self.calibration_data['predictions'],
            self.calibration_data['ground_truth'],
            s=0.1 * len(self.calibration_data['predictions']),
            k=3
        )
        
        logger.info(f"Calibração carregada de {filepath}")

# ============================================================================
# SISTEMA DE INFERÊNCIA PARA PRODUÇÃO
# ============================================================================

class WaterDistancePredictor:
    """Sistema completo de predição para deploy"""
    
    def __init__(self, 
                 model_path: str,
                 calibration_path: Optional[str] = None,
                 config: Optional[ModelConfig] = None):
        
        self.config = config or ModelConfig()
        self.model = keras.models.load_model(
            model_path,
            custom_objects={'PhysicalConstraintLayer': PhysicalConstraintLayer}
        )
        
        self.calibrator = BiasCalibrator()
        if calibration_path and Path(calibration_path).exists():
            self.calibrator.load(calibration_path)
        
        self.hsv_processor = HSVProcessor()
        self.feature_extractor = WaterFeatureExtractor()
        
        logger.info(f"Modelo carregado de {model_path}")
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocessa imagem para inferência
        
        Args:
            image: Imagem RGB ou BGR
            
        Returns:
            Tupla (imagem processada, vetor de características)
        """
        # Garantir formato RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # Redimensionar
        image = cv2.resize(image, self.config.image_size)
        
        # Extrair canal V
        value_channel = self.hsv_processor.extract_value_channel(image)
        
        # Extrair características
        features = self.feature_extractor.extract_features(value_channel)
        feature_vector = np.array(list(features.values()))
        
        # Preparar para batch
        image_batch = value_channel[np.newaxis, ..., np.newaxis]
        feature_batch = feature_vector[np.newaxis, :]
        
        return image_batch, feature_batch
    
    def predict(self, 
                image: np.ndarray,
                return_confidence: bool = False) -> Dict[str, float]:
        """
        Realiza predição de distância
        
        Args:
            image: Imagem de entrada
            return_confidence: Se True, retorna intervalo de confiança
            
        Returns:
            Dicionário com resultados
        """
        # Preprocessar
        image_batch, feature_batch = self.preprocess_image(image)
        
        # Predição
        inputs = {
            'image_input': image_batch,
            'feature_input': feature_batch
        }
        
        prediction_norm = self.model.predict(inputs, verbose=0)[0, 0]
        
        # Converter para distância real
        prediction = prediction_norm * self.config.max_distance
        
        # Aplicar calibração se disponível
        if self.calibrator.calibration_curve:
            prediction_calibrated = self.calibrator.correct(prediction_norm)[0]
            prediction_calibrated *= self.config.max_distance
        else:
            prediction_calibrated = prediction
        
        result = {
            'distance_raw': float(prediction),
            'distance_calibrated': float(prediction_calibrated),
            'normalized_value': float(prediction_norm)
        }
        
        # Estimar incerteza (usando dropout durante inferência)
        if return_confidence:
            predictions_mc = []
            for _ in range(10):  # Monte Carlo dropout
                pred = self.model(inputs, training=True).numpy()[0, 0]
                predictions_mc.append(pred * self.config.max_distance)
            
            predictions_mc = np.array(predictions_mc)
            result['confidence_interval'] = {
                'lower': float(np.percentile(predictions_mc, 5)),
                'upper': float(np.percentile(predictions_mc, 95)),
                'std': float(np.std(predictions_mc))
            }
        
        return result
    
    def predict_batch(self, 
                      images: List[np.ndarray],
                      batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Realiza predição em lote
        
        Args:
            images: Lista de imagens
            batch_size: Tamanho do batch
            
        Returns:
            Lista de resultados
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            image_batches = []
            feature_batches = []
            
            for img in batch_images:
                img_batch, feat_batch = self.preprocess_image(img)
                image_batches.append(img_batch[0])
                feature_batches.append(feat_batch[0])
            
            inputs = {
                'image_input': np.array(image_batches),
                'feature_input': np.array(feature_batches)
            }
            
            predictions = self.model.predict(inputs, verbose=0)
            
            for pred in predictions:
                distance = pred[0] * self.config.max_distance
                
                if self.calibrator.calibration_curve:
                    distance_cal = self.calibrator.correct(pred[0])[0]
                    distance_cal *= self.config.max_distance
                else:
                    distance_cal = distance
                
                results.append({
                    'distance_raw': float(distance),
                    'distance_calibrated': float(distance_cal)
                })
        
        return results

# ============================================================================
# AVALIAÇÃO E MÉTRICAS
# ============================================================================

class ModelEvaluator:
    """Sistema de avaliação detalhada do modelo"""
    
    @staticmethod
    def evaluate_by_ranges(predictions: np.ndarray,
                          ground_truth: np.ndarray,
                          ranges: Optional[List[Tuple[float, float]]] = None) -> pd.DataFrame:
        """
        Avalia performance por faixas de distância
        
        Args:
            predictions: Predições do modelo
            ground_truth: Valores verdadeiros
            ranges: Faixas de distância para análise
            
        Returns:
            DataFrame com métricas por faixa
        """
        if ranges is None:
            ranges = [(50, 100), (100, 200), (200, 400), (400, 800)]
        
        results = []
        
        for min_d, max_d in ranges:
            mask = (ground_truth >= min_d) & (ground_truth < max_d)
            
            if not mask.any():
                continue
            
            pred_range = predictions[mask]
            truth_range = ground_truth[mask]
            
            mae = np.mean(np.abs(pred_range - truth_range))
            rmse = np.sqrt(np.mean((pred_range - truth_range) ** 2))
            mape = np.mean(np.abs((pred_range - truth_range) / truth_range)) * 100
            
            # Percentis para análise de robustez
            errors = np.abs(pred_range - truth_range)
            p50 = np.percentile(errors, 50)
            p95 = np.percentile(errors, 95)
            
            results.append({
                'range': f'{min_d}-{max_d}cm',
                'samples': mask.sum(),
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'median_error': p50,
                'p95_error': p95,
                'std_error': np.std(errors)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def plot_calibration_curve(calibrator: BiasCalibrator,
                               save_path: Optional[str] = None):
        """
        Plota curva de calibração
        
        Args:
            calibrator: Objeto calibrador treinado
            save_path: Caminho para salvar o gráfico
        """
        import matplotlib.pyplot as plt
        
        if calibrator.calibration_data is None:
            raise ValueError("Calibrador não foi treinado")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot original
        axes[0].scatter(calibrator.calibration_data['predictions'],
                       calibrator.calibration_data['ground_truth'],
                       alpha=0.5, s=1)
        axes[0].plot([0, 1], [0, 1], 'r--', label='Ideal')
        axes[0].set_xlabel('Predições')
        axes[0].set_ylabel('Valores Reais')
        axes[0].set_title('Calibração: Antes')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Curva de calibração
        x_smooth = np.linspace(
            calibrator.calibration_data['predictions'].min(),
            calibrator.calibration_data['predictions'].max(),
            100
        )
        y_smooth = calibrator.correct(x_smooth)
        
        axes[1].plot(x_smooth, y_smooth, 'b-', label='Curva de Calibração')
        axes[1].plot([0, 1], [0, 1], 'r--', label='Sem Calibração')
        axes[1].set_xlabel('Predições')
        axes[1].set_ylabel('Predições Calibradas')
        axes[1].set_title('Função de Calibração')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em {save_path}")
        
        plt.show()

# ============================================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# ============================================================================

def train_model(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                data_dir: str,
                config: Optional[ModelConfig] = None) -> Tuple[WaterDistanceModel, dict]:
    """
    Função principal para treinar o modelo completo
    
    Args:
        train_df: DataFrame de treino
        val_df: DataFrame de validação
        data_dir: Diretório com as imagens
        config: Configurações do modelo
    
    Returns:
        Tupla (modelo treinado, histórico)
    """
    config = config or ModelConfig()
    
    # Criar augmentação
    augmentation = WaterAugmentation()
    
    # Criar geradores
    train_generator = WaterDistanceDataGenerator(
        train_df, data_dir, config, 
        augmentation=augmentation,
        training=True
    )
    
    val_generator = WaterDistanceDataGenerator(
        val_df, data_dir, config,
        augmentation=None,
        training=False
    )
    
    # Construir e compilar modelo
    model_wrapper = WaterDistanceModel(config)
    model_wrapper.build_model(num_features=8)
    model_wrapper.compile_model()
    
    # Mostrar sumário
    model_wrapper.model.summary()
    
    # Treinar
    history = model_wrapper.fit(
        train_generator,
        val_generator
    )
    
    # Salvar modelo final
    Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model_wrapper.model.save(config.model_save_path)
    logger.info(f"Modelo salvo em {config.model_save_path}")
    
    return model_wrapper, history.history

# ============================================================================
# EXEMPLO DE USO PARA DEPLOY
# ============================================================================

def main_training_pipeline():
    """Pipeline completo de treinamento"""
    
    # Configurações
    config = ModelConfig(
        batch_size=32,
        epochs=100,
        learning_rate=0.001,
        patience=10
    )
    
    # Carregar dados (exemplo)
    # train_df = pd.read_csv('train_data.csv')
    # val_df = pd.read_csv('val_data.csv')
    # data_dir = '/path/to/images'
    
    # Treinar modelo
    # model, history = train_model(train_df, val_df, data_dir, config)
    
    logger.info("Pipeline de treinamento configurado")

def main_inference_example():
    """Exemplo de uso para inferência em produção"""
    
    # Carregar predictor
    predictor = WaterDistancePredictor(
        model_path='./models/final_model.h5',
        calibration_path='./models/calibration.npz'
    )
    
    # Exemplo de predição única
    # image = cv2.imread('test_image.jpg')
    # result = predictor.predict(image, return_confidence=True)
    # print(f"Distância estimada: {result['distance_calibrated']:.2f} cm")
    # print(f"Intervalo de confiança: [{result['confidence_interval']['lower']:.2f}, "
    #       f"{result['confidence_interval']['upper']:.2f}]")
    
    logger.info("Sistema de inferência configurado")

if __name__ == "__main__":
    # Configurar para modo de treinamento ou inferência
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        main_training_pipeline()
    else:
        main_inference_example()
