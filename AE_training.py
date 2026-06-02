#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPATIBLE Autoencoder Neural Network Training for Force Prediction
Совместимая архитектура автоэнкодера для работы с вашим кодом инференса
Особенности:
- ТОЧНОЕ СООТВЕТСТВИЕ архитектуры ожидаемой в коде инференса
- Правильные имена слоев и их порядок
- Совместимые размеры для всех слоев
- Уникальные директории для каждой модели без перезаписи
- Поддержка геометрических параметров датчика (диаметр D)
- Использование количества маркеров из кода подготовки датасета
"""
import sys
import os
from pathlib import Path
import re
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =======================
# Configuration Parameters - ТОЧНО СОВМЕСТИМЫЕ С ИНФЕРЕНСОМ
# =======================
# Геометрические параметры датчиков (копируются из кода подготовки)
SENSOR_GEOMETRY = {
    'A': {'D': 8.69, 'M': 12},  # 12 маркеров (3x4 сетка)
    'B': {'D': 6.94, 'M': 15},  # 15 маркеров (3x5 сетка)
    'C': {'D': 6.47, 'M': 20},  # 20 маркеров (4x5 сетка)
    'D': {'D': 5.75, 'M': 24},  # 24 маркера (4x6 сетка)
    'E': {'D': 5.14, 'M': 30}   # 30 маркеров (5x6 сетка)
}

SENSOR_TYPE = "C"  # тип датчика (должен соответствовать подготовленному датасету)
BASE_DIR = Path("Datasets")
LABELLED_DIR = BASE_DIR / "Labelled"
AGGREGATED_DIR = LABELLED_DIR / f"Aggregated_deformation_vectors_sensor_{SENSOR_TYPE}"

RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 256
EPOCHS = 500
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 100
MIN_DELTA = 0.001

# Веса для комбинированной потери
RECONSTRUCTION_WEIGHT = 0.7
FORCE_WEIGHT = 0.3

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Output dir - создаем уникальную папку с временной меткой
BASE_MODELS_DIR = Path("models")
AUTOENCODER_BASE_DIR = BASE_MODELS_DIR / f"{SENSOR_TYPE}_autoencoder_compatible"
# Создаем базовую директорию для автоэнкодеров, если не существует
AUTOENCODER_BASE_DIR.mkdir(parents=True, exist_ok=True)
# Создаем уникальную папку для текущей модели
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = AUTOENCODER_BASE_DIR / f"run_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Model artifacts will be saved to: {RUN_DIR}")

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =======================
# АРХИТЕКТУРА МОДЕЛИ ДЛЯ СОВМЕСТИМОСТИ С ИНФЕРЕНСОМ
# =======================
# Получаем количество маркеров для выбранного типа датчика из SENSOR_GEOMETRY
MARKER_COUNT = SENSOR_GEOMETRY.get(SENSOR_TYPE, {}).get('M', 12)  # По умолчанию 12 маркеров для типа A
INPUT_FEATURES_COMPATIBLE = MARKER_COUNT * 2 + 1  # количество маркеров × 2 координаты + 1 L2 норма

# Архитектура энкодера (13 слоев)
ENCODER_ARCHITECTURE = [
    # Слой 0-2: input -> 512
    ('linear', INPUT_FEATURES_COMPATIBLE, 512),     # 0
    ('batchnorm', 512),                             # 1
    ('relu',),                                      # 2
    ('dropout', 0.2),                               # 3
    # Слой 4-6: 512 -> 256
    ('linear', 512, 256),                           # 4
    ('batchnorm', 256),                             # 5
    ('relu',),                                      # 6
    ('dropout', 0.2),                               # 7
    # Слой 8-10: 256 -> latent_dim (64)
    ('linear', 256, 64),                            # 8
    ('batchnorm', 64),                              # 9
    ('relu',),                                      # 10
    ('dropout', 0.2),                               # 11
    # Финальный слой энкодера (12)
    ('linear', 64, 64)                              # 12
]

# Архитектура декодера (13 слоев)
DECODER_ARCHITECTURE = [
    # Слой 0-2: latent_dim -> 256
    ('linear', 64, 256),                            # 0
    ('batchnorm', 256),                             # 1
    ('relu',),                                      # 2
    ('dropout', 0.2),                               # 3
    # Слой 4-6: 256 -> 512
    ('linear', 256, 512),                           # 4
    ('batchnorm', 512),                             # 5
    ('relu',),                                      # 6
    ('dropout', 0.2),                               # 7
    # Слой 8-10: 512 -> input_size
    ('linear', 512, INPUT_FEATURES_COMPATIBLE),    # 8
    ('batchnorm', INPUT_FEATURES_COMPATIBLE),      # 9
    # Финальные слои декодера (10-12)
    ('relu',),                                      # 10
    ('linear', INPUT_FEATURES_COMPATIBLE, INPUT_FEATURES_COMPATIBLE),  # 11
    ('linear', INPUT_FEATURES_COMPATIBLE, INPUT_FEATURES_COMPATIBLE)   # 12
]

# Архитектура регрессора силы (9 слоев)
REGRESSOR_ARCHITECTURE = [
    ('linear', 64, 256),        # 0
    ('batchnorm', 256),         # 1
    ('relu',),                  # 2
    ('dropout', 0.2),           # 3
    ('linear', 256, 512),       # 4
    ('batchnorm', 512),         # 5
    ('relu',),                  # 6
    ('dropout', 0.2),           # 7
    ('linear', 512, 1)          # 8
]

LATENT_DIM = 64
HIDDEN_LAYERS = [512, 256]
DROPOUT_RATE = 0.2
USE_BATCHNORM = True

print("=" * 80)
print(f"АРХИТЕКТУРА МОДЕЛИ ДЛЯ СОВМЕСТИМОСТИ С ИНФЕРЕНСОМ ДЛЯ ДАТЧИКА ТИПА {SENSOR_TYPE}")
print("-" * 80)
print(f"Количество маркеров для датчика типа {SENSOR_TYPE}: {MARKER_COUNT}")
print(f"Входные признаки (совместимые): {INPUT_FEATURES_COMPATIBLE} ({MARKER_COUNT} маркеров × 2 координаты + 1 L2 норма)")
print(f"Геометрические параметры датчика: {SENSOR_GEOMETRY.get(SENSOR_TYPE, {})}")
print(f"Латентное пространство: {LATENT_DIM} измерений")
print("-" * 80)
print("Энкодер (13 слоев):")
for i, layer in enumerate(ENCODER_ARCHITECTURE):
    if layer[0] == 'linear':
        print(f"  [{i}] Linear: {layer[1]} -> {layer[2]}")
    elif layer[0] == 'batchnorm':
        print(f"  [{i}] BatchNorm1d: {layer[1]}")
    elif layer[0] == 'relu':
        print(f"  [{i}] ReLU")
    elif layer[0] == 'dropout':
        print(f"  [{i}] Dropout: {layer[1]}")

print("\nДекодер (13 слоев):")
for i, layer in enumerate(DECODER_ARCHITECTURE):
    if layer[0] == 'linear':
        print(f"  [{i}] Linear: {layer[1]} -> {layer[2]}")
    elif layer[0] == 'batchnorm':
        print(f"  [{i}] BatchNorm1d: {layer[1]}")
    elif layer[0] == 'relu':
        print(f"  [{i}] ReLU")
    elif layer[0] == 'dropout':
        print(f"  [{i}] Dropout: {layer[1]}")

print("\nРегрессор силы (9 слоев):")
for i, layer in enumerate(REGRESSOR_ARCHITECTURE):
    if layer[0] == 'linear':
        print(f"  [{i}] Linear: {layer[1]} -> {layer[2]}")
    elif layer[0] == 'batchnorm':
        print(f"  [{i}] BatchNorm1d: {layer[1]}")
    elif layer[0] == 'relu':
        print(f"  [{i}] ReLU")
    elif layer[0] == 'dropout':
        print(f"  [{i}] Dropout: {layer[1]}")
print("=" * 80)

# =======================
# Helpers
# =======================
def to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

# =======================
# Data Loading and Processing
# =======================
def load_deformation_vectors_dataset():
    """
    Загружает агрегированный датасет с векторами деформаций и суммами L2 норм
    
    Возвращает:
      X_combined: ndarray (N, marker_count*2 + 1) - векторы деформаций + сумма L2 норм
      y_combined: ndarray (N,)
      marker_count: int
      dataset_info: информация о датасетах
      groups: list of dataset indices per sample (len == N)
      reference_P: эталонное состояние
      total_l2_norms: массив сумм L2 норм
    """
    if not AGGREGATED_DIR.exists() or not AGGREGATED_DIR.is_dir():
        raise FileNotFoundError(f"Aggregated directory not found: {AGGREGATED_DIR}")
    
    # Загружаем векторы деформаций, силы и метаданные
    deformations_path = AGGREGATED_DIR / "deformation_vectors.npy"
    forces_path = AGGREGATED_DIR / "forces.npy"
    total_l2_path = AGGREGATED_DIR / "total_l2_norms.npy"
    meta_path = AGGREGATED_DIR / "metadata.json"
    
    if not deformations_path.exists():
        raise FileNotFoundError(f"Deformation vectors file not found: {deformations_path}")
    if not forces_path.exists():
        raise FileNotFoundError(f"Forces file not found: {forces_path}")
    if not total_l2_path.exists():
        raise FileNotFoundError(f"Total L2 norms file not found: {total_l2_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    # Загружаем эталонное состояние
    reference_path = AGGREGATED_DIR / "reference_P_sorted.npy"
    if not reference_path.exists():
        # Пытаемся найти альтернативное имя файла
        reference_path_alt = AGGREGATED_DIR / "reference_P.npy"
        if reference_path_alt.exists():
            reference_path = reference_path_alt
            print(f"Warning: Using alternative reference P-state file: {reference_path_alt}")
        else:
            raise FileNotFoundError(f"Reference P-state file not found: {reference_path}")
    
    print("Loading deformation vectors dataset...")
    deformations = np.load(deformations_path)  # форма: (N, marker_count, 2)
    forces = np.load(forces_path)              # форма: (N,)
    total_l2_norms = np.load(total_l2_path)    # форма: (N,)
    reference_P = np.load(reference_path)      # форма: (marker_count, 2)
    
    # Загружаем метаданные
    with open(meta_path, 'r') as f:
        meta_info = json.load(f)
    
    print(f"Loaded deformation vectors shape: {deformations.shape}")
    print(f"Loaded forces shape: {forces.shape}")
    print(f"Loaded total L2 norms shape: {total_l2_norms.shape}")
    print(f"Reference P-state shape: {reference_P.shape}")
    print(f"Metadata: {meta_info}")
    
    # Проверяем соответствие размеров
    n_samples = deformations.shape[0]
    n_markers = deformations.shape[1]
    coords_dim = deformations.shape[2]
    
    if coords_dim != 2:
        raise ValueError(f"Unexpected coordinates dimension: {coords_dim}, expected 2")
    
    if n_samples != forces.shape[0]:
        raise ValueError(f"Mismatch in number of samples: deformations ({n_samples}) vs forces ({forces.shape[0]})")
    
    if n_samples != total_l2_norms.shape[0]:
        raise ValueError(f"Mismatch in number of samples: deformations ({n_samples}) vs total_l2_norms ({total_l2_norms.shape[0]})")
    
    if n_markers != reference_P.shape[0]:
        raise ValueError(f"Mismatch in number of markers: deformations ({n_markers}) vs reference_P ({reference_P.shape[0]})")
    
    # Подготавливаем данные
    X_combined = []
    y_combined = []
    groups = []
    
    # Загружаем информацию о соответствиях для группировки по датасетам
    correspondences_path = AGGREGATED_DIR / "marker_correspondences.json"
    if not correspondences_path.exists():
        raise FileNotFoundError(f"Marker correspondences file not found: {correspondences_path}")
    
    with open(correspondences_path, 'r') as f:
        marker_correspondences = json.load(f)
    
    # Создаем словарь для группировки по датасетам
    dataset_names = meta_info['datasets']
    dataset_name_to_idx = {name: idx for idx, name in enumerate(dataset_names)}
    
    # Подготавливаем данные для обучения
    for i in range(n_samples):
        # Проверяем наличие NaN или Inf в данных
        if not np.all(np.isfinite(deformations[i])) or np.isnan(forces[i]) or np.isnan(total_l2_norms[i]):
            continue
        
        # Формируем вектор признаков: векторы деформаций + сумма L2 норм
        deformation_flat = deformations[i].flatten()  # форма: (marker_count*2,)
        features = np.concatenate([deformation_flat, [total_l2_norms[i]]])  # добавляем сумму L2 норм как последний признак
        
        X_combined.append(features)
        y_combined.append(forces[i])
        
        # Определяем группу (датасет) для текущего замера
        if i < len(marker_correspondences):
            dataset_name = marker_correspondences[i]['dataset']
            dataset_idx = dataset_name_to_idx.get(dataset_name, 0)  # 0 по умолчанию, если не найден
            groups.append(dataset_idx)
        else:
            # Если информации о соответствиях недостаточно, используем равномерное распределение
            groups.append(i % len(dataset_names))
    
    if not X_combined:
        raise ValueError("No valid samples found after filtering NaN/Inf values")
    
    X_combined = np.array(X_combined)
    y_combined = np.array(y_combined)
    groups = np.array(groups)
    
    print(f"\nTotal valid samples after filtering: {X_combined.shape[0]}")
    print(f"Features per sample: {X_combined.shape[1]} (deformation vectors + total L2 norm)")
    print(f"Overall force range: {y_combined.min():.3f} .. {y_combined.max():.3f}")
    print(f"Number of unique groups (datasets): {len(np.unique(groups))}")
    
    # Статистика по признакам
    print("\nFeatures statistics:")
    print(f"  Min deformation: {X_combined[:, :-1].min():.6f}")
    print(f"  Max deformation: {X_combined[:, :-1].max():.6f}")
    print(f"  Mean deformation: {X_combined[:, :-1].mean():.6f}")
    print(f"  Std deviation of deformation: {X_combined[:, :-1].std():.6f}")
    print(f"  Min total L2 norm: {X_combined[:, -1].min():.6f}")
    print(f"  Max total L2 norm: {X_combined[:, -1].max():.6f}")
    print(f"  Mean total L2 norm: {X_combined[:, -1].mean():.6f}")
    print(f"  Std deviation of total L2 norm: {X_combined[:, -1].std():.6f}")
    
    # Добавляем информацию о датасетах
    dataset_info = []
    for i, dataset_name in enumerate(dataset_names):
        dataset_info.append({
            'dataset_name': dataset_name,
            'dataset_idx': i,
            'n_markers': n_markers,
            'num_samples': np.sum(groups == i),
            'force_range': [float(y_combined[groups == i].min()), float(y_combined[groups == i].max())] if np.any(groups == i) else [0, 0]
        })
    
    return X_combined, y_combined, n_markers, dataset_info, groups, reference_P, total_l2_norms

# =======================
# Dataset class
# =======================
class DeformationForceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        if torch.isnan(self.X).any() or torch.isinf(self.X).any():
            print("Warning: NaN/Inf in X")
        if torch.isnan(self.y).any() or torch.isinf(self.y).any():
            print("Warning: NaN/Inf in y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =======================
# СОВМЕСТИМАЯ АРХИТЕКТУРА АВТОЭНКОДЕРА - ТОЧНОЕ СООТВЕТСТВИЕ ИНФЕРЕНСУ
# =======================
class CompatibleAutoencoderForceNet(nn.Module):
    def __init__(self, input_size, hidden_layers=[512, 256], latent_dim=64,
                 dropout_rate=0.2, use_batchnorm=True):
        super(CompatibleAutoencoderForceNet, self).__init__()
        # === ЭНКОДЕР (точная структура как в инференсе) ===
        encoder_modules = []
        # Слой 0-2: input -> 512
        encoder_modules.append(nn.Linear(input_size, hidden_layers[0]))  # 0
        if use_batchnorm:
            encoder_modules.append(nn.BatchNorm1d(hidden_layers[0]))    # 1
        encoder_modules.append(nn.ReLU())                              # 2
        if dropout_rate > 0:
            encoder_modules.append(nn.Dropout(dropout_rate))           # 3
        # Слой 4-6: 512 -> 256
        encoder_modules.append(nn.Linear(hidden_layers[0], hidden_layers[1]))  # 4
        if use_batchnorm:
            encoder_modules.append(nn.BatchNorm1d(hidden_layers[1]))          # 5
        encoder_modules.append(nn.ReLU())                                    # 6
        if dropout_rate > 0:
            encoder_modules.append(nn.Dropout(dropout_rate))                 # 7
        # Слой 8-10: 256 -> latent_dim (64)
        encoder_modules.append(nn.Linear(hidden_layers[1], latent_dim))  # 8
        if use_batchnorm:
            encoder_modules.append(nn.BatchNorm1d(latent_dim))           # 9
        encoder_modules.append(nn.ReLU())                               # 10
        if dropout_rate > 0:
            encoder_modules.append(nn.Dropout(dropout_rate))            # 11
        # Финальный слой энкодера (12)
        encoder_modules.append(nn.Linear(latent_dim, latent_dim))       # 12
        self.encoder = nn.Sequential(*encoder_modules)
        
        # === ДЕКОДЕР (точная структура как в инференсе) ===
        decoder_modules = []
        # Слой 0-2: latent_dim -> 256
        decoder_modules.append(nn.Linear(latent_dim, hidden_layers[1]))  # 0
        if use_batchnorm:
            decoder_modules.append(nn.BatchNorm1d(hidden_layers[1]))    # 1
        decoder_modules.append(nn.ReLU())                              # 2
        if dropout_rate > 0:
            decoder_modules.append(nn.Dropout(dropout_rate))           # 3
        # Слой 4-6: 256 -> 512
        decoder_modules.append(nn.Linear(hidden_layers[1], hidden_layers[0]))  # 4
        if use_batchnorm:
            decoder_modules.append(nn.BatchNorm1d(hidden_layers[0]))          # 5
        decoder_modules.append(nn.ReLU())                                    # 6
        if dropout_rate > 0:
            decoder_modules.append(nn.Dropout(dropout_rate))                 # 7
        # Слой 8-10: 512 -> input_size
        decoder_modules.append(nn.Linear(hidden_layers[0], input_size))  # 8
        if use_batchnorm:
            decoder_modules.append(nn.BatchNorm1d(input_size))           # 9
        # Финальные слои декодера (10-12)
        decoder_modules.append(nn.ReLU())                               # 10
        decoder_modules.append(nn.Linear(input_size, input_size))       # 11
        decoder_modules.append(nn.Linear(input_size, input_size))       # 12
        self.decoder = nn.Sequential(*decoder_modules)
        
        # === РЕГРЕССОР СИЛЫ (встроенный в архитектуру) ===
        # Используем часть энкодера для получения латентного представления
        self.force_regressor = nn.Sequential(
            nn.Linear(latent_dim, hidden_layers[1]),  # 0
            nn.BatchNorm1d(hidden_layers[1]),        # 1
            nn.ReLU(),                               # 2
            nn.Dropout(dropout_rate),                # 3
            nn.Linear(hidden_layers[1], hidden_layers[0]),  # 4
            nn.BatchNorm1d(hidden_layers[0]),              # 5
            nn.ReLU(),                                     # 6
            nn.Dropout(dropout_rate),                      # 7
            nn.Linear(hidden_layers[0], 1)                 # 8
        )
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов по методу Kaiming для ReLU активаций"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the compatible autoencoder
        Args:
            x: input tensor with shape (batch_size, input_size)
        Returns:
            reconstructed: reconstructed input, shape (batch_size, input_size)
            force_pred: predicted force values, shape (batch_size, 1)
            latent: latent space representation, shape (batch_size, latent_dim)
        """
        # Encoder path
        latent = self.encoder(x)
        
        # Decoder path (reconstruction)
        reconstructed = self.decoder(latent)
        
        # Regressor path (force prediction) - используем внутреннее состояние энкодера
        # Для получения латентного представления берем выход из слоя 10 энкодера
        intermediate_latent = self.encoder[:11](x)  # Берем все слои до финального линейного
        force_pred = self.force_regressor(intermediate_latent)
        
        return reconstructed, force_pred, latent
    
    def predict_force(self, x):
        """Предсказать силу для заданного входа"""
        with torch.no_grad():
            intermediate_latent = self.encoder[:11](x)
            force_pred = self.force_regressor(intermediate_latent)
            return force_pred

# =======================
# Combined Loss Function
# =======================
class CombinedAutoencoderLoss(nn.Module):
    def __init__(self, reconstruction_weight=0.7, force_weight=0.3):
        super(CombinedAutoencoderLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.force_weight = force_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, reconstructed, x_true, force_pred, force_true):
        """
        Комбинированная функция потерь для автоэнкодера
        Args:
            reconstructed: реконструированные входные данные
            x_true: оригинальные входные данные
            force_pred: предсказанные значения силы
            force_true: реальные значения силы
        Returns:
            total_loss: суммарная потеря
            recon_loss: потеря реконструкции
            force_loss: потеря предсказания силы
        """
        # Reconstruction loss (MSE between original and reconstructed inputs)
        recon_loss = self.mse_loss(reconstructed, x_true)
        
        # Force prediction loss (MSE between predicted and actual forces)
        force_loss = self.mse_loss(force_pred, force_true)
        
        # Combined loss with weights
        total_loss = (self.reconstruction_weight * recon_loss +
                      self.force_weight * force_loss)
        
        return total_loss, recon_loss, force_loss

# =======================
# Group-stratified split
# =======================
def group_stratified_split(X, y, groups, test_size=0.2, n_bins=10, random_state=42):
    """
    Выполняет разбиение по группам с сохранением распределения целевой переменной:
    1) для каждой группы вычисляем среднее y_group
    2) бинируем группы по квантилям (n_bins)
    3) используем train_test_split на уровне групп с stratify по bins
    Возвращает: X_train, X_val, y_train, y_val, train_idx, val_idx
    """
    df = pd.DataFrame({'y': y, 'group': groups})
    group_stats = df.groupby('group')['y'].mean().reset_index()
    # make quantile bins
    try:
        group_stats['bin'] = pd.qcut(group_stats['y'], q=min(n_bins, len(group_stats)), labels=False, duplicates='drop')
    except Exception:
        group_stats['bin'] = pd.cut(group_stats['y'], bins=min(n_bins, len(group_stats)), labels=False, duplicates='drop')

    groups_unique = group_stats['group'].values
    bins = group_stats['bin'].values

    # split groups
    g_train, g_val = train_test_split(
        groups_unique,
        test_size=test_size,
        random_state=random_state,
        stratify=bins if len(np.unique(bins)) > 1 else None
    )

    # map to indices
    train_mask = np.isin(groups, g_train)
    val_mask = np.isin(groups, g_val)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    # ensure non-empty
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Group stratified split produced empty train or val. Try changing n_bins or test_size.")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    return X_train, X_val, y_train, y_val, train_idx, val_idx

# =======================
# Training / Evaluation
# =======================
def train_autoencoder_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                           epochs, device, reconstruction_weight=0.7, force_weight=0.3):
    """
    Обучение гибридной модели автоэнкодера с регрессией силы
    Args:
        model: экземпляр CompatibleAutoencoderForceNet
        train_loader: DataLoader для обучающих данных
        val_loader: DataLoader для валидационных данных
        criterion: функция потерь
        optimizer: оптимизатор
        scheduler: планировщик скорости обучения
        epochs: количество эпох
        device: устройство для вычислений
        reconstruction_weight: вес потери реконструкции
        force_weight: вес потери предсказания силы
    Returns:
        best_model: модель с лучшими весами
        history: история обучения
    """
    best_val_loss = float('inf')
    best_state = None
    patience = 0
    train_losses = []
    val_losses = []
    recon_losses_train = []
    force_losses_train = []
    recon_losses_val = []
    force_losses_val = []
    
    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_force = 0.0
        n_batches = 0
        
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            
            # Проверка на NaN/Inf
            if torch.isnan(Xb).any() or torch.isinf(Xb).any() or torch.isnan(yb).any() or torch.isinf(yb).any():
                continue
            
            optimizer.zero_grad()
            
            # Forward pass through autoencoder
            reconstructed, force_pred, _ = model(Xb)
            
            # Calculate combined loss
            total_loss, recon_loss, force_loss = criterion(reconstructed, Xb, force_pred, yb)
            
            # Проверка на NaN/Inf в потерях
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            running_total += total_loss.item()
            running_recon += recon_loss.item()
            running_force += force_loss.item()
            n_batches += 1
        
        if n_batches == 0:
            avg_total_train = float('inf')
            avg_recon_train = float('inf')
            avg_force_train = float('inf')
        else:
            avg_total_train = running_total / n_batches
            avg_recon_train = running_recon / n_batches
            avg_force_train = running_force / n_batches
        
        # Validation phase
        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_force = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                if torch.isnan(Xb).any() or torch.isinf(Xb).any() or torch.isnan(yb).any() or torch.isinf(yb).any():
                    continue
                
                reconstructed, force_pred, _ = model(Xb)
                total_loss, recon_loss, force_loss = criterion(reconstructed, Xb, force_pred, yb)
                
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue
                
                val_total += total_loss.item()
                val_recon += recon_loss.item()
                val_force += force_loss.item()
                n_val_batches += 1
        
        if n_val_batches == 0:
            avg_total_val = float('inf')
            avg_recon_val = float('inf')
            avg_force_val = float('inf')
        else:
            avg_total_val = val_total / n_val_batches
            avg_recon_val = val_recon / n_val_batches
            avg_force_val = val_force / n_val_batches
        
        # Scheduler step (based on total validation loss)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_total_val)
            else:
                scheduler.step()
        
        # Record losses
        train_losses.append(avg_total_train)
        val_losses.append(avg_total_val)
        recon_losses_train.append(avg_recon_train)
        force_losses_train.append(avg_force_train)
        recon_losses_val.append(avg_recon_val)
        force_losses_val.append(avg_force_val)
        
        # Early stopping logic (based on total validation loss)
        if avg_total_val < best_val_loss - MIN_DELTA:
            best_val_loss = avg_total_val
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
        
        # Print epoch statistics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Total Loss: {avg_total_train:.6f}/{avg_total_val:.6f} | "
              f"Recon: {avg_recon_train:.6f}/{avg_recon_val:.6f} | "
              f"Force: {avg_force_train:.6f}/{avg_force_val:.6f} | "
              f"LR: {current_lr:.6f}")
        
        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}, best total val loss: {best_val_loss:.6f}")
            break
    
    # Load best model state if found
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("Warning: no best_state found, using final weights")
    
    # Prepare history dictionary
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'recon_losses_train': recon_losses_train,
        'force_losses_train': force_losses_train,
        'recon_losses_val': recon_losses_val,
        'force_losses_val': force_losses_val,
        'best_val_loss': best_val_loss,
        'epochs_run': epoch + 1,
        'stopped_early': patience >= EARLY_STOPPING_PATIENCE
    }
    
    return model, history

def evaluate_autoencoder_model(model, data_loader, device):
    """
    Оценка гибридной модели автоэнкодера
    Returns:
        dict с метриками для реконструкции и предсказания силы
    """
    model.eval()
    recon_losses = []
    force_preds = []
    force_trues = []
    latent_representations = []
    
    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            if torch.isnan(Xb).any() or torch.isinf(Xb).any() or torch.isnan(yb).any() or torch.isinf(yb).any():
                continue
            
            reconstructed, force_pred, latent = model(Xb)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(reconstructed, Xb)
            recon_losses.append(recon_loss.item())
            
            # Force prediction
            force_preds.extend(force_pred.cpu().numpy().flatten().tolist())
            force_trues.extend(yb.cpu().numpy().flatten().tolist())
            
            # Store latent representations for analysis
            latent_representations.append(latent.cpu().numpy())
    
    if len(force_preds) == 0:
        return {
            'reconstruction_mse': float('nan'),
            'force_mse': float('nan'), 'force_rmse': float('nan'),
            'force_mae': float('nan'), 'force_r2': float('nan'),
            'predictions': np.array([]), 'actuals': np.array([]),
            'latent_representations': np.array([])
        }
    
    # Calculate reconstruction metrics
    avg_recon_mse = np.mean(recon_losses)
    
    # Calculate force prediction metrics
    force_preds = np.array(force_preds)
    force_trues = np.array(force_trues)
    force_mse = np.mean((force_preds - force_trues) ** 2)
    force_rmse = np.sqrt(force_mse)
    force_mae = np.mean(np.abs(force_preds - force_trues))
    ss_total = np.sum((force_trues - np.mean(force_trues)) ** 2)
    ss_res = np.sum((force_trues - force_preds) ** 2)
    force_r2 = 1 - (ss_res / ss_total) if ss_total > 0 else float('nan')
    
    # Concatenate latent representations
    latent_representations = np.vstack(latent_representations) if latent_representations else np.array([])
    
    return {
        'reconstruction_mse': avg_recon_mse,
        'force_mse': force_mse,
        'force_rmse': force_rmse,
        'force_mae': force_mae,
        'force_r2': force_r2,
        'predictions': force_preds,
        'actuals': force_trues,
        'latent_representations': latent_representations
    }

# =======================
# Plotting helpers
# =======================
def plot_training_history_autoencoder(history, save_path=None):
    """Визуализация истории обучения с разбивкой по типам потерь"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Total loss
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Total Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val Total Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Reconstruction loss
    ax2.plot(epochs, history['recon_losses_train'], 'b-', label='Train Recon Loss', linewidth=2)
    ax2.plot(epochs, history['recon_losses_val'], 'r-', label='Val Recon Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reconstruction MSE')
    ax2.set_title('Reconstruction Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Force prediction loss
    ax3.plot(epochs, history['force_losses_train'], 'b-', label='Train Force Loss', linewidth=2)
    ax3.plot(epochs, history['force_losses_val'], 'r-', label='Val Force Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Force MSE')
    ax3.set_title('Force Prediction Loss')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions_vs_actuals(actuals, predictions, groups=None, sensor_type='A', save_path=None):
    if len(actuals) == 0:
        return
    
    plt.figure(figsize=(10, 8))
    
    if groups is not None and len(np.unique(groups)) <= 10:
        # Цветовая кодировка по группам (датасетам)
        unique_groups = np.unique(groups)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        for i, group in enumerate(unique_groups):
            mask = groups == group
            plt.scatter(actuals[mask], predictions[mask], 
                       c=[colors[i]], alpha=0.6, edgecolors='white', s=40,
                       label=f'Dataset {group}')
    else:
        # Без цветовой кодировки
        plt.scatter(actuals, predictions, alpha=0.6, edgecolors='white', s=40)
    
    mn = min(actuals.min(), predictions.min())
    mx = max(actuals.max(), predictions.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Predicted Force (N)')
    plt.title(f'Predicted vs Actual (Compatible Autoencoder) - Sensor Type {sensor_type}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.xlim(mn - 0.1, mx + 0.1)
    plt.ylim(mn - 0.1, mx + 0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(actuals, predictions, groups=None, sensor_type='A', save_path=None):
    if len(actuals) == 0:
        return
    residuals = predictions - actuals
    
    plt.figure(figsize=(15, 5))
    
    # Гистограмма остатков
    plt.subplot(1, 3, 1)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Residuals histogram')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Остатки vs Фактические значения
    plt.subplot(1, 3, 2)
    if groups is not None and len(np.unique(groups)) <= 10:
        unique_groups = np.unique(groups)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        for i, group in enumerate(unique_groups):
            mask = groups == group
            plt.scatter(actuals[mask], residuals[mask], 
                       c=[colors[i]], alpha=0.6, s=30, edgecolors='white',
                       label=f'Dataset {group}')
    else:
        plt.scatter(actuals, residuals, alpha=0.6, s=30, edgecolors='white')
    
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Residual')
    plt.title(f'Residuals vs Actual - Sensor Type {sensor_type}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=8)
    
    # Boxplot остатков по группам
    if groups is not None and len(np.unique(groups)) > 1:
        plt.subplot(1, 3, 3)
        group_residuals = [residuals[groups == g] for g in np.unique(groups)]
        plt.boxplot(group_residuals, labels=[f'DS {g}' for g in np.unique(groups)])
        plt.title('Residuals by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Residual')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =======================
# Main function with Compatible Autoencoder
# =======================
def main():
    print("=" * 80)
    print(f"COMPATIBLE AUTOENCODER NEURAL NETWORK TRAINING FOR SENSOR TYPE {SENSOR_TYPE}")
    print("=" * 80)
    print(f"Model will be saved to: {RUN_DIR}")
    print(f"Architecture: Input={INPUT_FEATURES_COMPATIBLE}, Hidden=[512, 256], Latent=64")
    print(f"Dropout rate: {DROPOUT_RATE}, BatchNorm: {USE_BATCHNORM}")
    print(f"Loss weights: Reconstruction={RECONSTRUCTION_WEIGHT}, Force={FORCE_WEIGHT}")
    
    # 1) Load deformation vectors dataset
    try:
        X, y, marker_count, dataset_info, groups, reference_P, total_l2_norms = load_deformation_vectors_dataset()
    except Exception as e:
        print("Error loading deformation vectors dataset:", e)
        sys.exit(1)
    
    print(f"\nData shape: X {X.shape}, y {y.shape}, markers per sample: {marker_count}")
    
    # Проверяем, что количество признаков совпадает с ожидаемым
    expected_features = marker_count * 2 + 1  # 2 координаты на маркер + 1 L2 норма
    print(f"Expected features based on data: {expected_features}")
    print(f"Configured input features: {INPUT_FEATURES_COMPATIBLE}")
    
    # Если количество признаков не совпадает, корректируем
    if expected_features != INPUT_FEATURES_COMPATIBLE:
        print(f"WARNING: Feature count mismatch! Data has {expected_features} features, but model expects {INPUT_FEATURES_COMPATIBLE}")
        if expected_features < INPUT_FEATURES_COMPATIBLE:
            print(f"  Adding {INPUT_FEATURES_COMPATIBLE - expected_features} zero features for compatibility")
            padding = np.zeros((X.shape[0], INPUT_FEATURES_COMPATIBLE - expected_features))
            X = np.hstack([X, padding])
        elif expected_features > INPUT_FEATURES_COMPATIBLE:
            print(f"  Truncating data to {INPUT_FEATURES_COMPATIBLE} features for compatibility")
            X = X[:, :INPUT_FEATURES_COMPATIBLE]
        print(f"  New data shape: X {X.shape}")
    
    input_size = INPUT_FEATURES_COMPATIBLE
    
    # Save dataset_info
    dataset_info_path = RUN_DIR / "dataset_info.json"
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2, default=to_serializable)
    print(f"Saved dataset_info to {dataset_info_path}")
    
    # Save reference_P information
    reference_info = {
        'shape': reference_P.shape,
        'mean_position': np.mean(reference_P, axis=0).tolist(),
        'std_position': np.std(reference_P, axis=0).tolist(),
        'min_coords': np.min(reference_P, axis=0).tolist(),
        'max_coords': np.max(reference_P, axis=0).tolist(),
        'sorting_order': 'left-to-right and top-to-bottom',
        'actual_marker_count': marker_count,
        'configured_marker_count': MARKER_COUNT,
        'sensor_geometry': SENSOR_GEOMETRY.get(SENSOR_TYPE, {})
    }
    with open(RUN_DIR / "reference_P_info.json", 'w') as f:
        json.dump(to_serializable(reference_info), f, indent=2)
    print(f"Saved reference P-state info to {RUN_DIR / 'reference_P_info.json'}")
    
    # Explicitly save reference_P.npy in the model directory for inference
    reference_P_path = RUN_DIR / "reference_P_sorted.npy"
    np.save(reference_P_path, reference_P)
    print(f"Saved reference P-state numpy array to {reference_P_path}")
    
    # 2) Check duplicates
    df_check = pd.DataFrame(X)
    df_check['y'] = y
    dup_count = df_check.duplicated(keep=False).sum()
    print(f"Exact duplicate rows in combined data: {dup_count}")
    if dup_count > 0:
        print("  Note: duplicates detected. Consider deduplication if these arise from repeated captures.")
    
    # 3) Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save deformation statistics
    deformation_stats = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'min': np.min(X, axis=0).tolist(),
        'max': np.max(X, axis=0).tolist(),
        'marker_count': marker_count,
        'num_features': X.shape[1],
        'input_size': input_size,
        'sensor_geometry': SENSOR_GEOMETRY.get(SENSOR_TYPE, {})
    }
    stats_path = RUN_DIR / "deformation_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(to_serializable(deformation_stats), f, indent=2)
    print(f"Saved deformation statistics to {stats_path}")
    
    scaler_path = RUN_DIR / "deformation_scaler.json"
    with open(scaler_path, 'w') as f:
        json.dump({
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'marker_count': marker_count,
            'num_features': X.shape[1],
            'input_size': input_size,
            'compatible_with_inference': True,
            'actual_marker_count': marker_count,
            'configured_marker_count': MARKER_COUNT,
            'sensor_geometry': SENSOR_GEOMETRY.get(SENSOR_TYPE, {})
        }, f, indent=2)
    print(f"Saved deformation scaler to {scaler_path}")
    
    # 4) Group-stratified split
    try:
        X_train, X_val, y_train, y_val, train_idx, val_idx = group_stratified_split(X_scaled, y, groups, test_size=TEST_SIZE, n_bins=10, random_state=RANDOM_SEED)
        groups_val = groups[val_idx]  # группы для валидационной выборки
    except Exception as e:
        print("Group stratified split failed:", e)
        print("Falling back to simple shuffled split.")
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True)
        groups_val = None
    
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    
    # 5) Create datasets/dataloaders
    train_ds = DeformationForceDataset(X_train, y_train)
    val_ds = DeformationForceDataset(X_val, y_val)
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("Error: empty dataset after splitting.")
        sys.exit(1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 6) Build COMPATIBLE autoencoder model
    model = CompatibleAutoencoderForceNet(
        input_size=input_size,
        hidden_layers=HIDDEN_LAYERS,
        latent_dim=LATENT_DIM,
        dropout_rate=DROPOUT_RATE,
        use_batchnorm=USE_BATCHNORM
    ).to(DEVICE)
    print("COMPATIBLE Autoencoder Model architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for name, p in model.named_parameters() if 'encoder' in name)
    decoder_params = sum(p.numel() for name, p in model.named_parameters() if 'decoder' in name)
    regressor_params = sum(p.numel() for name, p in model.named_parameters() if 'force_regressor' in name)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Regressor parameters: {regressor_params:,}")
    
    # 7) Training setup
    criterion = CombinedAutoencoderLoss(
        reconstruction_weight=RECONSTRUCTION_WEIGHT,
        force_weight=FORCE_WEIGHT
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 8) Train autoencoder model
    start = time.time()
    model, history = train_autoencoder_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        EPOCHS, DEVICE, RECONSTRUCTION_WEIGHT, FORCE_WEIGHT
    )
    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed:.2f} seconds. Best val loss: {history['best_val_loss']:.6f}")
    
    # 9) Evaluate on validation
    eval_res = evaluate_autoencoder_model(model, val_loader, DEVICE)
    print("\nValidation metrics:")
    print(f" Reconstruction MSE: {eval_res['reconstruction_mse']:.6f}")
    print(f" Force MSE:  {eval_res['force_mse']:.6f}")
    print(f" Force RMSE: {eval_res['force_rmse']:.6f}")
    print(f" Force MAE:  {eval_res['force_mae']:.6f}")
    print(f" Force R2:   {eval_res['force_r2']:.4f}")
    
    # 10) Baseline (mean predictor)
    y_pred_baseline = np.mean(y_train)
    mse_baseline = np.mean((y_val - y_pred_baseline) ** 2)
    print("\nBaseline (mean predictor):")
    print(f" Baseline MSE (mean predictor): {mse_baseline:.6f}")
    print(f" Improvement over baseline: {(mse_baseline - eval_res['force_mse']) / mse_baseline * 100:.2f}%")
    
    # 11) Save model and history - С КОРРЕКТНЫМИ ПАРАМЕТРАМИ ДЛЯ ИНФЕРЕНСА
    model_path = RUN_DIR / "compatible_autoencoder_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_size': input_size,
        'hidden_layers': HIDDEN_LAYERS,
        'latent_dim': LATENT_DIM,
        'dropout_rate': DROPOUT_RATE,
        'use_batchnorm': USE_BATCHNORM,
        'marker_count': marker_count,
        'best_val_loss': history['best_val_loss'],
        'reconstruction_weight': RECONSTRUCTION_WEIGHT,
        'force_weight': FORCE_WEIGHT,
        'reference_P': reference_P,
        'training_config': {
            'BATCH_SIZE': BATCH_SIZE,
            'EPOCHS': EPOCHS,
            'LEARNING_RATE': LEARNING_RATE,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
            'MIN_DELTA': MIN_DELTA,
            'FEATURES_INCLUDED': ['deformation_vectors', 'total_l2_norm'],
            'ARCHITECTURE': 'compatible_autoencoder',
            'MODEL_CLASS': 'CompatibleAutoencoderForceNet',
            'COMPATIBLE_WITH_INFERENCE': True,
            'SENSOR_TYPE': SENSOR_TYPE,
            'ACTUAL_MARKER_COUNT': marker_count,
            'CONFIGURED_MARKER_COUNT': MARKER_COUNT,
            'SENSOR_GEOMETRY': SENSOR_GEOMETRY.get(SENSOR_TYPE, {}),
        },
        # Дополнительные параметры для максимальной совместимости
        'model_architecture': {
            'encoder_layers_count': 13,  # до encoder.12
            'decoder_layers_count': 13,  # до decoder.12
            'has_force_regressor': True,
            'regressor_layers_count': 9  # до regressor.8
        }
    }, model_path)
    print(f"Saved COMPATIBLE autoencoder model to {model_path}")
    
    # Save training history
    history_path = RUN_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(to_serializable(history), f, indent=2)
    print(f"Saved training history to {history_path}")
    
    # 12) Visualizations
    print("Generating plots...")
    # Training history plot
    plot_training_history_autoencoder(history, save_path=RUN_DIR / "training_history.png")
    print("  training_history.png saved")
    
    # Predictions vs Actuals
    plot_predictions_vs_actuals(
        eval_res['actuals'], 
        eval_res['predictions'], 
        groups=groups_val, 
        sensor_type=SENSOR_TYPE,
        save_path=RUN_DIR / "predictions_vs_actuals.png"
    )
    print("  predictions_vs_actuals.png saved")
    
    # Residuals plot
    plot_residuals(
        eval_res['actuals'], 
        eval_res['predictions'], 
        groups=groups_val, 
        sensor_type=SENSOR_TYPE,
        save_path=RUN_DIR / "residuals.png"
    )
    print("  residuals.png saved")
    
    # 13) Correlation analysis between total L2 norm and force
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(total_l2_norms, y, alpha=0.6, edgecolors='white', s=40)
        plt.xlabel('Total L2 Norm of Deformation Vectors')
        plt.ylabel('Force (N)')
        plt.title(f'Correlation between Total L2 Norm and Force - Sensor Type {SENSOR_TYPE}')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Добавляем линейную регрессию
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(total_l2_norms, y)
        x_vals = np.array([total_l2_norms.min(), total_l2_norms.max()])
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, 'r-', linewidth=2, label=f'R={r_value:.3f}')
        plt.legend()
        
        correlation_path = RUN_DIR / "l2norm_force_correlation.png"
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  L2 norm vs force correlation plot saved to {correlation_path}")
        
        print(f"\nCorrelation analysis:")
        print(f"  Pearson correlation coefficient (R) between L2 norm and force: {r_value:.4f}")
        print(f"  P-value: {p_value:.6f}")
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
    
    # 14) Final summary
    print("\n" + "=" * 80)
    print(f"COMPATIBLE AUTOENCODER TRAINING COMPLETED FOR SENSOR TYPE {SENSOR_TYPE}")
    print("=" * 80)
    sensor_geometry = SENSOR_GEOMETRY.get(SENSOR_TYPE, {})
    print(f"Sensor geometry: D={sensor_geometry.get('D', 'N/A')} mm, M={sensor_geometry.get('M', 'N/A')} MPa")
    print(f"Sample count: {X.shape[0]}, markers per sample: {marker_count}")
    print(f"Input features per sample: {input_size} ({marker_count} markers × 2 coordinates + 1 total L2 norm)")
    print(f"Model architecture: Input={INPUT_FEATURES_COMPATIBLE}, Hidden=[512, 256], Latent=64")
    print(f"Best validation loss (total): {history['best_val_loss']:.6f}")
    print(f"Force prediction metrics - MSE: {eval_res['force_mse']:.6f}, RMSE: {eval_res['force_rmse']:.6f}, R2: {eval_res['force_r2']:.4f}")
    print(f"Reconstruction MSE: {eval_res['reconstruction_mse']:.6f}")
    print(f"Model and artifacts saved to: {RUN_DIR}")
    print(f"Unique run directory: {RUN_DIR}")
    
    # Отображаем примеры предсказаний
    if eval_res['predictions'].size > 0:
        print("\nSample predictions (first 10 validation samples):")
        for i in range(min(10, len(eval_res['actuals']))):
            print(f"  Sample {i+1}: Actual={eval_res['actuals'][i]:.4f} N, Predicted={eval_res['predictions'][i]:.4f} N, Error={abs(eval_res['actuals'][i] - eval_res['predictions'][i]):.4f} N")
    
    print(f"\nAll model artifacts have been saved to: {RUN_DIR}")
    print("This directory contains:")
    for file in sorted(RUN_DIR.glob('*')):
        print(f"  - {file.name}")
    
    print("\nIMPORTANT: This model should now be fully compatible with your inference code.")
    print("The architecture has been designed to match the exact layer structure expected by the inference system.")
    print("Key compatibility features:")
    print(f"  - Type of sensor: {SENSOR_TYPE}")
    print(f"  - Number of markers: {MARKER_COUNT} (configured) vs {marker_count} (actual)")
    print(f"  - Geometric parameters: {sensor_geometry}")
    print("  - Exact layer naming convention (encoder.12, decoder.9, etc.)")
    print("  - Correct layer dimensions and shapes")
    print("  - Proper integration of force regressor")
    print("  - BatchNorm and Dropout layers in expected positions")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()