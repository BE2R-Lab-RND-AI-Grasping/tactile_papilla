#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RBF Regression Training for Force Prediction from Relative Deformation Data
Обучение RBF регрессии на относительных деформациях (векторы Q-P) и суммах L2 норм
Особенности:
- Загрузка агрегированных данных с датчиком типа B
- Использование готовых векторов деформаций (Q-P) вместо вычисления из координат
- Использование сумм L2 норм векторов деформаций как дополнительного признака
- Обработка данных без NaN значений (уже отфильтровано при агрегации)
- Групповой стратифицированный сплит по исходным датасетам
- RBF регрессия с гауссовыми базисными функциями и автоматической настройкой гиперпараметров
- Визуализация результатов обучения и предсказаний
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import joblib

# =======================
# Configuration Parameters
# =======================
BASE_DIR = Path("Datasets")
LABELLED_DIR = BASE_DIR / "Labelled"
AGGREGATED_DIR = LABELLED_DIR / "Aggregated_deformation_vectors_sensor_B"
SENSOR_TYPE = "A"  # тип датчика
RANDOM_SEED = 42
TEST_SIZE = 0.2  # доля для валидации (групповая стратификация)
N_JOBS = -1  # использовать все ядра для GridSearchCV

# RBF Hyperparameters for GridSearchCV
PARAM_GRID = {
    'rbfregressor__num_centers': [50, 100, 200, 300],
    'rbfregressor__gamma': [0.001, 0.01, 0.1, 1.0],
    'rbfregressor__alpha': [1e-6, 1e-4, 1e-2, 0.1, 1.0]
}

# Для быстрого обучения можно использовать упрощенную сетку
FAST_PARAM_GRID = {
    'rbfregressor__num_centers': [50, 100],
    'rbfregressor__gamma': [0.01, 0.1],
    'rbfregressor__alpha': [1e-4, 1e-2]
}

USE_FAST_GRID = False  # False для полного поиска, True для быстрого прототипирования

# =======================
# Output Configuration
# =======================
OUTPUT_DIR = Path("models") / f"{SENSOR_TYPE}_rbf_regression_force_prediction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# Custom RBF Regressor
# =======================
class RBFRegressor:
    """
    RBF Regressor using Gaussian radial basis functions
    
    Parameters:
    -----------
    num_centers : int, default=100
        Number of RBF centers to use (selected via K-means clustering)
    gamma : float, default=0.1
        Width parameter for the Gaussian RBF functions
    alpha : float, default=1e-4  
        Regularization strength for Ridge regression
    random_state : int, default=42
        Random state for reproducibility
    """
    def __init__(self, num_centers=100, gamma=0.1, alpha=1e-4, random_state=42):
        self.num_centers = num_centers
        self.gamma = gamma
        self.alpha = alpha
        self.random_state = random_state
        self.centers_ = None
        self.weights_ = None
        self.scaler_ = StandardScaler()
        self.ridge_ = Ridge(alpha=alpha, random_state=random_state)
        self.is_fitted_ = False
    
    def _compute_rbf_matrix(self, X):
        """
        Compute RBF matrix for input data X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        Phi : array of shape (n_samples, num_centers)
            RBF matrix where Phi[i, j] = exp(-gamma * ||x_i - c_j||^2)
        """
        if self.centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        n_samples = X.shape[0]
        n_centers = self.centers_.shape[0]
        Phi = np.zeros((n_samples, n_centers))
        
        # Compute pairwise distances and apply RBF kernel
        for i in range(n_samples):
            # Vectorized computation of squared Euclidean distances
            dist_sq = np.sum((X[i] - self.centers_) ** 2, axis=1)
            Phi[i] = np.exp(-self.gamma * dist_sq)
        
        return Phi
    
    def fit(self, X, y):
        """
        Fit the RBF regressor model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Standardize the input features
        X_scaled = self.scaler_.fit_transform(X)
        
        # Select RBF centers using K-means clustering
        print(f"  Selecting {self.num_centers} RBF centers using K-means...")
        kmeans = KMeans(
            n_clusters=min(self.num_centers, X_scaled.shape[0]),
            random_state=self.random_state,
            n_init=10
        )
        kmeans.fit(X_scaled)
        self.centers_ = kmeans.cluster_centers_
        
        # Compute RBF features for training data
        print("  Computing RBF features...")
        Phi = self._compute_rbf_matrix(X_scaled)
        
        # Fit Ridge regression on the RBF features
        print("  Fitting Ridge regression on RBF features...")
        self.ridge_.fit(Phi, y)
        self.weights_ = self.ridge_.coef_
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Predict using the RBF regressor model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Standardize the input features
        X_scaled = self.scaler_.transform(X)
        
        # Compute RBF features
        Phi = self._compute_rbf_matrix(X_scaled)
        
        # Predict using the linear model on RBF features
        return self.ridge_.predict(Phi)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'num_centers': self.num_centers,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

# =======================
# Data Loading and Processing (reused from original code)
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
    
    print(f"Total valid samples after filtering: {X_combined.shape[0]}")
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
# Group-stratified split (reused from original code)
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
# Evaluation Functions
# =======================
def evaluate_model(model, X, y):
    """Evaluate model performance on given data"""
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'actuals': y
    }

# =======================
# Plotting helpers
# =======================
def plot_predictions_vs_actuals(actuals, predictions, groups=None, save_path=None):
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
    plt.title('RBF Regression: Predicted vs Actual (Deformation Vectors + L2 Norm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.xlim(mn - 0.1, mx + 0.1)
    plt.ylim(mn - 0.1, mx + 0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(actuals, predictions, groups=None, save_path=None):
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
    plt.title('Residuals vs Actual')
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

def plot_rbf_features(model, X_train, feature_names, save_path=None):
    """
    Visualize RBF features and their weights
    """
    if not hasattr(model, 'rbfregressor'):
        return
    
    rbf_model = model.named_steps['rbfregressor']
    if rbf_model.centers_ is None:
        return
    
    # Get RBF features for training data
    X_scaled = model.named_steps['standardscaler'].transform(X_train)
    Phi = np.zeros((X_scaled.shape[0], rbf_model.centers_.shape[0]))
    
    for i in range(X_scaled.shape[0]):
        dist_sq = np.sum((X_scaled[i] - rbf_model.centers_) ** 2, axis=1)
        Phi[i] = np.exp(-rbf_model.gamma * dist_sq)
    
    # Get feature weights from Ridge regression
    weights = rbf_model.ridge_.coef_
    
    # Plot weights
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(weights, 'b-', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('RBF Feature Weights')
    plt.xlabel('RBF Center Index')
    plt.ylabel('Weight Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot weight distribution
    plt.subplot(1, 2, 2)
    plt.hist(weights, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of RBF Feature Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rbf_centers(model, X_train, y_train, feature_names, save_path=None):
    """
    Visualize RBF centers in the feature space
    """
    if not hasattr(model, 'rbfregressor'):
        return
    
    rbf_model = model.named_steps['rbfregressor']
    if rbf_model.centers_ is None:
        return
    
    # Take only the first two features for visualization
    X_scaled = model.named_steps['standardscaler'].transform(X_train)
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot of training data colored by target values
    sc = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                   c=y_train, cmap='viridis', 
                   alpha=0.6, s=30, edgecolors='white')
    plt.colorbar(sc, label='Force (N)')
    
    # Plot RBF centers
    plt.scatter(rbf_model.centers_[:, 0], rbf_model.centers_[:, 1],
               s=100, c='red', marker='X', edgecolors='black',
               label=f'RBF Centers ({rbf_model.num_centers})')
    
    plt.title('RBF Centers in Feature Space (first two features)')
    plt.xlabel(f'Feature 1: {feature_names[0]}')
    plt.ylabel(f'Feature 2: {feature_names[1]}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =======================
# Main training function
# =======================
def main():
    print("=" * 60)
    print("RBF REGRESSION TRAINING")
    print("Force Prediction from Deformation Vectors and L2 Norms")
    print("=" * 60)
    
    # 1) Load deformation vectors dataset
    try:
        X, y, marker_count, dataset_info, groups, reference_P, total_l2_norms = load_deformation_vectors_dataset()
    except Exception as e:
        print("Error loading deformation vectors dataset:", e)
        sys.exit(1)
    
    print(f"\nData shape: X {X.shape}, y {y.shape}, markers per sample: {marker_count}")
    
    # Save dataset_info
    dataset_info_path = OUTPUT_DIR / "dataset_info.json"
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
        'sorting_order': 'left-to-right and top-to-bottom'
    }
    with open(OUTPUT_DIR / "reference_P_info.json", 'w') as f:
        json.dump(to_serializable(reference_info), f, indent=2)
    print(f"Saved reference P-state info to {OUTPUT_DIR / 'reference_P_info.json'}")
    
    # Explicitly save reference_P.npy in the model directory for inference
    reference_P_path = OUTPUT_DIR / "reference_P_sorted.npy"
    np.save(reference_P_path, reference_P)
    print(f"Saved reference P-state numpy array to {reference_P_path}")
    
    # 2) Check duplicates
    df_check = pd.DataFrame(X)
    df_check['y'] = y
    dup_count = df_check.duplicated(keep=False).sum()
    print(f"Exact duplicate rows in combined data: {dup_count}")
    if dup_count > 0:
        print("  Note: duplicates detected. Consider deduplication if these arise from repeated captures.")
    
    # 3) Group-stratified split
    try:
        X_train, X_val, y_train, y_val, train_idx, val_idx = group_stratified_split(
            X, y, groups, test_size=TEST_SIZE, n_bins=10, random_state=RANDOM_SEED
        )
        groups_val = groups[val_idx]  # группы для валидационной выборки
    except Exception as e:
        print("Group stratified split failed:", e)
        print("Falling back to simple shuffled split.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True)
        groups_val = None
    
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    
    # 4) Create RBF pipeline with standardization
    print("\nCreating RBF regression pipeline with StandardScaler...")
    rbf_pipeline = make_pipeline(
        StandardScaler(),
        RBFRegressor(
            num_centers=100, 
            gamma=0.1, 
            alpha=1e-4,
            random_state=RANDOM_SEED
        )
    )
    
    # 5) Hyperparameter tuning with GridSearchCV
    print("\nStarting hyperparameter tuning with GridSearchCV...")
    print(f"Using {'FAST' if USE_FAST_GRID else 'FULL'} parameter grid")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        rbf_pipeline,
        param_grid=FAST_PARAM_GRID if USE_FAST_GRID else PARAM_GRID,
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=N_JOBS,
        verbose=2,
        refit=True
    )
    
    grid_search.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Hyperparameter tuning completed in {training_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation MSE: {-grid_search.best_score_:.6f}")
    
    # 6) Evaluate on training and validation sets
    print("\nEvaluating model performance...")
    
    # Training set evaluation
    train_eval = evaluate_model(grid_search, X_train, y_train)
    print("\nTraining metrics:")
    print(f" MSE:  {train_eval['mse']:.6f}")
    print(f" RMSE: {train_eval['rmse']:.6f}")
    print(f" MAE:  {train_eval['mae']:.6f}")
    print(f" R2:   {train_eval['r2']:.4f}")
    
    # Validation set evaluation
    val_eval = evaluate_model(grid_search, X_val, y_val)
    print("\nValidation metrics:")
    print(f" MSE:  {val_eval['mse']:.6f}")
    print(f" RMSE: {val_eval['rmse']:.6f}")
    print(f" MAE:  {val_eval['mae']:.6f}")
    print(f" R2:   {val_eval['r2']:.4f}")
    
    # 7) Baseline (mean predictor)
    y_pred_baseline = np.mean(y_train)
    mse_baseline = np.mean((y_val - y_pred_baseline) ** 2)
    print("\nBaseline (mean predictor):")
    print(f" Baseline MSE (mean predictor): {mse_baseline:.6f}")
    print(f" RBF improvement over baseline: {(mse_baseline - val_eval['mse']) / mse_baseline * 100:.2f}%")
    
    # 8) Save the best RBF model
    model_path = OUTPUT_DIR / "rbf_regression_force_prediction_model.joblib"
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Saved best RBF model to {model_path}")
    
    # 9) Save scaler separately for inference
    scaler = grid_search.best_estimator_.named_steps['standardscaler']
    scaler_path = OUTPUT_DIR / "deformation_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved deformation scaler to {scaler_path}")
    
    # 10) Save training history and results
    training_results = {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,  # convert back to MSE
        'training_time_seconds': training_time,
        'cv_results': {k: to_serializable(v) for k, v in grid_search.cv_results_.items()},
        'training_metrics': {
            'mse': train_eval['mse'],
            'rmse': train_eval['rmse'],
            'mae': train_eval['mae'],
            'r2': train_eval['r2']
        },
        'validation_metrics': {
            'mse': val_eval['mse'],
            'rmse': val_eval['rmse'],
            'mae': val_eval['mae'],
            'r2': val_eval['r2'],
            'baseline_mse': mse_baseline,
            'improvement_percent': (mse_baseline - val_eval['mse']) / mse_baseline * 100
        },
        'data_info': {
            'total_samples': X.shape[0],
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'marker_count': marker_count,
            'features_per_sample': X.shape[1],
            'feature_description': f'{marker_count} markers × 2 coordinates + 1 total L2 norm'
        },
        'training_config': {
            'param_grid_used': 'FAST' if USE_FAST_GRID else 'FULL',
            'grid_search_params': FAST_PARAM_GRID if USE_FAST_GRID else PARAM_GRID,
            'cv_folds': 3,
            'random_seed': RANDOM_SEED
        }
    }
    
    history_path = OUTPUT_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(to_serializable(training_results), f, indent=2)
    print(f"Saved training history to {history_path}")
    
    # 11) Generate visualizations
    print("\nGenerating visualizations...")
    
    # Predictions vs Actuals plot
    plot_predictions_vs_actuals(val_eval['actuals'], val_eval['predictions'], 
                              groups=groups_val, save_path=OUTPUT_DIR / "predictions_vs_actuals.png")
    print("  predictions_vs_actuals.png saved")
    
    # Residuals plot
    plot_residuals(val_eval['actuals'], val_eval['predictions'], 
                  groups=groups_val, save_path=OUTPUT_DIR / "residuals.png")
    print("  residuals.png saved")
    
    # RBF features visualization
    feature_names = []
    for i in range(marker_count):
        feature_names.append(f'Marker_{i}_dx')
        feature_names.append(f'Marker_{i}_dy')
    feature_names.append('Total_L2_Norm')
    
    try:
        plot_rbf_features(grid_search.best_estimator_, X_train, feature_names,
                         save_path=OUTPUT_DIR / "rbf_feature_weights.png")
        print("  rbf_feature_weights.png saved")
    except Exception as e:
        print(f"Error generating RBF feature weights plot: {e}")
    
    try:
        plot_rbf_centers(grid_search.best_estimator_, X_train, y_train, feature_names,
                        save_path=OUTPUT_DIR / "rbf_centers.png")
        print("  rbf_centers.png saved")
    except Exception as e:
        print(f"Error generating RBF centers plot: {e}")
    
    # 12) Per-group evaluation
    try:
        if groups_val is not None:
            df_val = pd.DataFrame({
                'group': groups_val,
                'y_true': val_eval['actuals'],
                'y_pred': val_eval['predictions']
            })
            
            print("\nPer-group validation MSE:")
            group_results = {}
            for group_idx, group_df in df_val.groupby('group'):
                group_mse = mean_squared_error(group_df['y_true'], group_df['y_pred'])
                group_rmse = np.sqrt(group_mse)
                group_mae = mean_absolute_error(group_df['y_true'], group_df['y_pred'])
                group_r2 = r2_score(group_df['y_true'], group_df['y_pred'])
                
                print(f"  Dataset {group_idx}:")
                print(f"    Samples: {len(group_df)}")
                print(f"    MSE: {group_mse:.6f}")
                print(f"    RMSE: {group_rmse:.6f}")
                print(f"    MAE: {group_mae:.6f}")
                print(f"    R2: {group_r2:.4f}")
                print(f"    Force range: {group_df['y_true'].min():.3f} - {group_df['y_true'].max():.3f} N")
                
                group_results[str(group_idx)] = {
                    'samples': len(group_df),
                    'mse': group_mse,
                    'rmse': group_rmse,
                    'mae': group_mae,
                    'r2': group_r2,
                    'force_range': [group_df['y_true'].min(), group_df['y_true'].max()]
                }
            
            # Save per-group results
            group_results_path = OUTPUT_DIR / "group_results.json"
            with open(group_results_path, 'w') as f:
                json.dump(to_serializable(group_results), f, indent=2)
            print(f"  Per-group results saved to {group_results_path}")
    except Exception as e:
        print(f"Error in per-group evaluation: {e}")
    
    # 13) Correlation analysis between total L2 norm and force
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(total_l2_norms, y, alpha=0.6, edgecolors='white', s=40)
        plt.xlabel('Total L2 Norm of Deformation Vectors')
        plt.ylabel('Force (N)')
        plt.title('Correlation between Total L2 Norm and Force')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Добавляем линейную регрессию
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(total_l2_norms, y)
        x_vals = np.array([total_l2_norms.min(), total_l2_norms.max()])
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, 'r-', linewidth=2, label=f'R={r_value:.3f}')
        plt.legend()
        
        correlation_path = OUTPUT_DIR / "l2norm_force_correlation.png"
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  L2 norm vs force correlation plot saved to {correlation_path}")
        print(f"\nCorrelation analysis:")
        print(f"  Pearson correlation coefficient (R) between L2 norm and force: {r_value:.4f}")
        print(f"  P-value: {p_value:.6f}")
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
    
    # 14) Final summary
    print("\n" + "=" * 60)
    print("RBF REGRESSION TRAINING COMPLETED")
    print("=" * 60)
    print(f"Sample count: {X.shape[0]}, markers per sample: {marker_count}")
    print(f"Input features per sample: {X.shape[1]} ({marker_count} markers × 2 coordinates + 1 total L2 norm)")
    print(f"Best validation MSE: {val_eval['mse']:.6f}")
    print(f"Best validation RMSE: {val_eval['rmse']:.6f}")
    print(f"Best validation R2: {val_eval['r2']:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Model and artifacts saved to: {OUTPUT_DIR}")
    print("\nSaved files:")
    print(f"  - Best RBF model: {model_path}")
    print(f"  - Deformation scaler: {scaler_path}")
    print(f"  - Training history: {history_path}")
    print(f"  - Dataset information: {dataset_info_path}")
    print(f"  - Reference P-state info: {OUTPUT_DIR / 'reference_P_info.json'}")
    print(f"  - Reference P-state array: {reference_P_path}")
    print(f"  - Predictions vs Actuals plot: {OUTPUT_DIR / 'predictions_vs_actuals.png'}")
    print(f"  - Residuals plot: {OUTPUT_DIR / 'residuals.png'}")
    print(f"  - RBF feature weights plot: {OUTPUT_DIR / 'rbf_feature_weights.png'}")
    print(f"  - RBF centers plot: {OUTPUT_DIR / 'rbf_centers.png'}")
    
    # Display sample predictions
    if len(val_eval['actuals']) > 0:
        print("\nSample predictions (first 10 validation samples):")
        for i in range(min(10, len(val_eval['actuals']))):
            print(f"  Sample {i+1}: Actual={val_eval['actuals'][i]:.4f} N, "
                  f"Predicted={val_eval['predictions'][i]:.4f} N, "
                  f"Error={abs(val_eval['actuals'][i] - val_eval['predictions'][i]):.4f} N")
    
    print("\n" + "=" * 60)
    print("RBF training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()