#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АНАЛИЗ КОМПРОМИССА: РЕГРЕССИЯ СИЛЫ ↔ ЛОКАЛИЗАЦИЯ КОНТАКТА
Версия 11.6-mod — узкие виолончельные диаграммы + объединённая линия выбросов
  - Узкие виолончельные диаграммы (widths=0.35), основная часть обрезается по перцентилю
  - Индивидуальные выбросы отображаются крестиками; репрезентативная точка выбросов (медиана)
    соединяется линией для показа тенденции между датчиками
  - Подписи графиков и легенды на русском языке (без англицизмов, кроме общепринятых аббревиатур)
Автор: адаптация
Дата: 2026
"""
import os
import sys
import json
import warnings
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn

# ---------------- Config ----------------
SENSOR_GEOMETRY = {
    'A': {'D': 8.69, 'M': 12},  # D = диаметр в мм, M = количество маркеров
    'B': {'D': 6.94, 'M': 15},
    'C': {'D': 6.47, 'M': 20},
    'D': {'D': 5.75, 'M': 24},
    'E': {'D': 5.14, 'M': 30}
}
SENSOR_TYPES = list(SENSOR_GEOMETRY.keys())

BASE_DIR = Path("Datasets")
LABELLED_DIR = BASE_DIR / "Labelled"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("analysis_results_v11")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLORS = {
    'A': '#1f77b4',  # синий
    'B': '#ff7f0e',  # оранжевый
    'C': '#2ca02c',  # зелёный
    'D': '#d62728',  # красный
    'E': '#9467bd'   # фиолетовый
}

plt.rcParams.update({
    'figure.figsize': (11, 7),
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.alpha': 0.25,
    'lines.linewidth': 2.0,
    'lines.markersize': 6
})
warnings.filterwarnings('ignore')
RND = 42
np.random.seed(RND)
random.seed(RND)

# ---------------- Utilities ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def gaussian_kernel1d(sigma, radius=None):
    if radius is None:
        radius = max(3, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius+1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k

def lhs_sample(trials, dims):
    result = np.zeros((trials, dims))
    rng = np.random.RandomState(RND)
    for d in range(dims):
        strata = (np.arange(trials) + rng.uniform(size=trials)) / trials
        rng.shuffle(strata)
        result[:, d] = strata
    return result

# ---------------- Data loaders ----------------
def load_aggregated_dataset(sensor_type):
    try:
        agg_dir = LABELLED_DIR / f"Aggregated_deformation_vectors_sensor_{sensor_type}"
        if not agg_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {agg_dir}")
        deformations = np.load(agg_dir / "deformation_vectors.npy")  # (N, m, 2)
        forces = np.load(agg_dir / "forces.npy")                     # (N,)
        total_l2_norms = np.load(agg_dir / "total_l2_norms.npy")     # (N,)
        reference_P = np.load(agg_dir / "reference_P_sorted.npy")    # (m,2)
        with open(agg_dir / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"✓ Загружен датасет {sensor_type}: {deformations.shape[0]} образцов, {deformations.shape[1]} такселей")
        return {
            'deformations': deformations,
            'forces': forces,
            'total_l2_norms': total_l2_norms,
            'reference_P': reference_P,
            'metadata': metadata,
            'taxel_count': deformations.shape[1],
            'diameter_mm': SENSOR_GEOMETRY.get(sensor_type, {}).get('D', float('nan')),
            'marker_count': SENSOR_GEOMETRY.get(sensor_type, {}).get('M', float('nan'))
        }
    except Exception as e:
        print(f"❌ Ошибка загрузки {sensor_type}: {e}")
        return None

def load_trained_model(sensor_type):
    try:
        model_dir = MODELS_DIR / f"{sensor_type}_deformation_vectors_force_prediction"
        if not model_dir.exists():
            print(f"[ВНИМАНИЕ] Модель не найдена: {model_dir}")
            return None
        model_path = model_dir / "force_prediction_model.pth"
        if not model_path.exists():
            model_path = model_dir / "model.pth"
        if not model_path.exists():
            print(f"[ВНИМАНИЕ] Файл модели не найден в {model_dir}")
            return None

        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception:
            ckpt = torch.load(model_path, map_location='cpu')

        state = None
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                keys = list(ckpt.keys())
                if any(k.endswith('.weight') for k in keys):
                    state = ckpt

        if state is None:
            print(f"[ВНИМАНИЕ] Не найден state_dict в чекпойнте для {sensor_type}. Пропускаю загрузку модели.")
            return None

        weight_keys = [k for k in state.keys() if k.endswith('.weight')]
        if not weight_keys:
            print(f"[ВНИМАНИЕ] Не найден .weight в state_dict для {sensor_type}")
            return None

        first_w = state[weight_keys[0]]
        if isinstance(first_w, torch.Tensor):
            input_size = int(first_w.shape[1])
        else:
            arr = np.array(first_w)
            input_size = int(arr.shape[1])

        hidden_layers = ckpt.get('hidden_layers', ckpt.get('model_hidden_layers', [128, 64]))
        dropout_rate = ckpt.get('dropout_rate', 0.0)
        use_batchnorm = ckpt.get('use_batchnorm', False)

        class ForcePredictionNet(nn.Module):
            def __init__(self, input_size, hidden_layers, dropout_rate=0.0, use_batchnorm=False):
                super().__init__()
                layers = []
                prev = input_size
                for h in hidden_layers:
                    layers.append(nn.Linear(prev, h))
                    if use_batchnorm:
                        layers.append(nn.BatchNorm1d(h))
                    layers.append(nn.ReLU())
                    if dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.network = nn.Sequential(*layers)
            def forward(self, x):
                return self.network(x)

        model = ForcePredictionNet(input_size, hidden_layers, dropout_rate, use_batchnorm)

        load_sd = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                load_sd[k] = torch.tensor(v)
            elif isinstance(v, torch.Tensor):
                load_sd[k] = v
            else:
                try:
                    load_sd[k] = torch.tensor(np.array(v))
                except Exception:
                    pass
        try:
            model.load_state_dict(load_sd, strict=False)
            model.eval()
        except Exception as e:
            print(f"[ВНИМАНИЕ] Не удалось загрузить веса модели {sensor_type}: {e}")
            return None

        scaler_info = None
        scaler_path = model_dir / "deformation_scaler.json"
        if scaler_path.exists():
            try:
                with open(scaler_path, 'r', encoding='utf-8') as f:
                    scaler_info = json.load(f)
            except Exception:
                scaler_info = None

        train_history = None
        if isinstance(ckpt, dict):
            if 'history' in ckpt:
                train_history = ckpt['history']
            elif 'train_history' in ckpt:
                train_history = ckpt['train_history']
            elif 'loss_history' in ckpt:
                train_history = {'train_loss': ckpt['loss_history']}
            elif 'train_loss' in ckpt or 'val_loss' in ckpt:
                train_history = {'train_loss': ckpt.get('train_loss'), 'val_loss': ckpt.get('val_loss')}
        hist_path = model_dir / "training_history.json"
        if train_history is None and hist_path.exists():
            try:
                with open(hist_path, 'r', encoding='utf-8') as f:
                    train_history = json.load(f)
            except Exception:
                train_history = None

        print(f"✓ Загружена модель {sensor_type}: input={input_size}, hidden={hidden_layers}")
        return {'model': model, 'scaler_info': scaler_info, 'input_size': input_size, 'hidden_layers': hidden_layers, 'train_history': train_history}
    except Exception as e:
        print(f"[ОШИБКА] load_trained_model({sensor_type}): {e}")
        return None

# ---------------- Basic metrics ----------------
def calculate_force_metrics(predictions, actuals):
    try:
        r2 = r2_score(actuals, predictions)
    except Exception:
        r2 = float('nan')
    rmse = math.sqrt(mean_squared_error(actuals, predictions)) if len(actuals) > 0 else float('nan')
    mae = mean_absolute_error(actuals, predictions) if len(actuals) > 0 else float('nan')
    return {'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae), 'n_samples': len(actuals)}

# ---------------- Marker loss by number ----------------
def simulate_marker_loss_by_n(deformations, reference_P, max_n=5, trials_per_n=100):
    N, m, _ = deformations.shape
    deformation_magnitudes = np.linalg.norm(deformations, axis=2)
    true_centroids = np.zeros((N, 2))
    for i in range(N):
        w = deformation_magnitudes[i]
        if w.sum() > 1e-10:
            true_centroids[i] = np.average(reference_P, axis=0, weights=w)
        else:
            true_centroids[i] = reference_P.mean(axis=0)

    results = {}
    for n in range(0, max_n+1):
        errs = []
        for t in range(trials_per_n):
            idx = np.random.randint(0, N)
            if n == 0:
                err = 0.0
            else:
                if n >= m:
                    remaining = np.arange(m-1)
                else:
                    lost = np.random.choice(m, size=n, replace=False)
                    remaining = np.setdiff1d(np.arange(m), lost)
                w_rem = deformation_magnitudes[idx][remaining]
                pos_rem = reference_P[remaining]
                if w_rem.sum() > 1e-10:
                    centroid_partial = np.average(pos_rem, axis=0, weights=w_rem)
                else:
                    centroid_partial = pos_rem.mean(axis=0)
                err = float(np.linalg.norm(true_centroids[idx] - centroid_partial))
            errs.append(err)
        results[n] = np.array(errs, dtype=float)
    return results

# ---------------- Pixel-noise sensitivity (LHS) ----------------
def simulate_noise_pixel_lhs(deformations, forces, model_info, trials=1000, pixel_range=50):
    N, m, _ = deformations.shape
    if model_info is None:
        return np.full(trials, np.nan), None

    model = model_info['model']
    scaler = model_info.get('scaler_info', None)

    deformation_flat = deformations.reshape(N, -1)
    total_l2 = np.linalg.norm(deformations, axis=(1,2))
    features_all = np.column_stack([deformation_flat, total_l2])
    if scaler:
        mean = np.array(scaler.get('mean', np.zeros(features_all.shape[1])))
        scale = np.array(scaler.get('scale', np.ones(features_all.shape[1])))
        if mean.shape[0] == features_all.shape[1] and scale.shape[0] == features_all.shape[1]:
            features_scaled = (features_all - mean) / (scale + 1e-12)
            use_scaler = True
        else:
            features_scaled = features_all
            use_scaler = False
    else:
        features_scaled = features_all
        use_scaler = False

    with torch.no_grad():
        preds_all = model(torch.tensor(features_scaled, dtype=torch.float32)).numpy().flatten()

    dims = m * 2
    lhs = lhs_sample(trials, dims)
    lhs = (lhs * 2.0 - 1.0) * pixel_range

    rel_errors = np.zeros(trials)
    sample_idxs = np.zeros(trials, dtype=int)
    for t in range(trials):
        i = np.random.randint(0, N)
        sample_idxs[t] = i
        base_pred = preds_all[i]
        F_true = forces[i]
        noise = lhs[t].reshape(m, 2)
        noisy = deformations[i] + noise
        flat_noisy = noisy.reshape(1, -1)
        tot_noisy = np.linalg.norm(noisy)
        feat_noisy = np.hstack([flat_noisy, np.array([[tot_noisy]])])
        if use_scaler:
            feat_noisy_scaled = (feat_noisy - mean) / (scale + 1e-12)
        else:
            feat_noisy_scaled = feat_noisy
        with torch.no_grad():
            p_noisy = model(torch.tensor(feat_noisy_scaled, dtype=torch.float32)).numpy().flatten()[0]
        rel_errors[t] = abs(p_noisy - base_pred) / max(F_true, 0.1)
    return rel_errors, sample_idxs

# ---------------- Force range curves (no fill, smooth) ----------------
def force_range_curves(forces_list):
    overall_max = max([float(np.max(f)) for f in forces_list if len(f) > 0])
    max_edge = math.ceil(overall_max / 2.0) * 2
    bins = np.arange(0.0, max_edge + 2.0, 2.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    densities = []
    sigma = 1.0
    kernel = gaussian_kernel1d(sigma)
    for f in forces_list:
        if len(f) == 0:
            densities.append(np.zeros_like(centers))
            continue
        hist, _ = np.histogram(f, bins=bins, density=True)
        hist_smooth = np.convolve(hist, kernel, mode='same')
        densities.append(hist_smooth)
    return centers, densities

# ---------------- Analyze single sensor ----------------
def analyze_sensor_comprehensive(sensor_type,
                                 marker_trials_per_n=100,
                                 max_missing_markers=5,
                                 noise_pixel_trials=1000,
                                 pixel_noise_range=50):
    print("\n" + "="*60)
    print(f"АНАЛИЗ ДАТЧИКА {sensor_type}")
    print("="*60)
    ds = load_aggregated_dataset(sensor_type)
    if ds is None:
        return None
    model_info = load_trained_model(sensor_type)
    results = {
        'sensor_type': sensor_type,
        'taxel_count': ds['taxel_count'],
        'diameter_mm': ds['diameter_mm'],
        'marker_count': ds['marker_count'],
        'n_samples': ds['deformations'].shape[0]
    }
    deformations = ds['deformations']
    forces = ds['forces']
    total_l2_norms = ds['total_l2_norms']
    reference_P = ds['reference_P']

    if model_info is not None:
        print(" → Расчёт регрессионных метрик...")
        N = deformations.shape[0]
        deformation_flat = deformations.reshape(N, -1)
        features = np.column_stack([deformation_flat, total_l2_norms])
        scaler = model_info.get('scaler_info', None)
        if scaler:
            mean = np.array(scaler.get('mean', np.zeros(features.shape[1])))
            scale = np.array(scaler.get('scale', np.ones(features.shape[1])))
            if mean.shape[0] == features.shape[1] and scale.shape[0] == features.shape[1]:
                features_scaled = (features - mean) / (scale + 1e-12)
            else:
                features_scaled = features
        else:
            features_scaled = features
        with torch.no_grad():
            preds = model_info['model'](torch.tensor(features_scaled, dtype=torch.float32)).numpy().flatten()
        fm = calculate_force_metrics(preds, forces)
        results['force_metrics'] = fm
        results['predictions'] = preds.tolist()
        print(f"    R²={fm['r2']:.4f}, RMSE={fm['rmse']:.4f}, MAE={fm['mae']:.4f}")
    else:
        print(" ⚠ Модель недоступна — регрессия пропущена")
        results['force_metrics'] = {'r2': float('nan'), 'rmse': float('nan'), 'mae': float('nan')}
        results['predictions'] = []

    results['actual_forces'] = forces.tolist()
    results['force_range_analysis'] = {
        'median_force': float(np.median(forces)) if len(forces) > 0 else float('nan'),
        'max_force': float(np.max(forces)) if len(forces) > 0 else float('nan'),
        'total_samples': len(forces)
    }
    print(f" → Диапазоны сил: медиана={results['force_range_analysis']['median_force']:.2f} Н, макс={results['force_range_analysis']['max_force']:.2f} Н")

    marker_errors = simulate_marker_loss_by_n(deformations, reference_P, max_n=max_missing_markers, trials_per_n=marker_trials_per_n)
    results['marker_loss_by_n'] = {int(k): v.tolist() for k, v in marker_errors.items()}
    print("    Пример: медиана ошибки при n=3:", float(np.median(marker_errors[3])))

    noise_errs, sample_idxs = simulate_noise_pixel_lhs(deformations, forces, model_info, trials=noise_pixel_trials, pixel_range=pixel_noise_range)
    results['pixel_noise_trials'] = noise_errs.tolist()
    results['pixel_noise_sample_idxs'] = None if sample_idxs is None else sample_idxs.tolist()
    print(f"    Пиксельный шум: медиана относит. ошибки = {np.nanmedian(noise_errs):.4f}")

    if model_info is not None and model_info.get('train_history') is not None:
        results['train_history'] = model_info['train_history']
    else:
        results['train_history'] = None

    return results

# ---------------- Aggregate ----------------
def load_all_sensors():
    all_results = {}
    for st in SENSOR_TYPES:
        res = analyze_sensor_comprehensive(st,
                                           marker_trials_per_n=100,
                                           max_missing_markers=5,
                                           noise_pixel_trials=1000,
                                           pixel_noise_range=50)
        if res is not None:
            all_results[st] = res
    with open(RESULTS_DIR / 'all_results_v11.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    return all_results

# ---------------- PLOTS ----------------
def plot_regression_metrics(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sensor_labels = [st for st in SENSOR_TYPES if st in all_results]
    taxel_counts = [all_results[st]['taxel_count'] for st in sensor_labels]
    colors = [SENSOR_COLORS[st] for st in sensor_labels]

    r2_values = [all_results[st]['force_metrics']['r2'] for st in sensor_labels]
    bars = axes[0].bar(range(len(sensor_labels)), r2_values, color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(sensor_labels)))
    axes[0].set_xticklabels([f"{lbl}\n({tc} такс.)" for lbl, tc in zip(sensor_labels, taxel_counts)])
    axes[0].set_ylabel('R²')
    axes[0].set_title('Качество регрессии силы — R²')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].set_ylim(0.0, 1.0)
    for bar, val in zip(bars, r2_values):
        axes[0].text(bar.get_x()+bar.get_width()/2, (val if not np.isnan(val) else 0) + 0.02,
                     f"{val:.3f}" if not np.isnan(val) else "N/A", ha='center', fontsize=9)

    mae_values = [all_results[st]['force_metrics']['mae'] for st in sensor_labels]
    bars = axes[1].bar(range(len(sensor_labels)), mae_values, color=colors, edgecolor='black')
    axes[1].set_xticks(range(len(sensor_labels)))
    axes[1].set_xticklabels([f"{lbl}\n({tc} такс.)" for lbl, tc in zip(sensor_labels, taxel_counts)])
    axes[1].set_ylabel('MAE (Н)')
    axes[1].set_title('Средняя абсолютная ошибка (MAE), Н')
    axes[1].grid(axis='y', alpha=0.25)
    for bar, val in zip(bars, mae_values):
        axes[1].text(bar.get_x()+bar.get_width()/2, (val if not np.isnan(val) else 0) + 0.02,
                     f"{val:.3f}" if not np.isnan(val) else "N/A", ha='center', fontsize=9)

    rmse_values = [all_results[st]['force_metrics']['rmse'] for st in sensor_labels]
    bars = axes[2].bar(range(len(sensor_labels)), rmse_values, color=colors, edgecolor='black')
    axes[2].set_xticks(range(len(sensor_labels)))
    axes[2].set_xticklabels([f"{lbl}\n({tc} такс.)" for lbl, tc in zip(sensor_labels, taxel_counts)])
    axes[2].set_ylabel('RMSE (Н)')
    axes[2].set_title('Среднеквадратичная ошибка (RMSE), Н')
    axes[2].grid(axis='y', alpha=0.25)
    for bar, val in zip(bars, rmse_values):
        axes[2].text(bar.get_x()+bar.get_width()/2, (val if not np.isnan(val) else 0) + 0.02,
                     f"{val:.3f}" if not np.isnan(val) else "N/A", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'regression_metrics.png', dpi=200)
    plt.close()
    print("✅ regression_metrics.png сохранён")

def plot_marker_loss_grouped_bars(all_results, max_n=5):
    sensors_present = [st for st in SENSOR_TYPES if st in all_results]
    if not sensors_present:
        print("[ВНИМАНИЕ] Нет данных для marker_loss_grouped_bars")
        return
    n_groups = max_n
    n_series = len(sensors_present)
    means = np.zeros((n_groups, n_series))
    for i, st in enumerate(sensors_present):
        for n in range(1, max_n+1):
            arr = np.array(all_results[st]['marker_loss_by_n'][n], dtype=float)
            means[n-1, i] = float(np.mean(arr)) if arr.size > 0 else np.nan

    x = np.arange(1, max_n+1)
    total_width = 0.8
    bar_width = total_width / n_series
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, st in enumerate(sensors_present):
        offsets = x - total_width/2 + i*bar_width + bar_width/2
        ax.bar(offsets, means[:, i], width=bar_width, color=SENSOR_COLORS[st],
               label=f"Датчик {st} ({all_results[st]['taxel_count']} такс.)", edgecolor='black', alpha=0.95)
        for xi, val in zip(offsets, means[:, i]):
            if not np.isnan(val):
                ax.text(xi, val + 0.01 * np.nanmax(means), f"{val:.2f}", ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel('Число потерянных маркеров n')
    ax.set_ylabel('Средняя ошибка центроиды (пиксели)')
    ax.set_title('Робастность при потере маркеров — группированные столбцы (n=1..5)')
    ax.legend(title='Датчики', loc='upper left')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'marker_loss_grouped_bars_n1_5.png', dpi=200)
    plt.close()
    print("✅ marker_loss_grouped_bars_n1_5.png сохранён")

def plot_pixel_noise_distributions(all_results, clip_percentile=95, max_points_per_sensor=400):
    """
    Виолончельные диаграммы (узкие) по основному распределению (обрезаны по clip_percentile)
    + отдельные крестики для выбросов + линия медиан выбросов между датчиками.
    Подписи и заголовки — на русском языке.
    """
    labels = []
    data = []
    colors = []
    diameters = []
    for st in SENSOR_TYPES:
        if st not in all_results:
            continue
        errs = np.array(all_results[st]['pixel_noise_trials'], dtype=float)
        errs = errs[~np.isnan(errs)]
        data.append(errs)
        labels.append(f"{st}\n(M={int(all_results[st].get('marker_count', np.nan))}, t={int(all_results[st].get('taxel_count', np.nan))})")
        colors.append(SENSOR_COLORS[st])
        diameters.append(all_results[st].get('diameter_mm', SENSOR_GEOMETRY[st]['D']))

    if not data:
        print("[ВНИМАНИЕ] Нет данных для pixel_noise_distributions")
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    positions = np.arange(1, len(data) + 1)
    rng = np.random.RandomState(RND)

    # Обрезка и выделение выбросов
    clipped_data = []
    outliers_list = []
    for errs in data:
        if errs.size == 0:
            clipped_data.append(np.array([]))
            outliers_list.append(np.array([]))
            continue
        pct = np.percentile(errs, clip_percentile)
        clipped = errs[errs <= pct]
        outs = errs[errs > pct]
        if clipped.size == 0:
            clipped = errs
            outs = np.array([])
        clipped_data.append(clipped)
        outliers_list.append(outs)

    # Рисуем узкие виолончели
    violin_parts = ax.violinplot(clipped_data, positions=positions, widths=0.35, showmeans=False, showextrema=False, showmedians=False)
    for i_body, body in enumerate(violin_parts['bodies']):
        col = colors[i_body % len(colors)]
        body.set_facecolor(col)
        body.set_edgecolor('black')
        body.set_alpha(0.4)

    # IQR и медиана (по полным данным)
    for i, errs in enumerate(data):
        pos = positions[i]
        if errs.size == 0:
            continue
        q1 = np.percentile(errs, 25)
        q3 = np.percentile(errs, 75)
        med = np.median(errs)
        ax.vlines(pos, q1, q3, color='k', linewidth=3, zorder=11)
        ax.hlines(med, pos - 0.12, pos + 0.12, color='k', linewidth=1.6, zorder=12)

    # Точки внутри виолончели (узкий джиттер) — основная часть (clipped)
    for i, clipped in enumerate(clipped_data):
        if clipped.size == 0:
            continue
        nplot = min(len(clipped), max_points_per_sensor)
        if len(clipped) > nplot:
            idxs = rng.choice(len(clipped), size=nplot, replace=False)
            sample = clipped[idxs]
        else:
            sample = clipped
        jitter = (rng.rand(len(sample)) - 0.5) * 0.08
        x_positions = (i + 1) + jitter * 0.35
        ax.scatter(x_positions, sample, s=10, c=colors[i], alpha=0.45, edgecolors='none', zorder=13)

    # Выбросы — крестики; репрезентативная точка (медиана выбросов) соединяется линией
    outlier_reps = []
    for i, outs in enumerate(outliers_list):
        if outs.size == 0:
            outlier_reps.append(np.nan)
            if data[i].size > 0:
                y_annot_base = np.percentile(data[i], 99.0)
                ax.text(i+1, y_annot_base * 1.02 + 1e-12, "o:0", ha='center', va='bottom', fontsize=8, color='gray', alpha=0.6)
            continue
        n_out_plot = min(len(outs), 300)
        if len(outs) > n_out_plot:
            sel = rng.choice(len(outs), size=n_out_plot, replace=False)
            outs_plot = outs[sel]
        else:
            outs_plot = outs
        jitter = (rng.rand(len(outs_plot)) - 0.5) * 0.28
        x_positions = (i + 1) + jitter * 0.35
        ax.scatter(x_positions, outs_plot, s=24, c='none', edgecolors='black', marker='x', linewidths=0.7, alpha=0.7, zorder=14)
        rep = float(np.median(outs))
        outlier_reps.append(rep)
        y_annot_base = np.percentile(data[i], min(99.5, 100.0))
        ax.text(i+1, y_annot_base * 1.12 + 1e-12, f"o:{len(outs)}", ha='center', va='bottom', fontsize=8, color='black')

    # Соединяющая линия медиан выбросов (если есть)
    xs = np.array(positions, dtype=float)
    ys = np.array(outlier_reps, dtype=float)
    ax.plot(xs, ys, linestyle='-', color='black', linewidth=1.4, marker='x', markersize=8, zorder=20)

    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.5, len(data) + 0.5)
    ax.set_ylabel('Относительная ошибка (|ΔF|/F)')
    ax.set_xlabel('Датчик (M = маркеров, t = такселей)')
    ax.set_title(f'Распределение относительной ошибки при пиксельном шуме — виолончельные диаграммы (основная часть обрезана до {clip_percentile}%) и линия выбросов')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'pixel_noise_violin_clipped_with_outlier_trend.png', dpi=200)
    plt.close()
    print("✅ pixel_noise_violin_clipped_with_outlier_trend.png сохранён")

def plot_force_range_curves(all_results):
    forces_list = []
    labels = []
    colors = []
    for st in SENSOR_TYPES:
        if st not in all_results:
            continue
        forces_list.append(np.array(all_results[st]['actual_forces'], dtype=float))
        labels.append(st)
        colors.append(SENSOR_COLORS[st])
    if not forces_list:
        print("[ВНИМАНИЕ] Нет данных для графика диапазонов сил.")
        return
    centers, densities = force_range_curves(forces_list)
    fig, ax = plt.subplots(figsize=(10, 6))
    for dens, lab, col in zip(densities, labels, colors):
        ax.plot(centers, dens, '-', label=f"Датчик {lab}", color=col, linewidth=2.2)
    ax.set_xlabel('Сила (Н)')
    ax.set_ylabel('Плотность (сглаженная)')
    ax.set_title('Наложенные сглаженные плотности распределения сил (шаг 2 Н)')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Датчики')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'force_range_curves_smooth.png', dpi=200)
    plt.close()
    print("✅ force_range_curves_smooth.png сохранён")

def plot_metrics_scatter(all_results):
    points = []
    labels = []
    sizes = []
    colors = []
    st_keys = []
    for st in SENSOR_TYPES:
        if st not in all_results:
            continue
        r = all_results[st]
        medians = []
        for n in range(1, 6):
            arr = np.array(r['marker_loss_by_n'][n], dtype=float)
            if arr.size > 0:
                medians.append(float(np.median(arr)))
        if len(medians) == 0:
            continue
        median_of_meds = float(np.median(np.array(medians)))
        robustness = 1.0 / (median_of_meds + 1e-10)
        noise_arr = np.array(r['pixel_noise_trials'], dtype=float)
        noise_arr = noise_arr[~np.isnan(noise_arr)]
        if noise_arr.size == 0:
            continue
        noise_med = float(np.median(noise_arr))
        regression_quality = 1.0 / (1.0 + noise_med)
        points.append((robustness, regression_quality))
        labels.append(st)
        st_keys.append(st)
        diam = float(r.get('diameter_mm', SENSOR_GEOMETRY.get(st, {}).get('D', 1.0)))
        sizes.append(diam * 120.0)
        colors.append(SENSOR_COLORS[st])

    if not points:
        print("[ВНИМАНИЕ] Нет точек для графика Робастность ↔ Качество регрессии")
        return

    points = np.array(points)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points[:,0], points[:,1], s=np.array(sizes), c=colors, edgecolors='black', zorder=10, alpha=0.95)
    for i, st in enumerate(st_keys):
        ax.annotate(f"{st}\nD={all_results[st].get('diameter_mm', SENSOR_GEOMETRY[st]['D']):.2f} мм\nM={int(all_results[st].get('marker_count', SENSOR_GEOMETRY[st]['M']))}",
                    (points[i,0], points[i,1]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

    xs = points[:,0]
    ys = points[:,1]
    n_pts = len(xs)
    deg = 3 if n_pts >= 4 else max(1, n_pts - 1)
    try:
        coeffs = np.polyfit(xs, ys, deg=deg)
        poly = np.poly1d(coeffs)
        x_min, x_max = xs.min(), xs.max()
        x_range = np.linspace(x_min - 0.05 * abs(x_min if x_min != 0 else 1.0),
                              x_max + 0.05 * abs(x_max if x_max != 0 else 1.0), 300)
        y_fit = poly(x_range)
        ax.plot(x_range, y_fit, linestyle='--', linewidth=2.2, color='k', zorder=5, label=f'Аппроксимация полиномом, степень {deg}')
        ax.legend(title='Аппроксимация')
    except Exception as e:
        print(f"[ВНИМАНИЕ] Не удалось подогнать аппроксимацию: {e}")

    ax.set_xlabel('Робастность (1 / медиана ошибок для n=1..5)')
    ax.set_ylabel('Качество регрессии (1 / (1 + медиана относит.ошибки))')
    ax.set_title('Робастность ↔ Качество регрессии (размер точки пропорционален диаметру такселя)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'metrics_scatter_points_diameter.png', dpi=200)
    plt.close()
    print("✅ metrics_scatter_points_diameter.png сохранён")

def plot_training_losses(all_results):
    sensors_with_hist = [st for st in SENSOR_TYPES if st in all_results and all_results[st].get('train_history') is not None]
    series = []
    for st in sensors_with_hist:
        hist = all_results[st]['train_history']
        train_loss = None
        if isinstance(hist, dict):
            train_loss = hist.get('train_loss') or hist.get('loss') or hist.get('train_losses')
        if train_loss is None and isinstance(hist, list):
            train_loss = hist
        if train_loss is None:
            continue
        try:
            train_loss = np.array(train_loss, dtype=float)
        except Exception:
            continue
        if train_loss.size == 0:
            continue
        series.append((st, train_loss))

    if not series:
        print("[ИНФО] Нет истории обучения в моделях — пропускаю график истории обучения.")
        return

    all_pos_losses = np.hstack([s[1][s[1] > 0.0] for s in series if np.any(s[1] > 0.0)]) if series else np.array([])
    if all_pos_losses.size == 0:
        y_min_pos = 1e-8
    else:
        y_min_pos = float(np.min(all_pos_losses))
    eps = max(1e-12, y_min_pos * 1e-3)

    fig, ax = plt.subplots(figsize=(12, 6))
    for st, train_loss in series:
        yvals = train_loss.copy()
        yvals = np.where(yvals <= 0, eps, yvals)
        ax.semilogy(np.arange(1, len(yvals)+1), yvals, '-', color=SENSOR_COLORS.get(st, 'k'),
                    linewidth=2.0, label=f"Датчик {st} ({all_results[st]['taxel_count']} такс.)", alpha=0.9)

    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Ошибка обучения (логарифмический масштаб)')
    ax.set_title('Истории обучения (только train_loss) — совмещённый график по датчикам')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(title='Датчики', loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'training_losses_per_sensor.png', dpi=200)
    plt.close()
    print("✅ training_losses_per_sensor.png сохранён (semilogy)")

# ---------------- Report ----------------
def generate_report(all_results):
    rows = []
    for st in SENSOR_TYPES:
        if st not in all_results:
            continue
        r = all_results[st]
        noise_arr = np.array(r['pixel_noise_trials'], dtype=float)
        noise_arr = noise_arr[~np.isnan(noise_arr)]
        median_noise = float(np.median(noise_arr)) if noise_arr.size > 0 else float('nan')

        medians = []
        for n in range(1,6):
            arr = np.array(r['marker_loss_by_n'][n], dtype=float)
            if arr.size > 0:
                medians.append(float(np.median(arr)))
        agg_marker_median = float(np.median(medians)) if len(medians) > 0 else float('nan')

        row = {
            'датчик': st,
            'таксели': r['taxel_count'],
            'маркеров_M': r.get('marker_count', np.nan),
            'диаметр_mm': r['diameter_mm'],
            'R2': r['force_metrics'].get('r2', float('nan')),
            'RMSE_N': r['force_metrics'].get('rmse', float('nan')),
            'MAE_N': r['force_metrics'].get('mae', float('nan')),
            'медиана_силы_N': r['force_range_analysis']['median_force'],
            'максимум_силы_N': r['force_range_analysis']['max_force'],
            'marker_loss_median_avg_n1_5_px': agg_marker_median,
            'pixel_noise_median_relerr': median_noise,
            'has_train_history': bool(r.get('train_history') is not None)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'comprehensive_metrics_v11.csv', index=False, encoding='utf-8-sig')
    df.to_excel(TABLES_DIR / 'comprehensive_metrics_v11.xlsx', index=False)
    with open(RESULTS_DIR / 'analysis_report_v11.txt', 'w', encoding='utf-8') as f:
        f.write(f"Отчёт анализа v11 — {datetime.now().isoformat()}\n\n")
        f.write(df.to_string(index=False))
    print("✅ Отчёты сохранены (.csv, .xlsx, .txt)")

# ---------------- Main ----------------
def main():
    print("="*80)
    print("ANALYSIS v11.6-mod — узкие виолончельные диаграммы + линия тенденции выбросов (подписи на русском)")
    print("="*80)
    all_results = load_all_sensors()
    if not all_results:
        print("Нет результатов — выход.")
        return
    plot_regression_metrics(all_results)
    plot_marker_loss_grouped_bars(all_results, max_n=5)
    plot_pixel_noise_distributions(all_results, clip_percentile=95, max_points_per_sensor=400)
    plot_force_range_curves(all_results)
    plot_metrics_scatter(all_results)
    plot_training_losses(all_results)
    generate_report(all_results)
    print("\nГотово. Результаты: ", RESULTS_DIR)
    print("Графики: ", PLOTS_DIR)
    print("Таблицы: ", TABLES_DIR)

if __name__ == "__main__":
    main()
