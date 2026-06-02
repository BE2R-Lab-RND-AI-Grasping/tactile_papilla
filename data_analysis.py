#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АНАЛИЗ: РЕГРЕССИЯ ПРОТИВ ЛОКАЛИЗАЦИИ — v12.1 (обновлён)
- Русские подписи, короткие названия графиков.
- Модификации:
* marker_loss_grouped_bars_n1_5 строит кривые A..E по процентам потерь (10..70%).
* force_histograms логирует µ, σ и их нормированные значения (min->0, max->1).
* marker_loss_grouped_bars_n1_5 логирует медианы и их нормировку 0..1 (по всем сенсорам и процентам).
* plot_pixel_noise_summary_colored логирует нормировку "внешних полос" (середина полосы -> 0..1).
- Автор адаптации: Егор Ракшин / адаптация
- Версия: 12.1 — 2026 (обновлён для: 1) новый scatter: "Компромисс качества регрессии и локализации" с осями в пикселях и медианами; 2) все графики сохраняются в .png и .svg)
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
import matplotlib.patheffects as path_effects
from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
# ---------------- Конфигурация ----------------
SENSOR_TYPES = ['A', 'B', 'C', 'D', 'E']
SENSOR_GEOMETRY = {
'A': {'taxel_count': 12, 'diameter': 8.69, 'name': 'A'},
'B': {'taxel_count': 15, 'diameter': 6.94, 'name': 'B'},
'C': {'taxel_count': 20, 'diameter': 6.47, 'name': 'C'},
'D': {'taxel_count': 24, 'diameter': 5.75, 'name': 'D'},
'E': {'taxel_count': 30, 'diameter': 5.14, 'name': 'E'}
}
BASE_DIR = Path("Datasets")
LABELLED_DIR = BASE_DIR / "Labelled"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("analysis_results_v12_rewrite")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
# Цвета для сенсоров
SENSOR_COLORS = {
'A': '#1f77b4',
'B': '#ff7f0e',
'C': '#2ca02c',
'D': '#d62728',
'E': '#9467bd'
}
# ---------------- Глобальная конфигурация шрифтов (увеличенные) ----------------
plt.rcParams.update({
'figure.figsize': (11, 7),
'figure.dpi': 150,
'font.family': 'DejaVu Sans',
'axes.titlesize': 16,      # увеличено с 14
'axes.labelsize': 14,      # увеличено с 12
'xtick.labelsize': 12,     # увеличено с 10
'ytick.labelsize': 12,     # увеличено с 10
'legend.fontsize': 11,     # увеличено с 10
'grid.alpha': 0.25,
'lines.linewidth': 2.0,
'lines.markersize': 6
})
warnings.filterwarnings('ignore')
RND = 42
np.random.seed(RND)
random.seed(RND)
# ---------------- Утилиты ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
def save_fig(fig, name_base):
    """Save figure as PNG and SVG into PLOTS_DIR and close it."""
    out_png = PLOTS_DIR / f"{name_base}.png"
    out_svg = PLOTS_DIR / f"{name_base}.svg"
    out_pdf = PLOTS_DIR / f"{name_base}.pdf"
    try:
        fig.savefig(out_png, dpi=200)
        fig.savefig(out_svg, dpi=200)
        fig.savefig(out_pdf, dpi=200)
    except Exception as e:
        print(f"[WARN] Error saving {name_base}: {e}")
    plt.close(fig)
    print(f"✅ {out_png.name} saved")
    print(f"✅ {out_svg.name} saved")
    print(f"✅ {out_pdf.name} saved")
def lhs_sample(trials, dims):
    result = np.zeros((trials, dims))
    rng = np.random.RandomState(RND)
    for d in range(dims):
        strata = (np.arange(trials) + rng.uniform(size=trials)) / trials
        rng.shuffle(strata)
        result[:, d] = strata
    return result
# ---------------- Загрузчики данных ----------------
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
            'diameter': SENSOR_GEOMETRY[sensor_type]['diameter']
        }
    except Exception as e:
        print(f"❌ Ошибка загрузки {sensor_type}: {e}")
        return None
def load_trained_model(sensor_type):
    try:
        model_dir = MODELS_DIR / f"{sensor_type}_deformation_vectors_force_prediction"
        if not model_dir.exists():
            print(f"[WARN] Модель не найдена: {model_dir}")
            return None
        model_path = model_dir / "force_prediction_model.pth"
        if not model_path.exists():
            model_path = model_dir / "model.pth"
        if not model_path.exists():
            print(f"[WARN] Файл модели не найден в {model_dir}")
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
            print(f"[WARN] Не найден state_dict в чекпойнте для {sensor_type}. Пропускаю загрузку модели.")
            return None
        weight_keys = [k for k in state.keys() if k.endswith('.weight')]
        if not weight_keys:
            print(f"[WARN] Не найден .weight в state_dict для {sensor_type}")
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
            print(f"[WARN] Не удалось загрузить веса модели {sensor_type}: {e}")
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
        print(f"[WARN] Ошибка load_trained_model({sensor_type}): {e}")
        return None
# ---------------- Базовые метрики ----------------
def calculate_force_metrics(predictions, actuals):
    try:
        r2 = r2_score(actuals, predictions)
    except Exception:
        r2 = float('nan')
    rmse = math.sqrt(mean_squared_error(actuals, predictions)) if len(actuals) > 0 else float('nan')
    mae = mean_absolute_error(actuals, predictions) if len(actuals) > 0 else float('nan')
    return {'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae), 'n_samples': len(actuals)}
# ---------------- Потеря маркеров — процентный вариант ----------------
def simulate_marker_loss_by_percentage(deformations, reference_P, percents=None, repeats_per_sample=50):
    """
    Симуляция потерь маркеров как процентов от всех маркеров.
    - deformations: np.array (N, m, 2)
    - reference_P: np.array (m, 2)
    - percents: list of integers процентов (например [10,20,...,70])
    - repeats_per_sample: количество случайных конфигураций потерь для каждого исходного образца
    Возвращает словарь: percent -> np.array(errors) длины N * repeats_per_sample
    """
    if percents is None:
        percents = [10,20,30,40,50,60,70]
    N, m, _ = deformations.shape
    deformation_magnitudes = np.linalg.norm(deformations, axis=2)  # (N, m)
    true_centroids = np.zeros((N, 2))
    for i in range(N):
        w = deformation_magnitudes[i]
        if w.sum() > 1e-10:
            true_centroids[i] = np.average(reference_P, axis=0, weights=w)
        else:
            true_centroids[i] = reference_P.mean(axis=0)
    results = {}
    rng_global = np.random.RandomState(RND)
    for pct in percents:
        k_lost = int(round((pct / 100.0) * m))
        k_lost = max(1, min(k_lost, m-1))
        errors_list = []
        for i in range(N):
            w_i = deformation_magnitudes[i]
            pos = reference_P
            for rep in range(repeats_per_sample):
                lost = rng_global.choice(m, size=k_lost, replace=False)
                remaining = np.setdiff1d(np.arange(m), lost)
                w_rem = w_i[remaining]
                pos_rem = pos[remaining]
                if w_rem.sum() > 1e-10:
                    centroid_partial = np.average(pos_rem, axis=0, weights=w_rem)
                else:
                    centroid_partial = pos_rem.mean(axis=0)
                err = float(np.linalg.norm(true_centroids[i] - centroid_partial))
                errors_list.append(err)
        results[int(pct)] = np.array(errors_list, dtype=float)
        print(f"  simulated pct={pct}%: lost k={k_lost}, errors_count={len(errors_list)}")
    return results
# ---------------- Чувствительность к пиксельному шуму (LHS) ----------------
def simulate_noise_pixel_lhs(deformations, forces, model_info, trials=1000, pixel_range=50):
    N, m, _ = deformations.shape
    if model_info is None:
        return np.full(trials, np.nan), None
    model = model_info['model']
    scaler = model_info.get('scaler_info', None)
    deformation_flat = deformations.reshape(N, -1)
    total_l2 = np.linalg.norm(deformations, axis=(1, 2))
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
# ---------------- Утилиты для графиков ----------------
def force_histograms_for_sensors(forces_dict, bins=None):
    """
    Построение столбчатой гистограммы по датчикам.
    Возвращает centers, hist_list, bins, а также статистику (mu, sigma) для каждого набора.
    """
    overall_max = 0.0
    for f in forces_dict:
        if len(f) > 0:
            overall_max = max(overall_max, float(np.max(f)))
    if bins is None:
        max_edge = math.ceil(overall_max + 1e-9) if overall_max > 0 else 10
        max_edge = int(math.ceil(max_edge / 3.0) * 3)
        bins = np.arange(0.0, max_edge + 3.0, 3.0)  # шаг = 3 Н
    hist_list = []
    mus = []
    sigmas = []
    for f in forces_dict:
        if len(f) == 0:
            hist_list.append(np.zeros(len(bins) - 1))
            mus.append(float('nan'))
            sigmas.append(float('nan'))
            continue
        hist, _ = np.histogram(f, bins=bins)
        hist_list.append(hist)
        mus.append(float(np.mean(f)))
        sigmas.append(float(np.std(f, ddof=1) if len(f) > 1 else 0.0))
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, hist_list, bins, np.array(mus), np.array(sigmas)
# ---------------- Анализ одного датчика ----------------
def analyze_sensor_comprehensive(sensor_type,
                                 percent_list=None,
                                 repeats_per_sample=50,
                                 noise_pixel_trials=1000,
                                 pixel_noise_range=50):
    """
    Анализ одного сенсора. Включает:
    - загрузку агрегированного датасета
    - расчёты регрессии (если модель есть)
    - симуляцию потерь маркеров в процентном виде (10..70%)
    - симуляцию пиксельного шума (LHS)
    Возвращает словарь results.
    """
    if percent_list is None:
        percent_list = [10,20,30,40,50,60,70]
    print("\n" + "=" * 60)
    print(f"АНАЛИЗ ДАТЧИКА {sensor_type}")
    print("=" * 60)
    ds = load_aggregated_dataset(sensor_type)
    if ds is None:
        return None
    model_info = load_trained_model(sensor_type)
    results = {'sensor_type': sensor_type,
               'taxel_count': ds['taxel_count'],
               'diameter_mm': ds['diameter'],
               'n_samples': ds['deformations'].shape[0]}
    deformations = ds['deformations']
    forces = ds['forces']
    total_l2_norms = ds['total_l2_norms']
    reference_P = ds['reference_P']
    # regression (if model available)
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
        print(f"    Коэффициент детерминации R2={fm['r2']:.4f}, RMSE={fm['rmse']:.4f}, MAE={fm['mae']:.4f}")
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
    # simulate marker loss as percentage
    print(f" → Симуляция процентных потерь маркеров: percents={percent_list}, repeats_per_sample={repeats_per_sample}")
    pct_results = simulate_marker_loss_by_percentage(deformations, reference_P, percents=percent_list, repeats_per_sample=repeats_per_sample)
    results['marker_loss_by_pct'] = {int(k): v.tolist() for k, v in pct_results.items()}
    # For backward compatibility, compute small marker_loss_by_n (0..5)
    try:
        N, m, _ = deformations.shape
        deformation_magnitudes = np.linalg.norm(deformations, axis=2)
        true_centroids = np.zeros((N, 2))
        for i in range(N):
            w = deformation_magnitudes[i]
            if w.sum() > 1e-10:
                true_centroids[i] = np.average(reference_P, axis=0, weights=w)
            else:
                true_centroids[i] = reference_P.mean(axis=0)
        results['marker_loss_by_n'] = {}
        for n in range(0, 6):
            errs = []
            trials_local = min(200, max(50, int(np.ceil(N * 0.02))))
            rng = np.random.RandomState(RND + n)
            for t in range(trials_local):
                idx = rng.randint(0, N)
                if n == 0:
                    err = 0.0
                else:
                    if n >= m:
                        remaining = np.arange(m - 1)
                    else:
                        lost = rng.choice(m, size=n, replace=False)
                        remaining = np.setdiff1d(np.arange(m), lost)
                    w_rem = deformation_magnitudes[idx][remaining]
                    pos_rem = reference_P[remaining]
                    if w_rem.sum() > 1e-10:
                        centroid_partial = np.average(pos_rem, axis=0, weights=w_rem)
                    else:
                        centroid_partial = pos_rem.mean(axis=0)
                    err = float(np.linalg.norm(true_centroids[idx] - centroid_partial))
                errs.append(err)
            results['marker_loss_by_n'][n] = np.array(errs, dtype=float).tolist()
    except Exception:
        results['marker_loss_by_n'] = {}
    # pixel noise simulation (LHS)
    print(f" → Симуляция пиксельного шума (LHS): trials={noise_pixel_trials}, range=±{pixel_noise_range} пикс.")
    noise_errs, sample_idxs = simulate_noise_pixel_lhs(deformations, forces, model_info, trials=noise_pixel_trials, pixel_range=pixel_noise_range)
    results['pixel_noise_trials'] = noise_errs.tolist()
    results['pixel_noise_sample_idxs'] = None if sample_idxs is None else sample_idxs.tolist()
    print(f"    Пиксельный шум: медиана относит. ошибки = {np.nanmedian(noise_errs):.4f}")
    if model_info is not None and model_info.get('train_history') is not None:
        results['train_history'] = model_info['train_history']
    else:
        results['train_history'] = None
    return results
# ---------------- Графики ----------------
def plot_force_histograms(all_results):
    forces_list = []
    labels = []
    colors = []
    for st in SENSOR_TYPES:
        if st not in all_results:
            continue
        arr = np.array(all_results[st]['actual_forces'], dtype=float)
        forces_list.append(arr)
        labels.append(st)
        colors.append(SENSOR_COLORS[st])
    if not forces_list:
        print("[WARN] Нет данных сил для гистограммы.")
        return
    centers, hist_list, bins, mus, sigmas = force_histograms_for_sensors(forces_list)
    bin_width = bins[1] - bins[0]  # здесь 3 Н
    width = bin_width * 0.8 / len(hist_list)
    # УЗКИЙ ПО ВЕРТИКАЛИ (было 7 → 3.5)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i, (hist, col, lab) in enumerate(zip(hist_list, colors, labels)):
        offsets = centers - 0.4 * bin_width + i * width + width / 2.0
        ax.bar(offsets, hist, width=width, alpha=0.85, label=lab, color=col, edgecolor='black')
    ax.set_xlabel('Сила, Н')
    ax.set_ylabel('Число наблюдений')
    ax.set_title('Гистограмма сил по датчикам')
    overall_max = bins[-1]
    major_ticks = np.arange(0, overall_max + 1, 5)  # метки каждые 5 Н
    minor_ticks = np.arange(0, overall_max + 1, 1)  # тики каждые 1 Н
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', which='minor', alpha=0.12)
    ax.legend(title='Датчик')
    plt.tight_layout()
    save_fig(fig, 'force_histograms')
    # --- Логи: µ и σ и их нормировки ---
    mu_vals = mus  # numpy array
    sigma_vals = sigmas
    # handle NaNs by ignoring them in min/max
    valid_mu_mask = np.isfinite(mu_vals)
    valid_sigma_mask = np.isfinite(sigma_vals)
    mu_min = float(np.nanmin(mu_vals)) if np.any(valid_mu_mask) else 0.0
    mu_max = float(np.nanmax(mu_vals)) if np.any(valid_mu_mask) else 0.0
    sigma_min = float(np.nanmin(sigma_vals)) if np.any(valid_sigma_mask) else 0.0
    sigma_max = float(np.nanmax(sigma_vals)) if np.any(valid_sigma_mask) else 0.0
    mu_range = mu_max - mu_min
    sigma_range = sigma_max - sigma_min
    normalized_mu = []
    normalized_sigma = []
    for mu in mu_vals:
        if not np.isfinite(mu) or mu_range == 0:
            normalized_mu.append(float('nan'))
        else:
            normalized_mu.append((mu - mu_min) / mu_range)
    for s in sigma_vals:
        if not np.isfinite(s) or sigma_range == 0:
            normalized_sigma.append(float('nan'))
        else:
            normalized_sigma.append((s - sigma_min) / sigma_range)
    # Log nicely
    print("\n[LOG] Force histograms statistics (raw and normalized):")
    print("Sensor | mu (N) | mu_norm(0..1) | sigma (N) | sigma_norm(0..1)")
    for i, lab in enumerate(labels):
        mu_raw = mu_vals[i]
        s_raw = sigma_vals[i]
        mu_n = normalized_mu[i]
        s_n = normalized_sigma[i]
        mu_raw_s = f"{mu_raw:.4f}" if np.isfinite(mu_raw) else "N/A"
        s_raw_s = f"{s_raw:.4f}" if np.isfinite(s_raw) else "N/A"
        mu_n_s = f"{mu_n:.4f}" if np.isfinite(mu_n) else "N/A"
        s_n_s = f"{s_n:.4f}" if np.isfinite(s_n) else "N/A"
        print(f"  {lab:>2s}   | {mu_raw_s:>7s} | {mu_n_s:>11s} | {s_raw_s:>8s} | {s_n_s:>12s}")
    print(f"  (mu_min={mu_min:.4f}, mu_max={mu_max:.4f}, sigma_min={sigma_min:.4f}, sigma_max={sigma_max:.4f})\n")
def plot_metrics_scatter_no_trends(all_results, percent_list=None):
    """
    Обновлённый scatter: строит "Компромисс качества регрессии и локализации".
    X (абсцисс) = усреднённая медиана ошибки локализации по всем percent_list (пкс).
    Y (ординат) = медиана относительной ошибки регрессора (пкс) из pixel_noise_trials.
    """
    if percent_list is None:
        percent_list = [10,20,30,40,50,60,70]
    x_vals = []
    y_vals = []
    labels = []
    colors = []
    diameters = []
    for st in SENSOR_TYPES:
        if st not in all_results:
            continue
        r = all_results[st]
        # collect medians for the percent_list
        medians = []
        for pct in percent_list:
            arr = np.array(r.get('marker_loss_by_pct', {}).get(int(pct), []), dtype=float)
            arr = arr[~np.isnan(arr)] if arr.size > 0 else np.array([])
            if arr.size > 0:
                medians.append(float(np.median(arr)))
        if len(medians) == 0:
            # skip sensor if no marker-loss data
            continue
        mean_median = float(np.mean(np.array(medians)))
        # Y: median of pixel noise trials (use median of pixel_noise_trials)
        noise_arr = np.array(r.get('pixel_noise_trials', []), dtype=float)
        noise_arr = noise_arr[~np.isnan(noise_arr)] if noise_arr.size>0 else noise_arr
        if noise_arr.size == 0:
            continue
        noise_med = float(np.median(noise_arr))
        x_vals.append(mean_median)
        y_vals.append(noise_med)
        labels.append(st)
        colors.append(SENSOR_COLORS[st])
        diam = float(r.get('diameter_mm', SENSOR_GEOMETRY[st]['diameter']))
        diameters.append(diam)
    if not x_vals:
        print("[WARN] Нет точек для разброса")
        return
    x = np.array(x_vals)
    y = np.array(y_vals)
    diam_array = np.array(diameters)
    diam_E = SENSOR_GEOMETRY['E']['diameter']
    base_marker_diameter_pt = 14.0
    diam_ratios = diam_array / float(diam_E)
    sizes_plot = (base_marker_diameter_pt * diam_ratios) ** 2
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(x, y, s=sizes_plot, c=colors, edgecolors='black', zorder=10, alpha=0.95)
    for i, lbl in enumerate(labels):
        tx = x[i]; ty = y[i]
        txt = ax.text(tx, ty, f"{lbl}", ha='center', va='center', fontsize=10, fontweight='bold', zorder=20, color='black')
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
    ax.set_xlabel('Усредненная ошибка локализации, пкс.')
    ax.set_ylabel('Абсолютная ошибка регрессора, пкс.')
    ax.set_title('Компромисс качества регрессии и локализации')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'metrics_scatter_compromise')
def plot_pixel_noise_summary_colored(all_results, max_points_jitter=100, jitter_amplitude=0.18, jitter_enabled=True):
    labels = []
    data = []
    colors = []
    p_low_vals = []
    p_high_vals = []
    medians = []
    for st in SENSOR_TYPES:
        if st not in all_results:
            continue
        errs = np.array(all_results[st]['pixel_noise_trials'], dtype=float)
        errs = errs[~np.isnan(errs)]
        labels.append(st)
        data.append(errs)
        colors.append(SENSOR_COLORS[st])
        if errs.size == 0:
            p_low_vals.append(float('nan'))
            p_high_vals.append(float('nan'))
            medians.append(float('nan'))
        else:
            p_low_vals.append(np.percentile(errs, 2.5))
            p_high_vals.append(np.percentile(errs, 97.5))
            medians.append(np.median(errs))
    if not data:
        print("[WARN] Нет данных для pixel_noise_summary")
        return
    n = len(data)
    # УЗКИЙ ПО ВЕРТИКАЛИ: высота уменьшена в 1.5 раза (было 7 → 4.67)
    fig, ax = plt.subplots(figsize=(5.5, 4.67))
    x_positions = np.arange(1, n + 1)
    q25 = np.zeros(n); q50 = np.zeros(n); q75 = np.zeros(n)
    for i, arr in enumerate(data):
        if arr.size == 0:
            q25[i] = q50[i] = q75[i] = np.nan
            continue
        q25[i] = np.percentile(arr, 25)
        q50[i] = np.percentile(arr, 50)
        q75[i] = np.percentile(arr, 75)
    p_low_vals = np.array(p_low_vals); p_high_vals = np.array(p_high_vals); medians = np.array(medians)
    for i in range(n):
        if np.isnan(p_low_vals[i]):
            continue
        ax.fill_between([x_positions[i] - 0.30, x_positions[i] + 0.30],
                         [p_low_vals[i], p_low_vals[i]],
                         [p_high_vals[i], p_high_vals[i]],
                         color=colors[i], alpha=0.16, linewidth=0)
        ax.fill_between([x_positions[i] - 0.20, x_positions[i] + 0.20],
                         [q25[i], q25[i]],
                         [q75[i], q75[i]],
                         color=colors[i], alpha=0.28, linewidth=0)
        ax.plot([x_positions[i] - 0.24, x_positions[i] + 0.24], [q50[i], q50[i]], color='k', linewidth=2)
    # Определяем максимальное значение для обрезки по Y
    all_vals = np.concatenate([arr for arr in data if arr.size > 0])
    if all_vals.size > 0:
        ymax = np.percentile(all_vals, 99)  # 99-й перцентиль для обрезки выбросов
        ax.set_ylim(bottom=0, top=ymax * 1.05)  # небольшой отступ сверху
    for i, arr in enumerate(data):
        if arr.size == 0:
            continue
        rng = np.random.RandomState(RND + i)
        nplot = min(arr.size, max_points_jitter)  # уменьшена плотность точек
        if arr.size > nplot:
            idxs = rng.choice(arr.size, size=nplot, replace=False)
            arr_plot = arr[idxs]
        else:
            arr_plot = arr
        # Джиттер ВКЛЮЧЁН по горизонтали (как в оригинале)
        if jitter_enabled:
            jitter = (rng.rand(len(arr_plot)) - 0.5) * jitter_amplitude
        else:
            jitter = np.zeros(len(arr_plot))
        ax.scatter(np.full_like(arr_plot, x_positions[i]) + jitter, arr_plot,
                   s=32, alpha=0.80, color=colors[i], edgecolors='black', linewidth=0.5, zorder=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{lab}" for lab in labels])
    ax.set_ylabel(r'Относительная ошибка регрессора $\eta$')
    ax.set_title('Чувствительность к пиксельному шуму')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'pixel_noise_summary_colored')
    # --- Логирование нормировки "полупрозрачных полос" ---
    mid_vals = (p_low_vals + p_high_vals) / 2.0
    valid_mask = np.isfinite(mid_vals)
    if np.any(valid_mask):
        min_mid = float(np.nanmin(mid_vals))
        max_mid = float(np.nanmax(mid_vals))
        range_mid = max_mid - min_mid
        print("\n[LOG] Pixel-noise outer-band middles and normalized (0..1):")
        print("Sensor | mid_outer | mid_norm")
        for i, lab in enumerate(labels):
            mid = mid_vals[i]
            if not np.isfinite(mid) or range_mid == 0:
                mid_norm = float('nan')
            else:
                mid_norm = (mid - min_mid) / range_mid
            mid_s = f"{mid:.6f}" if np.isfinite(mid) else "N/A"
            mid_n_s = f"{mid_norm:.4f}" if np.isfinite(mid_norm) else "N/A"
            print(f"  {lab:>2s}   | {mid_s:>10s} | {mid_n_s:>7s}")
        print(f"  (mid_min={min_mid:.6f}, mid_max={max_mid:.6f})\n")
    else:
        print("[LOG] Нет валидных значений для нормировки pixel-noise outer-band.\n")
# ---------------- График: marker_loss_grouped_bars_n1_5 (линии по процентам) ----------------
def marker_loss_grouped_bars_n1_5(all_results, percent_list=None, repeats_label='50x'):
    """
    Теперь строим КРИВЫЕ (линии) для каждого сенсора по процентам потерь.
    Также логируем медианные значения и их нормировку 0..1 (по всем сенсорам и процентам).
    """
    if percent_list is None:
        percent_list = [10,20,30,40,50,60,70]
    sensors = [s for s in SENSOR_TYPES if s in all_results]
    if not sensors:
        print("[WARN] Нет данных для marker_loss_grouped_bars_n1_5")
        return
    # collect medians matrix: shape (n_sensors, n_percents)
    medians_matrix = []
    counts_matrix = []
    for s in sensors:
        row = []
        cnt_row = []
        for pct in percent_list:
            arr = np.array(all_results[s].get('marker_loss_by_pct', {}).get(int(pct), []), dtype=float)
            arr = arr[~np.isnan(arr)] if arr.size > 0 else np.array([])
            med = float(np.median(arr)) if arr.size > 0 else float('nan')
            row.append(med)
            cnt_row.append(int(arr.size) if arr.size>0 else 0)
        medians_matrix.append(row)
        counts_matrix.append(cnt_row)
    medians_matrix = np.array(medians_matrix)  # (n_sensors, n_groups)
    counts_matrix = np.array(counts_matrix)
    # Normalize medians for logging (global min->0, max->1)
    flat = medians_matrix.flatten()
    finite_mask = np.isfinite(flat)
    if np.any(finite_mask):
        min_all = float(np.nanmin(flat))
        max_all = float(np.nanmax(flat))
        range_all = max_all - min_all
        if range_all == 0:
            norm_matrix = np.full_like(medians_matrix, 0.0)
        else:
            norm_matrix = (medians_matrix - min_all) / range_all
    else:
        norm_matrix = np.full_like(medians_matrix, np.nan)
    # Plotting: lines per sensor
    # УЗКИЙ ПО ВЕРТИКАЛИ: высота уменьшена в 1.5 раза (было 6 → 4.0)
    fig, ax = plt.subplots(figsize=(5, 4.0))
    x = np.array(percent_list)
    for i, s in enumerate(sensors):
        y = medians_matrix[i]
        # For plotting, mask NaNs so lines don't connect across NaNs
        mask = np.isfinite(y)
        if np.sum(mask) == 0:
            continue
        ax.plot(x[mask], y[mask], marker='o', linestyle='-', color=SENSOR_COLORS.get(s, '#777777'),
                label=s, linewidth=2, markersize=6, zorder=5)
    ax.set_xlabel('Потеря маркеров, %')
    ax.set_ylabel('Ошибка локализации, пкс.')
    ax.set_title('Робастность к процентной потере маркеров')
    ax.set_xticks(percent_list)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(title='Sensor')
    plt.tight_layout()
    save_fig(fig, 'marker_loss_grouped_pct_lines_10_70')
    # --- Логирование: печатаем исходные медианы и нормированные 0..1 ---
    print("\n[LOG] Marker-loss medians (raw) by sensor and percent:")
    header = "Sensor | " + " | ".join([f"{p:>3d}%" for p in percent_list])
    print(header)
    for i, s in enumerate(sensors):
        row_vals = medians_matrix[i]
        row_str = " | ".join([f"{v:.4f}" if np.isfinite(v) else "N/A   " for v in row_vals])
        print(f"  {s:>2s}   | {row_str}")
    print("\n[LOG] Marker-loss medians normalized to [0..1] across ALL sensors & percents:")
    header = "Sensor | " + " | ".join([f"{p:>3d}%" for p in percent_list])
    print(header)
    for i, s in enumerate(sensors):
        row_vals = norm_matrix[i]
        row_str = " | ".join([f"{v:.4f}" if np.isfinite(v) else "N/A   " for v in row_vals])
        print(f"  {s:>2s}   | {row_str}")
    print(f"  (global min = {min_all:.6f}, global max = {max_all:.6f})\n")
# ---------------- Дополнительные графики и отчёты ----------------
def plot_regression_metrics(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sensor_labels = [st for st in SENSOR_TYPES if st in all_results]
    colors = [SENSOR_COLORS[st] for st in sensor_labels]
    r2_values = [all_results[st]['force_metrics']['r2'] for st in sensor_labels]
    bars = axes[0].bar(range(len(sensor_labels)), r2_values, color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(sensor_labels)))
    axes[0].set_xticklabels([f"{lbl}" for lbl in sensor_labels])
    axes[0].set_ylabel('R2')
    axes[0].set_title('Коэфф. детерм. R2')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].set_ylim(0.0, 1.0)
    for bar, val in zip(bars, r2_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, (val if not np.isnan(val) else 0) + 0.02,
                     f"{val:.3f}" if not np.isnan(val) else "N/A", ha='center', fontsize=9)
    mae_values = [all_results[st]['force_metrics']['mae'] for st in sensor_labels]
    bars = axes[1].bar(range(len(sensor_labels)), mae_values, color=colors, edgecolor='black')
    axes[1].set_xticks(range(len(sensor_labels)))
    axes[1].set_xticklabels([f"{lbl}" for lbl in sensor_labels])
    axes[1].set_ylabel('MAE, Н')
    axes[1].set_title('Средняя абсолютная ошибка')
    axes[1].grid(axis='y', alpha=0.25)
    for bar, val in zip(bars, mae_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, (val if not np.isnan(val) else 0) + 0.02,
                     f"{val:.3f}" if not np.isnan(val) else "N/A", ha='center', fontsize=9)
    rmse_values = [all_results[st]['force_metrics']['rmse'] for st in sensor_labels]
    bars = axes[2].bar(range(len(sensor_labels)), rmse_values, color=colors, edgecolor='black')
    axes[2].set_xticks(range(len(sensor_labels)))
    axes[2].set_xticklabels([f"{lbl}" for lbl in sensor_labels])
    axes[2].set_ylabel('RMSE, Н')
    axes[2].set_title('Квадр.средняя ошибка')
    axes[2].grid(axis='y', alpha=0.25)
    for bar, val in zip(bars, rmse_values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, (val if not np.isnan(val) else 0) + 0.02,
                     f"{val:.3f}" if not np.isnan(val) else "N/A", ha='center', fontsize=9)
    plt.tight_layout()
    save_fig(fig, 'regression_metrics')
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
        for pct in [10,20,30,40,50]:
            arr = np.array(r.get('marker_loss_by_pct', {}).get(pct, []), dtype=float)
            if arr.size > 0:
                medians.append(float(np.median(arr)))
        agg_marker_median = float(np.median(medians)) if len(medians) > 0 else float('nan')
        row = {
            'sensor': st,
            'taxels': r['taxel_count'],
            'diameter_mm': r['diameter_mm'],
            'R2': r['force_metrics'].get('r2', float('nan')),
            'RMSE_N': r['force_metrics'].get('rmse', float('nan')),
            'MAE_N': r['force_metrics'].get('mae', float('nan')),
            'force_median_N': r['force_range_analysis']['median_force'],
            'force_max_N': r['force_range_analysis']['max_force'],
            'marker_loss_median_avg_pct10_50_px': agg_marker_median,
            'pixel_noise_median_relerr': median_noise,
            'has_train_history': bool(r.get('train_history') is not None)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'comprehensive_metrics_v12_rewrite.csv', index=False, encoding='utf-8-sig')
    df.to_excel(TABLES_DIR / 'comprehensive_metrics_v12_rewrite.xlsx', index=False)
    with open(RESULTS_DIR / 'analysis_report_v12_rewrite.txt', 'w', encoding='utf-8') as f:
        f.write(f"Analysis report v12 rewrite — {datetime.now().isoformat()}\n")
        f.write(df.to_string(index=False))
    print("✅ Reports saved (.csv, .xlsx, .txt)")
# ---------------- Главная функция ----------------
def main():
    print("=" * 80)
    print("ANALYSIS v12 — Regression vs Localization (rewrite) — marker loss by percent")
    print("=" * 80)
    all_results = {}
    percent_list = [10,20,30,40,50,60,70]
    repeats_per_sample = 50
    for st in SENSOR_TYPES:
        res = analyze_sensor_comprehensive(st,
                                           percent_list=percent_list,
                                           repeats_per_sample=repeats_per_sample,
                                           noise_pixel_trials=1000,
                                           pixel_noise_range=50)
        if res is not None:
            all_results[st] = res
    if not all_results:
        print("No results — exit.")
        return
    # other graphs
    plot_regression_metrics(all_results)
    plot_force_histograms(all_results)
    plot_metrics_scatter_no_trends(all_results, percent_list=percent_list)
    # updated marker-loss plot: lines per sensor
    marker_loss_grouped_bars_n1_5(all_results, percent_list, repeats_label=f"{repeats_per_sample}x")
    plot_pixel_noise_summary_colored(all_results, max_points_jitter=100, jitter_amplitude=0.18, jitter_enabled=True)
    generate_report(all_results)
    print("\nDone. Results in:", RESULTS_DIR)
    print("Plots in:", PLOTS_DIR)
    print("Tables in:", TABLES_DIR)
if __name__ == '__main__':
    main()