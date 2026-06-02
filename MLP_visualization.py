#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Visualization and Analysis (robust, bugfixed version)

- Scans model directories under BASE_DIR for trained model folders for each sensor.
- Loads training history, model_info, prediction arrays (if present) with multiple fallbacks.
- Computes robust metrics and parameter counts (can infer from saved .pth state_dict).
- Produces a set of publication-quality visualizations and a summary table.
- Saves all outputs to OUTPUT_DIR.

Usage: run the script in the project root where `models/` is located, or modify BASE_DIR.
"""
from pathlib import Path
import sys
import os
import json
import re
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from scipy import stats
from matplotlib.gridspec import GridSpec

# -----------------------
# Plot style / rcParams
# -----------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'errorbar.capsize': 2.0
})

np.random.seed(42)

# -----------------------
# Configuration
# -----------------------
BASE_DIR = Path("models")                       # where model folders live
OUTPUT_DIR = Path("results") / "visualizations" # where to put plots
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Which sensors to try to visualize (can be subset)
SENSOR_TYPES = ["A", "B", "C", "D", "E"]

# Force ranges for error-by-range plots (0..10 N in 0.5 steps by default)
FORCE_RANGES = [(i*0.5, (i+1)*0.5) for i in range(20)]

# Color palette
COLORS = {
    "A": "#1f77b4",  # Blue
    "B": "#ff7f0e",  # Orange
    "C": "#2ca02c",  # Green
    "D": "#d62728",  # Red
    "E": "#9467bd",  # Purple
    "background": "#f8f9fa",
    "grid": "#e9ecef",
    "text": "#343a40"
}

# -----------------------
# Helpers
# -----------------------
def to_serializable(obj):
    """Convert numpy types to Python native ones for JSON dumps."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj

def find_model_directory_for_sensor(sensor):
    """
    Heuristic search for a model directory for a given sensor under BASE_DIR.
    Looks for directories whose name starts with "<sensor>_" and contains "force_prediction"
    or whose name contains both sensor and "force_prediction". Returns Path or None.
    """
    if not BASE_DIR.exists():
        return None
    candidates = []
    for p in BASE_DIR.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if name.startswith(sensor.lower() + "_") and "force_prediction" in name:
            candidates.append(p)
        elif sensor.lower() in name and "force_prediction" in name:
            candidates.append(p)
        elif re.match(rf"^{sensor.lower()}_", name):
            # possible older folder naming
            candidates.append(p)
    # If exact matches not found, fallback to any folder that contains sensor char and "force"
    if not candidates:
        for p in BASE_DIR.iterdir():
            if not p.is_dir():
                continue
            name = p.name.lower()
            if sensor.lower() in name and "force" in name:
                candidates.append(p)
    # Return most relevant (prefer those with 'allmembranes' or 'force_prediction' explicitly)
    if not candidates:
        return None
    candidates_sorted = sorted(candidates, key=lambda x: (("force_prediction" in x.name.lower()) * -1,
                                                         ("allmembranes" in x.name.lower()) * -1,
                                                         len(x.name)))
    return candidates_sorted[0]

def safe_load_json(path):
    try:
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception:
        return None

def try_load_predictions_from_files(model_dir):
    """
    Try to locate prediction/actual arrays in a variety of file names/formats.
    Returns (actuals, predictions) or (None, None).
    """
    possible_files = [
        "predictions_vs_actuals.json",
        "predictions.json",
        "evaluation_predictions.json",
        "eval_predictions.json",
        "validation_predictions.json",
        "preds_actuals.json"
    ]
    for fname in possible_files:
        p = model_dir / fname
        if p.exists():
            obj = safe_load_json(p)
            if obj is None:
                continue
            # try common key names
            for a_keys in (("actuals", "predictions"), ("y_true", "y_pred"), ("actual", "pred"), ("validation_actuals","validation_predictions")):
                if all(k in obj for k in a_keys):
                    try:
                        actuals = np.array(obj[a_keys[0]], dtype=float).flatten()
                        predictions = np.array(obj[a_keys[1]], dtype=float).flatten()
                        return actuals, predictions
                    except Exception:
                        pass
            # try if file is a dict with 'data' list of pairs
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                data = obj["data"]
                try:
                    actuals = np.array([d[0] for d in data], dtype=float)
                    predictions = np.array([d[1] for d in data], dtype=float)
                    return actuals, predictions
                except Exception:
                    pass

    # check training_history.json and model_info.json
    hist = safe_load_json(model_dir / "training_history.json")
    if hist:
        for candidate_pairs in (("actuals","predictions"), ("validation_actuals","validation_predictions"), ("predictions","actuals")):
            if candidate_pairs[0] in hist and candidate_pairs[1] in hist:
                try:
                    actuals = np.array(hist[candidate_pairs[0]], dtype=float).flatten()
                    predictions = np.array(hist[candidate_pairs[1]], dtype=float).flatten()
                    return actuals, predictions
                except Exception:
                    pass
        if "evaluation" in hist and isinstance(hist["evaluation"], dict):
            evald = hist["evaluation"]
            if "actuals" in evald and "predictions" in evald:
                try:
                    actuals = np.array(evald["actuals"], dtype=float).flatten()
                    predictions = np.array(evald["predictions"], dtype=float).flatten()
                    return actuals, predictions
                except Exception:
                    pass

    mi = safe_load_json(model_dir / "model_info.json")
    if mi and "evaluation" in mi and isinstance(mi["evaluation"], dict):
        evald = mi["evaluation"]
        if "actuals" in evald and "predictions" in evald:
            try:
                actuals = np.array(evald["actuals"], dtype=float).flatten()
                predictions = np.array(evald["predictions"], dtype=float).flatten()
                return actuals, predictions
            except Exception:
                pass

    # try any JSON in folder that contains both arrays
    for p in model_dir.glob("*.json"):
        obj = safe_load_json(p)
        if not obj:
            continue
        if "actuals" in obj and "predictions" in obj:
            try:
                actuals = np.array(obj["actuals"], dtype=float).flatten()
                predictions = np.array(obj["predictions"], dtype=float).flatten()
                return actuals, predictions
            except Exception:
                pass

    return None, None

def calculate_model_parameters_from_state_dict(state_dict):
    """
    Given a PyTorch state_dict, estimate total number of parameters (weights + biases).
    """
    if state_dict is None:
        return 0
    total = 0
    try:
        for k, v in state_dict.items():
            # v could be Tensor or numpy array or list
            try:
                shp = getattr(v, 'shape', None)
                if shp is not None:
                    total += int(np.prod(shp))
            except Exception:
                # skip if weird
                continue
        return int(total)
    except Exception:
        return 0

def try_load_state_dict_and_count_params(model_dir):
    """
    Try to find .pth/.pt files and compute parameter count from saved state_dict.
    Uses weights_only=True when supported to reduce security warning; falls back otherwise.
    Returns int param_count or 0.
    """
    param_count = 0
    pths = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt")) + list(model_dir.glob("*checkpoint*.pth"))
    if not pths:
        return 0
    pths_sorted = sorted(pths, key=lambda p: p.stat().st_size, reverse=True)
    for p in pths_sorted:
        try:
            # attempt to pass weights_only if available (newer torch)
            try:
                obj = torch.load(p, map_location='cpu', weights_only=True)
            except TypeError:
                # older torch doesn't support weights_only arg
                obj = torch.load(p, map_location='cpu')
            # obj might be dict with 'model_state_dict' or already a state_dict
            if isinstance(obj, dict):
                if 'model_state_dict' in obj and isinstance(obj['model_state_dict'], dict):
                    param_count = calculate_model_parameters_from_state_dict(obj['model_state_dict'])
                else:
                    # maybe it's directly a state_dict-like dict
                    param_count = calculate_model_parameters_from_state_dict(obj)
            else:
                # maybe a state_dict-like object
                param_count = calculate_model_parameters_from_state_dict(obj)
            if param_count > 0:
                return param_count
        except Exception:
            # skip unreadable checkpoint
            continue
    return 0

def calculate_model_parameters(hidden_layers, input_size):
    """Count parameters of simple feed-forward MLP given hidden layers list and input size."""
    try:
        param_count = 0
        prev = int(input_size)
        for h in hidden_layers:
            h = int(h)
            param_count += prev * h  # weights
            param_count += h         # biases
            prev = h
        # output layer
        param_count += prev  # weights to single output
        param_count += 1     # bias
        return int(param_count)
    except Exception:
        return 0

def get_force_range_errors(actuals, residuals, force_ranges):
    """Calculate error statistics per force range."""
    errors_by_range = {}
    if actuals is None or residuals is None:
        return errors_by_range
    actuals = np.array(actuals).flatten()
    residuals = np.array(residuals).flatten()
    for (rmin, rmax) in force_ranges:
        mask = (actuals >= rmin) & (actuals < rmax)
        rr = residuals[mask]
        if rr.size > 0:
            errors_by_range[(rmin, rmax)] = {
                "count": int(rr.size),
                "mae": float(np.mean(np.abs(rr))),
                "rmse": float(np.sqrt(np.mean(rr**2))),
                "std": float(np.std(rr)),
                "residuals": rr.copy()
            }
    return errors_by_range

# -----------------------
# Core loading
# -----------------------
def load_all_model_results():
    """
    Iterate SENSOR_TYPES, find model dir for each, then load history/model_info/predictions/etc.
    Returns dict mapping sensor -> info dict
    """
    results = {}
    print("Scanning for model directories and loading results...")
    for sensor in SENSOR_TYPES:
        try:
            model_dir = find_model_directory_for_sensor(sensor)
            if model_dir is None:
                print(f"Warning: Could not find model directory for sensor {sensor} under {BASE_DIR}")
                continue
            print(f"\nProcessing sensor {sensor} -> {model_dir}")

            history = safe_load_json(model_dir / "training_history.json") or {}
            model_info = safe_load_json(model_dir / "model_info.json") or {}

            # Try to load predictions/actuals
            actuals, predictions = try_load_predictions_from_files(model_dir)
            if actuals is None or predictions is None:
                # attempt to load stored eval arrays in history/model_info/evaluation
                if "evaluation" in history and isinstance(history["evaluation"], dict):
                    ev = history["evaluation"]
                    if "actuals" in ev and "predictions" in ev:
                        actuals = np.array(ev["actuals"], dtype=float).flatten()
                        predictions = np.array(ev["predictions"], dtype=float).flatten()
                if (actuals is None or predictions is None) and "evaluation" in model_info and isinstance(model_info["evaluation"], dict):
                    ev = model_info["evaluation"]
                    if "actuals" in ev and "predictions" in ev:
                        actuals = np.array(ev["actuals"], dtype=float).flatten()
                        predictions = np.array(ev["predictions"], dtype=float).flatten()

            # If still missing, create synthetic but mark as synthetic
            synthetic = False
            if actuals is None or predictions is None:
                synthetic = True
                print("  Warning: No predictions found. Generating synthetic data for visualization (marked synthetic).")
                if "data_info" in model_info and isinstance(model_info["data_info"], dict):
                    di = model_info["data_info"]
                    fmin = di.get("force_min", 0.0)
                    fmax = di.get("force_max", 10.0)
                else:
                    fmin, fmax = 0.0, 10.0
                n = 200
                actuals = np.linspace(fmin, fmax, n)
                rng = np.random.RandomState(42 + (ord(sensor) - ord('A'))*11)
                predictions = actuals + rng.normal(scale=0.3 + 0.05*(ord(sensor)-ord('A')), size=n)
                predictions = np.clip(predictions, a_min=0.0, a_max=None)

            # Clean invalid values and make arrays 1d
            try:
                actuals = np.array(actuals, dtype=float).flatten()
                predictions = np.array(predictions, dtype=float).flatten()
            except Exception:
                actuals = None
                predictions = None

            if actuals is not None and predictions is not None:
                valid = (~np.isnan(actuals)) & (~np.isnan(predictions)) & (~np.isinf(actuals)) & (~np.isinf(predictions))
                if np.sum(~valid) > 0:
                    print(f"  Filtering {np.sum(~valid)} invalid pred/actual pairs.")
                actuals = actuals[valid]
                predictions = predictions[valid]
                residuals = predictions - actuals
            else:
                residuals = None

            # Attempt to determine architecture and parameter count
            param_count = 0
            input_size = None
            hidden_layers = []
            if isinstance(model_info, dict):
                if "model_config" in model_info and isinstance(model_info["model_config"], dict):
                    mc = model_info["model_config"]
                    hidden_layers = mc.get("hidden_layers", hidden_layers)
                    input_size = mc.get("input_size", input_size)
                if "architecture" in model_info and isinstance(model_info["architecture"], dict):
                    arch = model_info["architecture"]
                    hidden_layers = hidden_layers or arch.get("hidden_layers", hidden_layers)
                    input_size = input_size or model_info.get("input_features", input_size)
                if "hidden_layers" in model_info and input_size is None:
                    hidden_layers = model_info.get("hidden_layers", hidden_layers)
                    input_size = model_info.get("input_features", input_size)

            if (not hidden_layers or input_size is None) and isinstance(history, dict):
                if "model_config" in history and isinstance(history["model_config"], dict):
                    mc = history["model_config"]
                    hidden_layers = hidden_layers or mc.get("hidden_layers", hidden_layers)
                    input_size = input_size or mc.get("input_size", input_size)
                if "architecture" in history and isinstance(history["architecture"], dict):
                    arch = history["architecture"]
                    hidden_layers = hidden_layers or arch.get("hidden_layers", hidden_layers)
                    input_size = input_size or history.get("input_size", input_size)

            # try to compute param_count from .pth if available
            if param_count == 0:
                param_count = try_load_state_dict_and_count_params(model_dir)

            # if still zero but we have hidden_layers and input_size, compute
            if (not param_count) and hidden_layers and input_size:
                try:
                    param_count = calculate_model_parameters(hidden_layers, input_size)
                except Exception:
                    param_count = 0

            if not param_count:
                param_count = 0

            # collect metrics (prefer history.metrics then model_info.performance then calculated)
            metrics = {}
            metric_sources = []
            if isinstance(history, dict) and "metrics" in history and isinstance(history["metrics"], dict):
                metrics.update(history["metrics"])
                metric_sources.append("history.metrics")
            if isinstance(model_info, dict) and "performance" in model_info and isinstance(model_info["performance"], dict):
                metrics.update(model_info["performance"])
                metric_sources.append("model_info.performance")
            if "evaluation" in history and isinstance(history["evaluation"], dict):
                metrics.update(history["evaluation"])
                metric_sources.append("history.evaluation")

            # compute metrics from actuals/preds if missing
            if actuals is not None and predictions is not None:
                try:
                    calc_mse = float(np.mean((predictions - actuals)**2))
                    calc_rmse = float(np.sqrt(calc_mse))
                    calc_mae = float(np.mean(np.abs(predictions - actuals)))
                    ss_tot = float(np.sum((actuals - np.mean(actuals))**2))
                    ss_res = float(np.sum((actuals - predictions)**2))
                    calc_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    metrics.setdefault("mse", calc_mse)
                    metrics.setdefault("rmse", calc_rmse)
                    metrics.setdefault("mae", calc_mae)
                    metrics.setdefault("r2", calc_r2)
                    metric_sources.append("calculated_from_preds")
                except Exception:
                    pass

            if not metrics:
                metrics = {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "r2": float('nan')}

            training_samples = 0
            if isinstance(model_info, dict) and "data_info" in model_info and isinstance(model_info["data_info"], dict):
                training_samples = model_info["data_info"].get("training_samples", training_samples)
            elif isinstance(history, dict) and "training_time" in history:
                training_samples = training_samples
            else:
                if isinstance(model_info, dict) and "data_info" in model_info and isinstance(model_info["data_info"], dict):
                    training_samples = model_info["data_info"].get("total_samples", training_samples)

            results[sensor] = {
                "model_dir": str(model_dir),
                "history": history,
                "model_info": model_info,
                "actuals": actuals,
                "predictions": predictions,
                "residuals": residuals,
                "metrics": {k: float(v) if v is not None else float('nan') for k, v in metrics.items()},
                "param_count": int(param_count),
                "training_samples": int(training_samples) if training_samples else 0,
                "synthetic": bool(synthetic),
                "metric_sources": metric_sources
            }

            print(f"  Loaded sensor {sensor}: samples={len(actuals) if actuals is not None else 0}, params={param_count}, R2={results[sensor]['metrics'].get('r2', np.nan):.4f}")

        except Exception as e:
            print(f"Error processing sensor {sensor}: {e}")
            traceback.print_exc()
            continue

    if not results:
        raise RuntimeError("No model results loaded. Check BASE_DIR and folder structure.")
    return results

# -----------------------
# Utility for robust axes handling
# -----------------------
def _flatten_axes(axes):
    """Return 1D numpy array of axes objects regardless of how plt.subplots returned them."""
    try:
        return np.array(axes, dtype=object).ravel()
    except Exception:
        return np.array([axes], dtype=object)

# -----------------------
# Plotting utilities
# -----------------------
def plot_loss_curves(results_dict, save_path=None):
    sensors_with_loss = [s for s,d in results_dict.items() if 'train_losses' in d.get('history',{}) and 'val_losses' in d.get('history',{})]
    if not sensors_with_loss:
        print("No loss history found for any sensor. Skipping loss curves.")
        return
    n = len(sensors_with_loss)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), dpi=300)
    axes = _flatten_axes(axes)
    for i, sensor in enumerate(sensors_with_loss):
        ax = axes[i]
        hist = results_dict[sensor]['history']
        train_losses = hist.get('train_losses', [])
        val_losses = hist.get('val_losses', [])
        try:
            train_losses = [float(x) for x in train_losses if np.isfinite(float(x))]
            val_losses = [float(x) for x in val_losses if np.isfinite(float(x))]
        except Exception:
            train_losses, val_losses = [], []
        if not train_losses or not val_losses:
            ax.text(0.5,0.5,"No loss data",ha='center',va='center')
            ax.set_title(f"Sensor {sensor}")
            continue
        ax.plot(range(1,len(train_losses)+1), train_losses, label='Train', color='tab:blue')
        ax.plot(range(1,len(val_losses)+1), val_losses, label='Val', color='tab:orange')
        best_epoch = int(np.argmin(val_losses)) if len(val_losses)>0 else 0
        if len(val_losses)>0:
            ax.scatter(best_epoch+1, val_losses[best_epoch], color='green', s=40, zorder=5, label=f'Best (E{best_epoch+1})')
            ax.text(0.98, 0.95, f"Best Val: {val_losses[best_epoch]:.4f}", transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7))
        ax.set_title(f"Sensor {sensor}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)
    for j in range(i+1, len(axes)):
        try:
            fig.delaxes(axes[j])
        except Exception:
            pass
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")
    plt.close()

def plot_performance_metrics(results_dict, save_path=None):
    sensors = list(results_dict.keys())
    if not sensors:
        print("No sensors to plot performance metrics.")
        return
    rmse = [results_dict[s]['metrics'].get('rmse', np.nan) for s in sensors]
    mae  = [results_dict[s]['metrics'].get('mae', np.nan) for s in sensors]
    r2   = [results_dict[s]['metrics'].get('r2', np.nan) for s in sensors]
    colors = [COLORS.get(s, '#888888') for s in sensors]
    x = np.arange(len(sensors))
    fig, axes = plt.subplots(1,3, figsize=(15,5), dpi=300)
    axes = _flatten_axes(axes)
    width = 0.6
    for ax, vals, title, ylabel in zip(axes, [rmse, mae, r2], ['RMSE (N)','MAE (N)','R² Score'], ['RMSE (N)','MAE (N)','R²']):
        bars = ax.bar(x, vals, width, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Sensor {s}" for s in sensors], rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        if title == 'R² Score':
            ax.set_ylim(0,1.05)
        else:
            maxv = np.nanmax(vals)
            if np.isfinite(maxv) and maxv > 0:
                ax.set_ylim(0, maxv*1.2)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", xy=(b.get_x()+b.get_width()/2, v), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved performance metrics to {save_path}")
    plt.close()

def plot_predictions_vs_actuals(results_dict, save_path=None):
    sensors = [s for s,d in results_dict.items() if d.get('actuals') is not None and d.get('predictions') is not None]
    if not sensors:
        print("No prediction data available for predictions_vs_actuals plot.")
        return
    n = len(sensors)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols,4*n_rows), dpi=300)
    axes = _flatten_axes(axes)
    for i, sensor in enumerate(sensors):
        ax = axes[i]
        actuals = np.array(results_dict[sensor]['actuals']).flatten()
        preds   = np.array(results_dict[sensor]['predictions']).flatten()
        if actuals.size == 0 or preds.size == 0:
            ax.text(0.5,0.5,"No data",ha='center',va='center')
            ax.set_title(f"Sensor {sensor}")
            continue
        try:
            hb = ax.hexbin(actuals, preds, gridsize=40, cmap='viridis', mincnt=1)
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Count")
        except Exception:
            ax.scatter(actuals, preds, s=20, alpha=0.6, color=COLORS.get(sensor,'#777777'), edgecolors='w')
        mn = min(actuals.min(), preds.min())
        mx = max(actuals.max(), preds.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5)
        r2 = results_dict[sensor]['metrics'].get('r2', np.nan)
        ax.text(0.05, 0.95, f"R²={r2:.4f}", transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax.set_title(f"Sensor {sensor}")
        ax.set_xlabel("Actual Force (N)")
        ax.set_ylabel("Predicted Force (N)")
        ax.set_aspect('equal', adjustable='box')
        pad = (mx - mn) * 0.1 if mx>mn else 0.5
        ax.set_xlim(mn-pad, mx+pad)
        ax.set_ylim(mn-pad, mx+pad)
        ax.grid(True, linestyle='--', alpha=0.6)
    for j in range(i+1, len(axes)):
        try:
            fig.delaxes(axes[j])
        except Exception:
            pass
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved predictions vs actuals to {save_path}")
    plt.close()

def plot_residual_analysis(results_dict, save_path=None):
    sensors = [s for s,d in results_dict.items() if d.get('residuals') is not None and len(d.get('residuals'))>0]
    if not sensors:
        print("No residuals to analyze. Skipping residual analysis.")
        return
    fig = plt.figure(figsize=(15,10), dpi=300)
    gs = GridSpec(2,3, figure=fig)

    # Boxplots
    ax1 = fig.add_subplot(gs[0,:2])
    residuals_list = []
    labels = []
    for s in sensors:
        r = np.array(results_dict[s]['residuals'])
        if r.size == 0:
            continue
        q1, q3 = np.percentile(r, [25,75])
        iqr = q3 - q1
        lb, ub = q1 - 3*iqr, q3 + 3*iqr
        r_f = r[(r>=lb)&(r<=ub)]
        if r_f.size>0:
            residuals_list.append(r_f)
            labels.append(f"Sensor {s}")
    if residuals_list:
        sns.boxplot(data=residuals_list, ax=ax1, palette=[COLORS.get(s,'#888') for s in sensors])
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_title("Residuals by Sensor")
        ax1.set_ylabel("Residual (Pred - Actual) [N]")
        ax1.axhline(0, color='k', linestyle='--', alpha=0.6)
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
    else:
        ax1.text(0.5,0.5,"No residuals available",ha='center',va='center')

    # Residuals vs Actual for best sensor
    best_sensor = max(results_dict.items(), key=lambda kv: kv[1]['metrics'].get('r2', -np.inf))[0]
    ax2 = fig.add_subplot(gs[0,2])
    data_best = results_dict.get(best_sensor, {})
    if data_best and data_best.get('actuals') is not None and data_best.get('residuals') is not None:
        a = np.array(data_best['actuals']).flatten()
        r = np.array(data_best['residuals']).flatten()
        if a.size==r.size and a.size>0:
            try:
                hb = ax2.hexbin(a, r, gridsize=40, cmap='Blues', mincnt=1)
                cb = fig.colorbar(hb, ax=ax2)
                cb.set_label("Count")
            except Exception:
                ax2.scatter(a, r, s=20, alpha=0.6, color=COLORS.get(best_sensor,'#333'))
            ax2.axhline(0, color='r', linestyle='--')
            ax2.set_title(f"Residuals vs Actual (Best: Sensor {best_sensor})")
            ax2.set_xlabel("Actual Force (N)")
            ax2.set_ylabel("Residual (N)")
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.text(0.05,0.95, f"Mean={np.mean(r):.3f}\nStd={np.std(r):.3f}", transform=ax2.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax2.text(0.5,0.5,"No residuals for best sensor",ha='center',va='center')
    else:
        ax2.text(0.5,0.5,"No residuals for best sensor",ha='center',va='center')

    # KDE of residuals
    ax3 = fig.add_subplot(gs[1,:])
    any_plotted = False
    for s in sensors:
        r = np.array(results_dict[s]['residuals'])
        if r.size==0:
            continue
        q1, q3 = np.percentile(r, [25,75])
        iqr = q3 - q1
        lb, ub = q1 - 3*iqr, q3 + 3*iqr
        r_f = r[(r>=lb)&(r<=ub)]
        if r_f.size>1:
            sns.kdeplot(r_f, ax=ax3, label=f"Sensor {s}", color=COLORS.get(s,'#444'), linewidth=1.5)
            any_plotted = True
    if any_plotted:
        ax3.axvline(0, color='k', linestyle='--', alpha=0.6)
        ax3.set_title("Residual Distribution (KDE)")
        ax3.set_xlabel("Residual (N)")
        ax3.set_ylabel("Density")
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, linestyle='--', alpha=0.6)
    else:
        ax3.text(0.5,0.5,"No residual distributions to plot", ha='center', va='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved residual analysis to {save_path}")
    plt.close()

def plot_error_by_force_range(results_dict, save_path=None, force_ranges=FORCE_RANGES):
    sensors = [s for s,d in results_dict.items() if d.get('actuals') is not None and d.get('residuals') is not None]
    if not sensors:
        print("No data for error-by-force-range plot.")
        return
    x = np.arange(len(force_ranges))
    fig, ax = plt.subplots(figsize=(16,8), dpi=300)
    num = len(sensors)
    width = 0.8/num
    offsets = np.linspace(-width*(num-1)/2, width*(num-1)/2, num)
    any_data=False
    for i, s in enumerate(sensors):
        d = results_dict[s]
        errors = get_force_range_errors(d['actuals'], d['residuals'], force_ranges)
        mae_vals = []
        counts = []
        for fr in force_ranges:
            if fr in errors and errors[fr]['count']>=2:
                mae_vals.append(errors[fr]['mae'])
                counts.append(errors[fr]['count'])
                any_data=True
            else:
                mae_vals.append(np.nan)
                counts.append(0)
        ax.bar(x+offsets[i], mae_vals, width, label=f"Sensor {s}", color=COLORS.get(s,'#777'), edgecolor='black', alpha=0.85)
        for j, (mv, ct) in enumerate(zip(mae_vals, counts)):
            if not np.isnan(mv) and ct>0:
                if mv>0:
                    ax.text(x[j]+offsets[i], mv+0.005, f"n={ct}", ha='center', va='bottom', fontsize=7)
    if not any_data:
        ax.text(0.5,0.5,"No force-range data with >=2 samples", ha='center', va='center')
    else:
        labels = [f"{fr[0]}-{fr[1]} N" for fr in force_ranges]
        if len(labels)>12:
            step = max(1, len(labels)//12)
            labels = [lab if idx%step==0 else '' for idx,lab in enumerate(labels)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title("MAE by Force Range (0.5N bins)")
        ax.set_xlabel("Force Range (N)")
        ax.set_ylabel("MAE (N)")
        ax.legend(loc='best')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved error-by-force-range to {save_path}")
    plt.close()

def plot_model_complexity_vs_performance(results_dict, save_path=None):
    sensors = list(results_dict.keys())
    if not sensors:
        print("No sensors for complexity plot.")
        return
    xs = [max(1, results_dict[s]['param_count']) for s in sensors]
    ys = [results_dict[s]['metrics'].get('r2', np.nan) for s in sensors]
    colors = [COLORS.get(s,'#777') for s in sensors]
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    ax.scatter(xs, ys, s=120, c=colors, edgecolor='k', alpha=0.9)
    for s,x,y in zip(sensors,xs,ys):
        ax.annotate(f"Sensor {s}", (x,y), xytext=(6,0), textcoords='offset points', fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("R² Score")
    ax.set_ylim(0,1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    if len(xs)>2:
        try:
            valid = [i for i,(x,y) in enumerate(zip(xs,ys)) if x>0 and np.isfinite(y)]
            if len(valid)>2:
                logx = np.log10(np.array([xs[i] for i in valid]))
                yy = np.array([ys[i] for i in valid])
                coeffs = np.polyfit(logx, yy, 1)
                xp = np.logspace(np.min(logx), np.max(logx), 100)
                yp = np.poly1d(coeffs)(np.log10(xp))
                ax.plot(xp, yp, 'k--', alpha=0.7, label='trend')
                corr = np.corrcoef(logx, yy)[0,1]
                ax.text(0.05,0.95,f"corr={corr:.3f}", transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.8))
        except Exception:
            pass
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved model complexity vs performance to {save_path}")
    plt.close()

def create_performance_summary_table(results_dict, save_path=None):
    rows = []
    for s,d in results_dict.items():
        m = d['metrics']
        rows.append({
            'Sensor': f"Sensor {s}",
            'R2': m.get('r2', np.nan),
            'RMSE': m.get('rmse', np.nan),
            'MAE': m.get('mae', np.nan),
            'Params': d.get('param_count', 0),
            'TrainSamples': d.get('training_samples', 0),
            'SyntheticData': d.get('synthetic', False)
        })
    df = pd.DataFrame(rows)
    df = df[['Sensor','R2','RMSE','MAE','Params','TrainSamples','SyntheticData']]
    fig, ax = plt.subplots(figsize=(12, max(2, 0.3*len(df))), dpi=300)
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1,1.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved performance summary table to {save_path}")
    plt.close()
    if save_path:
        csv_path = Path(str(save_path)).with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved performance summary CSV to {csv_path}")

# -----------------------
# Main orchestration
# -----------------------
def main():
    print("Starting visualization generation...")
    results = load_all_model_results()

    sensors = list(results.keys())
    print(f"\nLoaded results for sensors: {sensors}")

    # debug summary (use explicit None checks to avoid ambiguous truthiness for numpy arrays)
    print("\nSummary of loaded results:")
    for s, d in results.items():
        actuals = d.get('actuals')
        n_samples = int(len(actuals)) if actuals is not None else 0
        params = d.get('param_count', 0)
        r2 = d['metrics'].get('r2', np.nan)
        synthetic = bool(d.get('synthetic', False))
        print(f" - Sensor {s}: samples={n_samples}, params={params}, R2={r2:.4f}, synthetic={synthetic}")

    # create output paths
    plot_loss_curves(results, OUTPUT_DIR / "loss_curves.png")
    plot_performance_metrics(results, OUTPUT_DIR / "performance_metrics.png")
    plot_predictions_vs_actuals(results, OUTPUT_DIR / "predictions_vs_actuals.png")
    plot_residual_analysis(results, OUTPUT_DIR / "residual_analysis.png")
    plot_error_by_force_range(results, OUTPUT_DIR / "error_by_force_range.png")
    plot_model_complexity_vs_performance(results, OUTPUT_DIR / "model_complexity_vs_performance.png")
    create_performance_summary_table(results, OUTPUT_DIR / "performance_summary.png")

    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("Visualization generation complete.")

if __name__ == "__main__":
    main()
