#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBTS FORCE PREDICTION COMPARISON TOOL

Stage 1: Alignment
 - Shows predicted force (blue) and ground truth force (red) on the same plot
 - Slider 1 adjusts horizontal shift of ground truth force data
 - Slider 2 adjusts horizontal scale (time compression/expansion) of ground truth force
 - Predicted force graph x-axis is fixed
 - Ground truth force can move outside visible area without affecting x-view
 - Y-axis automatically fits both graphs to ensure all data is visible
 - Finish alignment button closes this window and proceeds to stage 2

Stage 2: Comparison Visualization
 - Shows aligned data in two plots:
   * Top plot: Predicted force vs aligned ground truth force
   * Bottom plot: Force prediction error (difference)
 - Save button saves the visualization and aligned data to Inference/COMPARISON
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import json
from datetime import datetime
import re
import csv

# =======================
# Parameters (user can edit)
# =======================
FIG_WIDTH = 16                # Figure width in inches
DPI = 120                     # Figure DPI
TITLE_FONT = 12
AXIS_FONT = 10
TICK_FONT = 9
LEGEND_FONT = 9
scalemax = 2.0
EPS_FORCE = 0.0               # Threshold to zero out negative force values
MOVING_WINDOW = 21            # Moving average window size
APPLY_SMOOTH = True           # Apply smoothing to VBTS data

plt.rcParams.update({
    'figure.titlesize': TITLE_FONT,
    'axes.titlesize': TITLE_FONT,
    'axes.labelsize': AXIS_FONT,
    'xtick.labelsize': TICK_FONT,
    'ytick.labelsize': TICK_FONT,
    'legend.fontsize': LEGEND_FONT,
})

# -----------------------
# File configuration
# -----------------------
# Default filename (can be overridden by command line arguments)
DEFAULT_FILENAME = "frame_logs_A_20260122_211305"

# Parse command line arguments if provided
if len(sys.argv) > 1:
    FILENAME = sys.argv[1]
else:
    FILENAME = DEFAULT_FILENAME

print(f"Using filename: {FILENAME}")

# -----------------------
# Path handling: base directory is script location (not cwd)
# -----------------------
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # __file__ may not exist in some interactive environments; fallback to cwd
    SCRIPT_DIR = Path.cwd()

INFERENCE_ROOT = SCRIPT_DIR / "Inference"
INFERENCE_VBTS = INFERENCE_ROOT / "VBTS"
INFERENCE_FORCE = INFERENCE_ROOT / "FORCE"
INFERENCE_COMPARISON = INFERENCE_ROOT / "COMPARISON"

# Ensure directories exist
INFERENCE_COMPARISON.mkdir(parents=True, exist_ok=True)

# Print diagnostics
print(f"Script directory     : {SCRIPT_DIR}")
print(f"Inference root       : {INFERENCE_ROOT}")
print(f"VBTS inference dir   : {INFERENCE_VBTS}")
print(f"FORCE inference dir  : {INFERENCE_FORCE}")
print(f"COMPARISON output dir: {INFERENCE_COMPARISON}")

# =======================
# Helper Functions
# =======================
def _decorate_axes(ax):
    ax.minorticks_on()
    ax.grid(which='major', alpha=0.6)
    ax.grid(which='minor', alpha=0.5, linestyle=':')
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONT)

def moving_average_1d(x, window):
    """Apply moving average smoothing to 1D array"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if window <= 1 or n == 0:
        return x.copy()
    w = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    if pad > 0:
        x_pad = np.pad(x, pad_width=pad, mode='reflect')
    else:
        x_pad = x
    y = np.convolve(x_pad, w, mode='valid')
    if len(y) != n:
        y = y[:n]
    return y

# =======================
# Data Loading Functions
# =======================
def load_vbts_inference():
    """Load VBTS inference data from npz file"""
    # Build filename with .npz extension
    vbts_filename = f"{FILENAME}.npz"
    path = INFERENCE_VBTS / vbts_filename
    
    if not path.exists():
        # Try to find a matching file by prefix if exact match not found
        prefix = f"{FILENAME.split('_')[0]}_"
        candidates = list(INFERENCE_VBTS.glob(f"{prefix}*.npz")) if INFERENCE_VBTS.exists() else []
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
            chosen = candidates[0]
            print(f"[WARN] Exact VBTS inference filename not found: {path}")
            print(f"[INFO] Picking nearest VBTS inference file: {chosen.name}")
            path = chosen
        else:
            available = []
            if INFERENCE_VBTS.exists():
                for p in sorted(INFERENCE_VBTS.glob("*.npz")):
                    available.append(p.name)
            raise FileNotFoundError(f"VBTS inference file not found: {path}\nSearched in: {INFERENCE_VBTS}\nAvailable files: {available}")

    data = np.load(path, allow_pickle=True)
    
    # List available keys for debugging
    print(f"Available keys in VBTS inference file: {list(data.keys())}")
    
    # Check for force prediction keys
    force_keys = []
    if "force_mlp" in data:
        force_keys.append("force_mlp")
    if "force_autoencoder" in data:
        force_keys.append("force_autoencoder")
    if "force_svm" in data:
        force_keys.append("force_svm")
    
    if not force_keys:
        raise KeyError(f"No force prediction data found in {path}. Expected at least one of: force_mlp, force_autoencoder, force_svm")
    
    # Check for required keys
    required_keys = ["timestamps"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"VBTS inference NPZ missing required key '{k}' in file {path}")
    
    timestamps = data["timestamps"]          # (T,) - absolute timestamps
    
    # Get force predictions (use first available model)
    selected_model = force_keys[0]
    print(f"Using force prediction model: {selected_model}")
    force_pred = data[selected_model]  # (T,)
    
    # Convert timestamps to relative time in seconds
    if timestamps.ndim > 1:
        timestamps = timestamps.flatten()
    
    ts_rel = timestamps - timestamps[0]
    
    # Apply smoothing if configured
    if APPLY_SMOOTH and MOVING_WINDOW > 1:
        force_pred_smooth = moving_average_1d(force_pred, MOVING_WINDOW)
    else:
        force_pred_smooth = force_pred.copy()
    
    return {
        "fn": path.name,
        "ts_rel": ts_rel,
        "force_pred": force_pred_smooth,
        "force_model": selected_model,
        "T": len(timestamps),  # Number of frames
    }

def load_force_ground_truth():
    """Load ground truth force data from CSV file"""
    # Build force filename
    force_filename = f"{FILENAME}_force.csv"
    path = INFERENCE_FORCE / force_filename
    
    # Try alternative filename format if needed
    if not path.exists():
        force_filename = f"{FILENAME.split('_')[0]}_force.csv"
        path = INFERENCE_FORCE / force_filename
    
    if not path.exists():
        # Try to find a matching file by prefix
        candidates = list(INFERENCE_FORCE.glob(f"{FILENAME.split('_')[0]}*_force.csv")) if INFERENCE_FORCE.exists() else []
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
            chosen = candidates[0]
            print(f"[WARN] Exact FORCE ground truth filename not found: {path}")
            print(f"[INFO] Picking nearest FORCE ground truth file: {chosen.name}")
            path = chosen
        else:
            available = []
            if INFERENCE_FORCE.exists():
                for p in sorted(INFERENCE_FORCE.glob("*_force.csv")):
                    available.append(p.name)
            raise FileNotFoundError(f"FORCE ground truth file not found: {path}\nSearched in: {INFERENCE_FORCE}\nAvailable files: {available}")

    # Try to read the CSV file
    pids = []
    times_raw = []
    forces = []

    with path.open('r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        header_skipped = False
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
            if not header_skipped:
                first_cell = row[0].strip()
                # if first cell doesn't contain a number, assume header line and skip once
                if not re.search(r"-?\d+", first_cell):
                    header_skipped = True
                    continue
                else:
                    header_skipped = True

            pid_match = re.search(r"-?\d+", row[0])
            if not pid_match:
                continue

            pid = int(pid_match.group(0))
            time_raw = row[1].strip() if len(row) >= 2 else ''
            force_raw = row[-1].strip()

            # Clean force value
            force_clean = re.sub(r"[^0-9\-,.+]", "", force_raw)
            force_clean = force_clean.replace(',', '.')
            try:
                force_val = float(force_clean) if force_clean != '' else float('nan')
            except ValueError:
                force_val = float('nan')

            pids.append(pid)
            times_raw.append(time_raw)
            forces.append(force_val)

    # Create dataframe and process
    df = pd.DataFrame({'PID': pids, 'time_raw': times_raw, 'force_N': forces})
    if df.empty:
        raise ValueError(f"Empty or invalid force data in {path}")

    df = df.sort_values('PID').reset_index(drop=True)
    df['force_N'] = -1.0 * df['force_N']  # Invert force (compression positive)

    pid_arr = df['PID'].values.astype(int)
    force_arr = df['force_N'].values.astype(float)

    # Prepare for plotting: replace negative values (below EPS_FORCE) with zeros
    force_for_plot = force_arr.copy()
    neg_mask = np.isfinite(force_for_plot) & (force_for_plot < EPS_FORCE)
    if np.any(neg_mask):
        force_for_plot[neg_mask] = 0.0

    return {
        "fn": path.name,
        "pid_arr": pid_arr,
        "force_arr": force_arr,
        "force_for_plot": force_for_plot,
        "df": df,
    }

# =======================
# Stage 1: Alignment GUI with Scaling
# =======================
class AlignmentGUI:
    def __init__(self, vbts, force):
        self.vbts = vbts
        self.force = force
        self.shift = 0.0
        self.scale = 1.0  # Scale factor for time axis (1.0 = no scaling)
        self.completed = False
        self.paused = False

        # Create mapping from force PID to VBTS time range
        ts_vbts = self.vbts["ts_rel"]
        pid_force = self.force["pid_arr"]
        
        if pid_force.size == 0:
            raise ValueError("Force PID array is empty.")
        
        # Normalize force times to match VBTS time range for initial alignment
        self.force_times_base = np.interp(pid_force,
                                         (pid_force.min(), pid_force.max()),
                                         (float(ts_vbts.min()), float(ts_vbts.max())))
        
        # Create figure
        self.fig = plt.figure(figsize=(FIG_WIDTH, 8), dpi=DPI)
        gs = self.fig.add_gridspec(6, 1)
        
        # Main plot: force data
        self.ax_force = self.fig.add_subplot(gs[0:4, 0])
        
        # Control area
        control_ax = self.fig.add_subplot(gs[4:6, 0])
        control_ax.axis('off')

        # Plot predicted force (blue line) - fixed
        self.predicted_force_line, = self.ax_force.plot(self.vbts["ts_rel"], self.vbts["force_pred"],
                          'b-', linewidth=2.5, alpha=0.95, label=f'Predicted Force ({self.vbts["force_model"]})')
        self.ax_force.set_xlabel("Time (s)")
        self.ax_force.set_ylabel("Force (N)", color='k')
        self.ax_force.tick_params(axis='y', labelcolor='k')
        self.ax_force.set_title(f"Force Data Alignment\nVBTS: {vbts['fn']}, FORCE: {force['fn']}")

        # Plot ground truth force (red line) - initially unshifted and unscaled
        self.ground_truth_line, = self.ax_force.plot(self.force_times_base, self.force["force_for_plot"],
                                                    'r-', linewidth=2.0, alpha=0.95, label='Ground Truth Force')
        
        # Decorate axes
        _decorate_axes(self.ax_force)
        
        # Set fixed x-limits based on predicted force data
        self.x_min = float(self.vbts["ts_rel"].min())
        self.x_max = float(self.vbts["ts_rel"].max())
        self.ax_force.set_xlim(self.x_min, self.x_max)
        
        # Calculate initial y-limits to fit both graphs
        self.update_y_limits()
        
        # Add legend
        self.ax_force.legend(loc='upper right', fontsize=LEGEND_FONT)
        
        # Disable autoscaling on x-axis but enable on y-axis
        self.ax_force.autoscale(enable=False, axis='x')
        self.ax_force.autoscale(enable=True, axis='y')

        # Adjust layout for controls
        plt.subplots_adjust(bottom=0.25)

        # Create slider for time shift
        slider_shift_ax = plt.axes([0.2, 0.12, 0.65, 0.03])
        ts_span = float(self.vbts["ts_rel"].max() - self.vbts["ts_rel"].min())
        self.slider_shift = Slider(
            slider_shift_ax, 'Shift (s)', -ts_span*2, ts_span*2,
            valinit=0.0, valstep=0.001,
            color='#5DADE2'
        )
        self.slider_shift.on_changed(self.update_plot)

        # Create slider for time scale
        slider_scale_ax = plt.axes([0.2, 0.06, 0.65, 0.03])
        self.slider_scale = Slider(
            slider_scale_ax, 'Scale', 0.5, scalemax,
            valinit=1.0, valstep=0.0001,
            color='#AF7AC5'
        )
        self.slider_scale.on_changed(self.update_plot)

        # Create buttons
        finish_ax = plt.axes([0.4, 0.01, 0.2, 0.04])
        self.btn_finish = Button(finish_ax, 'Finish Alignment', color='#5DADE2', hovercolor='#85C1E9')
        self.btn_finish.on_clicked(self.finish_alignment)

        exit_ax = plt.axes([0.7, 0.01, 0.1, 0.04])
        self.btn_exit = Button(exit_ax, 'Exit', color='#E74C3C', hovercolor='#F1948A')
        self.btn_exit.on_clicked(self.exit_gui)

        # Add pause button
        pause_ax = plt.axes([0.1, 0.01, 0.1, 0.04])
        self.btn_pause = Button(pause_ax, 'Pause', color='#F39C12', hovercolor='#F8C471')
        self.btn_pause.on_clicked(self.toggle_pause)

        # Add reset button
        reset_ax = plt.axes([0.25, 0.01, 0.1, 0.04])
        self.btn_reset = Button(reset_ax, 'Reset', color='#8E44AD', hovercolor='#9B59B6')
        self.btn_reset.on_clicked(self.reset_alignment)

        # Show plot (block until closed)
        plt.tight_layout()
        plt.show(block=True)

    def update_y_limits(self):
        """Update y-axis limits to fit both graphs visible in current x-range"""
        # Get data visible in current x-range for predicted force
        mask_pred = (self.vbts["ts_rel"] >= self.x_min) & (self.vbts["ts_rel"] <= self.x_max)
        pred_values = self.vbts["force_pred"][mask_pred]
        
        # Get data visible in current x-range for ground truth force
        gt_x = self.ground_truth_line.get_xdata()
        gt_y = self.ground_truth_line.get_ydata()
        mask_gt = (gt_x >= self.x_min) & (gt_x <= self.x_max)
        gt_values = gt_y[mask_gt]
        
        # Combine all values to determine y-limits
        all_values = np.concatenate([pred_values, gt_values])
        all_values = all_values[np.isfinite(all_values)]
        
        if len(all_values) > 0:
            y_min = np.min(all_values)
            y_max = np.max(all_values)
            y_range = y_max - y_min
            pad = max(1e-6, 0.1 * y_range if y_range > 0 else 1.0)
            self.ax_force.set_ylim(y_min - pad, y_max + pad)
        else:
            # Fallback to full range if no data in current view
            y_min = min(np.nanmin(self.vbts["force_pred"]), np.nanmin(self.force["force_for_plot"]))
            y_max = max(np.nanmax(self.vbts["force_pred"]), np.nanmax(self.force["force_for_plot"]))
            y_range = y_max - y_min
            pad = max(1e-6, 0.1 * y_range if y_range > 0 else 1.0)
            self.ax_force.set_ylim(y_min - pad, y_max + pad)

    def update_plot(self, val):
        """Update ground truth force graph position based on slider values"""
        if self.paused:
            return
            
        self.shift = float(self.slider_shift.val)
        self.scale = float(self.slider_scale.val)

        # Apply scale then shift
        new_x = (self.force_times_base * self.scale) + self.shift
        self.ground_truth_line.set_xdata(new_x)

        # Update title with current parameters
        self.ax_force.set_title(f"Force Data Alignment (Shift: {self.shift:.3f}s, Scale: {self.scale:.4f})\n"
                              f"VBTS: {self.vbts['fn']}, FORCE: {self.force['fn']}")

        # Update y-axis limits to fit both graphs
        self.update_y_limits()

        self.fig.canvas.draw_idle()

    def finish_alignment(self, event):
        """Close alignment window and mark as completed"""
        self.completed = True
        plt.close(self.fig)

    def exit_gui(self, event):
        """Exit the entire application"""
        self.completed = False
        plt.close(self.fig)
        sys.exit(0)
    
    def toggle_pause(self, event):
        """Toggle pause state for sliders"""
        self.paused = not self.paused
        self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')
        self.fig.canvas.draw_idle()
    
    def reset_alignment(self, event):
        """Reset alignment parameters to default values"""
        self.slider_shift.set_val(0.0)
        self.slider_scale.set_val(1.0)
        self.shift = 0.0
        self.scale = 1.0
        self.update_plot(None)

# =======================
# Stage 2: Comparison Visualization and Saving
# =======================
class ComparisonGUI:
    def __init__(self, vbts, force, shift, scale):
        self.vbts = vbts
        self.force = force
        self.shift = shift
        self.scale = scale
        self.completed = False

        # Calculate aligned ground truth force times
        ts_vbts = self.vbts["ts_rel"]
        pid_force = self.force["pid_arr"]
        
        # Normalize force times to match VBTS time range
        force_times_base = np.interp(pid_force,
                                   (pid_force.min(), pid_force.max()),
                                   (float(ts_vbts.min()), float(ts_vbts.max())))
        
        # Apply scaling and shifting
        self.aligned_force_times = (force_times_base * self.scale) + self.shift
        
        # Create figure with two subplots
        self.fig = plt.figure(figsize=(FIG_WIDTH, 10), dpi=DPI)
        gs = self.fig.add_gridspec(8, 1)
        
        # Top plot: Force comparison
        self.ax_force = self.fig.add_subplot(gs[0:4, 0])
        
        # Bottom plot: Error analysis
        self.ax_error = self.fig.add_subplot(gs[4:6, 0], sharex=self.ax_force)
        
        # Control area
        control_ax = self.fig.add_subplot(gs[6:8, 0])
        control_ax.axis('off')

        # Plot predicted force and aligned ground truth force
        self.ax_force.plot(self.vbts["ts_rel"], self.vbts["force_pred"],
                          'b-', linewidth=2.5, alpha=0.95, label=f'Predicted Force ({self.vbts["force_model"]})')
        
        self.ax_force.plot(self.aligned_force_times, self.force["force_for_plot"],
                          'r-', linewidth=2.5, alpha=0.95, label='Ground Truth Force')
        
        self.ax_force.set_ylabel("Force (N)")
        self.ax_force.set_title(f"Force Comparison (Shift: {self.shift:.3f}s, Scale: {self.scale:.4f})")
        self.ax_force.legend(fontsize=LEGEND_FONT)
        _decorate_axes(self.ax_force)

        # Calculate force error at aligned points
        # Interpolate ground truth force to VBTS timestamps
        valid_mask = np.isfinite(self.force["force_for_plot"]) & np.isfinite(self.aligned_force_times)
        force_interp = np.full_like(self.vbts["ts_rel"], np.nan)
        
        if np.any(valid_mask):
            force_interp = np.interp(self.vbts["ts_rel"], 
                                   self.aligned_force_times[valid_mask], 
                                   self.force["force_for_plot"][valid_mask],
                                   left=np.nan, right=np.nan)
            force_error = self.vbts["force_pred"] - force_interp
            
            # Calculate error metrics
            valid_error_mask = np.isfinite(force_error)
            if np.any(valid_error_mask):
                mae = np.mean(np.abs(force_error[valid_error_mask]))
                rmse = np.sqrt(np.mean(force_error[valid_error_mask]**2))
                self.ax_force.text(0.02, 0.95, f"MAE: {mae:.3f} N, RMSE: {rmse:.3f} N", 
                                  transform=self.ax_force.transAxes, 
                                  bbox=dict(facecolor='white', alpha=0.8))
            
            # Plot error
            self.ax_error.plot(self.vbts["ts_rel"], force_error,
                             'g-', linewidth=1.8, alpha=0.95, label='Force Error (Pred - GT)')
            self.ax_error.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            self.ax_error.set_ylabel("Error (N)")
            self.ax_error.set_title("Force Prediction Error")
            self.ax_error.legend(fontsize=LEGEND_FONT)
        else:
            self.ax_error.text(0.5, 0.5, "Could not compute error: insufficient valid ground truth data",
                              ha='center', va='center', transform=self.ax_error.transAxes)
        
        _decorate_axes(self.ax_error)
        self.ax_error.set_xlabel("Time (s)")

        # Create buttons
        btn_w = 0.16
        margin = 0.01
        left = 0.02

        # Save button
        ax_save = self.fig.add_axes([left, 0.02, btn_w, 0.05])
        self.btn_save = Button(ax_save, 'Save Comparison', color='#2ECC71', hovercolor='#58D68D')
        self.btn_save.on_clicked(self.save_comparison)

        # Exit button
        left += btn_w + margin
        ax_exit = self.fig.add_axes([left, 0.02, btn_w, 0.05])
        self.btn_exit = Button(ax_exit, 'Exit', color='#95A5A6', hovercolor='#BDC3C7')
        self.btn_exit.on_clicked(self.exit_gui)

        # Add zoom reset button
        left += btn_w + margin
        ax_reset = self.fig.add_axes([left, 0.02, btn_w, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset Zoom', color='#3498DB', hovercolor='#5DADE2')
        self.btn_reset.on_clicked(self.reset_zoom)

        # Set overall title
        self.fig.suptitle(f"VBTS Force Prediction vs Ground Truth Comparison\n"
                         f"File: {FILENAME}",
                         fontsize=TITLE_FONT+2, fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for suptitle

        # Show plot
        plt.show(block=True)

    def save_comparison(self, event):
        """Save comparison plot and aligned data"""
        try:
            # Create experiment directory
            experiment_dirname = FILENAME
            experiment_dir = INFERENCE_COMPARISON / experiment_dirname
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Save plot as PNG
            plot_filename = experiment_dir / f"comparison_{experiment_dirname}.png"
            self.fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {plot_filename}")
            
            # Create aligned ground truth force array matching VBTS timestamps
            valid_mask = np.isfinite(self.force["force_for_plot"]) & np.isfinite(self.aligned_force_times)
            if np.any(valid_mask):
                force_aligned = np.interp(self.vbts["ts_rel"], 
                                         self.aligned_force_times[valid_mask], 
                                         self.force["force_for_plot"][valid_mask],
                                         left=np.nan, right=np.nan)
            else:
                force_aligned = np.full_like(self.vbts["ts_rel"], np.nan)
            
            # Calculate error metrics
            valid_force_mask = np.isfinite(force_aligned) & np.isfinite(self.vbts["force_pred"])
            if np.any(valid_force_mask):
                error = self.vbts["force_pred"][valid_force_mask] - force_aligned[valid_force_mask]
                mae = np.mean(np.abs(error))
                rmse = np.sqrt(np.mean(error**2))
            else:
                mae = np.nan
                rmse = np.nan
            
            # Save aligned data
            data_filename = experiment_dir / f"aligned_data_{experiment_dirname}.npz"
            np.savez_compressed(
                data_filename,
                vbts_timestamps=self.vbts["ts_rel"],
                predicted_force=self.vbts["force_pred"],
                force_model=self.vbts["force_model"],
                ground_truth_times=self.aligned_force_times,
                ground_truth_force=self.force["force_for_plot"],
                aligned_ground_truth_force=force_aligned,
                alignment_shift=self.shift,
                alignment_scale=self.scale,
                error_mae=mae,
                error_rmse=rmse,
                experiment_info={
                    'filename': FILENAME,
                    'vbts_file': self.vbts["fn"],
                    'force_file': self.force["fn"]
                }
            )
            print(f"Aligned data saved to: {data_filename}")
            
            # Save metadata as JSON
            metadata_filename = experiment_dir / f"metadata_{experiment_dirname}.json"
            metadata = {
                "experiment": experiment_dirname,
                "alignment_parameters": {
                    "shift_seconds": self.shift,
                    "scale_factor": self.scale
                },
                "error_metrics": {
                    "mae_newtons": float(mae) if not np.isnan(mae) else None,
                    "rmse_newtons": float(rmse) if not np.isnan(rmse) else None
                },
                "data_files": {
                    "vbts_inference": self.vbts["fn"],
                    "force_ground_truth": self.force["fn"],
                    "comparison_plot": plot_filename.name,
                    "aligned_data": data_filename.name
                },
                "timestamp": datetime.now().isoformat()
            }
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Metadata saved to: {metadata_filename}")
            
            print("\n" + "="*60)
            print("COMPARISON SUCCESSFULLY SAVED")
            print("="*60)
            print(f"Experiment: {experiment_dirname}")
            if not np.isnan(mae):
                print(f"MAE: {mae:.4f} N")
                print(f"RMSE: {rmse:.4f} N")
            print(f"Files saved to: {experiment_dir}")
            
            self.completed = True
            
        except Exception as e:
            print(f"Error saving comparison data: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def exit_gui(self, event):
        """Exit the application"""
        plt.close(self.fig)
    
    def reset_zoom(self, event):
        """Reset zoom to show all data"""
        # Get the x-axis limits from the predicted force
        x_min = float(self.vbts["ts_rel"].min())
        x_max = float(self.vbts["ts_rel"].max())
        
        # Set x-axis limits
        self.ax_force.set_xlim(x_min, x_max)
        
        # Get y-axis limits that include both force signals
        y_min = min(np.nanmin(self.vbts["force_pred"]), np.nanmin(self.force["force_for_plot"]))
        y_max = max(np.nanmax(self.vbts["force_pred"]), np.nanmax(self.force["force_for_plot"]))
        pad = max(1e-6, 0.1 * (y_max - y_min if y_max > y_min else 1.0))
        self.ax_force.set_ylim(y_min - pad, y_max + pad)
        
        # Reset error plot y-axis
        if hasattr(self, 'force_error') and self.force_error is not None:
            error_min = np.nanmin(self.force_error)
            error_max = np.nanmax(self.force_error)
            error_pad = max(1e-6, 0.1 * (error_max - error_min if error_max > error_min else 1.0))
            self.ax_error.set_ylim(error_min - error_pad, error_max + error_pad)
        
        # Keep x-axis limits same for error plot
        self.ax_error.set_xlim(x_min, x_max)
        
        self.fig.canvas.draw_idle()

# =======================
# Main Application
# =======================
def main():
    print("\n" + "="*60)
    print("STARTING FORCE PREDICTION COMPARISON TOOL")
    print("="*60)
    
    print("Loading VBTS inference data...")
    try:
        vbts_data = load_vbts_inference()
        print(f"VBTS inference data loaded successfully: {vbts_data['fn']}")
        print(f"Number of frames: {vbts_data['T']}")
        print(f"Using force prediction model: {vbts_data['force_model']}")
    except Exception as e:
        print(f"Error loading VBTS inference data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    print("Loading FORCE ground truth data...")
    try:
        force_data = load_force_ground_truth()
        print(f"FORCE ground truth data loaded successfully: {force_data['fn']}")
        print(f"Number of force measurements: {len(force_data['pid_arr'])}")
    except Exception as e:
        print(f"Error loading FORCE ground truth data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    print("\nStage 1: Alignment GUI")
    print("Adjust the sliders to align the ground truth force data with the predicted force")
    print("- Use the top slider to adjust horizontal shift")
    print("- Use the bottom slider to adjust time scale (compression/expansion)")
    print("- Use the Pause button to temporarily disable slider updates")
    print("- Use the Reset button to reset alignment parameters")
    print("Click 'Finish Alignment' when satisfied with the alignment")

    alignment_gui = AlignmentGUI(vbts_data, force_data)

    if not alignment_gui.completed:
        print("Alignment cancelled by user")
        return

    shift_value = alignment_gui.shift
    scale_value = alignment_gui.scale
    print(f"\nAlignment completed with shift: {shift_value:.3f} seconds and scale: {scale_value:.4f}")

    print("\nStage 2: Comparison Visualization")
    print("Viewing the aligned comparison of predicted force vs ground truth")
    print("- Use zoom tools to examine specific regions")
    print("- Click 'Reset Zoom' to show all data again")
    print("Click 'Save Comparison' to save the plot and aligned data")
    print("Click 'Exit' to close the application")

    comparison_gui = ComparisonGUI(vbts_data, force_data, shift_value, scale_value)

    if comparison_gui.completed:
        print("\nComparison successfully saved. Application terminated.")
    else:
        print("\nComparison cancelled. Application terminated.")

if __name__ == "__main__":
    main()