#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBTS + FORCE — Two-stage GUI for manual synchronization and peak annotation.

Stage 1: Alignment
 - Shows mean displacement (black) and force (blue) on twin axes for visual alignment
 - Slider 1 adjusts horizontal shift of force data
 - Slider 2 adjusts horizontal scale (time compression/expansion)
 - Finish alignment button closes this window and proceeds to stage 2

Stage 2: Annotation
 - Shows aligned data in two separate plots (VBTS markers & mean on top, force below)
 - Vertical line follows mouse cursor
 - Click to place synchronized markers (red dots with vertical line)
 - Finish labeling saves annotations and closes GUI

Notes:
 - Script resolves dataset paths relative to the script file location.
 - If exact file name is not found, script will try to pick a matching file by prefix.
 - Labelled outputs are saved under Datasets/Labelled (not Datasets/VBTS/Labelled).
"""

import sys
from pathlib import Path
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# =======================
# Parameters (user can edit)
# =======================
FIG_WIDTH = 16                # Figure width in inches
DPI = 120                     # Figure DPI
TITLE_FONT = 12
AXIS_FONT = 10
TICK_FONT = 9
LEGEND_FONT = 9
ANNOT_FONT = 8
scalemax = 2.0

plt.rcParams.update({
    'figure.titlesize': TITLE_FONT,
    'axes.titlesize': TITLE_FONT,
    'axes.labelsize': AXIS_FONT,
    'xtick.labelsize': TICK_FONT,
    'ytick.labelsize': TICK_FONT,
    'legend.fontsize': LEGEND_FONT,
})

# -----------------------
# Experiment string configuration
# -----------------------
EXPERIMENT_STRING = "E_m1_i7_19_01_26"

# Parse the experiment string
parts = EXPERIMENT_STRING.split('_')
if len(parts) != 6:
    raise ValueError(f"Invalid experiment string format: {EXPERIMENT_STRING}. Expected format: <substrate>_m<membrane>_i<indentor>_<day>_<month>_<year>")

VBTS_TYPE = parts[0]

# Parse membrane flag: second part must start with 'm'
if not parts[1].startswith('m'):
    raise ValueError(f"Second part of experiment string must start with 'm': {parts[1]}")
MEMBRANE_FLAG = parts[1][1:]  # remove the 'm'

# Parse indentor index: third part must start with 'i'
if not parts[2].startswith('i'):
    raise ValueError(f"Third part of experiment string must start with 'i': {parts[2]}")
try:
    INDENTOR_IDX = int(parts[2][1:])  # remove the 'i' and convert to integer
except ValueError as e:
    raise ValueError(f"Could not parse indentor index from '{parts[2]}': {e}")

# Parse date: day, month, year
try:
    DAY = int(parts[3])
    MONTH = int(parts[4])
    YEAR = int(parts[5])
except ValueError as e:
    raise ValueError(f"Could not parse date from parts[3:6] ({parts[3]}, {parts[4]}, {parts[5]}): {e}")

# -----------------------
# VBTS Processing Parameters (user can edit)
# -----------------------
MOVING_WINDOW = 21     # Moving average window size
APPLY_SMOOTH = True    # Apply smoothing to VBTS data

# -----------------------
# FORCE Processing Parameters (user can edit)
# -----------------------
EPS_FORCE = 0.0        # Threshold to zero out negative force values

# -----------------------
# Path handling: base directory is script location (not cwd)
# -----------------------
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # __file__ may not exist in some interactive environments; fallback to cwd
    SCRIPT_DIR = Path.cwd()

DATASETS_ROOT = SCRIPT_DIR / "Datasets"
DATASET_DIR_VBTS = DATASETS_ROOT / "VBTS"
DATASET_DIR_FORCE = DATASETS_ROOT / "FORCE"
# IMPORTANT CHANGE: labelled outputs are placed under Datasets/Labelled (top-level), not inside VBTS
DATASET_DIR_LABELLED = DATASETS_ROOT / "Labelled"  # save labelled outputs here

# Ensure labelled directory exists (will be used when saving)
DATASET_DIR_LABELLED.mkdir(parents=True, exist_ok=True)

# Print diagnostics
print(f"Script directory : {SCRIPT_DIR}")
print(f"Datasets root    : {DATASETS_ROOT}")
print(f"VBTS dir         : {DATASET_DIR_VBTS}")
print(f"FORCE dir        : {DATASET_DIR_FORCE}")
print(f"Labelled dir     : {DATASET_DIR_LABELLED}")

# =======================
# Helper Functions: Filename builders and formatting
# =======================
def _fmt_two(v):
    """Return 2-digit zero-padded string for day/month, preserving already string input."""
    try:
        vi = int(v)
        return f"{vi:02d}"
    except Exception:
        s = str(v)
        return s.zfill(2) if len(s) < 2 else s

def _fmt_membrane(m):
    """
    Format membrane flag exactly as needed:
    - If m is integer, convert to string without leading zeros
    - If m is string, use it exactly as provided (preserving any leading zeros)
    - Strip whitespace from string inputs
    """
    if isinstance(m, (int, np.integer)):
        # Convert integer to string without leading zeros
        return str(int(m))
    
    # For string inputs, strip whitespace and use exactly as provided
    s = str(m).strip()
    
    # If the string represents a number but has leading zeros that shouldn't be there,
    # we keep them only if they were explicitly provided by the user
    # (this preserves user intent - if they wrote "015", they want "015")
    return s

def build_npz_filename(substrate, membrane, indentor, dd, mm, yy):
    dd_s = _fmt_two(dd)
    mm_s = _fmt_two(mm)
    yy_s = str(yy)
    membrane_s = _fmt_membrane(membrane)
    return f"{substrate}_m{membrane_s}_i{int(indentor)}_{dd_s}_{mm_s}_{yy_s}.npz"

def build_csv_filename(substrate, membrane, indentor, dd, mm, yy):
    dd_s = _fmt_two(dd)
    mm_s = _fmt_two(mm)
    yy_s = str(yy)
    membrane_s = _fmt_membrane(membrane)
    return f"{substrate}_m{membrane_s}_i{int(indentor)}_{dd_s}_{mm_s}_{yy_s}.csv"

def get_unlabeled_experiments():
    """
    Get list of experiments that exist in VBTS directory but not in Labelled directory.
    Returns a list of experiment names that need labeling.
    """
    # Get all VBTS npz files
    vbts_files = list(DATASET_DIR_VBTS.glob("*.npz"))
    
    # Get all labeled experiment folders
    labeled_folders = [f.name for f in DATASET_DIR_LABELLED.iterdir() if f.is_dir()]
    
    # Extract experiment names from VBTS filenames
    vbts_experiments = []
    for file in vbts_files:
        # Remove extension and extract experiment name
        stem = file.stem  # filename without extension
        
        # Validate the format (should be like: A_m130_i4_14_01_26)
        parts = stem.split('_')
        if len(parts) >= 6:
            # Take the first 6 parts to form the experiment name
            experiment_name = '_'.join(parts[:6])
            vbts_experiments.append(experiment_name)
    
    # Find experiments that are in VBTS but not in Labelled
    unlabeled_experiments = []
    for exp in vbts_experiments:
        # Check if there's a corresponding folder in Labelled
        if exp not in labeled_folders:
            unlabeled_experiments.append(exp)
    
    return unlabeled_experiments, vbts_experiments, labeled_folders

# =======================
# Moving Average Helper (VBTS)
# =======================
def moving_average_1d(x, window):
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

def _decorate_axes(ax):
    ax.minorticks_on()
    ax.grid(which='major', alpha=0.6)
    ax.grid(which='minor', alpha=0.5, linestyle=':')
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONT)

# =======================
# Data Loading Functions
# =======================
def load_vbts():
    fn = build_npz_filename(VBTS_TYPE, MEMBRANE_FLAG, INDENTOR_IDX, DAY, MONTH, YEAR)
    path = DATASET_DIR_VBTS / fn
    if not path.exists():
        # try to find a matching file by prefix (substrate_mXXX_iY_*)
        membrane_s = _fmt_membrane(MEMBRANE_FLAG)
        prefix = f"{VBTS_TYPE}_m{membrane_s}_i{int(INDENTOR_IDX)}_"
        candidates = list(DATASET_DIR_VBTS.glob(f"{prefix}*.npz")) if DATASET_DIR_VBTS.exists() else []
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
            chosen = candidates[0]
            print(f"[WARN] Exact VBTS filename not found: {path}")
            print(f"[INFO] Picking nearest VBTS file: {chosen.name}")
            path = chosen
        else:
            available = []
            if DATASET_DIR_VBTS.exists():
                for p in sorted(DATASET_DIR_VBTS.glob("*.npz")):
                    available.append(p.name)
            raise FileNotFoundError(f"VBTS file not found: {path}\nSearched in: {DATASET_DIR_VBTS}\nAvailable .npz files: {available}")

    data = np.load(path, allow_pickle=True)
    for k in ("positions", "baseline_centroids", "ts"):
        if k not in data:
            raise KeyError(f"VBTS NPZ missing required key '{k}' in file {path}")

    positions = data["positions"]                    # (T, N, 2)
    baseline_centroids = data["baseline_centroids"]  # (N, 2)
    ts = data["ts"]                                  # (T,)
    meta = data["meta"].item() if "meta" in data else {}

    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(f"Unexpected positions shape in VBTS: {positions.shape} (expected (T,N,2))")

    T, N, _ = positions.shape
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1 or len(ts) != T:
        raise ValueError("TS length mismatch in VBTS")

    ts_rel = ts - ts[0]
    baseline = np.asarray(baseline_centroids, dtype=float)

    # Ensure baseline matches number of markers
    if baseline.shape[0] != N:
        if baseline.shape[0] > N:
            baseline = baseline[:N]
        else:
            extra = np.zeros((N - baseline.shape[0], 2), dtype=float)
            baseline = np.vstack([baseline, extra])

    # Calculate displacements and magnitudes
    displacements_raw = positions - baseline[None, :, :]
    disp_mag_raw = np.linalg.norm(displacements_raw, axis=2)   # (T, N)
    mean_disp_raw = np.nanmean(disp_mag_raw, axis=1)

    # Apply smoothing if configured
    if APPLY_SMOOTH and MOVING_WINDOW > 1:
        positions_smooth = np.empty_like(positions, dtype=float)
        for m in range(N):
            for c in range(2):
                positions_smooth[:, m, c] = moving_average_1d(positions[:, m, c], MOVING_WINDOW)
        displacements_smooth = positions_smooth - baseline[None, :, :]
        disp_mag_smooth = np.linalg.norm(displacements_smooth, axis=2)
        mean_disp_smooth = np.nanmean(disp_mag_smooth, axis=1)
    else:
        positions_smooth = positions.copy()
        displacements_smooth = displacements_raw.copy()
        disp_mag_smooth = disp_mag_raw.copy()
        mean_disp_smooth = mean_disp_raw.copy()

    return {
        "fn": path.name,
        "ts_rel": ts_rel,
        "disp_mag_smooth": disp_mag_smooth,
        "mean_disp_smooth": mean_disp_smooth,
        "positions_smooth": positions_smooth,
        "N": N,
        "T": T,
        "meta": meta,
    }

def load_force():
    fn = build_csv_filename(VBTS_TYPE, MEMBRANE_FLAG, INDENTOR_IDX, DAY, MONTH, YEAR)
    path = DATASET_DIR_FORCE / fn
    if not path.exists():
        # try to find matching CSV by prefix
        membrane_s = _fmt_membrane(MEMBRANE_FLAG)
        prefix = f"{VBTS_TYPE}_m{membrane_s}_i{int(INDENTOR_IDX)}_"
        candidates = list(DATASET_DIR_FORCE.glob(f"{prefix}*.csv")) if DATASET_DIR_FORCE.exists() else []
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
            chosen = candidates[0]
            print(f"[WARN] Exact FORCE filename not found: {path}")
            print(f"[INFO] Picking nearest FORCE file: {chosen.name}")
            path = chosen
        else:
            available = []
            if DATASET_DIR_FORCE.exists():
                for p in sorted(DATASET_DIR_FORCE.glob("*.csv")):
                    available.append(p.name)
            raise FileNotFoundError(f"FORCE file not found: {path}\nSearched in: {DATASET_DIR_FORCE}\nAvailable .csv files: {available}")

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

        # Create mapping from force PID to VBTS time range
        ts = self.vbts["ts_rel"]
        pid = self.force["pid_arr"]
        if pid.size == 0:
            raise ValueError("Force PID array is empty.")
        self.force_times_base = np.interp(pid,
                                         (pid.min(), pid.max()),
                                         (float(ts.min()), float(ts.max())))

        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(FIG_WIDTH, 6), dpi=DPI)
        self.ax2 = self.ax.twinx()  # Twin axes for force data

        # Plot VBTS mean displacement (black line)
        self.ax.plot(self.vbts["ts_rel"], self.vbts["mean_disp_smooth"],
                     'k-', linewidth=2.0, alpha=0.95, label='среднее сглаженное')
        self.ax.set_xlabel("Время, с")
        self.ax.set_ylabel("Смещение, пкс", color='k')
        self.ax.tick_params(axis='y', labelcolor='k')
        self.ax.set_title(f"Выравнивание временных шкал (смещение и масштаб)\nVBTS: {vbts['fn']}, FORCE: {force['fn']}")

        # Plot FORCE data (blue line) - initially unshifted and unscaled
        self.force_line, = self.ax2.plot(self.force_times_base, self.force["force_for_plot"],
                                         'b-', linewidth=1.6, alpha=0.95, label='Сила, Н')
        self.ax2.set_ylabel("Сила, Н", color='b')
        self.ax2.tick_params(axis='y', labelcolor='b')

        # Decorate axes
        _decorate_axes(self.ax)
        _decorate_axes(self.ax2)

        # Disable autoscaling for VBTS (keep fixed y-limits)
        y_min = np.nanmin(self.vbts["mean_disp_smooth"])
        y_max = np.nanmax(self.vbts["mean_disp_smooth"])
        # Add tiny padding
        pad = max(1e-6, 0.02 * (y_max - y_min if y_max > y_min else 1.0))
        self.ax.set_ylim(y_min - pad, y_max + pad)
        self.ax.set_xlim(float(self.vbts["ts_rel"].min()), float(self.vbts["ts_rel"].max()))
        self.ax.autoscale(enable=False)

        # Adjust layout for controls
        plt.subplots_adjust(bottom=0.35)

        # Create slider for time shift
        slider_shift_ax = plt.axes([0.2, 0.22, 0.65, 0.03])
        ts_span = float(self.vbts["ts_rel"].max() - self.vbts["ts_rel"].min())
        self.slider_shift = Slider(
            slider_shift_ax, 'Сдвиг (с)', -ts_span, ts_span,
            valinit=0.0, valstep=0.001,
            color='#5DADE2'
        )
        self.slider_shift.on_changed(self.update_plot)

        # Create slider for time scale
        slider_scale_ax = plt.axes([0.2, 0.12, 0.65, 0.03])
        self.slider_scale = Slider(
            slider_scale_ax, 'Масштаб', 0.6, scalemax,
            valinit=1.0, valstep=0.0001,
            color='#AF7AC5'
        )
        self.slider_scale.on_changed(self.update_plot)

        # Create buttons
        finish_ax = plt.axes([0.4, 0.02, 0.2, 0.04])
        self.btn_finish = Button(finish_ax, 'Finish alignment', color='#5DADE2', hovercolor='#85C1E9')
        self.btn_finish.on_clicked(self.finish_alignment)

        exit_ax = plt.axes([0.7, 0.02, 0.1, 0.04])
        self.btn_exit = Button(exit_ax, 'Exit', color='#E74C3C', hovercolor='#F1948A')
        self.btn_exit.on_clicked(self.exit_gui)

        # Force initial autoscaling for force plot's y-axis only
        self.ax2.relim()
        self.ax2.autoscale_view(scalex=False, scaley=True)

        # Show plot (block until closed)
        plt.show(block=True)

    def update_plot(self, val):
        """Update force graph position based on slider values"""
        self.shift = float(self.slider_shift.val)
        self.scale = float(self.slider_scale.val)

        # Apply scale then shift
        new_x = (self.force_times_base * self.scale) + self.shift
        self.force_line.set_xdata(new_x)

        # Autoscale only force y-axis
        self.ax2.relim()
        self.ax2.autoscale_view(scalex=False, scaley=True)

        # Update title with current parameters
        self.ax.set_title(f"Выравнивание временных шкал (смещение и масштаб)\nVBTS: {self.vbts['fn']}, FORCE: {self.force['fn']}\n"
                          f"Сдвиг: {self.shift:.3f} с, Масштаб: {self.scale:.4f}")

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

# =======================
# Stage 2: Annotation GUI
# =======================
class AnnotationGUI:
    def __init__(self, vbts, force, shift, scale):
        self.vbts = vbts
        self.force = force
        self.shift = shift
        self.scale = scale
        self.annotations = []

        # Calculate shifted and scaled force times
        ts = self.vbts["ts_rel"]
        pid = self.force["pid_arr"]
        self.force_times_base = np.interp(pid,
                                         (pid.min(), pid.max()),
                                         (float(ts.min()), float(ts.max())))

        # Apply scaling and shifting
        self.force_times = (self.force_times_base * self.scale) + self.shift

        # Create figure with two subplots
        self.fig = plt.figure(figsize=(FIG_WIDTH, 8), dpi=DPI)
        gs = self.fig.add_gridspec(6, 1)
        self.ax_vbts = self.fig.add_subplot(gs[0:3, 0])
        self.ax_force = self.fig.add_subplot(gs[3:5, 0], sharex=self.ax_vbts)
        control_ax = self.fig.add_subplot(gs[5, 0])
        control_ax.axis('off')

        # Plot VBTS data: all markers and mean
        N = self.vbts["N"]
        for i in range(N):
            self.ax_vbts.plot(self.vbts["ts_rel"], self.vbts["disp_mag_smooth"][:, i],
                              linewidth=0.9, alpha=0.7, label=f'm{i}')

        # Plot VBTS mean (thick black line)
        self.ax_vbts.plot(self.vbts["ts_rel"], self.vbts["mean_disp_smooth"],
                          linewidth=2.0, alpha=0.95, label='среднее сглаженное', color='k')
        self.ax_vbts.set_title(f"Модули смещений по маркерам (файл: {self.vbts['fn']})")
        self.ax_vbts.set_ylabel("смещ., пкс")
        _decorate_axes(self.ax_vbts)
        self.ax_vbts.legend(ncol=4, fontsize=LEGEND_FONT)

        # Plot FORCE data with applied shift and scale
        self.ax_force.plot(self.force_times, self.force["force_for_plot"],
                           linestyle='-', linewidth=1.6, alpha=0.95,
                           label='F, Н', color='#5DADE2')
        self.ax_force.set_title(f"Сигнал силы (файл: {self.force['fn']}, сдвиг: {self.shift:.3f} с, масштаб: {self.scale:.4f})")
        self.ax_force.set_xlabel("t, с")
        self.ax_force.set_ylabel("F, Н")
        _decorate_axes(self.ax_force)
        self.ax_force.legend(fontsize=LEGEND_FONT)

        # Vertical lines for mouse tracking
        self.vline_vbts = self.ax_vbts.axvline(x=0, color='gray', linestyle='--', alpha=0.6, visible=False)
        self.vline_force = self.ax_force.axvline(x=0, color='gray', linestyle='--', alpha=0.6, visible=False)

        # Storage for placed annotations
        self.placed_points = []

        # Create buttons
        btn_w = 0.16
        btn_h = 0.7
        margin = 0.01
        left = 0.02

        # Finish labeling button
        ax_finish = self.fig.add_axes([left, 0.02, btn_w, 0.05])
        self.btn_finish = Button(ax_finish, 'Finish labeling', color='#2ECC71', hovercolor='#58D68D')
        self.btn_finish.on_clicked(self.finish_labeling)

        # Clear last button
        left += btn_w + margin
        ax_clear = self.fig.add_axes([left, 0.02, btn_w, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear last', color='#F39C12', hovercolor='#F8C471')
        self.btn_clear.on_clicked(self.clear_last)

        # Clear all button
        left += btn_w + margin
        ax_clear_all = self.fig.add_axes([left, 0.02, btn_w, 0.05])
        self.btn_clear_all = Button(ax_clear_all, 'Clear all', color='#E74C3C', hovercolor='#F1948A')
        self.btn_clear_all.on_clicked(self.clear_all)

        # Exit button
        left += btn_w + margin
        ax_exit = self.fig.add_axes([left, 0.02, btn_w, 0.05])
        self.btn_exit = Button(ax_exit, 'Exit', color='#95A5A6', hovercolor='#BDC3C7')
        self.btn_exit.on_clicked(self.exit_gui)

        # Connect mouse events
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Set title and layout
        self.fig.suptitle(f"Ручная разметка синхронизированных данных (сдвиг: {self.shift:.3f} с, масштаб: {self.scale:.4f})")
        plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.12)

        # Show plot
        plt.show(block=True)

    def on_mouse_move(self, event):
        """Show vertical tracking line when mouse is over plots"""
        if event.inaxes in [self.ax_vbts, self.ax_force] and event.xdata is not None:
            self.vline_vbts.set_xdata(event.xdata)
            self.vline_vbts.set_visible(True)
            self.vline_force.set_xdata(event.xdata)
            self.vline_force.set_visible(True)
            self.fig.canvas.draw_idle()
        else:
            if self.vline_vbts.get_visible() or self.vline_force.get_visible():
                self.vline_vbts.set_visible(False)
                self.vline_force.set_visible(False)
                self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Place annotation markers on left mouse click"""
        if event.button != 1:  # Left click only
            return

        if event.inaxes not in [self.ax_vbts, self.ax_force] or event.xdata is None:
            return

        x_click = float(event.xdata)

        # Find nearest VBTS index
        idx_v = int(np.argmin(np.abs(self.vbts["ts_rel"] - x_click)))
        t_v = float(self.vbts["ts_rel"][idx_v])
        vb_val = float(self.vbts["mean_disp_smooth"][idx_v])

        # Find nearest FORCE index
        idx_f = int(np.argmin(np.abs(self.force_times - x_click)))
        t_f = float(self.force_times[idx_f])
        f_val = float(self.force["force_for_plot"][idx_f])

        # Draw red markers
        p1, = self.ax_vbts.plot(t_v, vb_val, marker='o', markersize=8,
                                markerfacecolor='red', markeredgecolor='k', zorder=20)
        p2, = self.ax_force.plot(t_f, f_val, marker='o', markersize=8,
                                 markerfacecolor='red', markeredgecolor='k', zorder=20)

        # Draw vertical line
        vline_vbts = self.ax_vbts.axvline(x=t_v, color='k', linewidth=1.0, alpha=0.7)
        vline_force = self.ax_force.axvline(x=t_v, color='k', linewidth=1.0, alpha=0.7)

        # Store annotation
        annot = {
            'vbts_time': t_v,
            'vbts_idx': idx_v,
            'vbts_val': vb_val,
            'force_time_shifted': t_f,
            'force_idx': idx_f,
            'force_val': f_val,
            'applied_shift': self.shift,
            'applied_scale': self.scale
        }
        self.annotations.append(annot)

        # Store plot elements for potential removal
        self.placed_points.append((p1, p2, vline_vbts, vline_force))

        self.fig.canvas.draw_idle()

    def clear_last(self, event):
        """Remove the last placed annotation"""
        if self.annotations:
            # Remove last annotation data
            self.annotations.pop()

            # Remove last plot elements
            p1, p2, vline_vbts, vline_force = self.placed_points.pop()
            try:
                p1.remove()
                p2.remove()
                vline_vbts.remove()
                vline_force.remove()
            except Exception:
                pass

            self.fig.canvas.draw_idle()

    def clear_all(self, event):
        """Remove all annotations"""
        # Clear data
        self.annotations.clear()

        # Clear plot elements
        for p1, p2, vline_vbts, vline_force in self.placed_points:
            try:
                p1.remove()
                p2.remove()
                vline_vbts.remove()
                vline_force.remove()
            except Exception:
                pass

        self.placed_points.clear()
        self.fig.canvas.draw_idle()

    def finish_labeling(self, event):
        """Save annotations and close GUI"""
        if not self.annotations:
            print("No annotations to save. Please mark at least one point before finishing.")
            plt.close(self.fig)
            return

        # Number of markers and annotations
        M = self.vbts["N"]  # Number of markers
        N = len(self.annotations)  # Number of annotations

        # Prepare inputs data: MxNx2 (marker_count x annotation_count x coordinates)
        inputs_data = np.zeros((M, N, 2), dtype=float)

        # Prepare outputs data: N (force values at annotation points)
        outputs_data = np.zeros(N, dtype=float)

        # Fill inputs and outputs data
        for j, annot in enumerate(self.annotations):
            vbts_idx = annot['vbts_idx']
            # Get positions for all markers at this time
            for i in range(M):
                inputs_data[i, j, 0] = self.vbts["positions_smooth"][vbts_idx, i, 0]
                inputs_data[i, j, 1] = self.vbts["positions_smooth"][vbts_idx, i, 1]
            # Get force value
            outputs_data[j] = annot['force_val']

        # Generate experiment folder name and create it under Datasets/Labelled (top-level)
        membrane_s = _fmt_membrane(MEMBRANE_FLAG)
        experiment_name = f"{VBTS_TYPE}_m{membrane_s}_i{INDENTOR_IDX}_{_fmt_two(DAY)}_{_fmt_two(MONTH)}_{str(YEAR)}"
        experiment_dir = DATASET_DIR_LABELLED / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Generate input and output filenames
        inputs_csv_out = experiment_dir / f"inputs_{experiment_name}.csv"
        outputs_csv_out = experiment_dir / f"outputs_{experiment_name}.csv"
        annotations_csv_out = experiment_dir / f"annotations_{experiment_name}.csv"

        # Save inputs data (flatten for CSV)
        inputs_flat = []
        for i in range(M):
            for j in range(N):
                inputs_flat.append([
                    f"marker_{i}",
                    float(inputs_data[i, j, 0]),
                    float(inputs_data[i, j, 1]),
                    int(j)  # annotation index
                ])

        inputs_df = pd.DataFrame(inputs_flat, columns=['marker_id', 'x_coord', 'y_coord', 'annotation_idx'])
        inputs_df.to_csv(inputs_csv_out, index=False, encoding='utf-8')

        # Save outputs data
        outputs_df = pd.DataFrame({
            'annotation_idx': list(range(N)),
            'force_value_N': outputs_data
        })
        outputs_df.to_csv(outputs_csv_out, index=False, encoding='utf-8')

        # Save annotations details
        annotations_df = pd.DataFrame(self.annotations)
        annotations_df.to_csv(annotations_csv_out, index=False, encoding='utf-8')

        print(f"Data saved successfully:")
        print(f"- Inputs (marker positions): {inputs_csv_out}")
        print(f"- Outputs (force values): {outputs_csv_out}")
        print(f"- Annotations details: {annotations_csv_out}")

        plt.close(self.fig)

    def exit_gui(self, event):
        """Exit without saving"""
        plt.close(self.fig)

# =======================
# Main Application
# =======================
def main():
    # Check for unlabeled experiments
    print("\n" + "="*60)
    print("ANALYZING DATASETS FOLDERS")
    print("="*60)
    
    unlabeled, all_vbts, labeled = get_unlabeled_experiments()
    
    print(f"Total VBTS experiments found: {len(all_vbts)}")
    print(f"Already labeled experiments: {len(labeled)}")
    print(f"Experiments needing labeling: {len(unlabeled)}")
    
    if unlabeled:
        print("\n" + "-"*40)
        print("UNLABELED EXPERIMENTS:")
        print("-"*40)
        for i, exp in enumerate(unlabeled, 1):
            print(f"{i}. {exp}")
    else:
        print("\nAll VBTS experiments have been labeled already!")
    
    print("\n" + "="*60)
    print("STARTING GUI FOR MANUAL LABELING")
    print("="*60)
    
    print("Loading VBTS data...")
    try:
        vbts_data = load_vbts()
        print(f"VBTS data loaded successfully: {vbts_data['fn']}")
    except Exception as e:
        print(f"Ошибка загрузки VBTS: {e}", file=sys.stderr)
        return

    print("Loading FORCE data...")
    try:
        force_data = load_force()
        print(f"FORCE data loaded successfully: {force_data['fn']}")
    except Exception as e:
        print(f"Ошибка загрузки FORCE: {e}", file=sys.stderr)
        return

    print("\nStage 1: Alignment GUI")
    print("Adjust the sliders to align the blue force graph with the black displacement graph")
    print("- Use the top slider to adjust horizontal shift")
    print("- Use the bottom slider to adjust time scale (compression/expansion)")
    print("Click 'Finish alignment' when satisfied with the alignment")

    alignment_gui = AlignmentGUI(vbts_data, force_data)

    if not alignment_gui.completed:
        print("Alignment cancelled by user")
        return

    shift_value = alignment_gui.shift
    scale_value = alignment_gui.scale
    print(f"\nAlignment completed with shift: {shift_value:.3f} seconds and scale: {scale_value:.4f}")

    print("\nStage 2: Annotation GUI")
    print("Move mouse to see alignment line")
    print("Click to place synchronized markers")
    print("Use 'Clear last' or 'Clear all' to remove markers")
    print("Click 'Finish labeling' to save annotations and exit")

    AnnotationGUI(vbts_data, force_data, shift_value, scale_value)

    print("\nAnnotation completed. Application terminated.")

if __name__ == "__main__":
    main()