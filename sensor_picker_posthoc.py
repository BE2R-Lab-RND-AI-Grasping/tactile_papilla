#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive GUI — normalized score bars for sensors A..E.

Изменения:
- Добавлены слайдеры для ввода желаемого усреднённого значения сил (desired_mu)
  и желаемого sigma (desired_sigma). Эти значения вводятся в тех же единицах,
  что и исходные сырые данные.
- Перед использованием введённые желаемые значения НЕ нормализуются заранее —
  они нормализуются относительно диапазона сырых данных (mu_min..mu_max и sigma_min..sigma_max)
  и затем используются для расчёта отклонения sensor_norm - desired_norm.
- Для µ и σ используется компонент ошибки = abs(sensor_norm - desired_norm).
  Эти компоненты умножаются на свои веса и участвуют в сумме для score.
- Остальные компоненты (pixel, marker) оставлены как раньше (marker умножается
  на процентный фактор).
- GUI обновляет результат "онлайн" при движении любых ползунков.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import DoubleVar, IntVar, BooleanVar
from math import isfinite

# ----------------- Константы (взяты из переданных логов) -----------------
SENSORS = ['A', 'B', 'C', 'D', 'E']
SENSOR_COLORS = {
    'A': '#1f77b4',  # blue
    'B': '#ff7f0e',  # orange
    'C': '#2ca02c',  # green
    'D': '#d62728',  # red
    'E': '#9467bd'   # purple
}

# Нормализованные значения mu и sigma (0..1) — как в ваших логах
FORCE_STATS = {
    'A': {'mu_norm': 0.0000, 'sigma_norm': 0.0000},
    'B': {'mu_norm': 0.3441, 'sigma_norm': 0.3289},
    'C': {'mu_norm': 0.5585, 'sigma_norm': 0.6098},
    'D': {'mu_norm': 0.9448, 'sigma_norm': 0.9591},
    'E': {'mu_norm': 1.0000, 'sigma_norm': 1.0000},
}

# Сырые (не-нормированные) значения mu и sigma — чтобы нормализовать введённые desired_* значения
FORCE_STATS_RAW = {
    'A': {'mu': 4.2269, 'sigma': 3.6958},
    'B': {'mu': 5.8997, 'sigma': 5.2292},
    'C': {'mu': 6.9413, 'sigma': 6.5392},
    'D': {'mu': 8.8194, 'sigma': 8.1677},
    'E': {'mu': 9.0876, 'sigma': 8.3586},
}
MU_MIN = min(FORCE_STATS_RAW[s]['mu'] for s in SENSORS)
MU_MAX = max(FORCE_STATS_RAW[s]['mu'] for s in SENSORS)
SIGMA_MIN = min(FORCE_STATS_RAW[s]['sigma'] for s in SENSORS)
SIGMA_MAX = max(FORCE_STATS_RAW[s]['sigma'] for s in SENSORS)

# Marker-loss normalized table (values already normalized to 0..1 in the logs)
MARKER_NORM = {
    'A': {10:0.0818, 20:0.1864, 30:0.3910, 40:0.4985, 50:0.6179, 60:0.7647, 70:0.9504},
    'B': {10:0.1015, 20:0.2003, 30:0.2931, 40:0.4849, 50:0.7040, 60:0.8383, 70:1.0000},
    'C': {10:0.0192, 20:0.1502, 30:0.2697, 40:0.3909, 50:0.5262, 60:0.6884, 70:0.9057},
    'D': {10:0.0000, 20:0.1377, 30:0.2186, 40:0.3463, 50:0.4428, 60:0.5575, 70:0.7970},
    'E': {10:0.0144, 20:0.1015, 30:0.1831, 40:0.2682, 50:0.3640, 60:0.4812, 70:0.6431},
}
AVAILABLE_PERCENTS = [10,20,30,40,50,60,70]

# Pixel-noise normalized values (0..1) from logs
PIXEL_NOISE_NORM = {
    'A': 0.0000,
    'B': 0.2016,
    'C': 0.2286,
    'D': 0.5050,
    'E': 1.0000
}

# ----------------- Helper functions -----------------
def choose_nearest_percent(pct):
    """Return nearest available percent from AVAILABLE_PERCENTS."""
    arr = AVAILABLE_PERCENTS
    return min(arr, key=lambda x: abs(x - pct))

def _normalize_value(val, vmin, vmax):
    """Normalize val to [0..1] relative to vmin..vmax. If range is zero, return 0.0."""
    try:
        vmin = float(vmin); vmax = float(vmax); val = float(val)
    except Exception:
        return 0.0
    if vmax <= vmin:
        return 0.0
    return max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))

def compute_scores(weights, percent, desired_mu_raw, desired_sigma_raw, normalize_weights=True):
    """
    weights: dict with keys 'mu','sigma','pixel','marker'
    percent: int (user-selected percent for marker-loss)
    desired_mu_raw, desired_sigma_raw: raw user inputs (in same units as FORCE_STATS_RAW)
    Returns dict sensor->score and breakdown.
    Lower score is better.
    """
    # normalize weights if requested
    wmu = float(weights.get('mu', 1.0))
    wsig = float(weights.get('sigma', 1.0))
    wpix = float(weights.get('pixel', 1.0))
    wmark = float(weights.get('marker', 1.0))
    if normalize_weights:
        s = wmu + wsig + wpix + wmark
        if s != 0:
            wmu, wsig, wpix, wmark = wmu / s, wsig / s, wpix / s, wmark / s

    # normalize desired raw inputs relative to dataset ranges
    desired_mu_norm = _normalize_value(desired_mu_raw, MU_MIN, MU_MAX)
    desired_sigma_norm = _normalize_value(desired_sigma_raw, SIGMA_MIN, SIGMA_MAX)

    pct_used = choose_nearest_percent(int(round(percent)))
    percent_factor = pct_used / 100.0

    results = {}
    for s in SENSORS:
        mu_n = FORCE_STATS[s]['mu_norm']
        sigma_n = FORCE_STATS[s]['sigma_norm']
        pixel_n = PIXEL_NOISE_NORM[s]
        marker_n = MARKER_NORM[s].get(pct_used, 0.0)

        # COMPONENTS:
        # For mu and sigma — use absolute deviation from desired normalized values
        mu_comp = abs(mu_n - desired_mu_norm)        # in [0..1]
        sigma_comp = abs(sigma_n - desired_sigma_norm)  # in [0..1]

        # Score formula: weighted sum of components (lower is better)
        score = (wmu * mu_comp) + (wsig * sigma_comp) + (wpix * pixel_n) + (wmark * percent_factor * marker_n)

        results[s] = {
            'score': float(score),
            'mu_norm': mu_n,
            'sigma_norm': sigma_n,
            'pixel_norm': pixel_n,
            'marker_norm_pct': marker_n,
            'percent_used': pct_used,
            'mu_comp': mu_comp,
            'sigma_comp': sigma_comp,
            'desired_mu_norm': desired_mu_norm,
            'desired_sigma_norm': desired_sigma_norm,
        }
    return results

# ----------------- GUI App -----------------
class ScoreBarsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor Selector — normalized score bars")
        self.root.geometry("980x520")

        # Variables (weights)
        self.w_mu = DoubleVar(value=1.0)
        self.w_sigma = DoubleVar(value=1.0)
        self.w_pixel = DoubleVar(value=1.0)
        self.w_marker = DoubleVar(value=1.0)
        self.percent = IntVar(value=30)
        self.normalize = BooleanVar(value=True)

        # Desired raw inputs (in same units as FORCE_STATS_RAW)
        default_mu = (MU_MIN + MU_MAX) / 2.0
        default_sigma = (SIGMA_MIN + SIGMA_MAX) / 2.0
        self.desired_mu = DoubleVar(value=round(default_mu, 4))
        self.desired_sigma = DoubleVar(value=round(default_sigma, 4))

        # Layout frames
        left = ttk.Frame(root, padding=10)
        left.pack(side='left', fill='y')
        right = ttk.Frame(root, padding=10)
        right.pack(side='right', fill='both', expand=True)

        # Controls (left)
        ttk.Label(left, text="Weights (move sliders)", font=('Helvetica', 11, 'bold')).pack(anchor='w', pady=(0,6))
        self._add_slider(left, "w_mu (µ weight)", self.w_mu, 0.0, 5.0)
        self._add_slider(left, "w_sigma (σ weight)", self.w_sigma, 0.0, 5.0)
        self._add_slider(left, "w_pixel", self.w_pixel, 0.0, 5.0)
        self._add_slider(left, "w_marker", self.w_marker, 0.0, 5.0)

        ttk.Label(left, text="Percent marker loss:", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(10,2))
        pct_scale = ttk.Scale(left, from_=10, to=70, orient='horizontal', variable=self.percent, command=self._on_change)
        pct_scale.pack(fill='x')
        self.pct_label = ttk.Label(left, text=f"{self.percent.get()} %")
        self.pct_label.pack(anchor='e', pady=(4,8))

        ttk.Checkbutton(left, text="Normalize weights (sum=1)", variable=self.normalize, command=self._on_change).pack(anchor='w', pady=(4,8))

        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=6)

        # Desired mu & sigma inputs
        ttk.Label(left, text="Desired (raw) values", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(6,4))
        # desired mu slider (range from MU_MIN..MU_MAX)
        self._add_slider(left, f"desired_mu (µ) [{MU_MIN:.4f}..{MU_MAX:.4f}]", self.desired_mu, MU_MIN, MU_MAX)
        # desired sigma slider
        self._add_slider(left, f"desired_sigma (σ) [{SIGMA_MIN:.4f}..{SIGMA_MAX:.4f}]", self.desired_sigma, SIGMA_MIN, SIGMA_MAX)

        ttk.Label(left, text="Notes:", font=('Helvetica', 9, 'bold')).pack(anchor='w')
        ttk.Label(left, text=("Lower score → better sensor.\n"
                              "µ/σ components = |sensor_norm - desired_norm| (чем меньше — тем ближе к желаемому).\n"
                              "Bar length = 1 - normalized_score (longer — лучше)."),
                  wraplength=260).pack(anchor='w', pady=(4,4))

        # Right: bars area + numeric table below
        self.canvas = tk.Canvas(right, bg='white', height=300)
        self.canvas.pack(fill='both', expand=False, pady=(0,8))
        self.canvas.update_idletasks()
        self.canvas_width = self.canvas.winfo_reqwidth()
        # We'll draw bars on the canvas
        self.bar_items = {}  # sensor -> dict of canvas items

        # Draw initial bars
        self._init_bars()

        # Numeric table (treeview)
        cols = ('sensor','score','mu_comp','sigma_comp','mu_norm','sigma_norm','pixel_norm','marker_norm')
        self.tree = ttk.Treeview(right, columns=cols, show='headings', height=6)
        col_widths = {'sensor':70, 'score':120, 'mu_comp':90, 'sigma_comp':90, 'mu_norm':80, 'sigma_norm':80, 'pixel_norm':80, 'marker_norm':100}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=col_widths.get(c, 100), anchor='center')
        self.tree.pack(fill='x')

        # trace variables for live update
        for var in (self.w_mu, self.w_sigma, self.w_pixel, self.w_marker, self.percent, self.desired_mu, self.desired_sigma):
            var.trace_add('write', lambda *args: self._on_change())

        # initial update
        self._on_change()

    def _add_slider(self, parent, label, var, frm, to):
        ttk.Label(parent, text=label).pack(anchor='w')
        scale = ttk.Scale(parent, from_=frm, to=to, orient='horizontal', variable=var, command=self._on_change)
        scale.pack(fill='x', pady=(0,6))
        val = ttk.Label(parent, textvariable=var)
        val.pack(anchor='e', pady=(0,4))

    def _init_bars(self):
        # Draw static labels and empty bars
        pad_x = 20
        pad_y = 20
        spacing = 38
        bar_height = 22
        left_label_w = 40
        max_w = 720  # maximum bar length in px
        y = pad_y
        self.max_bar_width = max_w
        self.left_label_w = left_label_w

        # header
        self.canvas.create_text(20, 8, anchor='nw', text="Sensor scores (normalized bars)", font=('Helvetica', 12, 'bold'))

        for i, s in enumerate(SENSORS):
            y_i = y + i * spacing
            # sensor label
            lbl = self.canvas.create_text(pad_x, y_i, anchor='nw', text=s, font=('Helvetica', 12, 'bold'))
            # background bar (grey) — full length
            bar_bg = self.canvas.create_rectangle(pad_x + left_label_w, y_i, pad_x + left_label_w + max_w, y_i + bar_height,
                                                 fill='#e6e6e6', outline='#cccccc')
            # filled bar (variable length)
            bar = self.canvas.create_rectangle(pad_x + left_label_w, y_i, pad_x + left_label_w + 10, y_i + bar_height,
                                               fill=SENSOR_COLORS[s], outline='')
            # score text at right of bar
            val_txt = self.canvas.create_text(pad_x + left_label_w + max_w + 12, y_i, anchor='nw', text='', font=('Helvetica', 10))
            # store
            self.bar_items[s] = {'label': lbl, 'bg': bar_bg, 'bar': bar, 'value': val_txt, 'y': y_i}

        # legend of normalization direction
        self.canvas.create_text(pad_x + left_label_w, y + len(SENSORS)*spacing + 8, anchor='nw',
                                text="Bar length = 1 - normalized_score (longer is better)", font=('Helvetica', 9, 'italic'))

    def _update_bars_and_table(self, results_dict):
        # results_dict: sensor -> {'score', 'mu_comp', ...}
        # compute min/max score among sensors
        scores = [results_dict[s]['score'] for s in SENSORS]
        finite_scores = [s for s in scores if isfinite(s)]
        if len(finite_scores) == 0:
            smin = 0.0; smax = 1.0
        else:
            smin = min(finite_scores)
            smax = max(finite_scores)
        srange = smax - smin

        # Update canvas bars
        for s in SENSORS:
            info = results_dict[s]
            score = info['score']
            # normalized in [0..1] where 0 = best (min), 1 = worst (max)
            if not isfinite(score):
                score_norm = 1.0
            else:
                if srange == 0:
                    score_norm = 0.0  # all equal => treat as best (full bars)
                else:
                    score_norm = (score - smin) / srange
                    score_norm = max(0.0, min(1.0, score_norm))
            # bar length = 1 - score_norm
            length_frac = 1.0 - score_norm
            bar_len = int(self.max_bar_width * length_frac)
            # update rectangle coords
            pad_x = 20; left_label_w = self.left_label_w
            y_i = self.bar_items[s]['y']
            bar_id = self.bar_items[s]['bar']
            x0 = pad_x + left_label_w
            y0 = y_i
            x1 = x0 + max(8, bar_len)  # at least small visible width
            y1 = y0 + 22
            self.canvas.coords(bar_id, x0, y0, x1, y1)
            self.canvas.itemconfigure(bar_id, fill=SENSOR_COLORS[s])
            # update numeric score text (show raw score and normalized fraction)
            score_text = f"{info['score']:.6f}  ({(1-length_frac):.3f})"
            self.canvas.itemconfigure(self.bar_items[s]['value'], text=score_text)

        # Update treeview table
        for it in self.tree.get_children():
            self.tree.delete(it)
        for s in SENSORS:
            info = results_dict[s]
            self.tree.insert('', 'end', values=(
                s,
                f"{info['score']:.6f}",
                f"{info['mu_comp']:.4f}",
                f"{info['sigma_comp']:.4f}",
                f"{info['mu_norm']:.4f}",
                f"{info['sigma_norm']:.4f}",
                f"{info['pixel_norm']:.4f}",
                f"{info['marker_norm_pct']:.4f}"
            ))

        # Highlight best sensor in title (minimal score)
        best_sensor = min(SENSORS, key=lambda x: results_dict[x]['score'] if isfinite(results_dict[x]['score']) else float('inf'))
        self.root.title(f"Sensor Selector — Best: {best_sensor} (lower score better)")

    def _on_change(self, *args):
        # gather parameters and compute
        weights = {'mu': self.w_mu.get(), 'sigma': self.w_sigma.get(), 'pixel': self.w_pixel.get(), 'marker': self.w_marker.get()}
        pct = int(round(self.percent.get()))
        normalize = bool(self.normalize.get())
        desired_mu_val = float(self.desired_mu.get())
        desired_sigma_val = float(self.desired_sigma.get())

        results = compute_scores(weights, pct, desired_mu_val, desired_sigma_val, normalize_weights=normalize)
        self._update_bars_and_table(results)

        # update percent label
        pct_used = choose_nearest_percent(pct)
        self.pct_label.config(text=f"{pct_used} % (used)")
        # optionally could display desired normalized values somewhere — skip for brevity

# ----------------- Run -----------------
def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    app = ScoreBarsApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
