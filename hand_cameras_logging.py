# Knowledge Base
## The content from [File](Pasted_Text_1770065997288.txt):
"""
VBTS TRI-CAMERA CONTACT TRACKER WITH FORCE PREDICTION AND CLUSTER-BASED NORMALIZATION
SINGLE-FRAME CAPTURE MODE: Press 'a' to capture ONE frame (mutually exclusive with continuous mode)
CONTINUOUS CAPTURE MODE: Press 'z' to toggle continuous frame-by-frame recording (mutually exclusive with single mode)
CLUSTER-BASED NORMALIZATION: All markers from all cameras clustered to find reference P-state (centers of mass)
DEFORMATION CALCULATION: Vectors computed relative to cluster-based reference state
POLYTOPE VISUALIZATION: REMOVED FROM FINAL PLOT (only centroids shown)
FORCE FILTERING: Outlier removal + median smoothing applied to force plots
IMPORTANT FIXES:
1. Fixed UnboundLocalError: heatmap_vis always initialized before use
2. Added missing compute_deformation_vectors_and_l2_norm() function
3. Fixed baseline capture crash when pressing 'b' during force prediction
4. Improved heatmap processing stability with proper initialization
5. Corrected directory path: 'rresults' -> 'results'
6. Two mutually exclusive capture modes: 'a' (single frame) and 'z' (continuous)
7. Cluster-based normalization across all cameras (KMeans centers of mass)
8. Deformation vectors computed relative to cluster-based reference state
9. POLYTOPES REMOVED FROM FINAL PLOT - ONLY CENTROIDS AND REFERENCE P-STATE SHOWN
10. OpenMP conflict resolved via environment variables set BEFORE imports
11. Raw centroid coordinates logged WITHOUT normalization for physics accuracy
12. Forces properly predicted and logged via neural network with dimension validation
13. AREA CONVERSION: Full camera view = 875 mm² → convert pixel area to mm²
14. GUI TEXT IN ENGLISH ONLY (to avoid encoding issues with Cyrillic in OpenCV)
15. ALL PLOT TEXT IN RUSSIAN ONLY (axes, legends, titles fully localized)
16. LINE-BASED force/area plots with small markers at data points, X-axis labeled as "Кадр"
17. FORCE FILTERING: Outlier removal (IQR method) + median smoothing (window=5) applied to plots
18. VERTICAL LEGEND layout for better readability in upper right corner
19. SEPARATE SVG EXPORT for each plot type + combined PNG
20. FIXED NameError: sensor_type → SENSOR_TYPE
21. INCREASED auto-baseline threshold frames for stability (10 → 30)
22. ALL DATA SAVED TO TIMESTAMPED SUBDIRECTORY (e.g., results/hand_cameras_inference/logdata_YYMMDD_HHMMSS/)
"""
import os
import sys
# CRITICAL: Set OpenMP environment variables BEFORE any other imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# Force matplotlib to use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import time
import threading
from collections import deque
from pathlib import Path
import json
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# ------------------ Sensor Type Configuration -----------------
SENSOR_TYPE = "A"  # Change this to match your sensor type
FORCE_BIAS = 2.8
NO_CONTACT_THRESHOLD_FRAMES = 30  # Increased from 10 to 30 for more stable auto-baseline
NO_CONTACT_THRESHOLD_FORCE = 0.3
# ------------------ Camera / capture parameters -----------------
CAM_IDS = [0, 1, 2]
BACKEND = getattr(cv2, "CAP_MSMF", None) if os.name == 'nt' else None
FOURCC = "MJPG"
REQ_W, REQ_H = 1920, 1080
REQ_FPS = 30.0
CAP_BUFFERSIZE = 1
# ------------------ Marker detection & tracking params -----------
MARKER_CONFIG = {
'A': {'count': 12, 'grid': '3x4'},
'B': {'count': 15, 'grid': '3x5'},
'C': {'count': 20, 'grid': '4x5'},
'D': {'count': 24, 'grid': '4x6'},
'E': {'count': 30, 'grid': '5x6'}
}
N_MARKERS = MARKER_CONFIG[SENSOR_TYPE]['count']
GRID_DESCRIPTION = MARKER_CONFIG[SENSOR_TYPE]['grid']
print(f"Selected sensor type: {SENSOR_TYPE} with {N_MARKERS} markers ({GRID_DESCRIPTION} grid)")
MIN_MARKER_AREA = 600
MAX_MARKER_AREA = 20000
MAX_MATCH_DIST = 50.0
PERSISTENT_FRAMES = 1
DISP_THRESHOLD = 2.0
EMA_ALPHA = 0.6
HEAT_ADD_SCALE = 1.0
HEAT_RADIUS = 60
HEAT_DECAY = 0.4
HEAT_THRESHOLD = 4.0
HEATMAP_BLUR_SIGMA = 12
CONTACT_CENTROID_EMA_ALPHA = 0.8
TOLERANCE_H = 15
TOLERANCE_S = 70
TOLERANCE_V = 70
AUTO_CALIB_FRAMES = 30
# ------------------ Area conversion parameters -----------------
TOTAL_CAMERA_VIEW_MM2 = 875.0  # Total visible area of camera view in mm²
TOTAL_CAMERA_VIEW_PIXELS = REQ_W * REQ_H  # Total pixels in camera frame
PIXELS_PER_MM2 = TOTAL_CAMERA_VIEW_PIXELS / TOTAL_CAMERA_VIEW_MM2
print(f"Camera view: {REQ_W}x{REQ_H} px = {TOTAL_CAMERA_VIEW_PIXELS} px²")
print(f"Full view area: {TOTAL_CAMERA_VIEW_MM2} mm²")
print(f"Conversion factor: 1 mm² = {PIXELS_PER_MM2:.2f} px²")
# ------------------ Force filtering parameters -----------------
FORCE_OUTLIER_IQR_FACTOR = 1.5  # IQR multiplier for outlier detection
FORCE_SMOOTHING_WINDOW = 5      # Median filter window size for smoothing
print(f"Force filtering: IQR factor={FORCE_OUTLIER_IQR_FACTOR}, smoothing window={FORCE_SMOOTHING_WINDOW}")
# ------------------ Default HSV masks (MUST be defined BEFORE GreenDetector class) ----------
DEFAULT_LOWER_HSV1 = np.array([35, 100, 50], dtype=np.uint8)
DEFAULT_UPPER_HSV1 = np.array([65, 255, 200], dtype=np.uint8)
DEFAULT_LOWER_HSV2 = np.array([70, 80, 70], dtype=np.uint8)
DEFAULT_UPPER_HSV2 = np.array([95, 200, 180], dtype=np.uint8)
# ------------------ Neural network inference parameters ------------------
MODEL_DIR = Path(f"models/{SENSOR_TYPE}_deformation_vectors_force_prediction")
MODEL_PATH = MODEL_DIR / "force_prediction_model.pth"
DEFORMATION_SCALER_PATH = MODEL_DIR / "deformation_scaler.json"
REFERENCE_P_PATH = MODEL_DIR / "reference_P.npy"
# ------------------ Base capture directory -----------------
BASE_CAPTURE_DIR = Path("results/hand_cameras_inference")  # Base directory for all sessions
BASE_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Base capture directory: {BASE_CAPTURE_DIR.absolute()}")

# ------------------ Session-specific capture directory (created at runtime) -----------------
SESSION_TIMESTAMP = time.strftime("%y%m%d_%H%M%S")  # Format: YYMMDD_HHMMSS
SESSION_CAPTURE_DIR = BASE_CAPTURE_DIR / f"logdata_{SESSION_TIMESTAMP}"
SESSION_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Session capture directory: {SESSION_CAPTURE_DIR.absolute()}")
print(f"All data will be saved to this subdirectory to avoid file clutter")
# ------------------ PyTorch Network Definition ------------------
class ForcePredictionNet(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate=0.0, use_batchnorm=False):
        super(ForcePredictionNet, self).__init__()
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

# ------------------ Force filtering utilities ------------------
def remove_outliers_iqr(data, factor=1.5):
    """Remove outliers using Interquartile Range (IQR) method"""
    if len(data) < 4:
        return data.copy()
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    filtered = np.where((data >= lower_bound) & (data <= upper_bound), data, np.nan)
    return filtered

def median_smooth(data, window_size=5):
    """Apply median smoothing to data array"""
    if len(data) < window_size:
        return data.copy()
    smoothed = np.copy(data)
    half_window = window_size // 2
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        # Filter out NaN values for median calculation
        valid_vals = window[~np.isnan(window)]
        if len(valid_vals) > 0:
            smoothed[i] = np.median(valid_vals)
        else:
            smoothed[i] = np.nan
    return smoothed

def filter_force_data(forces, iqr_factor=1.5, smoothing_window=5):
    """
    Apply complete force filtering pipeline:
    1. Remove outliers using IQR method per camera
    2. Apply median smoothing to remove noise
    Returns filtered forces array with same shape
    """
    n_frames, n_cameras = forces.shape
    filtered_forces = np.zeros_like(forces)
    for cam_idx in range(n_cameras):
        cam_forces = forces[:, cam_idx].copy()
        # Step 1: Remove outliers using IQR
        cam_forces_no_outliers = remove_outliers_iqr(cam_forces, factor=iqr_factor)
        # Step 2: Apply median smoothing
        cam_forces_smoothed = median_smooth(cam_forces_no_outliers, window_size=smoothing_window)
        # Fill any remaining NaNs with nearest valid value
        valid_indices = np.where(~np.isnan(cam_forces_smoothed))[0]
        if len(valid_indices) > 0:
            for i in range(len(cam_forces_smoothed)):
                if np.isnan(cam_forces_smoothed[i]):
                    # Find nearest valid index
                    nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - i))]
                    cam_forces_smoothed[i] = cam_forces_smoothed[nearest_idx]
        else:
            # If all values are NaN, fill with zeros
            cam_forces_smoothed = np.zeros_like(cam_forces_smoothed)
        filtered_forces[:, cam_idx] = cam_forces_smoothed
    return filtered_forces

# ------------------ Marker matching utilities ------------------
def match_points_hungarian(ref_points, cur_points, max_distance=MAX_MATCH_DIST):
    if len(ref_points) == 0 or len(cur_points) == 0:
        return [], []
    cost_matrix = cdist(ref_points, cur_points)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    confidences = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] <= max_distance:
            matches.append((i, j))
            confidence = 1.0 - min(1.0, cost_matrix[i, j] / max_distance)
            confidences.append(confidence)
    return matches, confidences

def sort_points_left_to_right_top_to_bottom(points):
    """Sort points left-to-right and top-to-bottom"""
    points_copy = points.copy()
    sorted_indices = np.lexsort((points_copy[:, 0], points_copy[:, 1]))
    sorted_points = points_copy[sorted_indices]
    return sorted_points, sorted_indices

def find_nearest_correspondences(reference_points, target_points):
    """Find nearest neighbor correspondences between reference and target points"""
    n_points = len(reference_points)
    finite_mask = np.all(np.isfinite(target_points), axis=1)
    if not np.all(finite_mask):
        num_finite = np.sum(finite_mask)
        if num_finite < n_points:
            return np.arange(n_points), np.full(n_points, np.nan)
        target_points_finite = target_points[finite_mask]
        original_indices = np.where(finite_mask)[0]
    else:
        target_points_finite = target_points
        original_indices = np.arange(len(target_points))
    
    try:
        from scipy.spatial import KDTree
        tree = KDTree(target_points_finite)
        distances, indices = tree.query(reference_points, k=1)
        original_indices_mapped = original_indices[indices]
        return original_indices_mapped, distances
    except Exception as e:
        return np.arange(n_points), np.full(n_points, np.nan)

def compute_deformation_vectors(reference_points, target_points, correspondences):
    """Compute deformation vectors as Q - P"""
    n_points = len(reference_points)
    deformation_vectors = np.zeros((n_points, 2))
    for i in range(n_points):
        ref_point = reference_points[i]
        target_idx = correspondences[i]
        if target_idx < len(target_points) and np.all(np.isfinite(target_points[target_idx])):
            target_point = target_points[target_idx]
            deformation_vectors[i] = target_point - ref_point
        else:
            deformation_vectors[i] = np.array([0.0, 0.0])
    return deformation_vectors

def compute_total_l2_norm(deformation_vectors):
    """Compute sum of L2 norms of deformation vectors"""
    l2_norms = np.linalg.norm(deformation_vectors, axis=1)
    return np.sum(l2_norms)

# CRITICAL FIX: Added missing function for deformation computation
def compute_deformation_vectors_and_l2_norm(current_positions, reference_positions):
    """Compute deformation vectors and total L2 norm relative to reference state"""
    if current_positions is None or reference_positions is None:
        return np.zeros((N_MARKERS, 2)), 0.0, False
    # Find correspondences between current and reference positions
    correspondences, _ = find_nearest_correspondences(reference_positions, current_positions)
    # Compute deformation vectors
    deformation_vectors = compute_deformation_vectors(reference_positions, current_positions, correspondences)
    # Compute total L2 norm
    total_l2_norm = compute_total_l2_norm(deformation_vectors)
    return deformation_vectors, total_l2_norm, True

# ------------------ Camera probing / reader -----------------------
def probe_available_cameras(max_probe=8, test_frames=3, wait_s=0.1):
    available = []
    for i in range(0, max_probe + 1):
        try:
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                try: cap.release()
                except Exception: pass
                continue
            frames_ok = 0
            for _ in range(test_frames):
                ok, _ = cap.read()
                if ok:
                    frames_ok += 1
                time.sleep(wait_s)
            cap.release()
            if frames_ok >= 2:
                available.append(i)
        except Exception:
            continue
    return available

def open_camera_with_settings(cam_id, req_w, req_h, req_fps, fourcc):
    if BACKEND is not None:
        cap = cv2.VideoCapture(int(cam_id), BACKEND)
    else:
        cap = cv2.VideoCapture(int(cam_id))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(req_w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(req_h))
        cap.set(cv2.CAP_PROP_FPS, float(req_fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(CAP_BUFFERSIZE))
    except Exception:
        pass
    time.sleep(0.3)
    for i in range(5):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            if w > 100 and h > 100:
                return cap
        time.sleep(0.1)
    cap.release()
    return None

class CameraReader(threading.Thread):
    def __init__(self, cam_id, cap):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.running = True
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
    
    def run(self):
        while self.running:
            try:
                ok, frame = self.cap.read()
                if ok and frame is not None and frame.size > 0:
                    with self.lock:
                        self.frame = frame.copy()
                        self.ok = True
                    self.fps_counter += 1
                    now = time.time()
                    if now - self.last_fps_time >= 1.0:
                        self.current_fps = self.fps_counter / (now - self.last_fps_time)
                        self.fps_counter = 0
                        self.last_fps_time = now
                else:
                    with self.lock:
                        self.ok = False
                time.sleep(0.01)
            except Exception:
                time.sleep(0.1)
                try:
                    self.cap.release()
                except Exception:
                    pass
    
    def get(self):
        with self.lock:
            if self.frame is None:
                return False, None, 0.0
            return self.ok, self.frame.copy(), self.current_fps
    
    def stop(self):
        self.running = False
        self.join(timeout=2.0)

def open_camera_reader(cam_id, req_w, req_h, req_fps, fourcc):
    cap = open_camera_with_settings(cam_id, req_w, req_h, req_fps, fourcc)
    if cap is None:
        return None
    reader = CameraReader(cam_id, cap)
    reader.start()
    start_wait = time.time()
    while time.time() - start_wait < 2.0:
        ok, frame, fps = reader.get()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            return reader
        time.sleep(0.05)
    reader.stop()
    return None

# ------------------ Heatmap utilities -----------------------------------
def add_heat_at(heatmap, center, magnitude, radius=HEAT_RADIUS):
    x, y = int(round(center[0])), int(round(center[1]))
    h, w = heatmap.shape
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    dx = xs - x
    dy = ys - y
    dist2 = dx.astype(np.float32)**2 + dy.astype(np.float32)**2
    sigma = max(1.0, radius / 2.0)
    gauss = np.exp(-dist2 / (2 * sigma * sigma))
    gauss *= magnitude
    heatmap[y0:y1, x0:x1] += gauss

def heatmap_to_colormap(heatmap):
    if heatmap is None or heatmap.size == 0:
        return None
    maxv = float(np.max(heatmap))
    if maxv <= 0:
        gray = np.zeros_like(heatmap, dtype=np.uint8)
    else:
        gray = np.clip((heatmap / maxv) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colored

# ------------------ GreenDetector (NOW WITH HSV CONSTANTS DEFINED ABOVE) -----------------
class GreenDetector:
    def __init__(self, lower1=DEFAULT_LOWER_HSV1.copy(), upper1=DEFAULT_UPPER_HSV1.copy(),
                 lower2=DEFAULT_LOWER_HSV2.copy(), upper2=DEFAULT_UPPER_HSV2.copy(),
                 min_area=MIN_MARKER_AREA, max_area=MAX_MARKER_AREA, apply_clahe=True):
        self.lower1 = lower1.copy()
        self.upper1 = upper1.copy()
        self.lower2 = lower2.copy()
        self.upper2 = upper2.copy()
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.apply_clahe = bool(apply_clahe)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    
    def set_masks(self, lower1, upper1, lower2=None, upper2=None):
        self.lower1 = lower1.copy(); self.upper1 = upper1.copy()
        if lower2 is not None and upper2 is not None:
            self.lower2 = lower2.copy(); self.upper2 = upper2.copy()
    
    def detect(self, frame_bgr, min_area=None, max_area=None):
        if frame_bgr is None or frame_bgr.size == 0:
            return [], np.zeros((1,1), dtype=np.uint8)
        min_area = min_area or self.min_area
        max_area = max_area or self.max_area
        blurred = cv2.GaussianBlur(frame_bgr, (7,7), 0)
        if self.apply_clahe:
            try:
                hsv_tmp = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                v = hsv_tmp[:, :, 2]
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                v_eq = clahe.apply(v); hsv_tmp[:, :, 2] = v_eq
                hsv = hsv_tmp
            except Exception:
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower1, self.upper1)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M.get("m00", 0) != 0:
                cx = float(M["m10"]/M["m00"]); cy = float(M["m01"]/M["m00"])
            else:
                cx, cy = x + w/2.0, y + h/2.0
            bbox = (int(x), int(y), int(w), int(h))
            centroid = (float(cx), float(cy))
            detections.append({"bbox": bbox, "area": float(area), "centroid": centroid})
        detections.sort(key=lambda d: d["area"], reverse=True)
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        return detections, mask_uint8
    
    def auto_calibrate_from_buffer(self, frames_preview):
        if frames_preview is None or len(frames_preview) == 0:
            return None
        broad_lower = np.array([30, 50, 20], dtype=np.uint8)
        broad_upper = np.array([100, 255, 255], dtype=np.uint8)
        collected = []
        got_any = False
        for fr in frames_preview:
            if fr is None or fr.size == 0:
                continue
            try:
                blurred = cv2.GaussianBlur(fr, (7,7), 0)
                hsv_tmp = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                v = hsv_tmp[:, :, 2]
                try:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    v_eq = clahe.apply(v)
                    hsv_tmp[:, :, 2] = v_eq
                except Exception:
                    pass
                mask = cv2.inRange(hsv_tmp, broad_lower, broad_upper)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
                pix = hsv_tmp[mask > 0]
                if pix.size > 0:
                    collected.append(pix.reshape(-1,3))
                    got_any = True
            except Exception:
                continue
        if not got_any:
            return None
        allpix = np.concatenate(collected, axis=0)
        h_med = int(np.median(allpix[:,0])); s_med = int(np.median(allpix[:,1])); v_med = int(np.median(allpix[:,2]))
        h_low = max(0, h_med - TOLERANCE_H); h_high = min(179, h_med + TOLERANCE_H)
        s_low = max(0, s_med - TOLERANCE_S); s_high = min(255, s_med + TOLERANCE_S)
        v_low = max(0, v_med - TOLERANCE_V); v_high = min(255, v_med + TOLERANCE_V)
        lower1 = np.array([max(30, h_low), max(50, s_low), max(20, v_low)], dtype=np.uint8)
        upper1 = np.array([min(80, h_high), min(255, s_high), min(200, v_high)], dtype=np.uint8)
        lower2 = np.array([max(70, h_low), max(50, s_low), max(50, v_low)], dtype=np.uint8)
        upper2 = np.array([min(100, h_high), min(220, s_high), min(180, v_high)], dtype=np.uint8)
        return lower1.copy(), upper1.copy(), lower2.copy(), upper2.copy()

# ------------------ Robust Marker Matcher --------------------------
class RobustMarkerMatcher:
    def __init__(self, max_match_distance=MAX_MATCH_DIST, min_confidence=0.5):
        self.max_match_distance = max_match_distance
        self.min_confidence = min_confidence
        self.reference_positions = None
        self.canonical_order = None
        self.last_matched_positions = None
    
    def set_reference(self, positions):
        if len(positions) < 2:
            return False
        self.reference_positions = positions.copy()
        self.canonical_order, _ = sort_points_left_to_right_top_to_bottom(positions)
        self.last_matched_positions = None
        return True
    
    def match_markers(self, current_positions):
        if self.reference_positions is None or len(current_positions) == 0:
            return None, None, None
        ref_points = self.canonical_order
        cur_points = np.array(current_positions)
        matches, confidences = match_points_hungarian(ref_points, cur_points, self.max_match_distance)
        if len(matches) == 0:
            return None, None, None
        n_ref = len(self.reference_positions)
        ordered_positions = np.zeros((n_ref, 2))
        match_flags = np.zeros(n_ref, dtype=bool)
        for (ref_idx, cur_idx), confidence in zip(matches, confidences):
            if confidence >= self.min_confidence:
                ordered_positions[ref_idx] = cur_points[cur_idx]
                match_flags[ref_idx] = True
        for i in range(n_ref):
            if not match_flags[i]:
                ordered_positions[i] = ref_points[i]
        if self.last_matched_positions is not None:
            ordered_positions, valid_flags = self._validate_consistency(
                self.last_matched_positions, ordered_positions, threshold=30.0)
            match_flags = match_flags & valid_flags
        self.last_matched_positions = ordered_positions.copy()
        return ordered_positions, match_flags, np.mean(confidences) if confidences else 0.0
    
    def _validate_consistency(self, prev_positions, curr_positions, threshold=30.0):
        if prev_positions is None or curr_positions is None or len(prev_positions) != len(curr_positions):
            return curr_positions, np.ones(len(curr_positions), dtype=bool)
        n_markers = len(prev_positions)
        valid_flags = np.ones(n_markers, dtype=bool)
        for i in range(n_markers):
            displacement = np.linalg.norm(curr_positions[i] - prev_positions[i])
            if displacement > threshold:
                valid_flags[i] = False
        return curr_positions, valid_flags
    
    def reset(self):
        self.last_matched_positions = None

# ------------------ Cluster-based normalization ------------------
def compute_cluster_based_reference(all_marker_positions, num_clusters):
    """
    Compute reference P-state using KMeans clustering on all marker positions
    Returns sorted cluster centers as reference state
    """
    if len(all_marker_positions) == 0:
        return None, None
    # Flatten all positions into single array
    all_points = np.vstack(all_marker_positions)
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(all_points)
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    # Sort centers left-to-right and top-to-bottom
    reference_P_sorted, _ = sort_points_left_to_right_top_to_bottom(cluster_centers)
    return reference_P_sorted, cluster_centers

# ------------------ Frame capture functions ------------------
def save_captured_frames(captured_frames, camera_ids, sensor_type, session_start_time, reference_P_sorted, cluster_centers, session_dir):
    """Save all captured frames with cluster-based normalization to session-specific directory"""
    if not captured_frames:
        print("No frames captured to save")
        return None
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = session_dir / f"captured_frames_{sensor_type}_{timestamp_str}.npz"
    # Prepare arrays
    timestamps = np.array([f['timestamp'] for f in captured_frames])
    frame_indices = np.array([f['frame_idx'] for f in captured_frames])
    forces = np.array([f['forces'] for f in captured_frames])
    areas_px = np.array([f['areas_px'] for f in captured_frames])
    areas_mm2 = np.array([f['areas_mm2'] for f in captured_frames])
    centroids_x = np.array([f['centroids_x'] for f in captured_frames])
    centroids_y = np.array([f['centroids_y'] for f in captured_frames])
    q_positions = np.array([f['q_positions'] for f in captured_frames], dtype=object)
    deformations = np.array([f['deformations'] for f in captured_frames], dtype=object)
    total_l2_norms = np.array([f['total_l2_norms'] for f in captured_frames])
    politopes = np.array([f['politopes'] for f in captured_frames], dtype=object)
    relative_timestamps = timestamps - session_start_time
    meta = {
        'sensor_type': sensor_type,
        'n_markers': N_MARKERS,
        'grid_description': GRID_DESCRIPTION,
        'force_bias': FORCE_BIAS,
        'camera_ids': camera_ids,
        'total_frames': len(captured_frames),
        'n_cameras': len(camera_ids),
        'start_time': timestamp_str,
        'session_start_time': session_start_time,
        'centroid_normalization': 'NONE - RAW PIXEL COORDINATES',
        'reference_P_method': 'CLUSTER_BASED_KMEANS',
        'reference_P_shape': list(reference_P_sorted.shape) if reference_P_sorted is not None else None,
        'cluster_centers_shape': list(cluster_centers.shape) if cluster_centers is not None else None,
        'total_camera_view_mm2': TOTAL_CAMERA_VIEW_MM2,
        'total_camera_view_pixels': TOTAL_CAMERA_VIEW_PIXELS,
        'pixels_per_mm2': PIXELS_PER_MM2,
        'force_filtering': {
            'iqr_factor': FORCE_OUTLIER_IQR_FACTOR,
            'smoothing_window': FORCE_SMOOTHING_WINDOW
        },
        'session_directory': str(session_dir.absolute())
    }
    np.savez_compressed(
        str(filename),
        timestamps=timestamps,
        relative_timestamps=relative_timestamps,
        frame_indices=frame_indices,
        forces=forces,
        areas_px=areas_px,
        areas_mm2=areas_mm2,
        centroids_x=centroids_x,
        centroids_y=centroids_y,
        q_positions=q_positions,
        deformations=deformations,
        total_l2_norms=total_l2_norms,
        politopes=politopes,
        reference_P_sorted=reference_P_sorted if reference_P_sorted is not None else np.array([]),
        cluster_centers=cluster_centers if cluster_centers is not None else np.array([]),
        meta=np.array([meta], dtype=object)
    )
    print("\n" + "="*60)
    print("ЗАХВАЧЕННЫЕ КАДРЫ СОХРАНЕНЫ")
    print("="*60)
    print(f"Файл: {filename}")
    print(f"Всего кадров: {len(captured_frames)}")
    print(f"ID камер: {camera_ids}")
    print(f"Тип датчика: {sensor_type}")
    print(f"Метод опорного состояния: Кластеризация KMeans ({N_MARKERS} кластеров)")
    print("Форма данных:")
    print(f"  Временные метки: {timestamps.shape}")
    print(f"  Силы: {forces.shape} ({len(camera_ids)} значений на кадр)")
    print(f"  Площади (пикс²): {areas_px.shape}")
    print(f"  Площади (мм²): {areas_mm2.shape}")
    print(f"  Центроиды X: {centroids_x.shape} (сырые координаты)")
    print(f"  Центроиды Y: {centroids_y.shape} (сырые координаты)")
    print(f"  Директория сессии: {session_dir}")
    print("="*60)
    return filename

def plot_force_graph(frame_indices, forces, camera_ids, save_path_png=None, save_path_svg=None):
    """Plot force values with filtering: outlier removal + median smoothing (FULL RUSSIAN TEXT)"""
    n_cameras = len(camera_ids)
    colors = ['b', 'g', 'r']
    # Apply force filtering
    filtered_forces = filter_force_data(forces, iqr_factor=FORCE_OUTLIER_IQR_FACTOR, smoothing_window=FORCE_SMOOTHING_WINDOW)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_cameras):
        ax.plot(frame_indices, filtered_forces[:, i],
                color=colors[i % len(colors)],
                linewidth=2.5,
                alpha=0.85,
                label=f'Камера {camera_ids[i]}',
                marker='o',
                markersize=4,
                markerfacecolor=colors[i % len(colors)],
                markeredgecolor='black',
                markeredgewidth=0.8)
    ax.set_ylabel('Сила, Н', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_xlabel('Кадр', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Измеренная сила контакта (фильтрованная)', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    if save_path_png:
        plt.savefig(str(save_path_png), dpi=300, bbox_inches='tight')
    if save_path_svg:
        plt.savefig(str(save_path_svg), format='svg', bbox_inches='tight')
    plt.close(fig)

def plot_area_graph(frame_indices, areas_mm2, camera_ids, save_path_png=None, save_path_svg=None):
    """Plot contact areas with lines and small markers (FULL RUSSIAN TEXT)"""
    n_cameras = len(camera_ids)
    colors = ['b', 'g', 'r']
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_cameras):
        ax.plot(frame_indices, areas_mm2[:, i],
                color=colors[i % len(colors)],
                linewidth=2.5,
                alpha=0.85,
                label=f'Камера {camera_ids[i]}',
                marker='s',
                markersize=4,
                markerfacecolor=colors[i % len(colors)],
                markeredgecolor='black',
                markeredgewidth=0.8)
    ax.set_ylabel('Площадь контакта, мм²', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_xlabel('Кадр', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Измеренная площадь контакта', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    if save_path_png:
        plt.savefig(str(save_path_png), dpi=300, bbox_inches='tight')
    if save_path_svg:
        plt.savefig(str(save_path_svg), format='svg', bbox_inches='tight')
    plt.close(fig)

def plot_polytope_graph(centroids_x, centroids_y, q_positions_all, camera_ids, reference_P_sorted,
                        cluster_centers, capture_mode='single', save_path_png=None, save_path_svg=None):
    """
    Plot centroids ONLY (NO polytopes) with reference P-state (FULL RUSSIAN TEXT)
    capture_mode: 'single' for 'a' key mode (show all frames),
    'continuous' for 'z' key mode (show only last frame)
    """
    n_cameras = len(camera_ids)
    n_frames = len(centroids_x)
    colors = ['b', 'g', 'r']
    fig, ax = plt.subplots(figsize=(18, 16))  # Significantly enlarged plot
    # 1. Plot reference P-state (cluster centers) as small green dots WITHOUT numeric labels
    if reference_P_sorted is not None:
        ax.scatter(reference_P_sorted[:, 0], reference_P_sorted[:, 1],
                   s=60, c='green', alpha=0.85,
                   marker='o', edgecolors='darkgreen', linewidths=1.5,
                   label='P-состояние', zorder=3)
    # 2. Determine which frames to show centroids for
    frames_to_show = []
    if capture_mode == 'continuous' and n_frames > 0:
        # Only last frame for continuous mode
        frames_to_show = [n_frames - 1]
    else:
        # All frames for single mode
        frames_to_show = range(n_frames)
    # 3. Plot centroids ONLY (NO polytopes) with X markers
    for frame_idx in frames_to_show:
        for cam_idx in range(n_cameras):
            # Plot centroid with X marker (same color as camera)
            cx = centroids_x[frame_idx, cam_idx]
            cy = centroids_y[frame_idx, cam_idx]
            if cx > 0 and cy > 0:
                ax.scatter(cx, cy,
                           s=220, c=colors[cam_idx % len(colors)], alpha=0.95,
                           marker='X', edgecolors='black', linewidths=2.5,
                           zorder=5,
                           label=f'Камера {camera_ids[cam_idx]}' if frame_idx == frames_to_show[0] and cam_idx == 0 else "")
    # Set labels and title (RUSSIAN ONLY)
    ax.set_xlabel('Координата X, пикс', fontsize=16, fontweight='bold', labelpad=14)
    ax.set_ylabel('Координата Y, пикс', fontsize=16, fontweight='bold', labelpad=14)
    ax.set_title('Опорное состояние и центроиды контакта',
                 fontsize=19, fontweight='bold', pad=22)
    # Clean legend - show only unique entries in vertical layout
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for handle, label in zip(handles, labels):
        if label not in by_label:
            by_label[label] = handle
    # Create vertical legend layout in upper right corner
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=14, loc='upper right', ncol=1, framealpha=0.95,
              fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()  # Match image coordinate system
    plt.tight_layout()
    if save_path_png:
        plt.savefig(str(save_path_png), dpi=300, bbox_inches='tight')
    if save_path_svg:
        plt.savefig(str(save_path_svg), format='svg', bbox_inches='tight')
    plt.close(fig)

def plot_captured_data(captured_frames, camera_ids, sensor_type, session_start_time,
                       reference_P_sorted, cluster_centers, capture_mode='single', save_path=None, session_dir=None):
    """
    Plot captured data with 3 subplots: filtered forces, areas, and centroids ONLY (NO polytopes) (FULL RUSSIAN TEXT)
    capture_mode: 'single' for 'a' key mode, 'continuous' for 'z' key mode
    """
    if not captured_frames:
        print("Нет данных для визуализации")
        return None, None, None, None
    
    if session_dir is None:
        session_dir = SESSION_CAPTURE_DIR
    
    frame_indices = np.arange(len(captured_frames))
    forces = np.array([f['forces'] for f in captured_frames])
    areas_mm2 = np.array([f['areas_mm2'] for f in captured_frames])
    centroids_x = np.array([f['centroids_x'] for f in captured_frames])
    centroids_y = np.array([f['centroids_y'] for f in captured_frames])
    q_positions_all = [f['q_positions'] for f in captured_frames]  # List of marker positions per frame
    n_cameras = len(camera_ids)
    n_frames = len(captured_frames)
    
    # Apply force filtering for the first subplot
    filtered_forces = filter_force_data(forces, iqr_factor=FORCE_OUTLIER_IQR_FACTOR, smoothing_window=FORCE_SMOOTHING_WINDOW)
    
    # Create figure with 3 subplots using GridSpec
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(3, 1, hspace=0.35, height_ratios=[1, 1, 1.4])
    
    # Plot 1: FILTERED forces over time (lines with small markers) - RUSSIAN TEXT
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['b', 'g', 'r']
    for i in range(n_cameras):
        ax1.plot(frame_indices, filtered_forces[:, i],
                 color=colors[i % len(colors)],
                 linewidth=2.8,
                 alpha=0.85,
                 label=f'Камера {camera_ids[i]}',
                 marker='o',
                 markersize=5,
                 markerfacecolor=colors[i % len(colors)],
                 markeredgecolor='black',
                 markeredgewidth=0.9)
    ax1.set_ylabel('Сила, Н', fontsize=16, fontweight='bold', labelpad=12)
    ax1.set_title('Измеренная сила контакта (фильтрованная)', fontsize=18, fontweight='bold', pad=18)
    ax1.legend(fontsize=14, loc='best', framealpha=0.95, ncol=3)
    ax1.grid(True, linestyle='--', alpha=0.75, linewidth=1.0)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel('Кадр', fontsize=15, fontweight='bold', labelpad=10)
    
    # Plot 2: Areas over time (lines with small markers) - RUSSIAN TEXT
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(n_cameras):
        ax2.plot(frame_indices, areas_mm2[:, i],
                 color=colors[i % len(colors)],
                 linewidth=2.8,
                 alpha=0.85,
                 label=f'Камера {camera_ids[i]}',
                 marker='s',
                 markersize=5,
                 markerfacecolor=colors[i % len(colors)],
                 markeredgecolor='black',
                 markeredgewidth=0.9)
    ax2.set_ylabel('Площадь контакта, мм²', fontsize=16, fontweight='bold', labelpad=12)
    ax2.set_title('Измеренная площадь контакта', fontsize=18, fontweight='bold', pad=18)
    ax2.legend(fontsize=14, loc='best', framealpha=0.95, ncol=3)
    ax2.grid(True, linestyle='--', alpha=0.75, linewidth=1.0)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel('Кадр', fontsize=15, fontweight='bold', labelpad=10)
    
    # Plot 3: CENTROIDS ONLY (NO polytopes) - RUSSIAN TEXT
    ax3 = fig.add_subplot(gs[2, 0])
    # Plot reference P-state as small green dots WITHOUT numeric labels
    if reference_P_sorted is not None:
        ax3.scatter(reference_P_sorted[:, 0], reference_P_sorted[:, 1],
                    s=70, c='green', alpha=0.85,
                    marker='o', edgecolors='darkgreen', linewidths=1.6,
                    label='P-состояние', zorder=3)
    # Determine frames to show based on capture mode
    frames_to_show = [n_frames - 1] if capture_mode == 'continuous' and n_frames > 0 else range(n_frames)
    # Plot centroids ONLY (NO polytopes) with X markers
    for frame_idx in frames_to_show:
        for cam_idx in range(n_cameras):
            # Plot centroid with X marker (same color as camera)
            cx = centroids_x[frame_idx, cam_idx]
            cy = centroids_y[frame_idx, cam_idx]
            if cx > 0 and cy > 0:
                ax3.scatter(cx, cy,
                            s=240, c=colors[cam_idx % len(colors)], alpha=0.95,
                            marker='X', edgecolors='black', linewidths=2.6,
                            zorder=5,
                            label=f'Камера {camera_ids[cam_idx]}' if frame_idx == frames_to_show[0] and cam_idx == 0 else "")
    ax3.set_xlabel('Координата X, пикс', fontsize=17, fontweight='bold', labelpad=14)
    ax3.set_ylabel('Координата Y, пикс', fontsize=17, fontweight='bold', labelpad=14)
    ax3.set_title('Опорное состояние и центроиды контакта',
                  fontsize=19, fontweight='bold', pad=22)
    # Clean legend - avoid duplicates, vertical layout
    handles, labels = ax3.get_legend_handles_labels()
    by_label = {}
    for handle, label in zip(handles, labels):
        if label not in by_label:
            by_label[label] = handle
    ax3.legend(by_label.values(), by_label.keys(),
               fontsize=15, loc='upper right', ncol=1, framealpha=0.95,
               fancybox=True, shadow=True)
    ax3.grid(True, linestyle='--', alpha=0.75, linewidth=1.0)
    ax3.set_aspect('equal', adjustable='box')
    ax3.invert_yaxis()
    ax3.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    
    # Save combined PNG
    if save_path is None:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        plot_filename_png = session_dir / f"captured_data_plot_{SENSOR_TYPE}_{timestamp_str}.png"  # FIXED: SENSOR_TYPE uppercase
    else:
        plot_filename_png = Path(save_path)
    plt.savefig(str(plot_filename_png), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save individual SVGs
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    force_svg = session_dir / f"force_plot_{SENSOR_TYPE}_{timestamp_str}.svg"
    area_svg = session_dir / f"area_plot_{SENSOR_TYPE}_{timestamp_str}.svg"
    polytope_svg = session_dir / f"polytope_plot_{SENSOR_TYPE}_{timestamp_str}.svg"
    
    plot_force_graph(frame_indices, forces, camera_ids,
                     save_path_svg=force_svg)
    plot_area_graph(frame_indices, areas_mm2, camera_ids,
                    save_path_svg=area_svg)
    plot_polytope_graph(centroids_x, centroids_y, q_positions_all, camera_ids,
                        reference_P_sorted, cluster_centers,
                        capture_mode=capture_mode, save_path_svg=polytope_svg)
    
    # Also save individual PNGs for completeness
    force_png = session_dir / f"force_plot_{SENSOR_TYPE}_{timestamp_str}.png"
    area_png = session_dir / f"area_plot_{SENSOR_TYPE}_{timestamp_str}.png"
    polytope_png = session_dir / f"polytope_plot_{SENSOR_TYPE}_{timestamp_str}.png"
    
    plot_force_graph(frame_indices, forces, camera_ids,
                     save_path_png=force_png, save_path_svg=None)
    plot_area_graph(frame_indices, areas_mm2, camera_ids,
                    save_path_png=area_png, save_path_svg=None)
    plot_polytope_graph(centroids_x, centroids_y, q_positions_all, camera_ids,
                        reference_P_sorted, cluster_centers,
                        capture_mode=capture_mode, save_path_png=polytope_png, save_path_svg=None)
    
    print("\nГрафики сохранены:")
    print(f"  Комбинированный PNG: {plot_filename_png}")
    print(f"  Сила (фильтрованная, SVG): {force_svg}")
    print(f"  Площадь (SVG): {area_svg}")
    print(f"  Центроиды (без политопов, SVG): {polytope_svg}")
    print(f"  Все файлы сохранены в: {session_dir}")
    return plot_filename_png, force_svg, area_svg, polytope_svg

# ------------------ Neural Network Inference Functions ------------------
def load_trained_models():
    """Load trained neural network model with robust error handling and dimension validation"""
    models = {}
    scalers = {}
    reference_P = None
    print("\n" + "="*60)
    print(f"ЗАГРУЗКА МОДЕЛЕЙ ДЛЯ ТИПА ДАТЧИКА {SENSOR_TYPE} С {N_MARKERS} МАРКЕРАМИ")
    print("="*60)
    print(f"Директория модели: {MODEL_DIR.absolute()}")
    print(f"Путь к модели: {MODEL_PATH.absolute()}")
    print(f"Путь к скалеру: {DEFORMATION_SCALER_PATH.absolute()}")
    print(f"Путь к опорному состоянию P: {REFERENCE_P_PATH.absolute()}")
    
    # Check if files exist
    if not MODEL_PATH.exists():
        print("\n" + "="*60)
        print(f"ОШИБКА: Файл модели НЕ НАЙДЕН по пути {MODEL_PATH}")
        print("Пожалуйста, обучите модель или проверьте конфигурацию типа датчика.")
        print("="*60 + "\n")
        return models, scalers, reference_P
    
    if not DEFORMATION_SCALER_PATH.exists():
        print("\n" + "="*60)
        print(f"ОШИБКА: Файл скалера деформаций НЕ НАЙДЕН по пути {DEFORMATION_SCALER_PATH}")
        print("="*60 + "\n")
        return models, scalers, reference_P
    
    if not REFERENCE_P_PATH.exists():
        print("\n" + "="*60)
        print(f"ОШИБКА: Файл опорного состояния P НЕ НАЙДЕН по пути {REFERENCE_P_PATH}")
        print("="*60 + "\n")
        return models, scalers, reference_P
    
    print("Все необходимые файлы найдены. Загрузка модели...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используемое устройство: {device}")
        # CRITICAL FIX: Use weights_only=False for compatibility with older checkpoints
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        # Validate model structure matches current configuration
        expected_input_size = N_MARKERS * 2 + 1  # 2D vectors for each marker + total L2 norm
        actual_input_size = checkpoint.get('input_size', -1)
        print("\nМетаданные модели:")
        print(f"  Количество маркеров при обучении: {checkpoint.get('marker_count', 'НЕИЗВЕСТНО')}")
        print(f"  Ожидаемый размер входа для {N_MARKERS} маркеров: {expected_input_size}")
        print(f"  Фактический размер входа модели: {actual_input_size}")
        
        if checkpoint.get('marker_count', -1) != N_MARKERS:
            print("\nВНИМАНИЕ: Модель обучена на {} маркерах, "
                  "но текущая конфигурация использует {} маркеров.".format(
                checkpoint.get('marker_count', 'НЕИЗВЕСТНО'), N_MARKERS))
            print(f"Прогноз силы может быть неточным. Рассмотрите возможность переобучения модели для типа датчика {SENSOR_TYPE}.")
        
        with open(DEFORMATION_SCALER_PATH, 'r') as f:
            scaler_info = json.load(f)
        reference_P = np.load(REFERENCE_P_PATH)
        
        # Create and load model
        mlp_model = ForcePredictionNet(
            input_size=checkpoint['input_size'],
            hidden_layers=checkpoint['hidden_layers'],
            dropout_rate=checkpoint['dropout_rate'],
            use_batchnorm=checkpoint['use_batchnorm']
        )
        mlp_model.load_state_dict(checkpoint['model_state_dict'])
        mlp_model.eval()
        mlp_model.to(device)
        
        print("\n" + "="*60)
        print("МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА")
        print("="*60)
        print(f"  Размер входа: {checkpoint['input_size']}")
        print(f"  Скрытые слои: {checkpoint['hidden_layers']}")
        print(f"  Dropout: {checkpoint['dropout_rate']}")
        print(f"  BatchNorm: {'Да' if checkpoint['use_batchnorm'] else 'Нет'}")
        print(f"  Количество маркеров (обучение): {checkpoint['marker_count']}")
        print(f"  Текущее количество маркеров: {N_MARKERS}")
        print(f"  Устройство: {device}")
        print("="*60 + "\n")
        
        models['mlp'] = mlp_model
        scalers['mlp'] = scaler_info
        return models, scalers, reference_P
    except Exception as e:
        print("\n" + "="*60)
        print(f"КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАГРУЗКЕ МОДЕЛИ: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return models, scalers, reference_P

def predict_force_mlp(model, deformation_vectors, total_l2_norm, scaler_info):
    """Predict force using MLP model with proper error handling and dimension matching"""
    try:
        # Flatten deformation vectors and append total L2 norm
        features = deformation_vectors.flatten()
        features = np.concatenate([features, [total_l2_norm]])
        # Get scaler parameters
        mean = np.array(scaler_info['mean'])
        scale = np.array(scaler_info['scale'])
        # Handle dimension mismatch gracefully
        if len(features) != len(mean):
            print("ПРЕДУПРЕЖДЕНИЕ: Несоответствие размерности признаков - признаки: {}, среднее скалера: {}".format(
                len(features), len(mean)))
            min_dim = min(len(features), len(mean))
            features = features[:min_dim]
            mean = mean[:min_dim]
            scale = scale[:min_dim]
            if len(features) < len(mean):
                features = np.pad(features, (0, len(mean) - len(features)), 'constant', constant_values=0.0)
        # Avoid division by zero
        valid_scale = np.where(np.abs(scale) > 1e-8, scale, 1.0)
        scaled_features = (features - mean) / valid_scale
        # Convert to tensor and predict
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
            if next(model.parameters()).is_cuda:
                input_tensor = input_tensor.cuda()
            prediction = model(input_tensor)
            predicted_force = prediction.item()
            calibrated_force = predicted_force - FORCE_BIAS
            # Clamp to valid range
            if np.isnan(calibrated_force) or np.isinf(calibrated_force):
                calibrated_force = 0.0
            elif calibrated_force < 0:
                calibrated_force = 0.0
            return calibrated_force
    except Exception as e:
        print(f"Ошибка при прогнозировании силы: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

# ------------------ Camera Session Class ------------------
class CameraSession:
    def __init__(self, camera_id, shared_model, shared_scaler):
        self.camera_id = camera_id
        self.model = shared_model
        self.scaler = shared_scaler
        self.hsv_lock = threading.Lock()
        self.lower_hsv1 = DEFAULT_LOWER_HSV1.copy()
        self.upper_hsv1 = DEFAULT_UPPER_HSV1.copy()
        self.lower_hsv2 = DEFAULT_LOWER_HSV2.copy()
        self.upper_hsv2 = DEFAULT_UPPER_HSV2.copy()
        self.detector = GreenDetector(
            lower1=self.lower_hsv1.copy(), upper1=self.upper_hsv1.copy(),
            lower2=self.lower_hsv2.copy(), upper2=self.upper_hsv2.copy(),
            min_area=MIN_MARKER_AREA, max_area=MAX_MARKER_AREA, apply_clahe=True
        )
        self.preview_buffer = deque(maxlen=AUTO_CALIB_FRAMES)
        self.baseline_visual = None
        self.p_state_recorded = False
        self.track_displacements = {}
        self.heatmap_full = None
        self.smoothed_contact_centroid = None
        self.contact_area_px = 0.0
        self.contact_area_mm2 = 0.0
        self.contact_centroid = (0.0, 0.0)  # RAW coordinates
        self.no_contact_frame_counter = 0
        self.force_prediction_mlp = 0.0
        self.total_l2_norm = 0.0
        self.prediction_ready_mlp = False
        self.session_start_time = time.time()
        self.marker_matcher = RobustMarkerMatcher(max_match_distance=MAX_MATCH_DIST * 2)
        self.current_marker_positions = None
        self.marker_match_confidence = 0.0
        self.marker_position_history = deque(maxlen=10)
        self.frame_width = 0
        self.frame_height = 0
    
    def set_hsv(self, new_lower1, new_upper1, new_lower2=None, new_upper2=None):
        with self.hsv_lock:
            self.lower_hsv1 = new_lower1.copy(); self.upper_hsv1 = new_upper1.copy()
            if new_lower2 is not None and new_upper2 is not None:
                self.lower_hsv2 = new_lower2.copy(); self.upper_hsv2 = new_upper2.copy()
    
    def get_hsv(self):
        with self.hsv_lock:
            return self.lower_hsv1.copy(), self.upper_hsv1.copy(), self.lower_hsv2.copy(), self.upper_hsv2.copy()
    
    def reset_hsv_to_defaults(self):
        self.set_hsv(DEFAULT_LOWER_HSV1, DEFAULT_UPPER_HSV1, DEFAULT_LOWER_HSV2, DEFAULT_UPPER_HSV2)
        self.marker_matcher.reset()
        print(f"Камера {self.camera_id}: Маски HSV сброшены к значениям по умолчанию. Сопоставление маркеров очищено.")
    
    def auto_calibrate(self, frames_for_calib):
        ac = self.detector.auto_calibrate_from_buffer(frames_for_calib)
        if ac is not None:
            self.set_hsv(ac[0], ac[1], ac[2], ac[3])
            print(f"Камера {self.camera_id}: Автокалибровка применена")
            return True
        else:
            print(f"Камера {self.camera_id}: Автокалибровка не удалась")
            return False
    
    def capture_baseline(self, detections):
        if len(detections) == 0:
            print(f"Камера {self.camera_id}: Захват состояния P не удался - не обнаружено маркеров")
            return False
        print("\n" + "="*60)
        print(f"КАМЕРА {self.camera_id}: ЗАХВАТ СОСТОЯНИЯ P (ОПОРНОЕ НЕДЕФОРМИРОВАННОЕ СОСТОЯНИЕ)")
        print(f"Тип датчика: {SENSOR_TYPE} с {N_MARKERS} маркерами")
        print("="*60)
        print(f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        p_positions = np.array([d["centroid"] for d in detections[:N_MARKERS]])
        if not self.marker_matcher.set_reference(p_positions):
            print(f"Камера {self.camera_id}: Не удалось установить надежную опору маркеров")
            return False
        self.baseline_visual = self.marker_matcher.canonical_order.copy()
        self.p_state_recorded = True
        print("\nПозиции маркеров состояния P (канонический порядок):")
        for i, (x, y) in enumerate(self.baseline_visual):
            print(f"Маркер {i:2d}: X={x:7.2f}, Y={y:7.2f}")
        self.track_displacements = {}
        for i in range(len(self.baseline_visual)):
            self.track_displacements[i] = {'disp_ema': 0.0, 'persistent_frames': 0}
        self.heatmap_full = None
        self.smoothed_contact_centroid = None
        self.contact_area_px = 0.0
        self.contact_area_mm2 = 0.0
        self.contact_centroid = (0.0, 0.0)
        self.force_prediction_mlp = 0.0
        self.total_l2_norm = 0.0
        self.prediction_ready_mlp = False
        self.no_contact_frame_counter = 0
        self.session_start_time = time.time()
        self.marker_position_history.clear()
        print(f"\nСостояние P успешно захвачено для камеры {self.camera_id} с {len(self.baseline_visual)} маркерами.")
        print("Включено надежное сопоставление маркеров.")
        return True
    
    def process_frame(self, frame, current_time, elapsed_time):
        """Process frame and return overlay, contact centroid (RAW), area (px² and mm²), force, and prediction status (ENGLISH TEXT ONLY)"""
        if frame is None or frame.size == 0:
            h_full, w_full = 1080, 1920
            blank = np.zeros((h_full, w_full, 3), dtype=np.uint8)
            cv2.putText(blank, f"CAMERA {self.camera_id}: NO FRAME", (50, h_full//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank, (0.0, 0.0), 0.0, 0.0, 0.0, False, None
        h_full, w_full = frame.shape[:2]
        self.frame_height = h_full
        self.frame_width = w_full
        # CRITICAL FIX: Always initialize heatmap_full at start of frame processing
        if self.heatmap_full is None or self.heatmap_full.shape != (h_full, w_full):
            self.heatmap_full = np.zeros((h_full, w_full), dtype=np.float32)
        self.preview_buffer.append(frame.copy())
        cur_l1, cur_u1, cur_l2, cur_u2 = self.get_hsv()
        self.detector.set_masks(cur_l1, cur_u1, cur_l2, cur_u2)
        detections_all, mask_full = self.detector.detect(frame, min_area=MIN_MARKER_AREA, max_area=MAX_MARKER_AREA)
        detection_centroids = [d["centroid"] for d in detections_all]
        overlay = frame.copy()
        # Draw coordinate system (ENGLISH TEXT ONLY)
        origin = (50, 50)
        axis_length = 50
        cv2.line(overlay, origin, (origin[0] + axis_length, origin[1]), (0, 0, 255), 2)
        cv2.putText(overlay, "X", (origin[0] + axis_length + 5, origin[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, origin, (origin[0], origin[1] + axis_length), (0, 255, 0), 2)
        cv2.putText(overlay, "Y", (origin[0] - 15, origin[1] + axis_length + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(overlay, origin, 5, (255, 0, 0), -1)
        for i, d in enumerate(detections_all):
            x, y, w, h = d["bbox"]
            cx, cy = int(round(d["centroid"][0])), int(round(d["centroid"][1]))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(overlay, f"{int(d['area'])}", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        # Reset contact metrics
        self.contact_area_px = 0.0
        self.contact_area_mm2 = 0.0
        self.contact_centroid = (0.0, 0.0)
        self.force_prediction_mlp = 0.0
        self.total_l2_norm = 0.0
        self.prediction_ready_mlp = False
        current_marker_positions = None
        # Force prediction if baseline captured
        if self.p_state_recorded and self.baseline_visual is not None and len(detection_centroids) > 0:
            try:
                ordered_positions, match_flags, match_confidence = self.marker_matcher.match_markers(detection_centroids)
                self.marker_match_confidence = match_confidence if match_confidence else 0.0
                if ordered_positions is not None:
                    self.current_marker_positions = ordered_positions.copy()
                    current_marker_positions = ordered_positions.copy()
                    self.marker_position_history.append(ordered_positions.copy())
                    # CRITICAL FIX: Compute deformation and predict force using added function
                    deformations, self.total_l2_norm, valid = compute_deformation_vectors_and_l2_norm(
                        ordered_positions, self.baseline_visual)
                    if valid and self.model is not None and self.scaler is not None:
                        self.force_prediction_mlp = predict_force_mlp(
                            self.model, deformations, self.total_l2_norm, self.scaler)
                        self.prediction_ready_mlp = True
                    # Heatmap processing for contact detection
                    for i in range(min(len(self.baseline_visual), len(ordered_positions))):
                        base_pt = self.baseline_visual[i]
                        curr_pt = ordered_positions[i]
                        disp_vec = np.array(curr_pt) - np.array(base_pt)
                        disp = np.linalg.norm(disp_vec)
                        if i not in self.track_displacements:
                            self.track_displacements[i] = {'disp_ema': 0.0, 'persistent_frames': 0}
                        # Update EMA of displacement
                        self.track_displacements[i]['disp_ema'] = (
                            EMA_ALPHA * disp + (1 - EMA_ALPHA) * self.track_displacements[i]['disp_ema'])
                        # Update persistent frames counter
                        if self.track_displacements[i]['disp_ema'] > DISP_THRESHOLD:
                            self.track_displacements[i]['persistent_frames'] = min(
                                self.track_displacements[i]['persistent_frames'] + 1, PERSISTENT_FRAMES)
                        else:
                            self.track_displacements[i]['persistent_frames'] = max(
                                self.track_displacements[i]['persistent_frames'] - 1, 0)
                        # If persistent contact, add to heatmap
                        if self.track_displacements[i]['persistent_frames'] == PERSISTENT_FRAMES:
                            magnitude = self.track_displacements[i]['disp_ema'] * HEAT_ADD_SCALE
                            add_heat_at(self.heatmap_full, curr_pt, magnitude, radius=HEAT_RADIUS)
            except Exception as e:
                print(f"Ошибка камеры {self.camera_id} при прогнозировании силы: {e}")
                import traceback
                traceback.print_exc()
        # Auto-baseline logic with INCREASED threshold (30 frames)
        if self.p_state_recorded and self.prediction_ready_mlp:
            if self.force_prediction_mlp < NO_CONTACT_THRESHOLD_FORCE:
                self.no_contact_frame_counter += 1
                if self.no_contact_frame_counter >= NO_CONTACT_THRESHOLD_FRAMES:
                    print("\n" + "="*60)
                    print(f"КАМЕРА {self.camera_id}: АВТОМАТИЧЕСКИЙ СБРОС ОПОРНОГО СОСТОЯНИЯ")
                    print(f"Сила: {self.force_prediction_mlp:.4f} Н < {NO_CONTACT_THRESHOLD_FORCE} Н в течение {NO_CONTACT_THRESHOLD_FRAMES} кадров")
                    print("="*60)
                    if len(detections_all) >= N_MARKERS:
                        self.capture_baseline(detections_all[:N_MARKERS])
                        self.no_contact_frame_counter = 0
                    else:
                        self.no_contact_frame_counter = 0
                else:
                    self.no_contact_frame_counter = 0
            else:
                self.no_contact_frame_counter = 0
        else:
            self.no_contact_frame_counter = 0
        # CRITICAL FIX: Always initialize heatmap_vis BEFORE conditional blocks
        heatmap_vis = None
        # Heatmap decay and contact region extraction
        if self.heatmap_full is not None:
            self.heatmap_full *= HEAT_DECAY
            heatmap_vis = heatmap_to_colormap(self.heatmap_full)
            if heatmap_vis is not None:
                heat_blurred = cv2.GaussianBlur(self.heatmap_full, (0, 0),
                                                sigmaX=HEATMAP_BLUR_SIGMA,
                                                sigmaY=HEATMAP_BLUR_SIGMA)
                heat_thresh = (heat_blurred > HEAT_THRESHOLD).astype(np.uint8) * 255
                contours, _ = cv2.findContours(heat_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_pts = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 10:
                        continue
                    cv2.drawContours(overlay, [c], -1, (0, 180, 255), 2)
                    all_pts.extend(c.reshape(-1, 2).tolist())
                if len(all_pts) >= 3:
                    pts = np.array(all_pts, dtype=np.int32)
                    try:
                        hull = ConvexHull(pts)
                        cv2.drawContours(overlay, [pts[hull.vertices]], -1, (255, 255, 0), 2)
                        self.contact_area_px = cv2.contourArea(pts[hull.vertices])
                        # Convert area to mm²
                        self.contact_area_mm2 = self.contact_area_px / PIXELS_PER_MM2
                        M = cv2.moments(pts[hull.vertices])
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            current_centroid = (cx, cy)
                            if self.smoothed_contact_centroid is None:
                                self.smoothed_contact_centroid = current_centroid
                            else:
                                self.smoothed_contact_centroid = (
                                    int(CONTACT_CENTROID_EMA_ALPHA * current_centroid[0] +
                                        (1 - CONTACT_CENTROID_EMA_ALPHA) * self.smoothed_contact_centroid[0]),
                                    int(CONTACT_CENTROID_EMA_ALPHA * current_centroid[1] +
                                        (1 - CONTACT_CENTROID_EMA_ALPHA) * self.smoothed_contact_centroid[1])
                                )
                            self.contact_centroid = (float(self.smoothed_contact_centroid[0]),
                                                     float(self.smoothed_contact_centroid[1]))
                            cv2.circle(overlay, self.smoothed_contact_centroid, 12, (255, 255, 0), 2)
                            cv2.circle(overlay, self.smoothed_contact_centroid, 4, (0, 0, 0), -1)
                            cv2.putText(overlay, f"{self.contact_area_mm2:.1f} mm2",
                                        (self.smoothed_contact_centroid[0] + 15, self.smoothed_contact_centroid[1] - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    except Exception:
                        pass
        # CRITICAL FIX: Safe blending - heatmap_vis is always defined (None or valid array)
        if heatmap_vis is not None:
            overlay = cv2.addWeighted(overlay, 0.7, heatmap_vis, 0.3, 0)
        # Draw prediction results (ENGLISH TEXT ONLY)
        y_position = 24
        line_height = 26
        cv2.putText(overlay, f"CAMERA {self.camera_id}", (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y_position += line_height
        if self.prediction_ready_mlp:
            cv2.putText(overlay, f"Force: {self.force_prediction_mlp:.2f} N",
                        (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            y_position += line_height
            cv2.putText(overlay, f"L2 norm: {self.total_l2_norm:.3f}",
                        (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
            y_position += line_height
        if self.p_state_recorded:
            cv2.putText(overlay, f"Match: {self.marker_match_confidence:.2f}",
                        (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            y_position += line_height
        if self.contact_area_px > 0:
            cv2.putText(overlay, f"Area: {self.contact_area_mm2:.1f} mm2",
                        (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
            y_position += line_height
        if self.p_state_recorded:
            status_text = "P STATE RECORDED"
            color = (0, 255, 0)
            cv2.putText(overlay, status_text, (w_full - 220, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if self.baseline_visual is not None:
            for i, (bx, by) in enumerate(self.baseline_visual):
                cv2.circle(overlay, (int(round(bx)), int(round(by))), 8, color, -1)
                cv2.putText(overlay, f"{i}", (int(round(bx))+6, int(round(by))-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return overlay, self.contact_centroid, self.contact_area_px, self.contact_area_mm2, self.force_prediction_mlp, self.prediction_ready_mlp, current_marker_positions

# ------------------ Main application -------------------------------------
def main():
    print("="*60)
    print("СИСТЕМА ОТСЛЕЖИВАНИЯ КОНТАКТА С ТРЕХ КАМЕР С ПРОГНОЗИРОВАНИЕМ СИЛЫ")
    print(f"Тип датчика: {SENSOR_TYPE} с {N_MARKERS} маркерами ({GRID_DESCRIPTION} сетка)")
    print(f"Сессия: {SESSION_CAPTURE_DIR.name}")
    print("="*60)
    print("РЕЖИМЫ ЗАХВАТА (ВЗАИМОИСКЛЮЧАЮЩИЕ):")
    print("  'a' - ЗАХВАТ ОДНОГО КАДРА (одно нажатие = один кадр со всех камер)")
    print("  'z' - РЕЖИМ НЕПРЕРЫВНОГО ЗАХВАТА (переключение вкл/выкл для покадровой записи)")
    print("КЛАСТЕРНАЯ НОРМАЛИЗАЦИЯ:")
    print("  - Все маркеры со всех камер кластеризуются методом KMeans")
    print("  - Центры кластеров сортируются слева-направо/сверху-вниз как опорное состояние P")
    print("  - Деформации вычисляются относительно кластерного опорного состояния")
    print("КОНВЕРТАЦИЯ ПЛОЩАДЕЙ:")
    print(f"  - Полная видимая область камеры: {TOTAL_CAMERA_VIEW_MM2} мм²")
    print(f"  - Конвертация: площадь контакта (пикс²) → мм² с коэффициентом {PIXELS_PER_MM2:.2f} пикс²/мм²")
    print("ФИЛЬТРАЦИЯ СИЛЫ:")
    print(f"  - Удаление аномалий методом IQR (множитель={FORCE_OUTLIER_IQR_FACTOR})")
    print(f"  - Сглаживание медианным фильтром (окно={FORCE_SMOOTHING_WINDOW})")
    print("ЦЕНТРОИДЫ:")
    print("  - На последнем графике отображаются ТОЛЬКО центроиды (маркеры X) и опорное состояние P")
    print("  - Политопы (выпуклые оболочки) УДАЛЕНЫ с последнего графика")
    print("="*60)
    print("\nУправление:")
    print("  1/2/3 - Выбор активной камеры (зеленая рамка указывает активную)")
    print("  a     - ЗАХВАТ ОДНОГО КАДРА (взаимоисключающе с 'z')")
    print("  z     - ПЕРЕКЛЮЧЕНИЕ НЕПРЕРЫВНОГО ЗАХВАТА (взаимоисключающе с 'a')")
    print("  b     - Захват опорного состояния (P-state) на АКТИВНОЙ камере")
    print("  B     - Захват опорного состояния на ВСЕХ камерах")
    print("  c     - Автокалибровка HSV на АКТИВНОЙ камере")
    print("  C     - Автокалибровка ВСЕХ камер")
    print("  r     - Сброс масок HSV на АКТИВНОЙ камере")
    print("  R     - Сброс масок HSV на ВСЕХ камерах")
    print("  s     - Сохранить снимок всех видов камер В СЕССИОННУЮ ДИРЕКТОРИЮ")
    print("  q     - Выход (сохраняет все захваченные кадры и показывает визуализацию)")
    print(f"\nN_MARKERS = {N_MARKERS}, СМЕЩЕНИЕ СИЛЫ = {FORCE_BIAS:.2f} Н")
    print(f"Автосброс опорного состояния: {NO_CONTACT_THRESHOLD_FRAMES} кадров ниже {NO_CONTACT_THRESHOLD_FORCE:.2f} Н (увеличенная задержка)")
    print(f"\nВСЕ ДАННЫЕ БУДУТ СОХРАНЕНЫ В: {SESSION_CAPTURE_DIR}")
    print("="*60)
    
    # Probe available cameras
    print("\nПоиск камер до ID {} ...".format(max(CAM_IDS)))
    avail = probe_available_cameras(max(CAM_IDS), test_frames=3, wait_s=0.1)
    print(f"Доступные камеры: {avail}")
    
    # Initialize camera readers
    readers = []
    for cam_id in CAM_IDS:
        if cam_id in avail:
            print(f"\nИнициализация камеры {cam_id}...")
            reader = open_camera_reader(cam_id, REQ_W, REQ_H, REQ_FPS, FOURCC)
            if reader is not None:
                readers.append(reader)
                print(f"Камера {cam_id} успешно инициализирована")
            else:
                print(f"ВНИМАНИЕ: Камера {cam_id} не была корректно инициализирована")
        else:
            print(f"ВНИМАНИЕ: Камера {cam_id} отсутствует в списке доступных {avail}")
    
    if len(readers) == 0:
        print("\nОШИБКА: Не удалось инициализировать ни одну камеру. Выход.")
        return
    
    print(f"\nУспешно инициализировано {len(readers)} камер(ы)")
    
    # Load shared neural network model - CRITICAL: weights_only=False for force prediction
    models, scalers, reference_P = load_trained_models()
    shared_model = models.get('mlp', None)
    shared_scaler = scalers.get('mlp', None)
    
    if shared_model is None:
        print("\n" + "="*60)
        print("ВНИМАНИЕ: МОДЕЛЬ MLP НЕ ЗАГРУЖЕНА")
        print("="*60)
        print("Прогнозирование силы будет НЕДОСТУПНО.")
        print("Проверьте:")
        print(f"  1. Существует ли директория модели: {MODEL_DIR}")
        print(f"  2. Существует ли файл модели: {MODEL_PATH}")
        print(f"  3. Существует ли файл скалера: {DEFORMATION_SCALER_PATH}")
        print(f"  4. Существует ли файл опорного состояния P: {REFERENCE_P_PATH}")
        print(f"  5. Совпадает ли тип датчика с обученной моделью: {SENSOR_TYPE}")
        print("="*60 + "\n")
    else:
        print("Модель MLP успешно загружена и готова к прогнозированию силы.")
    
    # Create camera sessions
    sessions = []
    for i, reader in enumerate(readers):
        session = CameraSession(i, shared_model, shared_scaler)
        sessions.append(session)
        print(f"Сессия камеры {i} создана")
    
    # Create display window
    main_window = "vbts_tri_camera_force_prediction"
    cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(main_window, 1280, 720)
    
    # Active camera selection
    active_camera_idx = 0
    
    # Capture modes (mutually exclusive)
    single_capture_mode = False   # 'a' key pressed - capture one frame
    continuous_capture_mode = False  # 'z' key pressed - continuous capture
    capture_mode_indicator = 'none'  # 'single' or 'continuous' for plot function
    
    # Storage for captured frames
    captured_frames = []
    session_start_time = time.time()
    
    # Wait for first valid frames
    print("\nОжидание первых корректных кадров от всех камер (до 10 секунд)...")
    timeout = time.time() + 10.0
    all_ready = False
    while time.time() < timeout and not all_ready:
        frames_ready = 0
        for reader in readers:
            ok, frame, fps = reader.get()
            if ok and frame is not None and frame.size > 0:
                frames_ready += 1
        if frames_ready == len(readers):
            all_ready = True
            print("Все камеры готовы.")
        time.sleep(0.1)
    
    if not all_ready:
        print("ВНИМАНИЕ: Таймаут ожидания кадров с камер. Продолжение с доступными камерами...")
    
    # Get frame dimensions
    ok, sample_frame, _ = readers[0].get()
    if ok and sample_frame is not None:
        h_full, w_full = sample_frame.shape[:2]
    else:
        h_full, w_full = 1080, 1920
    
    print("\nСистема готова.")
    print("Нажмите 'a' для ЗАХВАТА ОДНОГО КАДРА или 'z' для переключения режима НЕПРЕРЫВНОГО захвата.")
    print("Нажмите 'q' для выхода и сохранения всех захваченных данных.")
    print(f"Все файлы будут сохранены в: {SESSION_CAPTURE_DIR}")
    print("="*60)
    
    try:
        frame_count = 0
        all_marker_positions = []  # For cluster-based normalization
        
        while True:
            overlays = []
            current_time = time.time()
            elapsed_times = [current_time - session_start_time] * len(sessions)
            
            # Storage for current frame data
            current_frame_marker_positions = []
            current_frame_forces = []
            current_frame_areas_px = []
            current_frame_areas_mm2 = []
            current_frame_centroids_x = []
            current_frame_centroids_y = []
            
            for i, (reader, session) in enumerate(zip(readers, sessions)):
                ok, frame, fps = reader.get()
                if not ok or frame is None or frame.size == 0:
                    blank = np.zeros((h_full, w_full, 3), dtype=np.uint8)
                    cv2.putText(blank, f"CAMERA {i} DISCONNECTED/NO FRAME", (50, h_full//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    overlays.append(blank)
                    current_frame_marker_positions.append(None)
                    current_frame_forces.append(0.0)
                    current_frame_areas_px.append(0.0)
                    current_frame_areas_mm2.append(0.0)
                    current_frame_centroids_x.append(0.0)
                    current_frame_centroids_y.append(0.0)
                    continue
                
                # Process frame - CRITICAL: Get all metrics including force prediction (ENGLISH TEXT ONLY)
                overlay, centroid, area_px, area_mm2, force, ready, marker_positions = session.process_frame(
                    frame, current_time, elapsed_times[i])
                
                # Store marker positions for cluster-based normalization
                if marker_positions is not None:
                    current_frame_marker_positions.append(marker_positions.copy())
                    all_marker_positions.append(marker_positions.copy())
                else:
                    current_frame_marker_positions.append(None)
                
                # Store metrics - FORCE IS NOW CORRECTLY LOGGED
                current_frame_forces.append(force if ready else 0.0)
                current_frame_areas_px.append(area_px)
                current_frame_areas_mm2.append(area_mm2)
                current_frame_centroids_x.append(centroid[0])
                current_frame_centroids_y.append(centroid[1])
                
                # Add active camera indicator (ENGLISH TEXT ONLY)
                if i == active_camera_idx:
                    cv2.rectangle(overlay, (5, 5), (w_full-6, h_full-6), (0, 255, 0), 4)
                    cv2.putText(overlay, "ACTIVE", (w_full - 120, h_full - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Add FPS counter (ENGLISH TEXT ONLY)
                cv2.putText(overlay, f"FPS: {fps:.1f}", (w_full - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                # Add capture mode indicator (ENGLISH TEXT ONLY)
                if single_capture_mode:
                    cv2.putText(overlay, "SINGLE CAPTURE MODE", (20, h_full - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif continuous_capture_mode:
                    cv2.putText(overlay, "CONTINUOUS CAPTURE", (20, h_full - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(overlay, f"Frames: {len(captured_frames)}", (20, h_full - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                overlays.append(overlay)
            
            composite = np.vstack(overlays) if len(overlays) > 1 else overlays[0]
            
            # Resize composite for display
            max_display_height = 1600
            max_display_width = 1920
            h_comp, w_comp = composite.shape[:2]
            if h_comp > max_display_height or w_comp > max_display_width:
                scale_h = max_display_height / h_comp
                scale_w = max_display_width / w_comp
                scale = min(scale_h, scale_w)
                composite_display = cv2.resize(composite, (int(w_comp * scale), int(h_comp * scale)))
            else:
                composite_display = composite
            
            # Show capture counter on display (ENGLISH TEXT ONLY)
            cv2.putText(composite_display, f"Captured frames: {len(captured_frames)}",
                        (20, composite_display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(main_window, composite_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                print("\n" + "="*60)
                print("ЗАПРОС ЗАВЕРШЕНИЯ - СОХРАНЕНИЕ ВСЕХ ЗАХВАЧЕННЫХ КАДРОВ")
                print("="*60)
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                fname = SESSION_CAPTURE_DIR / f"vbts_tri_camera_snapshot_{SENSOR_TYPE}_{timestamp}.png"
                cv2.imwrite(str(fname), composite)
                print(f"Снимок сохранен: {fname}")
            elif key == ord('1') and len(sessions) > 0:
                active_camera_idx = 0
                print(f"Активная камера установлена: Камера 0")
            elif key == ord('2') and len(sessions) > 1:
                active_camera_idx = 1
                print(f"Активная камера установлена: Камера 1")
            elif key == ord('3') and len(sessions) > 2:
                active_camera_idx = 2
                print(f"Активная камера установлена: Камера 2")
            elif key == ord('a'):
                # SINGLE FRAME CAPTURE MODE - mutually exclusive with continuous mode
                if continuous_capture_mode:
                    print("Невозможно активировать режим одиночного захвата, пока активен непрерывный режим. Сначала нажмите 'z' для отключения непрерывного режима.")
                else:
                    single_capture_mode = True
                    capture_mode_indicator = 'single'
                    print("\n" + "="*60)
                    print("АКТИВИРОВАН ЗАХВАТ ОДИНОЧНОГО КАДРА")
                    print("="*60)
                    # Verify all cameras have P-state recorded
                    all_ready_for_capture = all(session.p_state_recorded for session in sessions)
                    if not all_ready_for_capture:
                        print("Невозможно захватить кадр: сначала захватите состояние P на всех камерах (нажмите 'B').")
                        single_capture_mode = False
                        capture_mode_indicator = 'none'
                        continue
                    # Capture data from all cameras
                    frame_data = {
                        'timestamp': current_time,
                        'frame_idx': len(captured_frames),
                        'forces': np.array(current_frame_forces),
                        'areas_px': np.array(current_frame_areas_px),
                        'areas_mm2': np.array(current_frame_areas_mm2),
                        'centroids_x': np.array(current_frame_centroids_x),
                        'centroids_y': np.array(current_frame_centroids_y),
                        'q_positions': current_frame_marker_positions,  # ALL marker positions per camera
                        'deformations': [],
                        'total_l2_norms': [],
                        'politopes': []
                    }
                    # Compute cluster-based reference if we have enough data
                    if len(all_marker_positions) >= N_MARKERS * 2:
                        reference_P_sorted, cluster_centers = compute_cluster_based_reference(
                            all_marker_positions, N_MARKERS)
                    else:
                        reference_P_sorted = None
                        cluster_centers = None
                        print("Предупреждение: Недостаточно позиций маркеров для кластерной нормализации.")
                    # Compute deformations relative to reference state
                    if reference_P_sorted is not None:
                        for cam_idx, positions in enumerate(current_frame_marker_positions):
                            if positions is not None:
                                # Find correspondences
                                correspondences, _ = find_nearest_correspondences(
                                    reference_P_sorted, positions)
                                # Compute deformations
                                deformations = compute_deformation_vectors(
                                    reference_P_sorted, positions, correspondences)
                                total_l2 = compute_total_l2_norm(deformations)
                                frame_data['deformations'].append(deformations)
                                frame_data['total_l2_norms'].append(total_l2)
                            else:
                                frame_data['deformations'].append(np.array([]))
                                frame_data['total_l2_norms'].append(0.0)
                    else:
                        # Fallback to per-camera baseline if no cluster reference
                        for cam_idx, session in enumerate(sessions):
                            if session.current_marker_positions is not None and session.baseline_visual is not None:
                                deformations, total_l2, _ = compute_deformation_vectors_and_l2_norm(
                                    session.current_marker_positions, session.baseline_visual)
                                frame_data['deformations'].append(deformations)
                                frame_data['total_l2_norms'].append(total_l2)
                            else:
                                frame_data['deformations'].append(np.array([]))
                                frame_data['total_l2_norms'].append(0.0)
                    # Compute politopes (convex hulls) for valid centroids
                    for cam_idx in range(len(sessions)):
                        cx = current_frame_centroids_x[cam_idx]
                        cy = current_frame_centroids_y[cam_idx]
                        if cx > 0 and cy > 0:
                            frame_data['politopes'].append([(cx, cy)])
                        else:
                            frame_data['politopes'].append([])
                    captured_frames.append(frame_data)
                    # Print capture summary with FORCE VALUES
                    print(f"\nКАДР ЗАХВАЧЕН: №{len(captured_frames)}")
                    print(f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    for i in range(len(current_frame_centroids_x)):
                        print("  Камера {}: X={:.1f}, Y={:.1f}, Сила={:.2f}Н, Площадь={:.2f}мм²".format(
                            i, current_frame_centroids_x[i], current_frame_centroids_y[i],
                            current_frame_forces[i], current_frame_areas_mm2[i]))
                    print("="*60)
                    single_capture_mode = False  # Reset after capture
            elif key == ord('z'):
                # TOGGLE CONTINUOUS CAPTURE MODE - mutually exclusive with single mode
                if single_capture_mode:
                    print("Невозможно активировать непрерывный режим, пока активен одиночный захват.")
                else:
                    continuous_capture_mode = not continuous_capture_mode
                    capture_mode_indicator = 'continuous' if continuous_capture_mode else 'none'
                    if continuous_capture_mode:
                        print("\n" + "="*60)
                        print("АКТИВИРОВАН РЕЖИМ НЕПРЕРЫВНОГО ЗАХВАТА")
                        print("="*60)
                        print("Каждый кадр будет захватываться до отключения режима.")
                        print("Нажмите 'z' снова для остановки непрерывного захвата.")
                        print("="*60)
                    else:
                        print("\n" + "="*60)
                        print("РЕЖИМ НЕПРЕРЫВНОГО ЗАХВАТА ОТКЛЮЧЕН")
                        print(f"Всего захвачено кадров: {len(captured_frames)}")
                        print("="*60)
            elif key == ord('b') and sessions and readers:
                ok, frame, _ = readers[active_camera_idx].get()
                if ok and frame is not None:
                    cur_l1, cur_u1, cur_l2, cur_u2 = sessions[active_camera_idx].get_hsv()
                    sessions[active_camera_idx].detector.set_masks(cur_l1, cur_u1, cur_l2, cur_u2)
                    detections_all, _ = sessions[active_camera_idx].detector.detect(
                        frame, min_area=MIN_MARKER_AREA, max_area=MAX_MARKER_AREA)
                    if len(detections_all) >= N_MARKERS:
                        sessions[active_camera_idx].capture_baseline(detections_all[:N_MARKERS])
                    else:
                        print("Камера {}: Обнаружено недостаточно маркеров ({}/{})".format(
                            active_camera_idx, len(detections_all), N_MARKERS))
            elif key == ord('B'):
                for i, (reader, session) in enumerate(zip(readers, sessions)):
                    ok, frame, _ = reader.get()
                    if ok and frame is not None:
                        cur_l1, cur_u1, cur_l2, cur_u2 = session.get_hsv()
                        session.detector.set_masks(cur_l1, cur_u1, cur_l2, cur_u2)
                        detections_all, _ = session.detector.detect(
                            frame, min_area=MIN_MARKER_AREA, max_area=MAX_MARKER_AREA)
                        if len(detections_all) >= N_MARKERS:
                            session.capture_baseline(detections_all[:N_MARKERS])
                        else:
                            print("Камера {}: Обнаружено недостаточно маркеров ({}/{})".format(
                                i, len(detections_all), N_MARKERS))
            elif key == ord('c') and sessions:
                frames_for_calib = list(sessions[active_camera_idx].preview_buffer)
                sessions[active_camera_idx].auto_calibrate(frames_for_calib)
            elif key == ord('C'):
                for i, session in enumerate(sessions):
                    frames_for_calib = list(session.preview_buffer)
                    session.auto_calibrate(frames_for_calib)
            elif key == ord('r') and sessions:
                sessions[active_camera_idx].reset_hsv_to_defaults()
            elif key == ord('R'):
                for session in sessions:
                    session.reset_hsv_to_defaults()
            
            # Continuous capture logic
            if continuous_capture_mode and all(session.p_state_recorded for session in sessions):
                frame_data = {
                    'timestamp': current_time,
                    'frame_idx': len(captured_frames),
                    'forces': np.array(current_frame_forces),
                    'areas_px': np.array(current_frame_areas_px),
                    'areas_mm2': np.array(current_frame_areas_mm2),
                    'centroids_x': np.array(current_frame_centroids_x),
                    'centroids_y': np.array(current_frame_centroids_y),
                    'q_positions': current_frame_marker_positions,  # ALL marker positions per camera
                    'deformations': [],
                    'total_l2_norms': [],
                    'politopes': []
                }
                # Compute cluster-based reference if we have enough data
                if len(all_marker_positions) >= N_MARKERS * 2:
                    reference_P_sorted, cluster_centers = compute_cluster_based_reference(
                        all_marker_positions, N_MARKERS)
                else:
                    reference_P_sorted = None
                    cluster_centers = None
                # Compute deformations
                if reference_P_sorted is not None:
                    for cam_idx, positions in enumerate(current_frame_marker_positions):
                        if positions is not None:
                            correspondences, _ = find_nearest_correspondences(
                                reference_P_sorted, positions)
                            deformations = compute_deformation_vectors(
                                reference_P_sorted, positions, correspondences)
                            total_l2 = compute_total_l2_norm(deformations)
                            frame_data['deformations'].append(deformations)
                            frame_data['total_l2_norms'].append(total_l2)
                        else:
                            frame_data['deformations'].append(np.array([]))
                            frame_data['total_l2_norms'].append(0.0)
                else:
                    for cam_idx, session in enumerate(sessions):
                        if session.current_marker_positions is not None and session.baseline_visual is not None:
                            deformations, total_l2, _ = compute_deformation_vectors_and_l2_norm(
                                session.current_marker_positions, session.baseline_visual)
                            frame_data['deformations'].append(deformations)
                            frame_data['total_l2_norms'].append(total_l2)
                        else:
                            frame_data['deformations'].append(np.array([]))
                            frame_data['total_l2_norms'].append(0.0)
                # Compute politopes
                for cam_idx in range(len(sessions)):
                    cx = current_frame_centroids_x[cam_idx]
                    cy = current_frame_centroids_y[cam_idx]
                    if cx > 0 and cy > 0:
                        frame_data['politopes'].append([(cx, cy)])
                    else:
                        frame_data['politopes'].append([])
                captured_frames.append(frame_data)
                if len(captured_frames) % 10 == 0:
                    print("Непрерывный захват: захвачено {} кадров".format(len(captured_frames)))
            
            frame_count += 1
        
        # Cleanup and save captured data
        print("\nОсвобождение ресурсов...")
        # Compute final cluster-based reference for all captured data
        if all_marker_positions and len(all_marker_positions) >= N_MARKERS:
            reference_P_sorted, cluster_centers = compute_cluster_based_reference(
                all_marker_positions, N_MARKERS)
            print("\nКластерное опорное состояние вычислено с {} кластерами".format(N_MARKERS))
            print("Форма опорного состояния P: {}".format(reference_P_sorted.shape))
        else:
            reference_P_sorted = None
            cluster_centers = None
            print("\nПредупреждение: Недостаточно данных для кластерной нормализации")
        
        # Save captured frames if any
        if captured_frames:
            camera_ids = [session.camera_id for session in sessions]
            save_captured_frames(captured_frames, camera_ids, SENSOR_TYPE, session_start_time,
                                 reference_P_sorted, cluster_centers, SESSION_CAPTURE_DIR)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            plot_filename_png = SESSION_CAPTURE_DIR / f"captured_data_plot_{SENSOR_TYPE}_{timestamp_str}.png"  # FIXED: SENSOR_TYPE uppercase
            # Use appropriate capture mode for plotting
            mode_for_plot = capture_mode_indicator if capture_mode_indicator in ['single', 'continuous'] else 'single'
            plot_captured_data(captured_frames, camera_ids, SENSOR_TYPE, session_start_time,
                               reference_P_sorted, cluster_centers,
                               capture_mode=mode_for_plot, save_path=plot_filename_png, session_dir=SESSION_CAPTURE_DIR)
            print("\n" + "="*60)
            print("СЕАНС ЗАВЕРШЕН - СОХРАНЕНО {} КАДРОВ".format(len(captured_frames)))
            print("="*60)
            print(f"Все данные сохранены в директорию: {SESSION_CAPTURE_DIR}")
            print("Созданные файлы:")
            print("  - captured_frames_*.npz (сырые данные с кластерной нормализацией)")
            print("  - captured_data_plot_*.png (комбинированный график из 3 сабплотов)")
            print("  - force_plot_*.svg (график силы с фильтрацией в векторном формате)")
            print("  - area_plot_*.svg (график площади в векторном формате)")
            print("  - polytope_plot_*.svg (график центроидов без политопов в векторном формате)")
            print("  - vbts_tri_camera_snapshot_*.png (снимки при нажатии 's')")
            print("="*60)
        else:
            print("\nЗа время сеанса не было захвачено ни одного кадра.")
    
    finally:
        # Critical: Proper cleanup sequence
        cv2.destroyAllWindows()
        for reader in readers:
            reader.stop(2)
        time.sleep(0.5)
        print("Завершено корректно.")

if __name__ == "__main__":
    main()