""""
Подготовка нормализованного датасета для заданного типа датчика с геометрическими параметрами
Этот скрипт:
1. Принимает тип датчика (A, B, C, D, E) как параметр
2. Загружает все датасеты указанного типа датчика
3. Собирает все состояния (как P, так и Q) для кластеризации
4. Находит центры масс кластеров всех состояний
5. Использует центры масс кластеров как эталонное P-состояние
6. Сортирует маркеры в эталонном состоянии в порядке слева-направо и сверху-вниз
7. Для каждого состояния сопоставляет маркеры с эталонным состоянием по принципу ближайшего соседа
8. Вычисляет векторы деформаций как разность между сопоставленными точками (Q-P)
9. Вычисляет сумму L2 норм векторов деформаций для каждого замера
10. Добавляет геометрические параметры датчика (D, M) из SENSOR_GEOMETRY
11. Сохраняет агрегированный датасет со всеми этими данными в формате, совместимом с кодом обучения
12. В конце отображает два графика:
- Эталонное нормированное P-состояние (центры масс кластеров)
- Все нормированные состояния всех датасетов, наложенные друг на друга, с центрами масс кластеров
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from tqdm import tqdm
import json
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
# =======================
# ГЕОМЕТРИЧЕСКИЕ ПАРАМЕТРЫ ДАТЧИКОВ И КОЛИЧЕСТВО МАРКЕРОВ
# =======================
SENSOR_GEOMETRY = {
    'A': {'D': 8.69, 'M': 12},  # 12 маркеров (3x4 сетка)
    'B': {'D': 6.94, 'M': 15},  # 15 маркеров (3x5 сетка)
    'C': {'D': 6.47, 'M': 20},  # 20 маркеров (4x5 сетка)
    'D': {'D': 5.75, 'M': 24},  # 24 маркера (4x6 сетка)
    'E': {'D': 5.14, 'M': 30}   # 30 маркеров (5x6 сетка)
}
# =======================
# Конфигурационные параметры
# =======================
# Пути к данным
BASE_DIR = Path("Datasets")
LABELLED_DIR = BASE_DIR / "Labelled"
# Параметр типа датчика (можно изменить здесь или передать в функцию)
SENSOR_TYPE = 'D'  # Может быть 'A', 'B', 'C', 'D', 'E'
# Параметры определения начальных состояний
P_THRESHOLD = 0.1  # Порог силы для начальных состояний (Н)
MIN_P_STATES = 1   # Минимальное количество P-состояний в датасете
# Параметры кластеризации
MIN_CLUSTER_SIZE = 3          # Минимальное количество точек в кластере для его учета
# Параметры визуализации
FIG_WIDTH = 25
FIG_HEIGHT = 8
DPI = 100
TITLE_FONT = 14
AXIS_FONT = 12
TICK_FONT = 10
ANNOT_FONT = 8
P_MARKER_SIZE = 80
Q_MARKER_SIZE = 15
P_ALPHA = 0.9
Q_ALPHA = 0.1
P_COLOR = '#1f77b4'           # синий для нормализованных P состояний
Q_COLOR = '#ff7f0e'           # оранжевый для нормализованных Q состояний
CLUSTER_CENTER_COLOR = '#2ca02c'  # зеленый для центров масс кластеров
ADJUSTED_P_COLOR = '#d62728'  # красный для скорректированных P состояний
GRID_ALPHA = 0.3
MARKER_CORRESPONDENCE_COLOR = '#e377c2'  # розовый для линий соответствия точек
CORRESPONDENCE_ALPHA = 0.5
CORRESPONDENCE_LINEWIDTH = 1.0
plt.rcParams.update({
    'figure.titlesize': TITLE_FONT,
    'axes.titlesize': TITLE_FONT,
    'axes.labelsize': AXIS_FONT,
    'xtick.labelsize': TICK_FONT,
    'ytick.labelsize': TICK_FONT,
    'legend.fontsize': ANNOT_FONT,
    'font.family': 'sans-serif'
})
# =======================
# Вспомогательные функции
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
    Normalize membrane flag to a string with at least 3 digits, preserving leading zeros
    if the user provided them as string (e.g. "015") or when provided as int (15 -> "015").
    """
    if isinstance(m, (int, np.integer)):
        return f"{int(m):03d}"
    s = str(m)
    # strip possible whitespace
    s = s.strip()
    # if s represents an integer (e.g. "015" or "15"), zero-pad to 3 digits
    if re.fullmatch(r"\d+", s):
        return s.zfill(3)
    # otherwise, return as-is
    return s
def sort_points_left_to_right_top_to_bottom(points):
    """
    Сортирует точки в порядке слева-направо и сверху-вниз
    Args:
        points: массив точек формы (n_points, 2)
    Returns:
        sorted_points: отсортированные точки
        sorted_indices: исходные индексы точек после сортировки
    """
    # Копируем точки для сортировки
    points_copy = points.copy()
    # Сначала сортируем по Y (сверху вниз, т.е. по возрастанию Y)
    # Затем сортируем по X (слева направо, т.е. по возрастанию X)
    sorted_indices = np.lexsort((points_copy[:, 0], points_copy[:, 1]))
    sorted_points = points_copy[sorted_indices]
    return sorted_points, sorted_indices
def find_nearest_correspondences(reference_points, target_points):
    """
    Находит ближайшие соответствия между точками эталонного состояния и целевого состояния
    Args:
        reference_points: эталонные точки (уже отсортированные в нужном порядке), форма (n_points, 2)
        target_points: целевые точки, форма (n_points, 2)
    Returns:
        correspondences: массив индексов соответствий, где correspondences[i] = j означает,
        что i-тая точка из reference_points соответствует j-той точке из target_points
        distances: расстояния до ближайших соседей
    """
    n_points = len(reference_points)
    # Проверяем наличие NaN и Inf в данных
    finite_mask = np.all(np.isfinite(target_points), axis=1)
    if not np.all(finite_mask):
        print(f"Предупреждение: target_points содержит NaN или Inf значения.")
        num_finite = np.sum(finite_mask)
        print(f"  Количество корректных точек: {num_finite} из {n_points}")
        if num_finite < n_points:
            print(f"  Недостаточно корректных точек для точного сопоставления. Используется запасной метод.")
            # Если недостаточно корректных точек, возвращаем соответствие по порядку
            return np.arange(n_points), np.full(n_points, np.nan)
    # Фильтруем точки
    target_points_finite = target_points[finite_mask]
    original_indices = np.where(finite_mask)[0]
    try:
        # Создаем KDTree для целевых точек для быстрого поиска ближайших соседей
        tree = KDTree(target_points_finite)
        # Находим ближайшего соседа для каждой точки из reference_points
        distances, indices = tree.query(reference_points, k=1)
        # Преобразуем индексы обратно к оригинальным
        original_indices_mapped = original_indices[indices]
        return original_indices_mapped, distances
    except Exception as e:
        print(f"Ошибка при работе с KDTree: {e}")
        print(f"reference_points shape: {reference_points.shape}, all finite: {np.all(np.isfinite(reference_points))}")
        print(f"target_points_finite shape: {target_points_finite.shape}, all finite: {np.all(np.isfinite(target_points_finite))}")
        # Возвращаем соответствие по порядку при ошибке
        return np.arange(n_points), np.full(n_points, np.nan)
def compute_deformation_vectors(reference_points, target_points, correspondences):
    """
    Вычисляет векторы деформаций между эталонными и целевыми точками с учетом соответствий
    Args:
        reference_points: эталонные точки, форма (n_points, 2)
        target_points: целевые точки, форма (n_points, 2)
        correspondences: массив соответствий, где correspondences[i] = j означает,
        что i-тая точка из reference_points соответствует j-той точке из target_points
    Returns:
        deformation_vectors: массив векторов деформаций, форма (n_points, 2)
    """
    n_points = len(reference_points)
    deformation_vectors = np.zeros((n_points, 2))
    for i in range(n_points):
        ref_point = reference_points[i]
        if correspondences[i] < len(target_points):
            target_point = target_points[correspondences[i]]
        else:
            # Если индекс выходит за границы, используем ближайшую доступную точку
            target_point = target_points[-1]
            print(f"Предупреждение: индекс {correspondences[i]} выходит за границы массива размером {len(target_points)}")
        deformation_vectors[i] = target_point - ref_point
    return deformation_vectors
def compute_total_l2_norm(deformation_vectors):
    """
    Вычисляет сумму L2 норм (евклидовых расстояний) для всех векторов деформаций
    Args:
        deformation_vectors: массив векторов деформаций, форма (n_points, 2)
    Returns:
        float: сумма L2 норм всех векторов
    """
    # Вычисляем L2 норму для каждого вектора и суммируем
    l2_norms = np.linalg.norm(deformation_vectors, axis=1)
    total_l2 = np.sum(l2_norms)
    return total_l2
def find_all_datasets_by_type(sensor_type):
    """
    Находит все датасеты указанного типа датчика
    Args:
        sensor_type: тип датчика ('A', 'B', 'C', 'D', 'E')
    Returns:
        list: Список путей к директориям датасетов
    """
    if not LABELLED_DIR.exists():
        print(f"Директория {LABELLED_DIR} не существует")
        return []
    datasets = []
    pattern = f"{sensor_type}_m*"
    # Ищем датасеты в директории Labelled
    for item in LABELLED_DIR.iterdir():
        if item.is_dir() and item.name.startswith(f"{sensor_type}_m"):
            datasets.append(item)
    # Если не нашли, ищем в поддиректориях
    if not datasets:
        for item in LABELLED_DIR.rglob("*"):
            if item.is_dir() and item.name.startswith(f"{sensor_type}_m"):
                datasets.append(item.parent / item.name)
    if not datasets:
        print(f"Не найдено датасетов с датчиком типа {sensor_type} в {LABELLED_DIR}")
        return []
    # Сортируем по имени для последовательного отображения
    datasets.sort(key=lambda x: x.name)
    print(f"Найдено датасетов с датчиком типа {sensor_type}: {len(datasets)}")
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds.name}")
    return datasets
def load_dataset_data(dataset_dir, sensor_type):
    """
    Загружает данные из датасета с определением начальных состояний по силе
    Args:
        dataset_dir (Path): Путь к директории датасета
        sensor_type (str): Тип датчика для определения количества маркеров
    Returns:
        tuple: (markers_by_idx, dataset_info) или (None, None) в случае ошибки
    """
    dataset_name = dataset_dir.name
    # Получаем количество маркеров из значения 'M' в SENSOR_GEOMETRY
    sensor_params = SENSOR_GEOMETRY.get(sensor_type, {})
    num_markers = sensor_params.get('M', 15)  # По умолчанию 15 маркеров
    # Поиск файлов с данными
    inputs_path = None
    outputs_path = None
    # Ищем файлы inputs и outputs
    for file in dataset_dir.iterdir():
        if file.name.lower().startswith('inputs') and file.suffix == '.csv':
            inputs_path = file
        elif file.name.lower().startswith('outputs') and file.suffix == '.csv':
            outputs_path = file
    # Если не нашли, пробуем глубокий поиск
    if inputs_path is None:
        inputs_candidates = list(dataset_dir.glob("**/*inputs*.csv"))
        if inputs_candidates:
            inputs_path = inputs_candidates[0]
    if outputs_path is None:
        outputs_candidates = list(dataset_dir.glob("**/*outputs*.csv"))
        if outputs_candidates:
            outputs_path = outputs_candidates[0]
    if inputs_path is None or outputs_path is None:
        print(f"Не удалось найти файлы с данными в {dataset_dir}")
        return None, None
    # Загружаем данные
    try:
        marker_positions = pd.read_csv(inputs_path)
        force_data = pd.read_csv(outputs_path)
    except Exception as e:
        print(f"Ошибка загрузки данных из {dataset_dir}: {e}")
        return None, None
    # Проверяем наличие нужных столбцов
    required_cols_inputs = ['marker_id', 'x_coord', 'y_coord', 'annotation_idx']
    for col in required_cols_inputs:
        if col not in marker_positions.columns:
            print(f"В файле {inputs_path.name} отсутствует необходимый столбец: {col}")
            return None, None
    # Определяем колонку с силой
    force_col = None
    possible_force_cols = ['force_value_N', 'force_value', 'force_N', 'force', 'force_value_N.1']
    for col in possible_force_cols:
        if col in force_data.columns:
            force_col = col
            break
    if force_col is None:
        # Берем первую числовую колонку, кроме индекса
        numeric_cols = force_data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            force_col = numeric_cols[0]
    if force_col is None:
        print(f"Не удалось определить колонку с силой в {outputs_path}")
        print(f"Доступные столбцы: {force_data.columns.tolist()}")
        return None, None
    # Определяем колонку с индексами замеров в force_data
    idx_col = None
    possible_idx_cols = ['annotation_idx', 'frame_idx', 'idx', 'index']
    for col in possible_idx_cols:
        if col in force_data.columns:
            idx_col = col
            break
    if idx_col is None:
        # Если нет явной колонки индексов, предполагаем соответствие порядку
        if len(force_data) == marker_positions['annotation_idx'].nunique():
            force_data['annotation_idx'] = sorted(marker_positions['annotation_idx'].unique())
            idx_col = 'annotation_idx'
        else:
            print(f"Не удалось сопоставить замеры по индексам в {dataset_dir}")
            print(f"Количество уникальных индексов в inputs: {marker_positions['annotation_idx'].nunique()}")
            print(f"Количество строк в outputs: {len(force_data)}")
            return None, None
    # Создаем словарь сопоставления индексов и сил
    force_dict = {}
    for _, row in force_data.iterrows():
        ann_idx = row[idx_col]
        # Приводим к целому числу если возможно
        try:
            ann_idx = int(ann_idx)
        except (TypeError, ValueError):
            pass
        force_value = row[force_col]
        force_dict[ann_idx] = force_value
    # Сортируем данные по annotation_idx
    marker_positions = marker_positions.sort_values(['annotation_idx', 'marker_id'])
    # Получаем уникальные индексы измерений
    unique_indices = sorted(marker_positions['annotation_idx'].unique())
    M = len(unique_indices)  # Количество замеров
    # Извлекаем позиции маркеров для каждого замера
    markers_by_idx = {}
    original_markers_by_idx = {}  # Сохраняем исходные координаты для визуализации
    for idx in unique_indices:
        frame_data = marker_positions[marker_positions['annotation_idx'] == idx]
        # Сортируем по marker_id
        frame_data = frame_data.sort_values('marker_id')
        coords = frame_data[['x_coord', 'y_coord']].values
        # Проверяем количество маркеров
        if coords.shape[0] != num_markers:
            print(f"Предупреждение: в замере {idx} датасета {dataset_name} "
                  f"обнаружено {coords.shape[0]} маркеров вместо ожидаемых {num_markers}. "
                  f"Пропускаем этот замер.")
            continue
        # Проверяем наличие NaN или Inf в координатах
        if not np.all(np.isfinite(coords)):
            print(f"Предупреждение: в замере {idx} датасета {dataset_name} "
                  f"обнаружены NaN или Inf значения в координатах. Пропускаем этот замер.")
            continue
        # Сохраняем как нормализованные (будут преобразованы позже), так и исходные координаты
        markers_by_idx[idx] = coords.copy()
        original_markers_by_idx[idx] = coords.copy()
    # Определяем начальные состояния (P) по порогу силы
    P_indices = [idx for idx in unique_indices if idx in markers_by_idx and force_dict.get(idx, np.inf) < P_THRESHOLD]
    # Если недостаточно P-состояний, берем замеры с минимальной силой
    if len(P_indices) < MIN_P_STATES:
        # Сортируем индексы по возрастанию силы
        sorted_indices = sorted([idx for idx in unique_indices if idx in markers_by_idx],
                               key=lambda idx: force_dict.get(idx, np.inf))
        P_indices = sorted_indices[:MIN_P_STATES]
        min_forces = [force_dict.get(idx, np.inf) for idx in P_indices]
        print(f"Предупреждение: в датасете {dataset_name} недостаточно замеров с силой < {P_THRESHOLD} Н")
        print(f"Взяты {len(P_indices)} замеров с минимальной силой: {min_forces}")
    # Информация о датасете
    dataset_info = {
        'name': dataset_name,
        'markers_count': num_markers,  # Количество маркеров для данного типа датчика
        'measurements_count': M,
        'directory': str(dataset_dir),
        'annotation_indices': unique_indices,
        'marker_ids': list(range(num_markers)),  # Стандартная нумерация маркеров
        'force_col': force_col,
        'force_dict': force_dict,
        'P_indices': P_indices,
        'original_coordinates': original_markers_by_idx  # Исходные координаты для визуализации
    }
    print(f"\nЗагружен датасет: {dataset_name}")
    print(f"  Количество маркеров: {dataset_info['markers_count']}")
    print(f"  Количество измерений: {dataset_info['measurements_count']}")
    print(f"  Количество P-состояний (начальных): {len(P_indices)}")
    print(f"  Силы P-состояний: {[force_dict.get(idx, np.inf) for idx in P_indices]}")
    return markers_by_idx, dataset_info
def find_cluster_centers(all_states, num_clusters):
    """
    Находит центры масс кластеров для всех состояний
    Args:
        all_states: список всех состояний
        num_clusters: количество кластеров (равно количеству маркеров)
    Returns:
        cluster_centers: массив координат центров кластеров
    """
    print("\nПоиск центров масс кластеров всех состояний...")
    print(f"Общее количество состояний: {len(all_states)}")
    # Объединяем все точки из всех состояний в один массив
    all_points = np.vstack(all_states)
    print(f"Общее количество точек для кластеризации: {len(all_points)}")
    # Применяем KMeans для кластеризации
    print(f"Применение KMeans с количеством кластеров: {num_clusters}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(all_points)
    # Получаем координаты центров кластеров
    cluster_centers = kmeans.cluster_centers_
    # Вычисляем инерцию (сумму квадратов расстояний от точек до центров их кластеров)
    inertia = kmeans.inertia_
    print(f"Инерция кластеризации: {inertia:.2f}")
    # Вычисляем среднее расстояние от точек до их центров кластеров
    distances = []
    for i, point in enumerate(all_points):
        cluster_idx = kmeans.labels_[i]
        center = cluster_centers[cluster_idx]
        distance = np.linalg.norm(point - center)
        distances.append(distance)
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)
    print(f"Среднее расстояние от точек до центров кластеров: {avg_distance:.2f}")
    print(f"Максимальное расстояние от точек до центров кластеров: {max_distance:.2f}")
    return cluster_centers
def normalize_and_align_datasets(datasets, sensor_type):
    """
    Нормализует все датасеты с использованием центров масс кластеров как эталонного состояния
    и выравнивает маркеры в порядке слева-направо и сверху-вниз
    Args:
        datasets: список путей к датасетам
        sensor_type: тип датчика для определения количества маркеров
    Returns:
        tuple: (reference_P_sorted, all_deformations, all_forces, dataset_info_list,
               marker_correspondences, all_original_states, all_total_l2_norms,
               cluster_centers)
    """
    print("\n" + "="*60)
    print(f"НАЧАЛО НОРМАЛИЗАЦИИ И ВЫРАВНИВАНИЯ ДАННЫХ ДЛЯ ДАТЧИКА ТИПА {sensor_type}")
    print("="*60)
    # Собираем все состояния для кластеризации
    all_states = []  # Список всех состояний для всех датасетов
    all_original_states = []  # Список исходных состояний для визуализации
    dataset_full_info = []  # Полная информация о датасетах с markers_by_idx
    # Определяем количество маркеров для данного типа датчика
    sensor_params = SENSOR_GEOMETRY.get(sensor_type, {})
    num_markers = sensor_params.get('M', 15)
    print(f"Количество маркеров для датчика типа {sensor_type}: {num_markers}")
    print("Шаг 1: Загрузка данных и сбор всех состояний...")
    valid_datasets = 0
    for dataset_dir in tqdm(datasets, desc="Загрузка датасетов"):
        markers_by_idx, dataset_info = load_dataset_data(dataset_dir, sensor_type)
        if markers_by_idx is None:
            continue
        # Проверяем наличие хотя бы одного состояния
        valid_indices = [idx for idx in dataset_info['annotation_indices'] if idx in markers_by_idx]
        if len(valid_indices) == 0:
            print(f"Пропускаем датасет {dataset_dir.name}: нет валидных состояний")
            continue
        # Собираем все состояния датасета
        dataset_states = []
        dataset_original_states = []
        for idx in valid_indices:
            if idx in markers_by_idx:
                dataset_states.append(markers_by_idx[idx])
                dataset_original_states.append(dataset_info['original_coordinates'][idx])
        # Добавляем состояния в общий список
        all_states.extend(dataset_states)
        all_original_states.extend(dataset_original_states)
        # Сохраняем полную информацию о датасете с markers_by_idx
        full_info = {
            'name': dataset_info['name'],
            'markers_by_idx': markers_by_idx,
            'annotation_indices': dataset_info['annotation_indices'],
            'P_indices': dataset_info['P_indices'],
            'force_dict': dataset_info['force_dict'],
            'marker_ids': dataset_info['marker_ids'],
            'markers_count': dataset_info['markers_count'],
            'original_coordinates': dataset_info['original_coordinates']
        }
        dataset_full_info.append(full_info)
        valid_datasets += 1
    if not all_states:
        print("Ошибка: не удалось загрузить ни одного валидного датасета")
        return None, None, None, None, None, None, None, None
    print(f"\nВсего загружено валидных датасетов: {valid_datasets}")
    print(f"Количество собранных состояний: {len(all_states)}")
    print(f"Количество маркеров в каждом датасете: {num_markers}")
    # Шаг 2: Поиск центров масс кластеров всех состояний
    print("\nШаг 2: Поиск центров масс кластеров всех состояний...")
    cluster_centers = find_cluster_centers(all_states, num_markers)
    # Сортируем центры масс в порядке слева-направо и сверху-вниз
    reference_P_sorted, sorted_indices = sort_points_left_to_right_top_to_bottom(cluster_centers)
    print(f"Центры масс отсортированы. Исходные индексы после сортировки: {sorted_indices}")
    print(f"Эталонное состояние (центры масс) вычислено и отсортировано. Форма: {reference_P_sorted.shape}")
    # Подготавливаем данные для агрегированного датасета
    print("\nШаг 3: Обработка всех состояний и вычисление векторов деформаций...")
    all_deformations = []  # Список для всех векторов деформаций (Q-P)
    all_forces = []         # Список для всех значений сил
    marker_correspondences = []  # Список соответствий маркеров для каждого замера
    all_total_l2_norms = []  # Список сумм L2 норм для каждого замера
    dataset_info_list = []  # Информация о датасетах
    # Обрабатываем каждый датасет
    for dataset_info in tqdm(dataset_full_info, desc="Обработка датасетов"):
        dataset_name = dataset_info['name']
        markers_by_idx = dataset_info['markers_by_idx']
        annotation_indices = dataset_info['annotation_indices']
        force_dict = dataset_info['force_dict']
        # Обрабатываем каждый замер в датасете
        for idx in annotation_indices:
            if idx not in markers_by_idx:
                continue
            # Получаем текущее состояние
            current_state = markers_by_idx[idx]
            force_value = force_dict.get(idx, np.nan)
            # Проверяем наличие NaN или Inf в данных
            if not np.all(np.isfinite(current_state)) or np.isnan(force_value):
                print(f"Предупреждение: в датасете {dataset_name}, замер {idx} содержатся некорректные значения. Пропускаем.")
                continue
            # Находим соответствия между текущим состоянием и эталонным отсортированным состоянием
            correspondences, distances = find_nearest_correspondences(
                reference_P_sorted, current_state
            )
            # Вычисляем векторы деформаций
            deformation_vectors = compute_deformation_vectors(
                reference_P_sorted, current_state, correspondences
            )
            # Вычисляем сумму L2 норм векторов деформаций
            total_l2_norm = compute_total_l2_norm(deformation_vectors)
            # Добавляем в общие списки
            all_deformations.append(deformation_vectors)
            all_forces.append(force_value)
            all_total_l2_norms.append(total_l2_norm)
            marker_correspondences.append({
                'dataset': dataset_name,
                'measurement_idx': idx,
                'correspondences': correspondences.tolist(),
                'distances': distances.tolist()
            })
            # Добавляем информацию о датасете
            dataset_info_list.append({
                'name': dataset_name,
                'num_measurements': len([idx for idx in annotation_indices if idx in markers_by_idx]),
                'marker_ids': list(range(num_markers))
            })
    if not all_deformations:
        print("Ошибка: не удалось вычислить векторы деформаций для ни одного замера")
        return None, None, None, None, None, None, None, None
    print(f"\nВсего обработано замеров: {len(all_deformations)}")
    print(f"Форма векторов деформаций: {np.array(all_deformations).shape}")
    print(f"Диапазон сил: {min(all_forces):.3f} Н - {max(all_forces):.3f} Н")
    print(f"Диапазон сумм L2 норм: {min(all_total_l2_norms):.3f} - {max(all_total_l2_norms):.3f}")
    return (reference_P_sorted, np.array(all_deformations), np.array(all_forces),
            dataset_info_list, marker_correspondences, all_original_states,
            np.array(all_total_l2_norms), cluster_centers)
# Вспомогательная функция для преобразования numpy типов в стандартные Python типы
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
def save_aggregated_dataset(reference_P, deformations, forces, dataset_info_list,
                           marker_correspondences, total_l2_norms, cluster_centers,
                           sensor_type):
    """
    Сохраняет агрегированный нормализованный датасет с векторами деформаций, суммами L2 норм
    и геометрическими параметрами датчика, включая информацию о центрах масс кластеров
    Args:
        reference_P: отсортированные центры масс кластеров как эталонное состояние
        deformations: массив векторов деформаций
        forces: массив значений сил
        dataset_info_list: информация о датасетах
        marker_correspondences: соответствия маркеров для каждого замера
        total_l2_norms: массив сумм L2 норм векторов деформаций
        cluster_centers: центры масс кластеров
        sensor_type: тип датчика (A, B, C, D, E)
    Returns:
        Path: путь к сохраненному датасету
    """
    print("\n" + "="*60)
    print(f"СОХРАНЕНИЕ АГРЕГИРОВАННОГО ДАТАСЕТА ДЛЯ ДАТЧИКА ТИПА {sensor_type}")
    print("="*60)
    # Определяем количество датасетов и замеров
    num_measurements = len(forces)
    num_markers = reference_P.shape[0]
    print(f"Статистика по данным:")
    print(f"  Общее количество замеров: {num_measurements}")
    print(f"  Количество датасетов: {len(dataset_info_list)}")
    print(f"  Количество маркеров: {num_markers}")
    print(f"  Форма эталонного состояния: {reference_P.shape}")
    print(f"  Диапазон сумм L2 норм: {np.min(total_l2_norms):.3f} - {np.max(total_l2_norms):.3f}")
    # Получаем геометрические параметры для данного типа датчика
    sensor_params = SENSOR_GEOMETRY.get(sensor_type, {})
    sensor_geometry = {
        'D': sensor_params.get('D', 'N/A'),
        'M': sensor_params.get('M', 'N/A')
    }
    print(f"Геометрические параметры датчика типа {sensor_type}:")
    for param, value in sensor_geometry.items():
        print(f"  {param}: {value}")
    # Создаем директорию для сохранения
    aggregated_dir = LABELLED_DIR / f"Aggregated_deformation_vectors_sensor_{sensor_type}"
    aggregated_dir.mkdir(exist_ok=True)
    # Сохраняем эталонное состояние (центры масс кластеров) с правильным именем файла
    reference_file = aggregated_dir / "reference_P_sorted.npy"
    np.save(reference_file, reference_P)
    print(f"Эталонное состояние (центры масс кластеров) сохранено в: {reference_file}")
    # Сохраняем центры масс кластеров (до сортировки)
    cluster_centers_file = aggregated_dir / "cluster_centers_unsorted.npy"
    np.save(cluster_centers_file, cluster_centers)
    print(f"Центры масс кластеров (до сортировки) сохранены в: {cluster_centers_file}")
    # Сохраняем векторы деформаций
    deformations_file = aggregated_dir / "deformation_vectors.npy"
    np.save(deformations_file, deformations)
    print(f"Векторы деформаций сохранены в: {deformations_file}")
    # Сохраняем значения сил
    forces_file = aggregated_dir / "forces.npy"
    np.save(forces_file, forces)
    print(f"Значения сил сохранены в: {forces_file}")
    # Сохраняем суммы L2 норм
    total_l2_file = aggregated_dir / "total_l2_norms.npy"
    np.save(total_l2_file, total_l2_norms)
    print(f"Суммы L2 норм векторов деформаций сохранены в: {total_l2_file}")
    # Сохраняем соответствия маркеров
    correspondences_file = aggregated_dir / "marker_correspondences.json"
    with open(correspondences_file, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(marker_correspondences), f, indent=2, ensure_ascii=False)
    print(f"Соответствия маркеров сохранены в: {correspondences_file}")
    # Создаем метаинформацию о датасете
    total_samples_per_dataset = {}
    for corr in marker_correspondences:
        dataset_name = corr['dataset']
        total_samples_per_dataset[dataset_name] = total_samples_per_dataset.get(dataset_name, 0) + 1
    meta_info = {
        'sensor_type': sensor_type,
        'sensor_geometry': sensor_geometry,  # Геометрические параметры датчика
        'num_datasets': len(dataset_info_list),
        'num_markers': num_markers,
        'total_measurements': num_measurements,
        'measurements_per_dataset': total_samples_per_dataset,
        'datasets': [info['name'] for info in dataset_info_list],
        'marker_ids': list(range(num_markers)),
        'reference_P_file': str(reference_file),
        'cluster_centers_unsorted_file': str(cluster_centers_file),
        'reference_P_shape': list(reference_P.shape),
        'cluster_centers_unsorted_shape': list(cluster_centers.shape),
        'reference_P_mean': np.mean(reference_P, axis=0).tolist(),
        'cluster_centers_unsorted_mean': np.mean(cluster_centers, axis=0).tolist(),
        'reference_P_std': np.std(reference_P, axis=0).tolist(),
        'cluster_centers_unsorted_std': np.std(cluster_centers, axis=0).tolist(),
        'deformations_shape': list(deformations.shape),
        'total_l2_norms_shape': list(total_l2_norms.shape),
        'total_l2_norms_range': [float(np.min(total_l2_norms)), float(np.max(total_l2_norms))],
        'total_l2_norms_mean': float(np.mean(total_l2_norms)),
        'total_l2_norms_std': float(np.std(total_l2_norms)),
        'force_range': [float(np.min(forces)), float(np.max(forces))],
        'force_mean': float(np.mean(forces)),
        'force_std': float(np.std(forces)),
        'created_at': pd.Timestamp.now().isoformat(),
        'description': (f'Датасет с векторами деформаций относительно эталонного состояния, '
                        f'определенного как центры масс кластеров всех состояний для датчика типа {sensor_type}. '
                        f'Геометрические параметры: {sensor_geometry}. '
                        f'Эталонное состояние отсортировано в порядке слева-направо и сверху-вниз.')
    }
    # Сохраняем метаинформацию
    meta_file = aggregated_dir / "metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(meta_info), f, indent=2, ensure_ascii=False)
    print(f"Метаинформация сохранена в: {meta_file}")
    # Создаем файл с описанием формата данных
    format_description = f"""
Формат данных агрегированного датасета для датчика типа {sensor_type}:
1. reference_P_sorted.npy:
- Форма: ({num_markers}, 2)
- Описание: Эталонное состояние (P-состояние), определенное как центры масс кластеров всех состояний
- Точки отсортированы в порядке слева-направо и сверху-вниз
- Структура: [[x0,y0], [x1,y1], ..., [x{num_markers-1},y{num_markers-1}]]
2. cluster_centers_unsorted.npy:
- Форма: ({num_markers}, 2)
- Описание: Центры масс кластеров всех состояний до сортировки
- Структура: [[x0,y0], [x1,y1], ..., [x{num_markers-1},y{num_markers-1}]]
3. deformation_vectors.npy:
- Форма: ({num_measurements}, {num_markers}, 2)
- Описание: Векторы деформаций для каждого замера относительно эталонного состояния
- Структура: [[[dx0,dy0], [dx1,dy1], ..., [dx{num_markers-1},dy{num_markers-1}]], ...]
- Вектор деформации вычисляется как: Q - P, где Q - положение маркера в деформированном состоянии
4. total_l2_norms.npy:
- Форма: ({num_measurements},)
- Описание: Суммы L2 норм (евклидовых расстояний) всех векторов деформаций для каждого замера
- Это скалярное значение, характеризующее общую степень деформации мембраны
5. forces.npy:
- Форма: ({num_measurements},)
- Описание: Значения силы для каждого замера
- Единицы измерения: Ньютоны (Н)
6. marker_correspondences.json:
- Описание: Соответствия между маркерами в эталонном и деформированном состояниях
- Структура: Список объектов с информацией о датасете, индексе замера, соответствиях и расстояниях
7. metadata.json:
- Описание: Метаинформация о датасете, включая геометрические параметры датчика:
- sensor_geometry: {{'D': {sensor_geometry.get('D', 'N/A')}, 'M': {sensor_geometry.get('M', 'N/A')}}}
- D: диаметр активной области (мм)
- M: модуль упругости материала (МПа)
Индексация маркеров следует порядку для датчика типа {sensor_type}:
"""
    # Добавляем описание разметки маркеров для каждого типа датчика
    sensor_params = SENSOR_GEOMETRY.get(sensor_type, {})
    num_markers = sensor_params.get('M', 15)
    if sensor_type == 'A':
        format_description += """
- Сетка 3x4: слева-направо и сверху-вниз
- Порядок: [0,1,2,3] - верхняя строка, [4,5,6,7] - средняя строка, [8,9,10,11] - нижняя строка
"""
    elif sensor_type == 'B':
        format_description += """
- Сетка 3x5: слева-направо и сверху-вниз
- Порядок: [0,1,2,3,4] - верхняя строка, [5,6,7,8,9] - средняя строка, [10,11,12,13,14] - нижняя строка
"""
    elif sensor_type == 'C':
        format_description += """
- Сетка 4x5: слева-направо и сверху-вниз
- Порядок: [0,1,2,3,4] - первая строка, [5,6,7,8,9] - вторая строка,
[10,11,12,13,14] - третья строка, [15,16,17,18,19] - четвертая строка
"""
    elif sensor_type == 'D':
        format_description += """
- Сетка 4x6: слева-направо и сверху-вниз
- Порядок: [0,1,2,3,4,5] - первая строка, [6,7,8,9,10,11] - вторая строка,
[12,13,14,15,16,17] - третья строка, [18,19,20,21,22,23] - четвертая строка
"""
    elif sensor_type == 'E':
        format_description += """
- Сетка 5x6: слева-направо и сверху-вниз
- Порядок: [0,1,2,3,4,5] - первая строка, [6,7,8,9,10,11] - вторая строка,
[12,13,14,15,16,17] - третья строка, [18,19,20,21,22,23] - четвертая строка,
[24,25,26,27,28,29] - пятая строка
"""
    else:
        format_description += "- Стандартный порядок сортировки слева-направо и сверху-вниз\n"
    format_file = aggregated_dir / "format_description.txt"
    with open(format_file, 'w', encoding='utf-8') as f:
        f.write(format_description)
    print(f"Описание формата данных сохранено в: {format_file}")
    print(f"\nАгрегированный датасет сохранен в: {aggregated_dir}")
    print(f"  Эталонное состояние: {reference_file}")
    print(f"  Центры масс кластеров (до сортировки): {cluster_centers_file}")
    print(f"  Векторы деформаций: {deformations_file}")
    print(f"  Суммы L2 норм: {total_l2_file}")
    print(f"  Силы: {forces_file}")
    print(f"  Соответствия маркеров: {correspondences_file}")
    print(f"  Метаинформация: {meta_file}")
    print(f"  Описание формата: {format_file}")
    print(f"  Геометрические параметры: D={sensor_geometry.get('D', 'N/A')} мм, M={sensor_geometry.get('M', 'N/A')} МПа")
    return aggregated_dir
def visualize_marker_correspondences(reference_P, cluster_centers, states, correspondences, forces, num_samples=5, sensor_type='B'):
    """
    Визуализирует сопоставление маркеров между эталонным и деформированными состояниями
    с добавлением центров масс кластеров
    """
    print("\n" + "="*60)
    print(f"ВИЗУАЛИЗАЦИЯ СООТВЕТСТВИЙ МАРКЕРОВ ДЛЯ ДАТЧИКА ТИПА {sensor_type}")
    print("="*60)
    # Выбираем образцы с разными уровнями силы
    force_indices = np.argsort(forces)
    selected_indices = np.linspace(0, len(forces)-1, num_samples, dtype=int)
    selected_indices = force_indices[selected_indices]
    fig, axes = plt.subplots(1, num_samples+1, figsize=(5*(num_samples+1), 6), dpi=DPI)
    # Первый график - общее представление
    ax_overview = axes[0]
    # Строим центры масс кластеров (до сортировки)
    ax_overview.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                       s=P_MARKER_SIZE*1.5, c=CLUSTER_CENTER_COLOR, alpha=0.9,
                       marker='*', edgecolors='k', linewidths=1.5,
                       label='Центры масс кластеров')
    # Строим эталонное состояние (отсортированные центры масс)
    ax_overview.scatter(reference_P[:, 0], reference_P[:, 1],
                       s=P_MARKER_SIZE, c=P_COLOR, alpha=0.7,
                       marker='o', edgecolors='k', linewidths=1,
                       label='Эталонное P-состояние')
    # Рисуем линии соответствий между неотсортированными и отсортированными центрами масс
    # Создаем KDTree для поиска соответствий
    tree = KDTree(cluster_centers)
    _, indices = tree.query(reference_P)
    for i, center_idx in enumerate(indices):
        ax_overview.plot([reference_P[i, 0], cluster_centers[center_idx, 0]],
                        [reference_P[i, 1], cluster_centers[center_idx, 1]],
                        c=MARKER_CORRESPONDENCE_COLOR, alpha=CORRESPONDENCE_ALPHA*0.5,
                        linewidth=CORRESPONDENCE_LINEWIDTH, linestyle='--')
    ax_overview.set_title(f'Общий обзор эталонного состояния\nдля датчика {sensor_type}', fontsize=TITLE_FONT)
    ax_overview.set_xlabel('X', fontsize=AXIS_FONT)
    ax_overview.set_ylabel('Y', fontsize=AXIS_FONT)
    ax_overview.grid(True, alpha=GRID_ALPHA)
    ax_overview.legend(loc='best', fontsize=ANNOT_FONT)
    ax_overview.set_aspect('equal', adjustable='box')
    # Остальные графики - конкретные примеры
    for i, idx in enumerate(selected_indices, 1):
        ax = axes[i]
        state = states[idx]
        corr = correspondences[idx]['correspondences']
        force = forces[idx]
        # Строим эталонное состояние
        ax.scatter(reference_P[:, 0], reference_P[:, 1],
                  s=P_MARKER_SIZE, c=P_COLOR, alpha=0.8,
                  marker='o', edgecolors='k', linewidths=1,
                  label='Эталонное P')
        # Строим центры масс кластеров
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                  s=P_MARKER_SIZE*0.8, c=CLUSTER_CENTER_COLOR, alpha=0.7,
                  marker='*', edgecolors='k', linewidths=1,
                  label='Центры масс')
        # Строим деформированное состояние
        ax.scatter(state[:, 0], state[:, 1],
                  s=Q_MARKER_SIZE, c=Q_COLOR, alpha=0.8,
                  marker='s', edgecolors='k', linewidths=1,
                  label=f'Q-состояние\nСила: {force:.3f} Н')
        # Рисуем линии соответствий
        for ref_idx, target_idx in enumerate(corr):
            ref_point = reference_P[ref_idx]
            target_point = state[target_idx]
            ax.plot([ref_point[0], target_point[0]],
                   [ref_point[1], target_point[1]],
                   c=MARKER_CORRESPONDENCE_COLOR, alpha=CORRESPONDENCE_ALPHA,
                   linewidth=CORRESPONDENCE_LINEWIDTH, linestyle='--')
        # Добавляем номера маркеров для эталонного состояния
        for j, point in enumerate(reference_P):
            ax.text(point[0] + 0.2, point[1] + 0.2, str(j),
                   fontsize=ANNOT_FONT, fontweight='bold', color=P_COLOR)
        # Добавляем номера маркеров для деформированного состояния
        for j, point in enumerate(state):
            ax.text(point[0] + 0.2, point[1] + 0.2, str(j),
                   fontsize=ANNOT_FONT, fontweight='bold', color=Q_COLOR)
        ax.set_title(f'Сопоставление маркеров\nЗамер #{idx+1}', fontsize=TITLE_FONT)
        ax.set_xlabel('X', fontsize=AXIS_FONT)
        ax.set_ylabel('Y', fontsize=AXIS_FONT)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.set_aspect('equal', adjustable='box')
        if i == 1:
            ax.legend(loc='best', fontsize=ANNOT_FONT)
    fig.suptitle(f'Сопоставление маркеров для датчика типа {sensor_type}',
                fontsize=TITLE_FONT+2, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Сохраняем график
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"marker_correspondences_{sensor_type}_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"График сопоставления маркеров сохранен: {output_path}")
    plt.close()
    print("\nВизуализация завершена.")
def visualize_deformation_vectors(reference_P, deformations, forces, total_l2_norms, num_samples=5, sensor_type='B'):
    """
    Визуализирует векторы деформаций для выбранных образцов
    Args:
        reference_P: отсортированное эталонное состояние
        deformations: массив векторов деформаций
        forces: значения сил
        total_l2_norms: суммы L2 норм векторов деформаций
        num_samples: количество образцов для визуализации
        sensor_type: тип датчика для заголовка
    """
    print("\n" + "="*60)
    print(f"ВИЗУАЛИЗАЦИЯ ВЕКТОРОВ ДЕФОРМАЦИЙ ДЛЯ ДАТЧИКА ТИПА {sensor_type}")
    print("="*60)
    # Выбираем образцы с разными уровнями силы
    force_indices = np.argsort(forces)
    selected_indices = np.linspace(0, len(forces)-1, num_samples, dtype=int)
    selected_indices = force_indices[selected_indices]
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 6), dpi=DPI)
    if num_samples == 1:
        axes = [axes]
    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        deformation = deformations[idx]
        force = forces[idx]
        total_l2 = total_l2_norms[idx]
        # Строим эталонное состояние
        ax.scatter(reference_P[:, 0], reference_P[:, 1],
                  s=P_MARKER_SIZE, c=P_COLOR, alpha=0.8,
                  marker='o', edgecolors='k', linewidths=1,
                  label='Эталонное P')
        # Рисуем векторы деформаций
        for j, (point, vector) in enumerate(zip(reference_P, deformation)):
            # Рисуем вектор
            ax.arrow(point[0], point[1], vector[0], vector[1],
                    head_width=0.1, head_length=0.15, fc='red', ec='red',
                    alpha=0.7, linewidth=1.5)
            # Добавляем номер маркера
            ax.text(point[0] + 0.2, point[1] + 0.2, str(j),
                   fontsize=ANNOT_FONT, fontweight='bold', color=P_COLOR)
        ax.set_title(f'Векторы деформаций\nСила: {force:.3f} Н\nСумма L2: {total_l2:.3f}', fontsize=TITLE_FONT)
        ax.set_xlabel('X', fontsize=AXIS_FONT)
        ax.set_ylabel('Y', fontsize=AXIS_FONT)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.set_aspect('equal', adjustable='box')
        if i == 0:
            ax.legend(loc='best', fontsize=ANNOT_FONT)
    fig.suptitle(f'Векторы деформаций для датчика типа {sensor_type}',
                fontsize=TITLE_FONT+2, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Сохраняем график
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"deformation_vectors_{sensor_type}_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"График векторов деформаций сохранен: {output_path}")
    plt.close()
    print("\nВизуализация завершена.")
def visualize_final_plots(reference_P, cluster_centers, all_states, sensor_type='B'):
    """
    Отображает два финальных графика в терминале:
    1. Эталонное состояние (центры масс кластеров) и неотсортированные центры масс
    2. Все нормированные состояния всех датасетов, наложенные друг на друга, с эталонным состоянием
    """
    print("\n" + "="*80)
    print(f"ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ ДЛЯ ДАТЧИКА ТИПА {sensor_type}")
    print("="*80)
    print(f"График 1: Эталонное состояние (отсортированные центры масс) и неотсортированные центры масс")
    print(f"График 2: Все нормированные состояния, наложенные друг на друга, с эталонным состоянием")
    print("Закройте окна графиков для завершения программы")
    print("="*80)
    # Создаем фигуру для сравнения отсортированных и неотсортированных центров масс
    plt.figure(figsize=(12, 10), dpi=100)
    # Строим неотсортированные центры масс кластеров
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
               s=P_MARKER_SIZE*2, c=CLUSTER_CENTER_COLOR, alpha=0.95,
               marker='*', edgecolors='black', linewidths=2,
               label=f'Центры масс кластеров\n({len(cluster_centers)} точек)')
    # Строим отсортированное эталонное состояние
    plt.scatter(reference_P[:, 0], reference_P[:, 1],
               s=P_MARKER_SIZE*1.5, c=P_COLOR, alpha=0.85,
               marker='o', edgecolors='black', linewidths=1.5,
               label=f'Эталонное P-состояние\n({len(reference_P)} маркеров)')
    # Рисуем линии соответствий между неотсортированными и отсортированными центрами масс
    tree = KDTree(cluster_centers)
    _, indices = tree.query(reference_P)
    for i, center_idx in enumerate(indices):
        plt.plot([reference_P[i, 0], cluster_centers[center_idx, 0]],
                [reference_P[i, 1], cluster_centers[center_idx, 1]],
                c=MARKER_CORRESPONDENCE_COLOR, alpha=CORRESPONDENCE_ALPHA*0.5,
                linewidth=CORRESPONDENCE_LINEWIDTH, linestyle='--')
    # Добавляем номера для всех точек
    for i, (x, y) in enumerate(reference_P):
        plt.text(x + 0.15, y + 0.15, str(i),
                fontsize=9, fontweight='bold', color=P_COLOR)
    for i, (x, y) in enumerate(cluster_centers):
        plt.text(x + 0.15, y - 0.15, str(i),
                fontsize=9, fontweight='bold', color=CLUSTER_CENTER_COLOR)
    # Добавляем сетку и оси
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    # Статистика по расстояниям
    distances = np.linalg.norm(reference_P - cluster_centers[indices], axis=1)
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)
    title = f'Сравнение отсортированных и неотсортированных центров масс\nдля датчика типа {sensor_type}\n'
    title += f'Среднее расстояние между соответствующими точками: {avg_distance:.2f}\n'
    title += f'Макс. расстояние между соответствующими точками: {max_distance:.2f}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('X координата', fontsize=12)
    plt.ylabel('Y координата', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.axis('equal')
    plt.tight_layout()
    # Создаем вторую фигуру для всех состояний
    plt.figure(figsize=(12, 10), dpi=100)
    # Строим все состояния с очень низкой прозрачностью
    print(f"Отображение {len(all_states)} состояний...")
    for i, state in enumerate(all_states):
        if len(state) != len(reference_P):
            print(f"Предупреждение: состояние {i} имеет {len(state)} маркеров вместо {len(reference_P)}")
            continue
        plt.scatter(state[:, 0], state[:, 1],
                   s=Q_MARKER_SIZE*2, c=Q_COLOR, alpha=Q_ALPHA,
                   marker='s', edgecolors='none')
    # Строим эталонное состояние
    plt.scatter(reference_P[:, 0], reference_P[:, 1],
               s=P_MARKER_SIZE*1.5, c=P_COLOR, alpha=0.95,
               marker='o', edgecolors='black', linewidths=2,
               label=f'Эталонное P-состояние\n({len(reference_P)} маркеров)')
    # Добавляем сетку и оси
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    # Статистика по всем точкам
    all_points = np.vstack(all_states)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    title = f'Все нормированные состояния для датчика типа {sensor_type}\n'
    title += f'Всего состояний: {len(all_states)}, Общее количество точек: {len(all_points)}\n'
    title += f'Диапазон по X: {x_min:.2f} - {x_max:.2f} ({x_range:.2f})\n'
    title += f'Диапазон по Y: {y_min:.2f} - {y_max:.2f} ({y_range:.2f})'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('X координата', fontsize=12)
    plt.ylabel('Y координата', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.axis('equal')
    plt.tight_layout()
    # Отображаем оба графика и ждем их закрытия
    print("\nОтображение графиков. Закройте окна для завершения программы...")
    plt.show()
def main():
    """Основная функция для запуска нормализации и подготовки датасета"""
    print("="*80)
    print(f"ПОДГОТОВКА НОРМАЛИЗОВАННОГО ДАТАСЕТА ДЛЯ ДАТЧИКА ТИПА {SENSOR_TYPE}")
    print("="*80)
    print(f"Геометрические параметры для датчика типа {SENSOR_TYPE}:")
    sensor_params = SENSOR_GEOMETRY.get(SENSOR_TYPE, {})
    for param, value in sensor_params.items():
        print(f"  {param}: {value}")
    print(f"Метод определения эталонного состояния: центры масс кластеров всех состояний")
    # Находим все датасеты указанного типа датчика
    datasets = find_all_datasets_by_type(SENSOR_TYPE)
    if not datasets:
        print(f"Нет датасетов для обработки датчика типа {SENSOR_TYPE}. Проверьте директорию Labelled.")
        return
    # Нормализуем и выравниваем все датасеты с использованием центров масс кластеров как эталонного состояния
    result = normalize_and_align_datasets(datasets, SENSOR_TYPE)
    if result[0] is None:
        print("Ошибка при нормализации данных. Завершение работы.")
        return
    (reference_P, deformations, forces, dataset_info_list,
     marker_correspondences, all_original_states, total_l2_norms,
     cluster_centers) = result
    # Визуализируем сопоставление маркеров для нескольких образцов
    print("\nГенерация визуализаций...")
    visualize_marker_correspondences(
        reference_P,
        cluster_centers,
        all_original_states,
        marker_correspondences,
        forces,
        num_samples=5,
        sensor_type=SENSOR_TYPE
    )
    # Визуализируем векторы деформаций
    visualize_deformation_vectors(
        reference_P,
        deformations,
        forces,
        total_l2_norms,
        num_samples=5,
        sensor_type=SENSOR_TYPE
    )
    # Сохраняем агрегированный датасет
    save_aggregated_dataset(
        reference_P,
        deformations,
        forces,
        dataset_info_list,
        marker_correspondences,
        total_l2_norms,
        cluster_centers,
        sensor_type=SENSOR_TYPE
    )
    print("\n" + "="*80)
    print(f"ПОДГОТОВКА ДАТАСЕТА ЗАВЕРШЕНА ДЛЯ ДАТЧИКА ТИПА {SENSOR_TYPE}")
    print("="*80)
    # Выводим финальную статистику
    sensor_params = SENSOR_GEOMETRY.get(SENSOR_TYPE, {})
    num_markers = sensor_params.get('M', 15)
    print(f"\nФинальная статистика:")
    print(f"  Количество датасетов: {len(dataset_info_list)}")
    print(f"  Общее количество замеров: {len(forces)}")
    print(f"  Количество маркеров: {num_markers}")
    print(f"  Диапазон сил: {np.min(forces):.3f} Н - {np.max(forces):.3f} Н")
    print(f"  Средняя сила: {np.mean(forces):.3f} Н")
    print(f"  Стандартное отклонение силы: {np.std(forces):.3f} Н")
    print(f"  Диапазон сумм L2 норм: {np.min(total_l2_norms):.3f} - {np.max(total_l2_norms):.3f}")
    print(f"  Средняя сумма L2 норм: {np.mean(total_l2_norms):.3f}")
    print(f"  Стандартное отклонение L2 норм: {np.std(total_l2_norms):.3f}")
    # Проверяем распределение соответствий маркеров
    correspondences_stats = {}
    for corr in marker_correspondences:
        for ref_idx, target_idx in enumerate(corr['correspondences']):
            key = f"ref_{ref_idx}_to_target_{target_idx}"
            correspondences_stats[key] = correspondences_stats.get(key, 0) + 1
    print(f"\nСтатистика соответствий маркеров (топ-10 самых частых):")
    sorted_stats = sorted(correspondences_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    for key, count in sorted_stats:
        print(f"  {key}: {count} раз")
    # ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ: два графика в терминале
    print("\n" + "="*80)
    print("ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ: ОТКРЫТИЕ ГРАФИКОВ В ОКНАХ")
    print("="*80)
    visualize_final_plots(
        reference_P,
        cluster_centers,
        all_original_states,
        sensor_type=SENSOR_TYPE
    )
    print("\n" + "="*80)
    print("ПРОГРАММА ЗАВЕРШЕНА")
    print("="*80)
if __name__ == "__main__":
    main()