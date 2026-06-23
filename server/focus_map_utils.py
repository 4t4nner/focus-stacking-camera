#!/usr/bin/env python3
"""
Utilities for building HOMOGENEOUS focus maps (no 'grain' / speckle).

Pipeline:
  1) sharpness DENSITY fields (heavy gaussian) instead of sparse binary masks
  2) optimal label assignment via Graph Cut (PyMaxflow / pygco) if available,
     otherwise robust majority-vote smoothing
  3) small-island removal via connected components
  4) watershed-style fill of 'no man's land' (where nobody is sharp)
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# --- optional graph-cut backends -------------------------------------------
_HAS_MAXFLOW = False
_HAS_GCO = False
try:
    import maxflow  # PyMaxflow
    _HAS_MAXFLOW = True
except Exception:
    pass
try:
    # gco-wrapper: pip install gco-wrapper  ->  import gco
    import gco
    _HAS_GCO = True
except Exception:
    pass


# ===========================================================================
#  1. SHARPNESS DENSITY
# ===========================================================================

def sharpness_density(aligned_masks, density_sigma: float = 25.0):
    """
    Превращает разреженные бинарные DoG-маски в гладкие непрерывные поля
    'плотности резкости'. Большая sigma => крупные однородные зоны.

    Returns: np.ndarray (n, H, W) float32.
    """
    fields = []
    for mask in aligned_masks:
        fm = mask.astype(np.float32) / 255.0
        fm = cv2.GaussianBlur(fm, (0, 0), density_sigma)
        fields.append(fm)
    return np.stack(fields, axis=0)


# ===========================================================================
#  2. LABEL ASSIGNMENT
# ===========================================================================

def assign_graphcut(density, smooth_lambda: float = 18.0):
    """
    Оптимальное мультиметочное назначение через graph cut.
    density: (n, H, W) float32. Возвращает (H, W) int32 или None, если бэкенда нет.

    Data-term  : -density (чем выше резкость метки, тем дешевле её выбрать)
    Smooth-term: штраф за разные метки соседей (регуляризация => крупные зоны)
    """
    n, h, w = density.shape

    # --- случай 2 источников: PyMaxflow (точное решение) ---
    if n == 2 and _HAS_MAXFLOW:
        g = maxflow.Graph[float]()
        nodes = g.add_grid_nodes((h, w))
        # source=label1, sink=label0
        g.add_grid_tedges(nodes, density[1], density[0])
        structure = np.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]], dtype=np.float64)
        g.add_grid_edges(nodes, weights=smooth_lambda,
                         structure=structure, symmetric=True)
        g.maxflow()
        labels = g.get_grid_segments(nodes).astype(np.int32)  # bool -> 0/1
        return labels

    # --- N источников: gco-wrapper alpha-expansion ---
    if _HAS_GCO:
        try:
            return _assign_gco(density, smooth_lambda)
        except Exception as e:
            logger.warning(f"gco-wrapper assignment failed, fallback to "
                           f"majority-vote: {e}")
            return None

    return None

def _assign_gco(density, smooth_lambda: float):
    """
    Назначение меток через gco-wrapper.
    Пытается высокоуровневый хелпер cut_grid_graph_simple; если его нет
    в данной сборке — строит решётку вручную через GCO().
    Обе ветки решают одну и ту же энергию (alpha-expansion, Potts).
    """
    n, h, w = density.shape
    SCALE = 1000

    d = density.copy()
    dmax = float(d.max())
    if dmax > 0:
        d = d / dmax

    pairwise = (smooth_lambda * SCALE * (1 - np.eye(n))).astype(np.int32)
    pairwise = np.ascontiguousarray(pairwise)

    # --- путь 1: высокоуровневый хелпер (предпочтительно) ---
    if hasattr(gco, "cut_grid_graph_simple"):
        unary = ((1.0 - d).transpose(1, 2, 0) * SCALE).astype(np.int32)
        unary = np.ascontiguousarray(unary)
        labels = gco.cut_grid_graph_simple(
            unary, pairwise, n_iter=-1, algorithm='expansion'
        )
        return labels.reshape(h, w).astype(np.int32)

    # --- путь 2: ручная решётка через GCO() ---
    num_sites = h * w
    data_cost = ((1.0 - d).transpose(1, 2, 0).reshape(num_sites, n)
                 * SCALE).astype(np.int32)
    data_cost = np.ascontiguousarray(data_cost)

    g = gco.GCO()
    g.create_general_graph(num_sites, n, energy_is_float=False)
    g.set_data_cost(data_cost)
    g.set_smooth_cost(pairwise)

    idx = np.arange(num_sites).reshape(h, w)
    e_from = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()]).astype(np.int32)
    e_to   = np.concatenate([idx[:, 1:].ravel(),  idx[1:, :].ravel()]).astype(np.int32)
    e_w    = np.ones(e_from.shape[0], dtype=np.int32)
    g.set_all_neighbors(e_from, e_to, e_w)

    g.expansion(-1)
    labels = g.get_labels().astype(np.int32)
    g.destroy_graph()
    return labels.reshape(h, w)

def majority_smooth(assignment, n, ksize: int = 21, iterations: int = 3):
    """
    Fallback-сглаживание дискретной карты меток голосованием большинства.
    Убирает 'зерно' без graph-cut.
    """
    a = assignment.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    for _ in range(iterations):
        votes = np.zeros((n,) + a.shape, dtype=np.float32)
        for i in range(n):
            layer = (a == i).astype(np.float32)
            layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, kernel)
            votes[i] = cv2.GaussianBlur(layer, (0, 0), ksize / 3.0)
        a = np.argmax(votes, axis=0).astype(np.int32)
    return a


# ===========================================================================
#  3. SMALL-ISLAND REMOVAL
# ===========================================================================

def remove_small_islands(assignment, n, min_area: int):
    """
    Острова метки площадью < min_area поглощаются доминирующей соседней меткой.
    """
    a = assignment.copy()
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    changed = True
    guard = 0
    while changed and guard < 6:
        changed = False
        guard += 1
        for i in range(n):
            mask = (a == i).astype(np.uint8)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for comp in range(1, num):
                if stats[comp, cv2.CC_STAT_AREA] < min_area:
                    comp_mask = (labels == comp)
                    dil = cv2.dilate(comp_mask.astype(np.uint8), ring_kernel)
                    ring = (dil > 0) & (~comp_mask)
                    neigh = a[ring]
                    neigh = neigh[neigh != i]
                    if neigh.size:
                        a[comp_mask] = int(np.bincount(neigh, minlength=n).argmax())
                        changed = True
    return a


# ===========================================================================
#  4. FILL неопределённые пиксели
# ===========================================================================

def fill_undefined(assignment, undefined, n):
    """
    Заполняет undefined-пиксели ближайшей определённой меткой
    (геодезический рост => гладкие границы вместо рваного dilate).
    """
    if not np.any(undefined):
        return assignment
    a = assignment.copy().astype(np.int32)
    a[undefined] = -1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(max(a.shape)):
        holes = (a == -1)
        if not holes.any():
            break
        votes = np.zeros((n,) + a.shape, dtype=np.uint8)
        for i in range(n):
            layer = (a == i).astype(np.uint8)
            votes[i] = cv2.dilate(layer, kernel)
        any_grown = votes.max(axis=0) > 0
        best = np.argmax(votes, axis=0).astype(np.int32)
        fill_now = holes & any_grown
        a[fill_now] = best[fill_now]
    a[a == -1] = assignment[a == -1]
    return a