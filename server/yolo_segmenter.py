#!/usr/bin/env python3
"""
YOLO v11 instance segmentation for object-aware focus map refinement.

Downloads yolo11n-seg.pt on first run.
Produces per-object binary masks that are used to enforce object integrity
in the focus map (an object must not be split across focus zones).
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model = None
_model_lock = None

# Which model to use (env override possible)
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolo11x-seg.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.50"))
YOLO_MAX_DET = int(os.environ.get("YOLO_MAX_DET", "50"))


@dataclass
class SegmentedObject:
    """Single detected+segmented object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray  # bool mask, full image size (H, W)


def _ensure_model():
    """Lazy-load YOLO model (thread-safe)."""
    global _model, _model_lock
    import threading

    if _model_lock is None:
        _model_lock = threading.Lock()

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        logger.info(f"Loading YOLO model: {YOLO_MODEL}")
        try:
            from ultralytics import YOLO
            _model = YOLO(YOLO_MODEL)
            # Warm up with a dummy image
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _model.predict(dummy, verbose=False)
            logger.info(f"YOLO model loaded successfully: {YOLO_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            _model = None
            raise

    return _model


def segment_objects(
    image: np.ndarray,
    conf: float = None,
    iou: float = None,
    max_det: int = None,
    target_size: int = 1280,
) -> List[SegmentedObject]:
    """
    Run YOLOv11-seg on an image and return instance masks.

    Args:
        image: BGR numpy array (H, W, 3)
        conf: confidence threshold
        iou: NMS IoU threshold
        max_det: maximum detections
        target_size: YOLO inference size (longer side)

    Returns:
        List of SegmentedObject with masks at original image resolution.
    """
    if conf is None:
        conf = YOLO_CONF
    if iou is None:
        iou = YOLO_IOU
    if max_det is None:
        max_det = YOLO_MAX_DET

    model = _ensure_model()
    if model is None:
        return []

    h_orig, w_orig = image.shape[:2]

    results = model.predict(
        image,
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=target_size,
        verbose=False,
        retina_masks=True,  # full-resolution masks
    )

    objects = []

    if not results or len(results) == 0:
        return objects

    result = results[0]

    # Check if we have masks
    if result.masks is None or result.boxes is None:
        return objects

    masks_data = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
    boxes = result.boxes

    for i in range(len(boxes)):
        # Class info
        cls_id = int(boxes.cls[i].item())
        cls_conf = float(boxes.conf[i].item())
        cls_name = result.names.get(cls_id, f"class_{cls_id}")

        # Bounding box
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

        # Mask — resize to original if needed
        raw_mask = masks_data[i]  # (mask_h, mask_w), float 0..1
        if raw_mask.shape[0] != h_orig or raw_mask.shape[1] != w_orig:
            raw_mask = cv2.resize(
                raw_mask, (w_orig, h_orig),
                interpolation=cv2.INTER_LINEAR
            )

        bool_mask = raw_mask > 0.5

        objects.append(SegmentedObject(
            class_id=cls_id,
            class_name=cls_name,
            confidence=cls_conf,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            mask=bool_mask,
        ))

    logger.info(f"YOLO segmented {len(objects)} objects")
    return objects


def refine_focus_map_with_objects(
        focus_map: np.ndarray,
        objects: List[SegmentedObject],
        focus_points: Optional[List[Tuple[int, int]]] = None,
        forced_zone: Optional[np.ndarray] = None,
        session_dir: str = None,
) -> np.ndarray:
    """
    Refine focus_map so that each detected object belongs entirely to one zone.

    Правило выбора зоны объекта:
      1) если внутри объекта лежит точка фокуса i -> объект целиком отдаётся
         источнику i (объект РАСШИРЯЕТ область точки);
         если точек несколько — берётся та, чья зона уже покрывает больше
         пикселей объекта;
      2) иначе -> зона большинства пикселей (прежнее поведение).

    Args:
        focus_map: (H, W) int32, пиксель -> индекс источника
        objects: список SegmentedObject
        focus_points: список (cx, cy) в координатах focus_map; индекс = источник
        forced_zone: (H, W) int32, forced_zone[y,x] = i+1 для жёстко
                     закреплённых пикселей (из build_focus_map); может быть None
        session_dir: для отладочных изображений
    """
    if not objects:
        return focus_map

    h, w = focus_map.shape
    refined = focus_map.copy()
    n_sources = int(focus_map.max()) + 1

    focus_points = focus_points or []
    changes_total = 0

    for idx, obj in enumerate(objects):
        mask = obj.mask
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        obj_pixels = np.where(mask)
        n_pixels = len(obj_pixels[0])
        if n_pixels == 0:
            continue

        # --- какие точки фокуса попадают внутрь объекта? ---
        inside = []
        for pi, (px, py) in enumerate(focus_points):
            px = int(np.clip(px, 0, w - 1))
            py = int(np.clip(py, 0, h - 1))
            if pi < n_sources and mask[py, px]:
                inside.append(pi)

        zone_values = refined[obj_pixels]
        zone_counts = np.bincount(zone_values, minlength=n_sources)

        if inside:
            # выбираем источник точки, который уже покрывает больше пикселей объекта
            target_zone = max(inside, key=lambda s: zone_counts[s])
            reason = f"focus-point {target_zone}"
        else:
            target_zone = int(np.argmax(zone_counts))
            reason = "majority"

        need_change = n_pixels - zone_counts[target_zone]
        if need_change > 0:
            refined[obj_pixels] = target_zone
            changes_total += need_change
            logger.info(
                f"  Object {idx} '{obj.class_name}' ({obj.confidence:.2f}): "
                f"{n_pixels} px -> zone {target_zone} ({reason}, "
                f"changed {need_change} px)"
            )

    # --- повторно навязываем forced_zone: контуры точек неприкосновенны ---
    if forced_zone is not None:
        forced = forced_zone > 0
        refined[forced] = forced_zone[forced] - 1

    logger.info(f"  Focus map refinement: {changes_total} pixels changed "
                f"across {len(objects)} objects")

    if session_dir:
        _save_refinement_debug(focus_map, refined, objects, n_sources, session_dir)

    return refined

def _save_refinement_debug(
    original_map: np.ndarray,
    refined_map: np.ndarray,
    objects: List[SegmentedObject],
    n_sources: int,
    session_dir: str,
):
    """Save debug images showing YOLO detections and refinement diff."""
    h, w = original_map.shape

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255),
        (128, 255, 0),
    ]

    # 1. YOLO objects overlay
    obj_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i, obj in enumerate(objects):
        color = colors[i % len(colors)]
        mask = obj.mask
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        obj_vis[mask] = color

        # Draw bbox
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(obj_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = f"{obj.class_name} {obj.confidence:.2f}"
        cv2.putText(obj_vis, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(session_dir, "yolo_objects.png"), obj_vis)

    # 2. Diff map (where focus_map changed)
    diff = (original_map != refined_map).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(session_dir, "focusmap_yolo_diff.png"), diff)

    # 3. Refined focus map visualization
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n_sources):
        mask = refined_map == i
        color = colors[i % len(colors)]
        vis[mask] = color
    cv2.imwrite(os.path.join(session_dir, "focusmap_refined.png"), vis)

    logger.info(f"  [DEBUG] Saved YOLO debug images to {session_dir}")


def is_available() -> bool:
    """Check if YOLO can be loaded."""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        return False

