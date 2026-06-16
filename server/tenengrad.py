#!/usr/bin/env python3
"""
Tenengrad sharpness map extractor.
Usage: python tenengrad.py <path_to_image.jpg>
"""

import sys
import os
import cv2
import numpy as np


def tenengrad_map(image_path: str, ksize: int = 3, blur: int = 0) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if blur > 0:
        gray = cv2.GaussianBlur(gray, (blur | 1, blur | 1), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    sharpness = gx * gx + gy * gy
    return sharpness


def to_bright_thick(arr: np.ndarray,
                    percentile: float = 98.0,
                    dilate_iter: int = 2,
                    dilate_ksize: int = 3) -> np.ndarray:
    """
    Нормализует карту резкости с «растяжкой» до 255 и утолщает линии
    морфологической дилатацией.
    """
    arr = np.sqrt(arr)  # смягчаем динамический диапазон

    # Растяжка контраста: всё что выше перцентиля -> 255
    hi = np.percentile(arr, percentile)
    if hi < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)

    norm = np.clip(arr / hi, 0.0, 1.0) * 255.0
    out = norm.astype(np.uint8)

    # Гамма < 1 -> ещё ярче средние значения
    gamma = 0.6
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    out = cv2.LUT(out, lut)

    # Утолщение линий
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    out = cv2.dilate(out, kernel, iterations=dilate_iter)

    return out


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Стандартная нормализация для heatmap/overlay."""
    arr = np.sqrt(arr)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        print("Использование: python tenengrad.py <path_to_image.jpg>")
        sys.exit(1)

    in_path = sys.argv[1]
    base, _ = os.path.splitext(in_path)

    sharpness = tenengrad_map(in_path, ksize=3, blur=3)

    # 1) Яркая карта с утолщёнными линиями
    bright_map = to_bright_thick(sharpness,
                                 percentile=99.95,
                                 dilate_iter=2,
                                 dilate_ksize=3)
    out_gray = f"{base}_tenengrad.jpg"
    cv2.imwrite(out_gray, bright_map)

    # 2) Тепловая карта (на основе яркой версии)
    heatmap = cv2.applyColorMap(bright_map, cv2.COLORMAP_INFERNO)
    out_heat = f"{base}_tenengrad_heatmap.jpg"
    cv2.imwrite(out_heat, heatmap)

    # 3) Наложение на оригинал
    orig = cv2.imread(in_path, cv2.IMREAD_COLOR)
    mask3 = cv2.cvtColor(bright_map, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(orig, 0.5, mask3, 0.8, 0)
    out_overlay = f"{base}_tenengrad_overlay.jpg"
    cv2.imwrite(out_overlay, overlay)

    print("Сохранено:")
    print(f"  {out_gray}")
    print(f"  {out_heat}")
    print(f"  {out_overlay}")


if __name__ == "__main__":
    main()