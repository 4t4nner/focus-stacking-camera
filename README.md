# Focus stacking camera app

## Пайплайн и теория

### Общая цель проекта

Приложение реализует **автоматический focus stacking (фокус-стекинг) с детекцией объектов**.

Когда в кадре несколько объектов на разных расстояниях от камеры, обычная камера не может сфокусироваться на всех сразу (ограниченная глубина резкости — DoF, Depth of Field). Приложение детектирует все объекты, по очереди фокусируется на каждом, делает снимок, а затем программно склеивает из этих снимков одно изображение, в котором **все объекты резкие одновременно** (all-in-focus image).

---

### Теория

#### 1. Focus Stacking (фокус-стекинг)
**Проблема:** у объектива есть диафрагма (aperture). Чем она шире — тем меньше глубина резкости. В макросъёмке или при съёмке близких объектов DoF может быть меньше сантиметра. Объекты вне этой зоны получаются размытыми (bokeh).

**Решение:** сделать серию снимков с разными плоскостями фокуса, а затем для каждого пикселя финального изображения выбрать тот кадр, где этот пиксель самый резкий.

Математически: пусть есть N изображений $I_1, I_2, \ldots, I_N$. Для каждого пикселя $(x,y)$ вычисляется карта резкости $S_i(x,y)$. Финальное изображение:
$$I_{final}(x,y) = I_{k}(x,y), \quad \text{где } k = \arg\max_i S_i(x,y)$$

#### 2. Object Detection (детекция объектов)
Используется **ML Kit Object Detection** (Google) в режимах STREAM (live preview) и SINGLE_IMAGE. Также есть альтернатива через **YOLOv8/v10/v11 ONNX** через OpenCV DNN. Детектор возвращает bounding boxes, classId, confidence.
Центр bbox используется как точка фокуса (AF point).

Альтернатива YOLO в проекте использует letterbox preprocessing (resize с сохранением пропорций + padding до 640×640) и NMS (Non-Maximum Suppression) для устранения дубликатов.

#### 3. Измерение резкости (Sharpness Measurement)
В проекте используется **DoG (Difference of Gaussians) с enhancement**:
$$ DoG(x,y) = |G_{\sigma_1} * I - G_{\sigma_2} * I| $$
где $\sigma_1 < \sigma_2$. DoG — это аппроксимация LoG (Laplacian of Gaussian), даёт отклик на края/текстуру (высокочастотное содержимое = резкость).

Далее применяется **unsharp masking** для усиления:
$$S = DoG + \alpha \cdot (DoG - G_\sigma * DoG)$$

Результат бинаризуется по порогу → **details mask**: 1 = пиксель резкий, 0 = размытый. Потом морфология (OPEN → CLOSE) убирает шум и закрывает дырки.

**Альтернативные метрики резкости** (не используемые здесь): Laplacian variance, Sobel gradient magnitude, Tenengrad, модифицированный Laplacian.

#### 4. Image Alignment (выравнивание изображений)
Между снимками проходит ~350 мс + время фокусировки → рука дрожит, объекты двигаются. Нужно геометрически совместить кадры.

**Алгоритм:**
1. **ORB (Oriented FAST and Rotated BRIEF)** — детектор ключевых точек + бинарный дескриптор. Быстрее SIFT/SURF, свободен от патентов.
2. **Brute-Force Hamming matcher** — сравнивает ORB-дескрипторы (Hamming distance).
3. **Lowe's ratio test** — оставляет только матчи, где ближайший сосед значительно ближе второго (ratio < 0.75).
4. **RANSAC + findHomography** — устойчивая оценка гомографии (матрица 3×3, описывает проективное преобразование).

Гомография:
$$\begin{pmatrix} x' \\ y' \\ w' \end{pmatrix} = H \cdot \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}, \quad (x_{final}, y_{final}) = (x'/w', y'/w')$$

Затем `warpPerspective` переводит все кадры в систему координат первого (reference).

**Качество выравнивания** оценивается по `det(H)` (≈1 = хорошо) и по перспективным компонентам H[2,0], H[2,1] (≈0 = хорошо).

#### 5. Focus Map Building
После выравнивания для каждого пикселя решается: из какого исходного изображения его взять? Алгоритм:
- Для каждой маски детализации в точке (x,y) смотрим величину (после Gaussian blur для сглаживания).
- Пиксель получает индекс source с максимальной резкостью.
- Если во всех масках 0 (полностью несфокусированная область) — используется **Voronoi-подобный** подход: итеративная дилатация от focus points → пиксель получает индекс ближайшего focus point.
- **Mode filter** (кластерная медиана) сглаживает зубчатые границы.

#### 6. Compositing (склейка)
- **Direct composition**: прямая выборка пикселя по focus map.
- **Seam blending (feather blend)**: на границах между зонами разных source'ов используется **distance transform** — для каждого пикселя в зоне источника i считается расстояние до границы. Веса нормируются:
  $$w_i(x,y) = \frac{d_i(x,y)}{\sum_j d_j(x,y)}, \quad I(x,y) = \sum_i w_i(x,y) \cdot I_i(x,y)$$
  Это сглаживает швы (feathering).
- **Artifact detection**: если в одной точке цвета соседних кадров сильно отличаются (>60 в RGB distance), это ghosting — погрешность выравнивания.
- **Navier-Stokes inpainting** (`Photo.inpaint` с флагом INPAINT_NS) — заполняет артефактные пиксели, решая уравнение Навье-Стокса на изолинии яркости.

---

### Пайплайн
```

Фото_1 (фокус на объект 1) ──┐
Фото_2 (фокус на объект 2) ──┼── ImageAligner (ORB+Homography) ──► Aligned images
Фото_N (фокус на объект N) ──┘                                            │
                                                                           ▼
Маска_1 ──┐                                                    FocusMapBuilder
Маска_2 ──┼── warp masks ──► aligned masks ──────────────────► (per-pixel assignment)
Маска_N ──┘                                                            │
                                                                       ▼
                                                            FocusStackCompositor
                                                           ┌─ direct composition
                                                           ├─ feather blend at seams
                                                           └─ OpenCV inpainting (NS)
                                                                       │
                                                                       ▼
                                                              All-in-focus image
```
#### Фаза A: Live Preview (постоянно)
1. **CameraX** показывает preview через `PreviewView`.
2. `ImageAnalysis` получает кадры в RGBA_8888.
3. `processFrame()` → `imageProxyToBitmap()` (конверсия с учётом rowPadding и rotation).
4. `MlKitDetector.detectStream(bitmap)` → список `Detection` (classId, className, confidence, x1y1x2y2).
5. Detections отображаются в `OverlayView` поверх preview. Считается FPS.

#### Фаза B: Capture (по нажатию кнопки)
1. **Pause detection** (`detectionPaused = true`), UI переходит в "capturing mode" (красная рамка).
2. **Сессия**: создаётся `sessionTimestamp` для именования файлов.
3. **Для каждого детектированного объекта**:
    - Перевод координат центра bbox из image-space в preview-view-space (учитывая scale и offsets).
    - `focusOnPointAndWait(viewX, viewY)` — `FocusMeteringAction` (AF+AE), блокируется до 3 секунд.
    - `capturePhotoAsync()` → `imageCapture.takePicture()` сохраняет в MediaStore в папку `Pictures/CameraAppDetections/`.
    - Между снимками `Thread.sleep(350)` чтобы камера успела перестроиться.

#### Фаза C: Focus Stacking Pipeline
4. **Details mask generation** (`DetailsMaskGenerator.generateMask`):
    - Для каждого photo.path делается DoG + unsharp + threshold + morphology.
    - Сохраняется в cache.
5. **Duplicate filtering** (`computeSimilarity`):
    - Все маски попарно сравниваются через XOR.
    - Если маски похожи >95% (similarity ≥ 0.95, т.е. DIFF ≤ 0.05) — дубликат, отбрасывается.
    - Это нужно: если два объекта оказались в одной зоне DoF, их снимки избыточны.
6. **Alignment** (`ImageAligner.alignImages`):
    - Reference = первое фото.
    - Для остальных: ORB → match → Lowe → RANSAC → H → warpPerspective.
    - Downscale до 1600px для ускорения feature detection, координаты потом масштабируются обратно.
7. **Warp masks** (`ImageAligner.warpMask`): маски warp'аются той же гомографией (INTER_NEAREST, т.к. бинарные).
8. **Transform focus points**: координаты центров объектов переводятся в ref-space через homography.
9. **Focus map building** (`FocusMapBuilder.buildFocusMap`):
    - Blur масок (smoothing).
    - Per-pixel argmax по резкости → assignment map (CV_32SC1).
    - Voronoi-fill для unfocused areas.
    - Mode filter для сглаживания границ.
10. **Composition** (`FocusStackCompositor.compose`):
    - Direct pixel pick.
    - Seam detection (3×3 окрестность: разные assignment'ы = граница).
    - Feather blend с distance transform weights.
    - Artifact detection + inpainting.
    - Debug visualizations (seam mask, color-coded focus map).
11. **Save**: composite.jpg, focusmap.png, details_masks.png в MediaStore.
12. **Annotated photo**: последнее фото + рамки/подписи/точки фокуса.
13. UI: показывает composite (или annotated как fallback) в `capturedIV`.

---

### Ключевые алгоритмические детали

| Компонент            | Алгоритм                                             | Назначение                             |
| -------------------- | ---------------------------------------------------- | -------------------------------------- |
| YoloDetector         | ONNX + OpenCV DNN + letterbox + NMS                  | Альтернатива ML Kit                    |
| MlKitDetector        | Google ML Kit                                        | Основной детектор (быстрый, on-device) |
| DetailsMaskGenerator | DoG + unsharp + threshold + morphology               | Бинарная карта резких пикселей         |
| ImageAligner         | ORB + BF-Hamming + Lowe + RANSAC                     | Выравнивание кадров                    |
| FocusMapBuilder      | Per-pixel argmax + Voronoi fill + mode filter        | Карта "откуда брать пиксель"           |
| FocusStackCompositor | Distance-transform weighting + Navier-Stokes inpaint | Бесшовная склейка                      |

---

### Потенциальные проблемы и узкие места

1. **Производительность per-pixel loops в Kotlin** — в `FocusMapBuilder` и `FocusStackCompositor` много циклов `for y; for x; assignment.get(...)`. Это очень медленно на больших изображениях (4000×3000). Нужно использовать векторизованные операции OpenCV (setTo с маской, mixChannels, LUT).
2. **ORB может не найти фичей** на однотонных/размытых сценах → alignment fails.
3. **Focus breathing** — при перефокусе меняется FOV, одна гомография не всегда компенсирует (нужна thin-plate spline или optical flow).
4. **Motion artifacts** — если объект двигался, inpaint не спасёт.
5. **Memory pressure** — декодирование нескольких 12 Mpx фото сразу в RAM может привести к OOM.
