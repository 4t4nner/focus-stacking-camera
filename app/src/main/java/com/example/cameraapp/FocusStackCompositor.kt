package com.example.cameraapp

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import java.io.File
import java.io.FileOutputStream

/**
 * Объединяет несколько выровненных изображений в одно полностью сфокусированное изображение с использованием карты фокуса.
 * Использует OpenCV seamlessClone для смешивания швов и инпейнтинг Навье-Стокса для устранения артефактов.
 */
object FocusStackCompositor {

    private const val TAG = "FocusCompositor"

    data class CompositeResult(
        val bitmap: Bitmap,
        val seamMask: Bitmap?,  // отладка: показывает границы швов
        val focusMapVis: Bitmap?  // отладка: цветовое обозначение назначения источников
    )

    /**
     * Формирует итоговое полностью сфокусированное изображение.
     *
     * @param alignedImages список выровненных bitmap-изображений (в одном координатном пространстве)
     * @param focusMap попиксельное назначение источника
     * @param seamBlendRadius радиус для пуассоновского/мягкого смешивания на границах швов
     * @param inpaintRadius радиус для инпейнтинга зон с артефактами
     * @return CompositeResult или null
     */
    fun compose(
        alignedImages: List<Bitmap>,
        focusMap: FocusMapBuilder.FocusMap,
        seamBlendRadius: Int = 15,
        inpaintRadius: Int = 5
    ): CompositeResult? {
        if (!DetailsMaskGenerator.ensureOpenCV()) return null
        if (alignedImages.isEmpty()) return null

        try {
            val w = focusMap.width
            val h = focusMap.height
            val n = alignedImages.size

            Log.d(TAG, "Compositing ${w}x${h} from $n sources")

            // Преобразуем все изображения в Mat (RGBA)
            val srcMats = alignedImages.map { bmp ->
                val mat = Mat()
                Utils.bitmapToMat(bmp, mat)
                mat
            }

            // ========== Шаг 1: Прямая попиксельная сборка по карте фокуса ==========
            val composite = Mat(h, w, CvType.CV_8UC4)
            val assignData = IntArray(1)

            for (y in 0 until h) {
                for (x in 0 until w) {
                    focusMap.assignment.get(y, x, assignData)
                    val srcIdx = assignData[0].coerceIn(0, n - 1)
                    val pixel = srcMats[srcIdx].get(y, x)
                    composite.put(y, x, *pixel)
                }
            }

            Log.d(TAG, "Direct composition done")

            // ========== Шаг 2: Поиск границ швов ==========
            val seamMask = findSeamBoundaries(focusMap.assignment, seamBlendRadius)
            val seamCount = Core.countNonZero(seamMask)
            Log.d(TAG, "Seam pixels: $seamCount")

            // ========== Шаг 3: Мягкое смешивание на швах ==========
            if (seamCount > 0) {
                featherBlendAtSeams(composite, srcMats, focusMap, seamMask, seamBlendRadius)
                Log.d(TAG, "Feather blend done")
            }

            // ========== Шаг 4: Инпейнтинг оставшихся артефактов ==========
            // Обнаружение артефактов: пиксели, где выровненные изображения значительно различаются
            // (указывает на движение/смещение, которое гомография не смогла исправить)
            val artifactMask = detectArtifacts(srcMats, focusMap, composite)
            val artifactCount = Core.countNonZero(artifactMask)
            Log.d(TAG, "Artifact pixels: $artifactCount")

            if (artifactCount > 0) {
                // Используем инпейнтинг Навье-Стокса из OpenCV
                val compositeBgr = Mat()
                Imgproc.cvtColor(composite, compositeBgr, Imgproc.COLOR_RGBA2BGR)

                val inpainted = Mat()
                Photo.inpaint(compositeBgr, artifactMask, inpainted, inpaintRadius.toDouble(), Photo.INPAINT_NS)

                val inpaintedRgba = Mat()
                Imgproc.cvtColor(inpainted, inpaintedRgba, Imgproc.COLOR_BGR2RGBA)

                // Копируем восстановленные пиксели обратно только там, где были артефакты
                inpaintedRgba.copyTo(composite, artifactMask)

                compositeBgr.release()
                inpainted.release()
                inpaintedRgba.release()
                Log.d(TAG, "Inpainting done")
            }

            artifactMask.release()

            // ========== Преобразуем результаты в Bitmap ==========
            val resultBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(composite, resultBitmap)
            composite.release()

            // Отладка: визуализация швов
            val seamBitmap = createSeamVisualization(seamMask, w, h)
            seamMask.release()

            // Отладка: визуализация карты фокуса
            val focusVis = createFocusMapVisualization(focusMap, w, h, n)

            srcMats.forEach { it.release() }

            Log.d(TAG, "Composition complete")
            return CompositeResult(resultBitmap, seamBitmap, focusVis)

        } catch (e: Exception) {
            Log.e(TAG, "Composition error", e)
            return null
        }
    }

    /**
     * Сохраняет итоговый результат в файл.
     */
    fun saveResult(bitmap: Bitmap, outputPath: String, quality: Int = 95): Boolean {
        return try {
            FileOutputStream(File(outputPath)).use { fos ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, quality, fos)
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error saving result", e)
            false
        }
    }

    /**
     * Находит пиксели на границах между разными назначениями источников.
     */
    private fun findSeamBoundaries(assignment: Mat, radius: Int): Mat {
        val h = assignment.rows()
        val w = assignment.cols()
        val seamMask = Mat.zeros(h, w, CvType.CV_8UC1)

        val data = IntArray(1)
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                assignment.get(y, x, data)
                val center = data[0]

                var isBoundary = false
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (dy == 0 && dx == 0) continue
                        assignment.get(y + dy, x + dx, data)
                        if (data[0] != center) {
                            isBoundary = true
                            break
                        }
                    }
                    if (isBoundary) break
                }

                if (isBoundary) {
                    seamMask.put(y, x, 255.0)
                }
            }
        }

        // Расширяем (dilate), чтобы создать зону смешивания
        if (radius > 1) {
            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size(radius * 2.0 + 1, radius * 2.0 + 1)
            )
            Imgproc.dilate(seamMask, seamMask, kernel)
            kernel.release()
        }

        return seamMask
    }

    /**
     * Выполняет мягкое смешивание пикселей в зоне шва с использованием взвешенного среднего
     * на основе расстояния от сфокусированной области каждого источника.
     */
    private fun featherBlendAtSeams(
        composite: Mat,
        srcMats: List<Mat>,
        focusMap: FocusMapBuilder.FocusMap,
        seamMask: Mat,
        blendRadius: Int
    ) {
        val h = composite.rows()
        val w = composite.cols()
        val n = srcMats.size

        // Для каждого источника создаём бинарную маску его назначенной области,
        // затем вычисляем преобразование расстояний → карту весов
        val weightMaps = mutableListOf<Mat>()

        for (i in 0 until n) {
            val sourceMask = Mat.zeros(h, w, CvType.CV_8UC1)
            val assignData = IntArray(1)
            for (y in 0 until h) {
                for (x in 0 until w) {
                    focusMap.assignment.get(y, x, assignData)
                    if (assignData[0] == i) {
                        sourceMask.put(y, x, 255.0)
                    }
                }
            }

            // Преобразование расстояний: пиксели внутри области получают высокий вес,
            // пиксели у границы получают низкий вес
            val dist = Mat()
            Imgproc.distanceTransform(sourceMask, dist, Imgproc.DIST_L2, 3)
            sourceMask.release()

            // Нормализуем расстояния в диапазон 0-1, ограничиваем по blendRadius
            Core.min(dist, Scalar(blendRadius.toDouble()), dist)
            Core.divide(dist, Scalar(blendRadius.toDouble()), dist)

            weightMaps.add(dist)
        }

        // Нормализуем веса так, чтобы их сумма в каждом пикселе равнялась 1.0
        val weightSum = Mat.zeros(h, w, CvType.CV_32FC1)
        for (wm in weightMaps) {
            Core.add(weightSum, wm, weightSum)
        }
        // Избегаем деления на ноль
        val epsilon = Mat(h, w, CvType.CV_32FC1, Scalar(1e-6))
        Core.add(weightSum, epsilon, weightSum)
        epsilon.release()

        for (wm in weightMaps) {
            Core.divide(wm, weightSum, wm)
        }
        weightSum.release()

        // Смешиваем только в зоне шва
        val pixel = DoubleArray(4)
        val blended = DoubleArray(4)
        val assignData = IntArray(1)

        for (y in 0 until h) {
            for (x in 0 until w) {
                if (seamMask.get(y, x)[0] < 128.0) continue

                blended[0] = 0.0; blended[1] = 0.0; blended[2] = 0.0; blended[3] = 0.0

                for (i in 0 until n) {
                    val weight = weightMaps[i].get(y, x)[0]
                    if (weight < 1e-6) continue

                    val srcPixel = srcMats[i].get(y, x)
                    blended[0] += srcPixel[0] * weight
                    blended[1] += srcPixel[1] * weight
                    blended[2] += srcPixel[2] * weight
                    blended[3] += srcPixel[3] * weight
                }

                composite.put(y, x,
                    blended[0].coerceIn(0.0, 255.0),
                    blended[1].coerceIn(0.0, 255.0),
                    blended[2].coerceIn(0.0, 255.0),
                    blended[3].coerceIn(0.0, 255.0)
                )
            }
        }

        weightMaps.forEach { it.release() }
        Log.d(TAG, "Feather blend completed for $n sources")
    }

    /**
     * Обнаруживает пиксели-артефакты, где выравнивание не удалось (двоение, двойные края).
     * Сравнивает источник каждого пикселя с композитом — большая разница в цвете = артефакт.
     */
    private fun detectArtifacts(
        srcMats: List<Mat>,
        focusMap: FocusMapBuilder.FocusMap,
        composite: Mat
    ): Mat {
        val h = composite.rows()
        val w = composite.cols()
        val n = srcMats.size
        val artifactMask = Mat.zeros(h, w, CvType.CV_8UC1)

        // Для каждой пары соседних источников проверяем на двоение
        // на границах швов: если разница в цвете между назначенным источником
        // и соседним источником слишком велика — помечаем как артефакт
        val ARTIFACT_THRESHOLD = 60.0  // порог разницы цвета
        val assignData = IntArray(1)

        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                focusMap.assignment.get(y, x, assignData)
                val myIdx = assignData[0].coerceIn(0, n - 1)

                // Проверяем, находится ли этот пиксель рядом с границей
                var nearBoundary = false
                for (dy in -2..2) {
                    for (dx in -2..2) {
                        val ny = y + dy
                        val nx = x + dx
                        if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue
                        focusMap.assignment.get(ny, nx, assignData)
                        if (assignData[0] != myIdx) {
                            nearBoundary = true
                            break
                        }
                    }
                    if (nearBoundary) break
                }

                if (!nearBoundary) continue

                // Сравниваем пиксель по всем источникам — большой разброс = артефакт двоения
                var maxDiff = 0.0
                val refPixel = srcMats[myIdx].get(y, x)

                for (i in 0 until n) {
                    if (i == myIdx) continue
                    val otherPixel = srcMats[i].get(y, x)
                    val diff = kotlin.math.sqrt(
                        (refPixel[0] - otherPixel[0]).let { it * it } +
                                (refPixel[1] - otherPixel[1]).let { it * it } +
                                (refPixel[2] - otherPixel[2]).let { it * it }
                    )
                    if (diff > maxDiff) maxDiff = diff
                }

                if (maxDiff > ARTIFACT_THRESHOLD) {
                    artifactMask.put(y, x, 255.0)
                }
            }
        }

        // Немного расширяем маску артефактов, чтобы захватить соседние пиксели
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.dilate(artifactMask, artifactMask, kernel)
        kernel.release()

        return artifactMask
    }

    /**
     * Создаёт отладочную визуализацию границ швов (белые линии на чёрном фоне).
     */
    private fun createSeamVisualization(seamMask: Mat, w: Int, h: Int): Bitmap {
        val vis = Mat()
        Imgproc.cvtColor(seamMask, vis, Imgproc.COLOR_GRAY2RGBA)
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(vis, bitmap)
        vis.release()
        return bitmap
    }

    /**
     * Создаёт отладочную цветовую визуализацию карты фокуса.
     * Каждый источник получает отдельный цвет.
     */
    private fun createFocusMapVisualization(
        focusMap: FocusMapBuilder.FocusMap,
        w: Int, h: Int, n: Int
    ): Bitmap {
        // Предопределённые цвета для не более чем 10 источников
        val colors = arrayOf(
            doubleArrayOf(255.0, 0.0, 0.0, 200.0),     // красный
            doubleArrayOf(0.0, 255.0, 0.0, 200.0),     // зелёный
            doubleArrayOf(0.0, 0.0, 255.0, 200.0),     // синий
            doubleArrayOf(255.0, 255.0, 0.0, 200.0),   // жёлтый
            doubleArrayOf(255.0, 0.0, 255.0, 200.0),   // пурпурный
            doubleArrayOf(0.0, 255.0, 255.0, 200.0),   // голубой
            doubleArrayOf(255.0, 128.0, 0.0, 200.0),   // оранжевый
            doubleArrayOf(128.0, 0.0, 255.0, 200.0),   // фиолетовый
            doubleArrayOf(0.0, 255.0, 128.0, 200.0),   // весенне-зелёный
            doubleArrayOf(255.0, 128.0, 128.0, 200.0)  // розовый
        )

        val vis = Mat(h, w, CvType.CV_8UC4)
        val assignData = IntArray(1)

        for (y in 0 until h) {
            for (x in 0 until w) {
                focusMap.assignment.get(y, x, assignData)
                val idx = assignData[0].coerceIn(0, colors.size - 1)
                vis.put(y, x, *colors[idx])
            }
        }

        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(vis, bitmap)
        vis.release()
        return bitmap
    }
}