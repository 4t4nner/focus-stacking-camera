package com.example.cameraapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.util.Log
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc

/**
 * Выровняет несколько фотографий относительно эталонного изображения с использованием сопоставления признаков ORB и гомографии.
 * Обрабатывает движение камеры или объекта между снимками (окно около 3 секунд).
 */
object ImageAligner {

    private const val TAG = "ImageAligner"
    private const val MAX_FEATURES = 2000
    private const val GOOD_MATCH_RATIO = 0.75f
    private const val MIN_MATCHES = 10
    // Максимальный размер для вычисления выравнивания (ускорение)
    private const val ALIGN_MAX_DIM = 1600

    data class AlignedImage(
        val bitmap: Bitmap,
        val homography: Mat?,       // null for reference image
        val alignQuality: Double,   // 0.0-1.0, 1.0 = perfect match
        val sourceIndex: Int
    )

    /**
     * Выровнять все изображения относительно эталонного (индекс 0).
     * Возвращает список выровненных изображений в тех же координатах, что и эталон.
     *
     * @param imagePaths пути к сделанным фотографиям (первая - эталон)
     * @return список AlignedImage, либо пустой если не удалось выполнить
     */
    fun alignImages(imagePaths: List<String>): List<AlignedImage> {
        if (!DetailsMaskGenerator.ensureOpenCV()) return emptyList()
        if (imagePaths.isEmpty()) return emptyList()

        try {
            val refBitmap = loadBitmap(imagePaths[0]) ?: return emptyList()
            val refMat = bitmapToGrayMat(refBitmap)

            // Вычисляем коэффициент масштабирования для ускорения детекции признаков
            val scaleFactor = computeScaleFactor(refMat)
            val refSmall = if (scaleFactor < 1.0) {
                val small = Mat()
                Imgproc.resize(refMat, small, Size(0.0, 0.0), scaleFactor, scaleFactor)
                small
            } else refMat

            // Detect features on reference
            val orb = ORB.create(MAX_FEATURES)
            val refKeypoints = MatOfKeyPoint()
            val refDescriptors = Mat()
            orb.detectAndCompute(refSmall, Mat(), refKeypoints, refDescriptors)

            Log.d(TAG, "Reference: ${refKeypoints.toList().size} keypoints")
            val results = mutableListOf<AlignedImage>()

            // Эталонное изображение — трансформация не требуется

            for (i in 1 until imagePaths.size) {
                val srcBitmap = loadBitmap(imagePaths[i])
                if (srcBitmap == null) {
                    Log.e(TAG, "Failed to load image $i: ${imagePaths[i]}")
                    continue
                }

                val srcMat = bitmapToGrayMat(srcBitmap)
                val srcSmall = if (scaleFactor < 1.0) {
                    val small = Mat()
                    Imgproc.resize(srcMat, small, Size(0.0, 0.0), scaleFactor, scaleFactor)
                    small
                } else srcMat

                // Детектируем признаки
                val srcKeypoints = MatOfKeyPoint()
                val srcDescriptors = Mat()
                orb.detectAndCompute(srcSmall, Mat(), srcKeypoints, srcDescriptors)

                Log.d(TAG, "Image $i: ${srcKeypoints.toList().size} keypoints")

                // Match features
                val homography = findHomography(
                // Сопоставление признаков
                    srcDescriptors, srcKeypoints,
                    scaleFactor
                )

                if (homography != null) {
                    // Warp source image to reference coordinate space
                    val srcColor = Mat()
                    Utils.bitmapToMat(srcBitmap, srcColor)
                    srcBitmap.recycle()
                    // Трансформируем исходное изображение в координатное пространство эталонного
                    val warped = Mat()
                    Imgproc.warpPerspective(
                        srcColor, warped, homography,
                        Size(refBitmap.width.toDouble(), refBitmap.height.toDouble()),
                        Imgproc.INTER_LINEAR,
                        Core.BORDER_REFLECT
                    )
                    srcColor.release()

                    val warpedBitmap = Bitmap.createBitmap(
                        warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888
                    )
                    Utils.matToBitmap(warped, warpedBitmap)
                    warped.release()

                    // Оценка качества выравнивания по определителю гомографии
                    val quality = estimateAlignQuality(homography)
                    Log.d(TAG, "Image $i aligned, quality=${"%.3f".format(quality)}")

                    results.add(AlignedImage(warpedBitmap, homography, quality, i))
                } else {
                    Log.w(TAG, "Image $i: alignment failed, using unaligned")
                    results.add(AlignedImage(srcBitmap, null, 0.0, i))
                }

                if (srcSmall != srcMat) srcSmall.release()
                srcMat.release()
            }

            if (refSmall != refMat) refSmall.release()
            refMat.release()
//            orb.release()
            refKeypoints.release()
            refDescriptors.release()

            return results

        } catch (e: Exception) {
            Log.e(TAG, "Alignment error", e)
            return emptyList()
        }
    }

    /**
     * Выровнять одну маску с эталоном, используя заранее вычисленную гомографию.
     */
    fun warpMask(maskPath: String, homography: Mat?, refWidth: Int, refHeight: Int): Bitmap? {
        if (!DetailsMaskGenerator.ensureOpenCV()) return null
        if (homography == null) {
            // Эталонная маска — трансформация не требуется
            return BitmapFactory.decodeFile(maskPath)
        }

        try {
            val maskBitmap = BitmapFactory.decodeFile(maskPath) ?: return null
            val maskMat = Mat()
            Utils.bitmapToMat(maskBitmap, maskMat)
            maskBitmap.recycle()

            val warped = Mat()
            Imgproc.warpPerspective(
                maskMat, warped, homography,
                Size(refWidth.toDouble(), refHeight.toDouble()),
                Imgproc.INTER_NEAREST  // метод INTER_NEAREST для бинарной маски
            )
            maskMat.release()

            val result = Bitmap.createBitmap(warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(warped, result)
            warped.release()
            return result

        } catch (e: Exception) {
            Log.e(TAG, "Warp mask error", e)
            return null
        }
    }

    private fun findHomography(
        refDesc: Mat, refKp: MatOfKeyPoint,
        srcDesc: Mat, srcKp: MatOfKeyPoint,
        scaleFactor: Double
    ): Mat? {
        if (refDesc.empty() || srcDesc.empty()) return null

        val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
        val knnMatches = mutableListOf<MatOfDMatch>()
        matcher.knnMatch(srcDesc, refDesc, knnMatches, 2)

        // Критерий Лоу (соотношение ближайших соседей)
        val goodMatches = knnMatches.filter { m ->
            val matchList = m.toList()
            matchList.size == 2 && matchList[0].distance < GOOD_MATCH_RATIO * matchList[1].distance
        }.map { it.toList()[0] }

        Log.d(TAG, "Good matches: ${goodMatches.size}/${knnMatches.size}")

        if (goodMatches.size < MIN_MATCHES) return null

        val srcKpList = srcKp.toList()
        val refKpList = refKp.toList()

        val srcPts = MatOfPoint2f()
        val refPts = MatOfPoint2f()

        val invScale = 1.0 / scaleFactor

        srcPts.fromList(goodMatches.map { m ->
            val pt = srcKpList[m.queryIdx].pt
            Point(pt.x * invScale, pt.y * invScale)
        })
        refPts.fromList(goodMatches.map { m ->
            val pt = refKpList[m.trainIdx].pt
            Point(pt.x * invScale, pt.y * invScale)
        })

        val mask = Mat()
        val H = Calib3d.findHomography(srcPts, refPts, Calib3d.RANSAC, 5.0, mask)

        val inliers = Core.countNonZero(mask)
        mask.release()
        srcPts.release()
        refPts.release()

        Log.d(TAG, "Homography inliers: $inliers/${goodMatches.size}")

        return if (!H.empty() && inliers >= MIN_MATCHES) H else null
    }

    private fun estimateAlignQuality(H: Mat): Double {
        val det = Core.determinant(H)
        // det close to 1.0 = good, far from 1.0 = bad
        val detScore = 1.0 / (1.0 + kotlin.math.abs(det - 1.0) * 5.0)

        val h20 = H.get(2, 0)?.get(0) ?: 0.0
        val h21 = H.get(2, 1)?.get(0) ?: 0.0
        val perspScore = 1.0 / (1.0 + (kotlin.math.abs(h20) + kotlin.math.abs(h21)) * 1000.0)

        return (detScore + perspScore) / 2.0
    }

    private fun computeScaleFactor(mat: Mat): Double {
        val maxDim = maxOf(mat.cols(), mat.rows())
        return if (maxDim > ALIGN_MAX_DIM) ALIGN_MAX_DIM.toDouble() / maxDim else 1.0
    }

    private fun bitmapToGrayMat(bitmap: Bitmap): Mat {
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        rgba.release()
        return gray
    }

    private fun loadBitmap(path: String): Bitmap? {
        val bitmap = BitmapFactory.decodeFile(path) ?: return null
        val exif = ExifInterface(path)
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
        )
        val rotation = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (rotation == 0f) return bitmap
        val matrix = Matrix().apply { postRotate(rotation) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated != bitmap) bitmap.recycle()
        return rotated
    }

    /**
     * Удалить матрицы гомографии из списка AlignedImage.
     */
    fun releaseAlignedImages(images: List<AlignedImage>) {
        for (img in images) {
            img.homography?.release()
        }
    }
}

