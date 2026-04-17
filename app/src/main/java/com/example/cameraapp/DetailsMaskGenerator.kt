package com.example.cameraapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream

object DetailsMaskGenerator {

    private const val TAG = "DetailsMask"

    /** Минимальная разница (0.0–1.0) между масками, чтобы считать их различными */
    const val DEFAULT_SIMILARITY_THRESHOLD = 0.05

    @Volatile
    private var opencvInitialized = false

    fun ensureOpenCV(): Boolean {
        if (opencvInitialized) return true
        opencvInitialized = OpenCVLoader.initLocal()
        if (!opencvInitialized) {
            Log.e(TAG, "OpenCV initialization failed!")
        } else {
            Log.d(TAG, "OpenCV initialized successfully")
        }
        return opencvInitialized
    }

    fun generateMask(
        imagePath: String,
        outputPath: String,
        threshold: Int = 20,
        kernelSize: Int = 3,
        sigma1: Double = 1.0,
        sigma2: Double = 2.0,
        gain: Double = 2.0,
        thickness: Double = 1.5
    ): Boolean {
        if (!ensureOpenCV()) return false

        try {
            val bitmap = loadBitmapWithOrientation(imagePath)
            if (bitmap == null) {
                Log.e(TAG, "Failed to load bitmap: $imagePath")
                return false
            }

            val src = Mat()
            Utils.bitmapToMat(bitmap, src)
            bitmap.recycle()

            val bgr = Mat()
            Imgproc.cvtColor(src, bgr, Imgproc.COLOR_RGBA2BGR)
            src.release()

            val sharpness = dogSharpnessEnhanced(bgr, sigma1, sigma2, gain, thickness)
            bgr.release()

            val binary = Mat()
            Imgproc.threshold(
                sharpness, binary,
                threshold.toDouble(), 255.0,
                Imgproc.THRESH_BINARY
            )
            sharpness.release()

            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size(kernelSize.toDouble(), kernelSize.toDouble())
            )

            val opened = Mat()
            Imgproc.morphologyEx(binary, opened, Imgproc.MORPH_OPEN, kernel)
            binary.release()

            val closed = Mat()
            Imgproc.morphologyEx(opened, closed, Imgproc.MORPH_CLOSE, kernel)
            opened.release()
            kernel.release()

            val maskBitmap = Bitmap.createBitmap(
                closed.cols(), closed.rows(), Bitmap.Config.ARGB_8888
            )
            val rgba = Mat()
            Imgproc.cvtColor(closed, rgba, Imgproc.COLOR_GRAY2RGBA)
            closed.release()

            Utils.matToBitmap(rgba, maskBitmap)
            rgba.release()

            val outputFile = File(outputPath)
            FileOutputStream(outputFile).use { fos ->
                maskBitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
            }
            maskBitmap.recycle()

            val fileSize = outputFile.length()
            Log.d(TAG, "Mask saved: $outputPath (${fileSize / 1024}KB)")
            return fileSize > 0

        } catch (e: Exception) {
            Log.e(TAG, "Error generating mask for $imagePath", e)
            return false
        }
    }

    /**
     * Вычисляет степень схожести двух масок (0.0 = полностью разные, 1.0 = идентичные).
     * Использует отношение совпадающих пикселей к общему числу пикселей.
     */
    fun computeSimilarity(maskPath1: String, maskPath2: String): Double {
        if (!ensureOpenCV()) return -1.0

        try {
            val bmp1 = BitmapFactory.decodeFile(maskPath1) ?: return -1.0
            val bmp2 = BitmapFactory.decodeFile(maskPath2) ?: run { bmp1.recycle(); return -1.0 }

            val mat1 = Mat()
            val mat2 = Mat()
            Utils.bitmapToMat(bmp1, mat1)
            Utils.bitmapToMat(bmp2, mat2)
            bmp1.recycle()
            bmp2.recycle()

            // Конвертируем в grayscale
            val gray1 = Mat()
            val gray2 = Mat()
            Imgproc.cvtColor(mat1, gray1, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.cvtColor(mat2, gray2, Imgproc.COLOR_RGBA2GRAY)
            mat1.release()
            mat2.release()

            // Если размеры разные — ресайзим вторую к первой
            if (gray1.size() != gray2.size()) {
                Imgproc.resize(gray2, gray2, gray1.size())
            }

            // Бинаризуем (на случай если PNG сохранился с антиалиасингом)
            val bin1 = Mat()
            val bin2 = Mat()
            Imgproc.threshold(gray1, bin1, 127.0, 255.0, Imgproc.THRESH_BINARY)
            Imgproc.threshold(gray2, bin2, 127.0, 255.0, Imgproc.THRESH_BINARY)
            gray1.release()
            gray2.release()

            // XOR — различающиеся пиксели
            val xorMat = Mat()
            Core.bitwise_xor(bin1, bin2, xorMat)
            bin1.release()
            bin2.release()

            val totalPixels = xorMat.rows().toLong() * xorMat.cols().toLong()
            val differentPixels = Core.countNonZero(xorMat).toLong()
            xorMat.release()

            val similarity = 1.0 - (differentPixels.toDouble() / totalPixels.toDouble())
            return similarity

        } catch (e: Exception) {
            Log.e(TAG, "Error computing similarity", e)
            return -1.0
        }
    }

    private fun dogSharpnessEnhanced(
        bgr: Mat,
        sigma1: Double,
        sigma2: Double,
        gain: Double,
        thickness: Double
    ): Mat {
        val gray = Mat()
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)

        val grayF = Mat()
        gray.convertTo(grayF, CvType.CV_64F)
        gray.release()

        val g1 = Mat()
        Imgproc.GaussianBlur(grayF, g1, Size(0.0, 0.0), sigma1)

        val g2 = Mat()
        Imgproc.GaussianBlur(grayF, g2, Size(0.0, 0.0), sigma2)
        grayF.release()

        val diff = Mat()
        Core.subtract(g1, g2, diff)
        g1.release()
        g2.release()

        val dog = Mat()
        Core.absdiff(diff, Mat.zeros(diff.size(), diff.type()), dog)
        diff.release()

        Core.multiply(dog, org.opencv.core.Scalar(gain), dog)

        if (thickness > 1.0) {
            Imgproc.GaussianBlur(dog, dog, Size(0.0, 0.0), thickness - 1.0)
        }

        val blurred = Mat()
        Imgproc.GaussianBlur(dog, blurred, Size(0.0, 0.0), 1.0)

        val unsharpDiff = Mat()
        Core.subtract(dog, blurred, unsharpDiff)
        blurred.release()

        val scaled = Mat()
        Core.multiply(unsharpDiff, org.opencv.core.Scalar(0.7), scaled)
        unsharpDiff.release()

        Core.add(dog, scaled, dog)
        scaled.release()

        val minMax = Core.minMaxLoc(dog)
        val maxVal = minMax.maxVal
        if (maxVal > 0) {
            Core.multiply(dog, org.opencv.core.Scalar(255.0 / maxVal), dog)
        }

        val result = Mat()
        dog.convertTo(result, CvType.CV_8U)
        dog.release()

        return result
    }

    private fun loadBitmapWithOrientation(path: String): Bitmap? {
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
}