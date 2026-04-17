package com.example.cameraapp

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Builds a per-pixel assignment map: which source image provides the best-focused
 * pixel at each location.
 *
 * For each pixel, picks the source with the highest sharpness (from details_mask).
 * Ties broken by proximity to the focus point of that image.
 * Regions not sharp in any image get assigned to the nearest focused region.
 */
object FocusMapBuilder {

    private const val TAG = "FocusMapBuilder"

    /**
     * Per-pixel source assignment.
     * assignmentMap[y][x] = index into alignedMasks list (which source to use)
     */
    data class FocusMap(
        val assignment: Mat,  // CV_32SC1, values = source index
        val width: Int,
        val height: Int,
        val sourceCount: Int
    ) {
        fun release() {
            assignment.release()
        }
    }

    /**
     * Build focus map from aligned detail masks.
     *
     * @param alignedMasks list of aligned binary masks (Bitmap, ARGB_8888), one per source image
     * @param focusPoints list of (cx, cy) focus points in reference image coordinates, one per source
     * @param blurRadius Gaussian blur radius for smooth transitions between zones
     * @return FocusMap or null on failure
     */
    fun buildFocusMap(
        alignedMasks: List<Bitmap>,
        focusPoints: List<Pair<Float, Float>>,
        blurRadius: Int = 31
    ): FocusMap? {
        if (!DetailsMaskGenerator.ensureOpenCV()) return null
        if (alignedMasks.isEmpty()) return null

        try {
            val w = alignedMasks[0].width
            val h = alignedMasks[0].height
            val n = alignedMasks.size

            Log.d(TAG, "Building focus map: ${w}x${h}, $n sources")

            // Convert masks to grayscale float Mats
            val maskMats = alignedMasks.map { bmp ->
                val rgba = Mat()
                Utils.bitmapToMat(bmp, rgba)
                val gray = Mat()
                Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
                rgba.release()
                val floatMat = Mat()
                gray.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)
                gray.release()
                // Smooth to avoid pixel-level noise in assignment
                if (blurRadius > 1) {
                    val ksize = if (blurRadius % 2 == 0) blurRadius + 1 else blurRadius
                    Imgproc.GaussianBlur(floatMat, floatMat, Size(ksize.toDouble(), ksize.toDouble()), 0.0)
                }
                floatMat
            }

            // Assignment map: for each pixel, pick source with max sharpness
            val assignment = Mat(h, w, CvType.CV_32SC1, Scalar(0.0))
            val maxSharpness = Mat(h, w, CvType.CV_32FC1, Scalar(0.0))

            for (i in 0 until n) {
                // Compare current mask with running max
                val mask = maskMats[i]
                // Where this mask is sharper than current max
                val comparison = Mat()
                Core.compare(mask, maxSharpness, comparison, Core.CMP_GT)

                // Update assignment where this source wins
                assignment.setTo(Scalar(i.toDouble()), comparison)
                // Update max sharpness
                mask.copyTo(maxSharpness, comparison)

                comparison.release()
            }

            // For pixels where ALL masks have 0 sharpness → assign to nearest focus point
            val zeroMask = Mat()
            Core.compare(maxSharpness, Scalar(0.001), zeroMask, Core.CMP_LT)

            val zeroCount = Core.countNonZero(zeroMask)
            Log.d(TAG, "Unfocused pixels: $zeroCount / ${w * h} " +
                    "(${(zeroCount * 100.0 / (w * h)).toInt()}%)")

            if (zeroCount > 0 && focusPoints.isNotEmpty()) {
                assignByNearestFocusPoint(assignment, zeroMask, focusPoints, w, h)
            }

            zeroMask.release()
            maxSharpness.release()
            maskMats.forEach { it.release() }

            // Smooth assignment boundaries with mode filter (median-like)
            smoothAssignment(assignment, 5)

            Log.d(TAG, "Focus map built successfully")
            return FocusMap(assignment, w, h, n)

        } catch (e: Exception) {
            Log.e(TAG, "Error building focus map", e)
            return null
        }
    }

    /**
     * For unfocused pixels (zeroMask), assign to the source whose focus point is nearest.
     */
    private fun assignByNearestFocusPoint(
        assignment: Mat,
        zeroMask: Mat,
        focusPoints: List<Pair<Float, Float>>,
        w: Int, h: Int
    ) {
        // Create distance maps for each focus point, pick the nearest
        // For efficiency, use Voronoi-like approach: create seed image and distanceTransform

        val seeds = Mat.zeros(h, w, CvType.CV_8UC1)

        // Draw small circles at each focus point
        for ((i, fp) in focusPoints.withIndex()) {
            val cx = fp.first.toInt().coerceIn(0, w - 1)
            val cy = fp.second.toInt().coerceIn(0, h - 1)
            Imgproc.circle(seeds, Point(cx.toDouble(), cy.toDouble()), 3, Scalar((i + 1).toDouble()), -1)
        }

        // Dilate seeds iteratively to fill (simple Voronoi)
        // For large images this is fast enough with a few hundred iterations
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(7.0, 7.0))
        val maxIter = (maxOf(w, h) / 3) // enough to fill entire image
        var prev = seeds.clone()

        for (iter in 0 until maxIter) {
            val dilated = Mat()
            Imgproc.dilate(prev, dilated, kernel)
            // Only fill where still zero
            val stillZero = Mat()
            Core.compare(prev, Scalar(0.0), stillZero, Core.CMP_EQ)
            dilated.copyTo(prev, stillZero)
            stillZero.release()

            val remaining = Core.countNonZero(Mat().also {
                Core.compare(prev, Scalar(0.0), it, Core.CMP_EQ)
            })
            if (remaining == 0) {
                dilated.release()
                break
            }
            dilated.release()
        }

        // Now prev has values 1..N for each pixel (nearest focus point index+1)
        // Apply only to zero-mask pixels in assignment
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (zeroMask.get(y, x)[0] > 0) {
                    val voronoiVal = prev.get(y, x)[0].toInt()
                    if (voronoiVal > 0) {
                        assignment.put(y, x, (voronoiVal - 1).toDouble())
                    }
                }
            }
        }

        kernel.release()
        seeds.release()
        prev.release()
    }

    /**
     * Smooth assignment boundaries using local mode filter to reduce jagged edges.
     */
    private fun smoothAssignment(assignment: Mat, kernelSize: Int) {
        if (kernelSize <= 1) return

        val h = assignment.rows()
        val w = assignment.cols()
        val half = kernelSize / 2
        val result = assignment.clone()

        // Simple mode filter — for each pixel, pick most frequent value in neighborhood
        // Only process boundary pixels for speed
        for (y in half until h - half) {
            for (x in half until w - half) {
                val center = assignment.get(y, x)[0].toInt()

                // Quick check: if all neighbors same, skip
                var allSame = true
                outer@ for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (assignment.get(y + dy, x + dx)[0].toInt() != center) {
                            allSame = false
                            break@outer
                        }
                    }
                }
                if (allSame) continue

                // Count frequencies in kernel
                val counts = mutableMapOf<Int, Int>()
                for (dy in -half..half) {
                    for (dx in -half..half) {
                        val v = assignment.get(y + dy, x + dx)[0].toInt()
                        counts[v] = (counts[v] ?: 0) + 1
                    }
                }
                val mode = counts.maxByOrNull { it.value }?.key ?: center
                result.put(y, x, mode.toDouble())
            }
        }

        result.copyTo(assignment)
        result.release()
    }
}