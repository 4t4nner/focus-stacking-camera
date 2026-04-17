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
 * Composites multiple aligned images into one all-in-focus image using the focus map.
 * Uses OpenCV seamlessClone for seam blending and Navier-Stokes inpainting for artifacts.
 */
object FocusStackCompositor {

    private const val TAG = "FocusCompositor"

    data class CompositeResult(
        val bitmap: Bitmap,
        val seamMask: Bitmap?,  // debug: shows seam boundaries
        val focusMapVis: Bitmap?  // debug: color-coded source assignment
    )

    /**
     * Compose final all-in-focus image.
     *
     * @param alignedImages list of aligned bitmaps (same coordinate space)
     * @param focusMap per-pixel source assignment
     * @param seamBlendRadius radius for Poisson/feather blending at seam boundaries
     * @param inpaintRadius radius for inpainting artifact zones
     * @return CompositeResult or null
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

            // Convert all images to Mat (RGBA)
            val srcMats = alignedImages.map { bmp ->
                val mat = Mat()
                Utils.bitmapToMat(bmp, mat)
                mat
            }

            // ========== Step 1: Direct pixel composition from focus map ==========
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

            // ========== Step 2: Find seam boundaries ==========
            val seamMask = findSeamBoundaries(focusMap.assignment, seamBlendRadius)
            val seamCount = Core.countNonZero(seamMask)
            Log.d(TAG, "Seam pixels: $seamCount")

            // ========== Step 3: Feather blend at seams ==========
            if (seamCount > 0) {
                featherBlendAtSeams(composite, srcMats, focusMap, seamMask, seamBlendRadius)
                Log.d(TAG, "Feather blend done")
            }

            // ========== Step 4: Inpainting for remaining artifacts ==========
            // Detect artifacts: pixels where aligned images differ significantly
            // (indicates motion/misalignment that homography couldn't fix)
            val artifactMask = detectArtifacts(srcMats, focusMap, composite)
            val artifactCount = Core.countNonZero(artifactMask)
            Log.d(TAG, "Artifact pixels: $artifactCount")

            if (artifactCount > 0) {
                // Use OpenCV Navier-Stokes inpainting
                val compositeBgr = Mat()
                Imgproc.cvtColor(composite, compositeBgr, Imgproc.COLOR_RGBA2BGR)

                val inpainted = Mat()
                Photo.inpaint(compositeBgr, artifactMask, inpainted, inpaintRadius.toDouble(), Photo.INPAINT_NS)

                val inpaintedRgba = Mat()
                Imgproc.cvtColor(inpainted, inpaintedRgba, Imgproc.COLOR_BGR2RGBA)

                // Copy inpainted pixels back only where artifacts were
                inpaintedRgba.copyTo(composite, artifactMask)

                compositeBgr.release()
                inpainted.release()
                inpaintedRgba.release()
                Log.d(TAG, "Inpainting done")
            }

            artifactMask.release()

            // ========== Convert results to Bitmaps ==========
            val resultBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(composite, resultBitmap)
            composite.release()

            // Debug: seam visualization
            val seamBitmap = createSeamVisualization(seamMask, w, h)
            seamMask.release()

            // Debug: focus map visualization
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
     * Save composite result to file.
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
     * Find pixels at boundaries between different source assignments.
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

        // Dilate to create blend zone
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
     * Feather-blend pixels in the seam zone using weighted average
     * based on distance from each source's focused region.
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

        // For each source, create a binary mask of its assigned region
        // then compute distance transform → weight map
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

            // Distance transform: pixels inside the region get high weight,
            // pixels near the boundary get low weight
            val dist = Mat()
            Imgproc.distanceTransform(sourceMask, dist, Imgproc.DIST_L2, 3)
            sourceMask.release()

            // Normalize distances to 0-1 range, clamp at blendRadius
            Core.min(dist, Scalar(blendRadius.toDouble()), dist)
            Core.divide(dist, Scalar(blendRadius.toDouble()), dist)

            weightMaps.add(dist)
        }

        // Normalize weights so they sum to 1.0 at each pixel
        val weightSum = Mat.zeros(h, w, CvType.CV_32FC1)
        for (wm in weightMaps) {
            Core.add(weightSum, wm, weightSum)
        }
        // Avoid division by zero
        val epsilon = Mat(h, w, CvType.CV_32FC1, Scalar(1e-6))
        Core.add(weightSum, epsilon, weightSum)
        epsilon.release()

        for (wm in weightMaps) {
            Core.divide(wm, weightSum, wm)
        }
        weightSum.release()

        // Blend only in seam zone
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
     * Detect artifact pixels where alignment failed (ghosting, double edges).
     * Compares each pixel's source with the composite — large color difference = artifact.
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

        // For each pair of adjacent sources, check for ghosting
        // at seam boundaries: if color difference between assigned source
        // and neighboring source is too large, mark as artifact
        val ARTIFACT_THRESHOLD = 60.0  // color difference threshold
        val assignData = IntArray(1)

        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                focusMap.assignment.get(y, x, assignData)
                val myIdx = assignData[0].coerceIn(0, n - 1)

                // Check if this pixel is near a boundary
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

                // Compare pixel across all sources — large variance = ghosting artifact
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

        // Dilate artifact mask slightly to cover surrounding pixels
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.dilate(artifactMask, artifactMask, kernel)
        kernel.release()

        return artifactMask
    }

    /**
     * Create debug visualization of seam boundaries (white lines on black).
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
     * Create debug color-coded visualization of focus map.
     * Each source gets a distinct color.
     */
    private fun createFocusMapVisualization(
        focusMap: FocusMapBuilder.FocusMap,
        w: Int, h: Int, n: Int
    ): Bitmap {
        // Predefined colors for up to 10 sources
        val colors = arrayOf(
            doubleArrayOf(255.0, 0.0, 0.0, 200.0),     // red
            doubleArrayOf(0.0, 255.0, 0.0, 200.0),     // green
            doubleArrayOf(0.0, 0.0, 255.0, 200.0),     // blue
            doubleArrayOf(255.0, 255.0, 0.0, 200.0),   // yellow
            doubleArrayOf(255.0, 0.0, 255.0, 200.0),   // magenta
            doubleArrayOf(0.0, 255.0, 255.0, 200.0),   // cyan
            doubleArrayOf(255.0, 128.0, 0.0, 200.0),   // orange
            doubleArrayOf(128.0, 0.0, 255.0, 200.0),   // purple
            doubleArrayOf(0.0, 255.0, 128.0, 200.0),   // spring green
            doubleArrayOf(255.0, 128.0, 128.0, 200.0)  // pink
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