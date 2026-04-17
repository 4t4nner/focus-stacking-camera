package com.example.cameraapp

import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Typeface
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.doOnLayout
import com.example.cameraapp.databinding.ActivityMainBinding
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    private val mainBinding: ActivityMainBinding by lazy {
        ActivityMainBinding.inflate(layoutInflater)
    }

    private val multiplePermissionId = 14
    private val multiplePermissionNameList = if (Build.VERSION.SDK_INT >= 33) {
        arrayListOf(android.Manifest.permission.CAMERA)
    } else {
        arrayListOf(
            android.Manifest.permission.CAMERA,
            android.Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
    }

    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraProvider: ProcessCameraProvider
    private lateinit var camera: Camera
    private lateinit var cameraSelector: CameraSelector
    private var lensFacing = CameraSelector.LENS_FACING_BACK

    private lateinit var mlKitDetector: MlKitDetector
    private lateinit var analysisExecutor: ExecutorService

    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    private var currentDetections: List<Detection> = emptyList()
    @Volatile
    private var analysisImageWidth: Int = 1
    @Volatile
    private var analysisImageHeight: Int = 1

    @Volatile
    private var detectionPaused: Boolean = false

    // FPS
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()

    private val capturedPhotos = mutableListOf<CapturedPhoto>()

    @Volatile
    private var isCapturing = false

    private val APP_FOLDER = "CameraAppDetections"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(mainBinding.root)

        setupFullScreen()

        mlKitDetector = MlKitDetector()
        analysisExecutor = Executors.newSingleThreadExecutor()

        // Pre-init OpenCV in background so it's ready when needed
        Thread { DetailsMaskGenerator.ensureOpenCV() }.start()

        mainBinding.previewView.doOnLayout {
            if (checkMultiplePermission()) {
                startCamera()
            }
        }

        mainBinding.captureIB.setOnClickListener {
            if (!isCapturing) {
                startFocusAndCapture()
            }
        }

        mainBinding.closeIB.setOnClickListener {
            hideCapturedImage()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        mlKitDetector.close()
    }

    // ======================== FULL SCREEN ========================

    private fun setupFullScreen() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val windowInsetsController = WindowCompat.getInsetsController(window, window.decorView)
        windowInsetsController?.hide(WindowInsetsCompat.Type.systemBars())
        windowInsetsController?.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    }

    // ======================== PERMISSIONS ========================

    private fun checkMultiplePermission(): Boolean {
        val listPermissionNeeded = arrayListOf<String>()
        for (permission in multiplePermissionNameList) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                listPermissionNeeded.add(permission)
            }
        }
        if (listPermissionNeeded.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, listPermissionNeeded.toTypedArray(), multiplePermissionId)
            return false
        }
        return true
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == multiplePermissionId && grantResults.isNotEmpty()) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                startCamera()
            } else {
                appSettingOpen(this)
            }
        }
    }

    // ======================== CAMERA ========================

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e("CameraX", "Error", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val screenAspectRatio = aspectRatio(mainBinding.previewView.width, mainBinding.previewView.height)
        val rotation = mainBinding.previewView.display.rotation
        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(
                AspectRatioStrategy(screenAspectRatio, AspectRatioStrategy.FALLBACK_RULE_AUTO)
            ).build()

        val preview = Preview.Builder()
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(rotation)
            .build()
            .also { it.setSurfaceProvider(mainBinding.previewView.surfaceProvider) }

        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(rotation)
            .build()

        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy ->
            processFrame(imageProxy)
        }

        cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

        try {
            cameraProvider.unbindAll()
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageCapture, imageAnalysis
            )
        } catch (e: Exception) {
            Log.e("CameraX", "Error binding", e)
        }
    }

    private fun aspectRatio(width: Int, height: Int): Int {
        if (width == 0 || height == 0) return AspectRatio.RATIO_16_9
        val previewRatio = maxOf(width, height).toDouble() / minOf(width, height)
        return if (abs(previewRatio - 4.0 / 3.0) <= abs(previewRatio - 16.0 / 9.0))
            AspectRatio.RATIO_4_3 else AspectRatio.RATIO_16_9
    }

    // ======================== LIVE DETECTION ========================

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            if (detectionPaused) return

            val bitmap = imageProxyToBitmap(imageProxy) ?: return
            val w = bitmap.width
            val h = bitmap.height

            val detections = mlKitDetector.detectStream(bitmap)
            bitmap.recycle()

            currentDetections = detections
            analysisImageWidth = w
            analysisImageHeight = h

            frameCount++
            val now = System.currentTimeMillis()
            val elapsed = now - lastFpsTime

            runOnUiThread {
                mainBinding.overlayView.setDetections(detections, w, h)

                if (elapsed >= 1000) {
                    val fps = frameCount * 1000f / elapsed
                    mainBinding.fpsTV.text = "FPS: %.1f".format(fps)
                    frameCount = 0
                    lastFpsTime = now
                }
            }
        } catch (e: Exception) {
            Log.e("Analysis", "Error", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val planes = imageProxy.planes
        if (planes.isEmpty()) return null

        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * imageProxy.width

        val bitmapWidth = imageProxy.width + rowPadding / pixelStride
        val bitmap = Bitmap.createBitmap(bitmapWidth, imageProxy.height, Bitmap.Config.ARGB_8888)
        buffer.rewind()
        bitmap.copyPixelsFromBuffer(buffer)

        val cropped = if (bitmapWidth != imageProxy.width) {
            Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height).also {
                if (it != bitmap) bitmap.recycle()
            }
        } else bitmap

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        if (rotationDegrees == 0) return cropped

        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)
        if (rotated != cropped) cropped.recycle()
        return rotated
    }

    // ======================== FOCUS (MAIN THREAD SAFE) ========================

    private fun focusOnPointAndWait(viewX: Float, viewY: Float, timeoutMs: Long = 3000): Boolean {
        val resultLatch = CountDownLatch(1)
        var focusSuccess = false

        mainHandler.post {
            try {
                val factory = mainBinding.previewView.meteringPointFactory
                val point = factory.createPoint(viewX, viewY)
                val action = FocusMeteringAction.Builder(
                    point,
                    FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
                )
                    .setAutoCancelDuration(5, TimeUnit.SECONDS)
                    .build()

                val future = camera.cameraControl.startFocusAndMetering(action)
                future.addListener({
                    try {
                        val result = future.get()
                        focusSuccess = result.isFocusSuccessful
                        Log.d("Focus", "Focus result: success=$focusSuccess")
                    } catch (e: Exception) {
                        Log.e("Focus", "Focus future error", e)
                        focusSuccess = false
                    } finally {
                        resultLatch.countDown()
                    }
                }, ContextCompat.getMainExecutor(this))
            } catch (e: Exception) {
                Log.e("Focus", "Error setting up focus", e)
                resultLatch.countDown()
            }
        }

        val completed = resultLatch.await(timeoutMs, TimeUnit.MILLISECONDS)
        if (!completed) {
            Log.w("Focus", "Focus timed out after ${timeoutMs}ms")
        }
        return focusSuccess
    }

    // ======================== CAPTURE PIPELINE ========================

    private class PendingCapture(
        val index: Int,
        val detection: Detection,
        val objCx: Float,
        val objCy: Float,
        val latch: CountDownLatch = CountDownLatch(1),
        @Volatile var path: String? = null,
        @Volatile var error: String? = null
    )

    private fun capturePhotoAsync(
        fileName: String,
        shutterLatch: CountDownLatch,
        pending: PendingCapture
    ) {
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(
                    MediaStore.MediaColumns.RELATIVE_PATH,
                    "${Environment.DIRECTORY_PICTURES}/$APP_FOLDER"
                )
            }
        }

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            val dir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                APP_FOLDER
            )
            if (!dir.exists()) dir.mkdirs()
        }

        val outputOptions = ImageCapture.OutputFileOptions.Builder(
            contentResolver,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        ).build()

        mainHandler.post {
            imageCapture.takePicture(
                outputOptions,
                ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                        val uri = output.savedUri
                        if (uri != null) {
                            pending.path = getPathFromUri(uri) ?: uri.toString()
                        }
                        Log.d("Capture", "Save complete #${pending.index + 1}: ${pending.path}")
                        pending.latch.countDown()
                    }

                    override fun onError(exception: ImageCaptureException) {
                        Log.e("Capture", "Error #${pending.index + 1}: ${exception.message}", exception)
                        pending.error = exception.message
                        pending.latch.countDown()
                    }
                }
            )
        }
    }

    private fun startFocusAndCapture() {
        val detections = currentDetections.toList()
        if (detections.isEmpty()) {
            Toast.makeText(this, "No objects detected", Toast.LENGTH_SHORT).show()
            return
        }

        val startTime = System.currentTimeMillis()
        fun logTime(msg: String) {
            Log.d("CaptureTimer", "$msg: +${System.currentTimeMillis() - startTime}ms")
        }

        isCapturing = true
        detectionPaused = true
        logTime("Button pressed, detection paused")

        mainBinding.overlayView.setCapturingMode(true)
        mainBinding.captureIB.visibility = View.GONE
        mainBinding.focusInfoTV.text = "Capturing ${detections.size} object(s)..."
        mainBinding.focusInfoTV.visibility = View.VISIBLE

        Thread {
            val imgW = analysisImageWidth
            val imgH = analysisImageHeight
            val previewW = mainBinding.previewView.width.toFloat()
            val previewH = mainBinding.previewView.height.toFloat()

            val scaleX = previewW / imgW
            val scaleY = previewH / imgH
            val scale = maxOf(scaleX, scaleY)
            val offsetX = (previewW - imgW * scale) / 2f
            val offsetY = (previewH - imgH * scale) / 2f

            val sessionTimestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
                .format(System.currentTimeMillis())

            val pendingCaptures = mutableListOf<PendingCapture>()

            logTime("Starting capture loop")

            for ((index, det) in detections.withIndex()) {
                runOnUiThread {
                    mainBinding.focusInfoTV.text =
                        "Focusing on ${det.className} (${index + 1}/${detections.size})..."
                }

                val objCx = (det.x1 + det.x2) / 2f
                val objCy = (det.y1 + det.y2) / 2f

                val viewX = objCx * scale + offsetX
                val viewY = objCy * scale + offsetY

                logTime("Focus start obj #${index + 1} (${det.className}) viewXY=(${viewX.toInt()},${viewY.toInt()})")

                val focusOk = focusOnPointAndWait(viewX, viewY, 3000)
                logTime("Focus done obj #${index + 1}, success=$focusOk")

                val cx = objCx.toInt()
                val cy = objCy.toInt()
                val fileName = "DET_${sessionTimestamp}_${index + 1}_${cx}x${cy}.jpg"

                val shutterLatch = CountDownLatch(1)
                val pending = PendingCapture(index, det, objCx, objCy)
                pendingCaptures.add(pending)
                capturePhotoAsync(fileName, shutterLatch, pending)

                if (index < detections.size - 1) {
                    Thread.sleep(350)
                    logTime("Shutter done obj #${index + 1}, starting next focus")
                } else {
                    pending.latch.await(10, TimeUnit.SECONDS)
                    logTime("Captured obj #${index + 1} (last, fully saved)")
                }
            }

            logTime("Waiting for all saves to complete...")
            for (pending in pendingCaptures) {
                pending.latch.await(10, TimeUnit.SECONDS)
            }
            logTime("All saves complete")

            // Collect results
            val results = mutableListOf<CapturedPhoto>()
            for (pending in pendingCaptures) {
                val path = pending.path
                if (path != null) {
                    results.add(CapturedPhoto(path, pending.detection.className, pending.objCx, pending.objCy))
                    Log.d("Capture", "Saved: $path for ${pending.detection.className}")
                } else {
                    Log.e("Capture", "Failed obj #${pending.index + 1}: ${pending.error}")
                }
            }

            // =====================================================
            // FOCUS STACKING PIPELINE
            // =====================================================
            logTime("Starting focus stacking pipeline...")
            runOnUiThread {
                mainBinding.focusInfoTV.text = "Generating detail masks..."
            }

            // Step 1: Generate details masks (temp files)
            data class TempMask(val index: Int, val tempPath: String, val photo: CapturedPhoto)

            val tempMasks = mutableListOf<TempMask>()

            for ((i, photo) in results.withIndex()) {
                val tempMaskFile = File(cacheDir, "mask_temp_${sessionTimestamp}_${i + 1}.png")
                val success = DetailsMaskGenerator.generateMask(
                    imagePath = photo.path,
                    outputPath = tempMaskFile.absolutePath,
                    threshold = 20,
                    kernelSize = 3
                )
                if (success && tempMaskFile.length() > 0) {
                    tempMasks.add(TempMask(i, tempMaskFile.absolutePath, photo))
                    logTime("Mask ${i + 1} generated")
                } else {
                    tempMaskFile.delete()
                    Log.e("Capture", "Failed to generate mask ${i + 1}")
                }
            }

            // Step 1b: Filter duplicate masks
            val similarityThreshold = DetailsMaskGenerator.DEFAULT_SIMILARITY_THRESHOLD
            val uniqueMasks = mutableListOf<TempMask>()
            for (candidate in tempMasks) {
                var isDuplicate = false
                for (accepted in uniqueMasks) {
                    val similarity = DetailsMaskGenerator.computeSimilarity(
                        candidate.tempPath, accepted.tempPath
                    )
                    val isUnique = similarity < (1.0 - similarityThreshold)
                    Log.d(
                        "MaskFilter",
                        "Mask #${candidate.index + 1} vs #${accepted.index + 1}: " +
                                "similarity=%.4f, unique=$isUnique".format(similarity)
                    )
                    if (!isUnique) {
                        isDuplicate = true
                        Log.d("MaskFilter", "Mask #${candidate.index + 1} REJECTED")
                        break
                    }
                }
                if (!isDuplicate) {
                    uniqueMasks.add(candidate)
                    Log.d("MaskFilter", "Mask #${candidate.index + 1} ACCEPTED")
                }
            }
            Log.d("MaskFilter", "Kept ${uniqueMasks.size}/${tempMasks.size} masks")

            // Save unique masks to MediaStore
            val maskPaths = mutableListOf<String>()
            for (mask in uniqueMasks) {
                val tempFile = File(mask.tempPath)
                val maskName = "DET_${sessionTimestamp}_${mask.index + 1}_details_mask.png"
                val savedPath = saveTempFileToMediaStore(tempFile, maskName, "image/png")
                if (savedPath != null) {
                    maskPaths.add(savedPath)
                    logTime("Mask ${mask.index + 1} saved: $savedPath")
                }
            }

            // Step 2: Align images
            logTime("Aligning images...")
            runOnUiThread {
                mainBinding.focusInfoTV.text = "Aligning images..."
            }

            val photoPaths = uniqueMasks.map { it.photo.path }
            val alignedImages = if (photoPaths.size > 1) {
                ImageAligner.alignImages(photoPaths)
            } else {
                // Single image — no alignment needed
                val bmp = loadBitmapWithCorrectOrientation(photoPaths.firstOrNull() ?: "")
                if (bmp != null) listOf(ImageAligner.AlignedImage(bmp, null, 1.0, 0))
                else emptyList()
            }
            logTime("Alignment done: ${alignedImages.size} images")

            var compositeResultPath: String? = null
            var focusMapVisPath: String? = null

            if (alignedImages.size >= 2) {
                // Step 3: Warp masks to reference space
                logTime("Warping masks...")
                runOnUiThread {
                    mainBinding.focusInfoTV.text = "Building focus map..."
                }

                val refW = alignedImages[0].bitmap.width
                val refH = alignedImages[0].bitmap.height

                val alignedMaskBitmaps = mutableListOf<Bitmap>()
                for ((i, mask) in uniqueMasks.withIndex()) {
                    val ai = alignedImages.getOrNull(i)
                    val warpedMask = ImageAligner.warpMask(
                        mask.tempPath, ai?.homography, refW, refH
                    )
                    if (warpedMask != null) {
                        alignedMaskBitmaps.add(warpedMask)
                    } else {
                        // Fallback: load mask as-is
                        val fallback = BitmapFactory.decodeFile(mask.tempPath)
                        if (fallback != null) alignedMaskBitmaps.add(fallback)
                    }
                }

                // Focus points in reference coordinates
                val focusPoints = uniqueMasks.mapIndexed { i, m ->
                    val ai = alignedImages.getOrNull(i)
                    if (ai?.homography != null) {
                        transformPoint(m.photo.focusCx, m.photo.focusCy, ai.homography)
                    } else {
                        Pair(m.photo.focusCx, m.photo.focusCy)
                    }
                }

                // Step 4: Build focus map
                logTime("Building focus map...")
                val focusMap = FocusMapBuilder.buildFocusMap(alignedMaskBitmaps, focusPoints)

                if (focusMap != null) {
                    // Step 5: Composite
                    logTime("Compositing...")
                    runOnUiThread {
                        mainBinding.focusInfoTV.text = "Compositing final image..."
                    }

                    val alignedBitmaps = alignedImages.map { it.bitmap }
                    val compositeResult = FocusStackCompositor.compose(alignedBitmaps, focusMap)

                    Log.d("CaptureTimer", "compositeResult=$compositeResult")
                    if (compositeResult != null) {
                        // Save composite
                        val tempComposite = File(cacheDir, "composite_${sessionTimestamp}.jpg")
                        FocusStackCompositor.saveResult(
                            compositeResult.bitmap,
                            tempComposite.absolutePath
                        )
                        val compositeName = "DET_${sessionTimestamp}_composite.jpg"
                        compositeResultPath =
                            saveTempFileToMediaStore(tempComposite, compositeName, "image/jpeg")
                        tempComposite.delete()
                        logTime("Composite saved: $compositeResultPath")

                        // Save focus map visualization
                        compositeResult.focusMapVis?.let { vis ->
                            val tempVis =
                                File(cacheDir, "focusmap_vis_${sessionTimestamp}.png")
                            FileOutputStream(tempVis).use { fos ->
                                vis.compress(Bitmap.CompressFormat.PNG, 100, fos)
                            }
                            val visName = "DET_${sessionTimestamp}_focusmap.png"
                            focusMapVisPath =
                                saveTempFileToMediaStore(tempVis, visName, "image/png")
                            tempVis.delete()
                        }

                        compositeResult.bitmap.recycle()
                        compositeResult.seamMask?.recycle()
                        compositeResult.focusMapVis?.recycle()
                    }

                    focusMap.release()
                }

                alignedMaskBitmaps.forEach { it.recycle() }
            }

            // Cleanup temp masks
            for (mask in tempMasks) {
                File(mask.tempPath).delete()
            }

            ImageAligner.releaseAlignedImages(alignedImages)
            alignedImages.forEach { it.bitmap.recycle() }

            logTime("Focus stacking pipeline complete")

            // Annotated photo
            logTime("Saving annotated photo")
            val annotatedPath =
                saveAnnotatedFromLastPhoto(results, detections, imgW, imgH, sessionTimestamp)
            logTime("Annotated photo saved: $annotatedPath")

            synchronized(capturedPhotos) {
                capturedPhotos.addAll(results)
            }

            logTime("All done, total objects: ${results.size}")

            runOnUiThread {
                isCapturing = false
                detectionPaused = false
                mainBinding.overlayView.setCapturingMode(false)
                mainBinding.captureIB.visibility = View.VISIBLE

                if (results.isNotEmpty()) {
                    val info = results.mapIndexed { i, cp ->
                        "${i + 1}. ${cp.objectName}: ${cp.path.substringAfterLast('/')}"
                    }.joinToString("\n")
                    val maskInfo =
                        if (maskPaths.isNotEmpty()) "\nMasks: ${maskPaths.size}" else ""
                    val compInfo =
                        if (compositeResultPath != null) "\nComposite: ✓" else ""
                    val ann =
                        if (annotatedPath != null) "\nAnnotated: ${annotatedPath.substringAfterLast('/')}" else ""
                    mainBinding.focusInfoTV.text =
                        "Captured ${results.size} photo(s):\n$info$maskInfo$compInfo$ann"
                    mainBinding.focusInfoTV.visibility = View.VISIBLE
                    mainBinding.closeIB.visibility = View.VISIBLE

                    // Show composite if available, otherwise annotated
                    val showPath =
                        compositeResultPath ?: annotatedPath ?: results.last().path
                    val bmp = loadBitmapWithCorrectOrientation(showPath)
                    if (bmp != null) {
                        mainBinding.capturedIV.setImageBitmap(bmp)
                        mainBinding.capturedIV.visibility = View.VISIBLE
                    }
                } else {
                    mainBinding.focusInfoTV.text = "No photos captured"
                    mainBinding.focusInfoTV.visibility = View.VISIBLE
                }
            }
        }.start()
    }

    // ======================== TRANSFORM POINT BY HOMOGRAPHY ========================

    private fun transformPoint(
        x: Float,
        y: Float,
        homography: org.opencv.core.Mat
    ): Pair<Float, Float> {
        try {
            val h = homography
            val w =
                h.get(2, 0)[0] * x + h.get(2, 1)[0] * y + h.get(2, 2)[0]
            if (kotlin.math.abs(w) < 1e-10) return Pair(x, y)
            val tx =
                (h.get(0, 0)[0] * x + h.get(0, 1)[0] * y + h.get(0, 2)[0]) / w
            val ty =
                (h.get(1, 0)[0] * x + h.get(1, 1)[0] * y + h.get(1, 2)[0]) / w
            return Pair(tx.toFloat(), ty.toFloat())
        } catch (e: Exception) {
            return Pair(x, y)
        }
    }

    // ======================== SAVE TEMP FILE TO MEDIASTORE ========================

    private fun saveTempFileToMediaStore(
        tempFile: File,
        displayName: String,
        mimeType: String
    ): String? {
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
            put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(
                    MediaStore.MediaColumns.RELATIVE_PATH,
                    "${Environment.DIRECTORY_PICTURES}/$APP_FOLDER"
                )
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            val dir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                APP_FOLDER
            )
            if (!dir.exists()) dir.mkdirs()
        }

        val uri: Uri? = contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        )
        if (uri == null) {
            Log.e("MediaStore", "Failed to create MediaStore entry for $displayName")
            return null
        }

        return try {
            contentResolver.openOutputStream(uri)?.use { outputStream ->
                tempFile.inputStream().use { inputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentValues.clear()
                contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
                contentResolver.update(uri, contentValues, null, null)
            }
            val path = getPathFromUri(uri) ?: uri.toString()
            Log.d("MediaStore", "Saved $displayName → $path")
            path
        } catch (e: Exception) {
            Log.e("MediaStore", "Error saving $displayName", e)
            contentResolver.delete(uri, null, null)
            null
        }
    }

    // ======================== GET PATH FROM URI ========================

    private fun getPathFromUri(uri: Uri): String? {
        val projection = arrayOf(MediaStore.Images.Media.DATA)
        contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)
                return cursor.getString(columnIndex)
            }
        }
        return null
    }

    // ======================== LOAD BITMAP WITH EXIF ROTATION ========================

    private fun loadBitmapWithCorrectOrientation(path: String): Bitmap? {
        try {
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
            val rotated =
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated != bitmap) bitmap.recycle()
            return rotated
        } catch (e: Exception) {
            Log.e("LoadBitmap", "Error loading $path", e)
            return null
        }
    }

    // ======================== ANNOTATED PHOTO ========================

    private fun saveAnnotatedFromLastPhoto(
        results: List<CapturedPhoto>,
        detections: List<Detection>,
        imgW: Int,
        imgH: Int,
        sessionTimestamp: String
    ): String? {
        if (results.isEmpty()) return null

        try {
            // Use last captured photo as background
            val lastPhoto = results.last()
            val baseBitmap = loadBitmapWithCorrectOrientation(lastPhoto.path) ?: return null
            val annotated = baseBitmap.copy(Bitmap.Config.ARGB_8888, true)
            baseBitmap.recycle()

            val canvas = Canvas(annotated)
            val scaleXAnn = annotated.width.toFloat() / imgW
            val scaleYAnn = annotated.height.toFloat() / imgH

            val boxPaint = Paint().apply {
                color = Color.GREEN
                style = Paint.Style.STROKE
                strokeWidth = 6f
                isAntiAlias = true
            }
            val fillPaint = Paint().apply {
                color = Color.GREEN
                style = Paint.Style.FILL
            }
            val textPaint = Paint().apply {
                color = Color.WHITE
                textSize = 48f
                isAntiAlias = true
                typeface = Typeface.DEFAULT_BOLD
            }
            val pointPaint = Paint().apply {
                color = Color.RED
                style = Paint.Style.FILL
                isAntiAlias = true
            }
            val pointStroke = Paint().apply {
                color = Color.WHITE
                style = Paint.Style.STROKE
                strokeWidth = 4f
                isAntiAlias = true
            }
            val crossPaint = Paint().apply {
                color = Color.WHITE
                strokeWidth = 3f
                isAntiAlias = true
            }

            for ((i, det) in detections.withIndex()) {
                val left = det.x1 * scaleXAnn
                val top = det.y1 * scaleYAnn
                val right = det.x2 * scaleXAnn
                val bottom = det.y2 * scaleYAnn

                canvas.drawRect(left, top, right, bottom, boxPaint)

                val label = "${i + 1}. ${det.className} ${(det.confidence * 100).toInt()}%"
                val textW = textPaint.measureText(label)
                val textH = 52f
                canvas.drawRect(left, top - textH - 8, left + textW + 16, top, fillPaint)
                canvas.drawText(label, left + 8, top - 14, textPaint)

                // Focus point
                val cx = (left + right) / 2f
                val cy = (top + bottom) / 2f
                canvas.drawCircle(cx, cy, 20f, pointPaint)
                canvas.drawCircle(cx, cy, 22f, pointStroke)
                canvas.drawLine(cx - 30, cy, cx + 30, cy, crossPaint)
                canvas.drawLine(cx, cy - 30, cx, cy + 30, crossPaint)
            }

            // Save annotated
            val tempAnnotated = File(cacheDir, "annotated_${sessionTimestamp}.jpg")
            FileOutputStream(tempAnnotated).use { fos ->
                annotated.compress(Bitmap.CompressFormat.JPEG, 95, fos)
            }
            annotated.recycle()

            val annotatedName = "DET_${sessionTimestamp}_annotated.jpg"
            val savedPath = saveTempFileToMediaStore(tempAnnotated, annotatedName, "image/jpeg")
            tempAnnotated.delete()
            return savedPath

        } catch (e: Exception) {
            Log.e("Annotated", "Error creating annotated photo", e)
            return null
        }
    }

    // ======================== HIDE CAPTURED IMAGE ========================

    private fun hideCapturedImage() {
        mainBinding.capturedIV.visibility = View.GONE
        mainBinding.capturedIV.setImageBitmap(null)
        mainBinding.closeIB.visibility = View.GONE
        mainBinding.focusInfoTV.visibility = View.GONE
        mainBinding.overlayView.clear()
    }
}