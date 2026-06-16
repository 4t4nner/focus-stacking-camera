package com.example.cameraapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.net.Uri
import android.os.*
import android.util.Log
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.regex.Pattern
import androidx.camera.core.Camera as CameraX

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private lateinit var settingsIB: ImageButton
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var capturedIV: ImageView
    private lateinit var captureIB: ImageButton
    private lateinit var folderIB: ImageButton
    private lateinit var fpsTV: TextView
    private lateinit var serverStatusTV: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var progressTV: TextView

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var processingExecutor: ExecutorService

    private var remoteConfig: RemoteProcessor.RemoteConfig? = null
    private var isServerAvailable = false

    private var camera: CameraX? = null
    private var imageCapture: ImageCapture? = null
    private var isCapturing = false

    // ===== ДЕТЕКЦИЯ =====
    private var yoloDetector: YoloDetector? = null
    private var mlKitDetector: MlKitDetector? = null
    private var useYolo = false  // false = ML Kit, true = YOLO

    private val isAnalyzing = AtomicBoolean(false)
    private var lastFpsTime = 0L
    private var frameCount = 0

    // Object detection results (used for capture)
    @Volatile
    private var detectedObjects: List<DetectedObject> = emptyList()

    private val outputDir: File by lazy {
        val dir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            "CameraApp"
        )
        if (!dir.exists()) dir.mkdirs()
        dir
    }

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        if (uris.isNotEmpty()) {
            processPickedFiles(uris)
        }
    }

    data class DetectedObject(
        val label: String,
        val cx: Float,
        val cy: Float,
        val rect: RectF
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        settingsIB = findViewById(R.id.settingsIB)
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        capturedIV = findViewById(R.id.capturedIV)
        captureIB = findViewById(R.id.captureIB)
        folderIB = findViewById(R.id.folderIB)
        fpsTV = findViewById(R.id.fpsTV)
        serverStatusTV = findViewById(R.id.serverStatusTV)
        progressBar = findViewById(R.id.progressBar)
        progressTV = findViewById(R.id.progressTV)

        cameraExecutor = Executors.newSingleThreadExecutor()
        processingExecutor = Executors.newFixedThreadPool(2)

        // Инициализация детекторов
        initDetectors()

        refreshRemoteStatus()

        settingsIB.setOnClickListener {
            SettingsDialog(this) {
                refreshRemoteStatus()
            }.show()
        }

        captureIB.setOnClickListener {
            if (!isCapturing) {
                startFocusStackCapture()
            }
        }

        folderIB.setOnClickListener {
            openFilePicker()
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    // ======================== ДЕТЕКТОРЫ ========================

    private fun initDetectors() {
        // Попробовать YOLO, если не удалось — ML Kit
        processingExecutor.execute {
            try {
                val yolo = YoloDetector(this)
                if (yolo.initialize()) {
                    yoloDetector = yolo
                    useYolo = true
                    Log.d(TAG, "YOLO detector initialized")
                    runOnUiThread {
                        fpsTV.text = "YOLO ready"
                    }
                } else {
                    Log.w(TAG, "YOLO init failed, using ML Kit")
                    mlKitDetector = MlKitDetector()
                    useYolo = false
                    runOnUiThread {
                        fpsTV.text = "ML Kit ready"
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "YOLO init error, falling back to ML Kit", e)
                mlKitDetector = MlKitDetector()
                useYolo = false
            }
        }
    }

    // ======================== CAMERA SETUP с ДЕТЕКЦИЕЙ ========================

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // 1. Preview
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // 2. ImageCapture
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()

            // 3. ImageAnalysis — ДЕТЕКЦИЯ В РЕАЛЬНОМ ВРЕМЕНИ
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                analyzeFrame(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    // ======================== АНАЛИЗ КАДРОВ ========================

    private fun analyzeFrame(imageProxy: ImageProxy) {
        // Не анализируем во время захвата
        if (isCapturing) {
            imageProxy.close()
            return
        }

        // Пропускаем если предыдущий анализ ещё не завершён
        if (!isAnalyzing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            val srcWidth = imageProxy.width
            val srcHeight = imageProxy.height

            if (bitmap == null) {
                imageProxy.close()
                isAnalyzing.set(false)
                return
            }

            // Запускаем детекцию
            val detections: List<Detection> = if (useYolo && yoloDetector != null) {
                yoloDetector!!.detect(bitmap)
            } else if (mlKitDetector != null) {
                mlKitDetector!!.detectStream(bitmap)
            } else {
                emptyList()
            }

            bitmap.recycle()

            // Обновляем detectedObjects для захвата
            detectedObjects = detections.map { det ->
                DetectedObject(
                    label = det.className,
                    cx = (det.x1 + det.x2) / 2f,
                    cy = (det.y1 + det.y2) / 2f,
                    rect = RectF(det.x1, det.y1, det.x2, det.y2)
                )
            }

            // FPS counter
            frameCount++
            val now = System.currentTimeMillis()
            if (now - lastFpsTime >= 1000) {
                val fps = frameCount
                frameCount = 0
                lastFpsTime = now
                runOnUiThread {
                    val detectorName = if (useYolo) "YOLO" else "MLKit"
                    fpsTV.text = "$detectorName ${fps}fps | ${detections.size} obj"
                }
            }

            // Отрисовка оверлея
            runOnUiThread {
                overlayView.setDetections(detections, srcWidth, srcHeight)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Analysis error", e)
        } finally {
            imageProxy.close()
            isAnalyzing.set(false)
        }
    }

    /**
     * Конвертирует ImageProxy (YUV_420_888) в Bitmap.
     */
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        try {
            val yBuffer = imageProxy.planes[0].buffer
            val uBuffer = imageProxy.planes[1].buffer
            val vBuffer = imageProxy.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(
                nv21,
                ImageFormat.NV21,
                imageProxy.width,
                imageProxy.height,
                null
            )

            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                Rect(0, 0, imageProxy.width, imageProxy.height),
                80,
                out
            )

            val bytes = out.toByteArray()
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

            // Учитываем поворот камеры
            val rotation = imageProxy.imageInfo.rotationDegrees
            if (rotation != 0 && bitmap != null) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                val rotated = Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                )
                if (rotated != bitmap) bitmap.recycle()
                return rotated
            }
            return bitmap

        } catch (e: Exception) {
            Log.e(TAG, "imageProxyToBitmap error", e)
            return null
        }
    }

    // ======================== REMOTE CONFIG ========================

    private fun refreshRemoteStatus() {
        processingExecutor.execute {
            val config = AppSettings.getRemoteConfig(this@MainActivity)
            remoteConfig = config

            if (config.enabled) {
                val available = RemoteProcessor.isServerAvailable(config)
                isServerAvailable = available

                runOnUiThread {
                    serverStatusTV.visibility = View.VISIBLE
                    if (available) {
                        serverStatusTV.text = "🟢 ${config.serverHost}:${config.serverPort}"
                        serverStatusTV.setTextColor(Color.parseColor("#4CAF50"))
                    } else {
                        serverStatusTV.text = "🔴 Server offline"
                        serverStatusTV.setTextColor(Color.parseColor("#F44336"))
                    }
                }
            } else {
                isServerAvailable = false
                runOnUiThread {
                    serverStatusTV.visibility = View.GONE
                }
            }
        }
    }

    // ======================== FILE PICKER ========================

    private fun openFilePicker() {
        try {
            filePickerLauncher.launch(arrayOf("image/jpeg", "image/png"))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open file picker", e)
            Toast.makeText(this, "Cannot open file picker", Toast.LENGTH_SHORT).show()
        }
    }

    private fun processPickedFiles(uris: List<Uri>) {
        isCapturing = true
        showProgress("Loading selected images...")

        processingExecutor.execute {
            try {
                val tempDir = File(cacheDir, "picked_${System.currentTimeMillis()}")
                tempDir.mkdirs()

                data class PickedImage(
                    val path: String,
                    val cx: Float,
                    val cy: Float,
                    val index: Int
                )

                val pickedImages = mutableListOf<PickedImage>()
                val pattern = Pattern.compile("""DET_\d+_\d+_(\d+)_(\d+)x(\d+)\.\w+""")

                for ((i, uri) in uris.withIndex()) {
                    val filename = getFilenameFromUri(uri) ?: "image_$i.jpg"
                    Log.d(TAG, "Picked file: $filename")

                    val tempFile = File(tempDir, filename)
                    contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(tempFile).use { output ->
                            input.copyTo(output)
                        }
                    }

                    val matcher = pattern.matcher(filename)
                    if (matcher.find()) {
                        val index = matcher.group(1)!!.toInt()
                        val cx = matcher.group(2)!!.toFloat()
                        val cy = matcher.group(3)!!.toFloat()
                        pickedImages.add(PickedImage(tempFile.absolutePath, cx, cy, index))
                    } else {
                        val bmp = BitmapFactory.decodeFile(tempFile.absolutePath)
                        val cx = (bmp?.width ?: 1000) / 2f
                        val cy = (bmp?.height ?: 1000) / 2f
                        bmp?.recycle()
                        pickedImages.add(PickedImage(tempFile.absolutePath, cx, cy, i))
                    }
                }

                pickedImages.sortBy { it.index }
                val imagePaths = pickedImages.map { it.path }
                val focusPoints = pickedImages.map { Pair(it.cx, it.cy) }

                processImageStack(imagePaths, focusPoints)
                tempDir.deleteRecursively()

            } catch (e: Exception) {
                Log.e(TAG, "Error processing picked files", e)
                runOnUiThread {
                    hideProgress()
                    Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun getFilenameFromUri(uri: Uri): String? {
        var name: String? = null
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            if (cursor.moveToFirst() && nameIndex >= 0) {
                name = cursor.getString(nameIndex)
            }
        }
        return name
    }

    // ======================== PROCESSING PIPELINE ========================

    private fun processImageStack(imagePaths: List<String>, focusPoints: List<Pair<Float, Float>>) {
        runOnUiThread {
            showProgress("Processing focus stack...")
            folderIB.visibility = View.GONE
            captureIB.visibility = View.GONE
            overlayView.visibility = View.GONE
        }

        val config = remoteConfig ?: AppSettings.getRemoteConfig(this@MainActivity)
        remoteConfig = config
        Log.d(TAG, "=== processImageStack ===")
        Log.d(TAG, "config: $config")
        Log.d(TAG, "config.enabled: ${config?.enabled}")
        Log.d(TAG, "isServerAvailable (cached): $isServerAvailable")

        if (config != null && config.enabled) {
            runOnUiThread { updateProgress("Checking server...") }
            val freshCheck = RemoteProcessor.isServerAvailable(config)
            Log.d(TAG, "Fresh server check: $freshCheck")
            isServerAvailable = freshCheck
        }

        Log.d(TAG, "Final decision - remote: ${config != null && config.enabled && isServerAvailable}")

        var compositeBitmap: Bitmap? = null

        if (config != null && config.enabled && isServerAvailable) {
            runOnUiThread { updateProgress("Uploading to server ${config.serverHost}...") }

            // Создаём callback для обновления прогресса с сервера
            val progressCallback = object : RemoteProcessor.ProgressCallback {
                override fun onProgress(message: String) {
                    runOnUiThread { updateProgress(message) }
                }
            }

            compositeBitmap = RemoteProcessor.processRemotely(
                config, imagePaths, focusPoints, progressCallback
            )

            if (compositeBitmap != null) {
                runOnUiThread { updateProgress("Server processing complete!") }
            } else {
                runOnUiThread { updateProgress("Server failed, processing locally...") }
            }
        }

        if (compositeBitmap == null) {
            runOnUiThread { updateProgress("Processing locally on device...") }
            compositeBitmap = processLocally(imagePaths, focusPoints)
        }

        if (compositeBitmap != null) {
            val resultFile = saveComposite(compositeBitmap)
            runOnUiThread {
                hideProgress()
                capturedIV.setImageBitmap(compositeBitmap)
                capturedIV.visibility = View.VISIBLE
                previewView.visibility = View.GONE
                overlayView.visibility = View.GONE

                Toast.makeText(this, "Saved: ${resultFile?.name ?: "unknown"}", Toast.LENGTH_LONG).show()

                capturedIV.setOnClickListener {
                    capturedIV.visibility = View.GONE
                    previewView.visibility = View.VISIBLE
                    overlayView.visibility = View.VISIBLE
                    captureIB.visibility = View.VISIBLE
                    folderIB.visibility = View.VISIBLE
                    isCapturing = false
                }
            }
        } else {
            runOnUiThread {
                hideProgress()
                captureIB.visibility = View.VISIBLE
                folderIB.visibility = View.VISIBLE
                Toast.makeText(this, "Processing failed", Toast.LENGTH_LONG).show()
                isCapturing = false
            }
        }
    }
    private fun processLocally(
        imagePaths: List<String>,
        focusPoints: List<Pair<Float, Float>>
    ): Bitmap? {
        try {
            if (imagePaths.size < 2) {
                return BitmapFactory.decodeFile(imagePaths.firstOrNull() ?: return null)
            }

            val tempDir = File(cacheDir, "local_proc_${System.currentTimeMillis()}")
            tempDir.mkdirs()

            // Step 1: Generate detail masks
            val maskPaths = mutableListOf<String>()
            for ((i, imgPath) in imagePaths.withIndex()) {
                runOnUiThread { updateProgress("Detail mask ${i + 1}/${imagePaths.size}...") }
                val maskPath = File(tempDir, "mask_$i.png").absolutePath
                val success = DetailsMaskGenerator.generateMask(imgPath, maskPath)
                if (success) {
                    maskPaths.add(maskPath)
                } else {
                    maskPaths.add("")
                }
            }

            // Step 2: Filter duplicates
            runOnUiThread { updateProgress("Filtering duplicate masks...") }
            val uniqueIndices = filterDuplicateMasks(maskPaths)

            if (uniqueIndices.size < 2) {
                tempDir.deleteRecursively()
                return BitmapFactory.decodeFile(imagePaths[0])
            }

            val uImagePaths = uniqueIndices.map { imagePaths[it] }
            val uMaskPaths = uniqueIndices.map { maskPaths[it] }
            val uFocusPoints = uniqueIndices.map { i ->
                if (i < focusPoints.size) focusPoints[i]
                else {
                    val bmp = BitmapFactory.decodeFile(imagePaths[0])
                    val p = Pair(bmp?.width?.div(2f) ?: 500f, bmp?.height?.div(2f) ?: 500f)
                    bmp?.recycle()
                    p
                }
            }

            // Step 3: Align images
            runOnUiThread { updateProgress("Aligning ${uImagePaths.size} images...") }
            val alignedImages = ImageAligner.alignImages(uImagePaths)

            if (alignedImages.size < 2) {
                tempDir.deleteRecursively()
                return BitmapFactory.decodeFile(imagePaths[0])
            }

            // Step 4: Warp masks
            runOnUiThread { updateProgress("Warping masks...") }
            val refW = alignedImages[0].bitmap.width
            val refH = alignedImages[0].bitmap.height

            val alignedMasks = mutableListOf<Bitmap>()
            for ((i, aligned) in alignedImages.withIndex()) {
                if (i < uMaskPaths.size && uMaskPaths[i].isNotEmpty()) {
                    val warpedMask = ImageAligner.warpMask(uMaskPaths[i], aligned.homography, refW, refH)
                    alignedMasks.add(warpedMask ?: Bitmap.createBitmap(refW, refH, Bitmap.Config.ARGB_8888))
                } else {
                    alignedMasks.add(Bitmap.createBitmap(refW, refH, Bitmap.Config.ARGB_8888))
                }
            }

            // Step 5: Focus map
            runOnUiThread { updateProgress("Building focus map...") }
            val focusMap = FocusMapBuilder.buildFocusMap(alignedMasks, uFocusPoints)

            if (focusMap == null) {
                ImageAligner.releaseAlignedImages(alignedImages)
                tempDir.deleteRecursively()
                return BitmapFactory.decodeFile(imagePaths[0])
            }

            // Step 6: Composite
            runOnUiThread { updateProgress("Compositing...") }
            val alignedBitmaps = alignedImages.map { it.bitmap }
            val compositeResult = FocusStackCompositor.compose(alignedBitmaps, focusMap)

            focusMap.release()
            ImageAligner.releaseAlignedImages(alignedImages)
            alignedMasks.forEach { it.recycle() }
            tempDir.deleteRecursively()

            if (compositeResult == null) {
                return BitmapFactory.decodeFile(imagePaths[0])
            }

            return compositeResult.bitmap

        } catch (e: Exception) {
            Log.e(TAG, "Local processing failed", e)
            return null
        }
    }

    private fun filterDuplicateMasks(maskPaths: List<String>): List<Int> {
        if (maskPaths.isEmpty()) return emptyList()

        val unique = mutableListOf(0)
        for (i in 1 until maskPaths.size) {
            if (maskPaths[i].isEmpty()) continue
            var isDuplicate = false
            for (j in unique) {
                if (maskPaths[j].isEmpty()) continue
                val similarity = DetailsMaskGenerator.computeSimilarity(maskPaths[i], maskPaths[j])
                if (similarity >= 0 && similarity > (1.0 - DetailsMaskGenerator.DEFAULT_SIMILARITY_THRESHOLD)) {
                    isDuplicate = true
                    break
                }
            }
            if (!isDuplicate) {
                unique.add(i)
            }
        }
        return unique
    }

    // ======================== SAVE ========================

    private fun saveComposite(bitmap: Bitmap): File? {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val file = File(outputDir, "FS_${timestamp}.jpg")
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
            val intent = Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE)
            intent.data = Uri.fromFile(file)
            sendBroadcast(intent)
            return file
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save composite", e)
            return null
        }
    }

    // ======================== CAPTURE ========================

    private fun startFocusStackCapture() {
        if (detectedObjects.isEmpty()) {
            Toast.makeText(this, "No objects detected to focus on", Toast.LENGTH_SHORT).show()
            return
        }

        isCapturing = true
        captureIB.visibility = View.GONE
        folderIB.visibility = View.GONE

        // Показать overlay в режиме захвата (нумерация объектов)
        overlayView.setCapturingMode(true)
        showProgress("Capturing focus stack...")

        processingExecutor.execute {
            captureSequence(detectedObjects)
        }
    }

    private fun captureSequence(objects: List<DetectedObject>) {
        val capturedPaths = mutableListOf<String>()
        val focusPoints = mutableListOf<Pair<Float, Float>>()
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

        for ((i, obj) in objects.withIndex()) {
            runOnUiThread {
                updateProgress("Capturing ${i + 1}/${objects.size}: ${obj.label}")
            }

            try {
                // Фокусировка на объекте
                val meteringPoint = previewView.meteringPointFactory
                    .createPoint(obj.cx, obj.cy)
                val focusAction = FocusMeteringAction.Builder(meteringPoint)
                    .setAutoCancelDuration(3, java.util.concurrent.TimeUnit.SECONDS)
                    .build()

                camera?.cameraControl?.startFocusAndMetering(focusAction)

                // Ждём фокусировку
                Thread.sleep(800)

                // Захват
                val filename = "DET_${timestamp}_${i + 1}_${obj.cx.toInt()}x${obj.cy.toInt()}.jpg"
                val file = File(outputDir, filename)
                val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()

                val latch = java.util.concurrent.CountDownLatch(1)

                imageCapture?.takePicture(
                    outputOptions,
                    cameraExecutor,
                    object : ImageCapture.OnImageSavedCallback {
                        override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                            capturedPaths.add(file.absolutePath)
                            focusPoints.add(Pair(obj.cx, obj.cy))
                            Log.d(TAG, "Captured: $filename")
                            latch.countDown()
                        }

                        override fun onError(e: ImageCaptureException) {
                            Log.e(TAG, "Capture failed for $filename", e)
                            latch.countDown()
                        }
                    }
                )

                latch.await(10, java.util.concurrent.TimeUnit.SECONDS)

            } catch (e: Exception) {
                Log.e(TAG, "Error capturing frame $i", e)
            }
        }

        runOnUiThread {
            overlayView.setCapturingMode(false)
        }

        if (capturedPaths.size >= 2) {
            processImageStack(capturedPaths, focusPoints)
        } else {
            runOnUiThread {
                hideProgress()
                captureIB.visibility = View.VISIBLE
                folderIB.visibility = View.VISIBLE
                isCapturing = false
                Toast.makeText(this, "Not enough images captured", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ======================== UI HELPERS ========================

    private fun showProgress(message: String) {
        runOnUiThread {
            progressBar.visibility = View.VISIBLE
            progressTV.visibility = View.VISIBLE
            progressTV.text = message
        }
    }

    private fun updateProgress(message: String) {
        progressTV.text = message
    }

    private fun hideProgress() {
        progressBar.visibility = View.GONE
        progressTV.visibility = View.GONE
    }

    // ======================== PERMISSIONS ========================

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        processingExecutor.shutdown()
        mlKitDetector?.close()
    }
}