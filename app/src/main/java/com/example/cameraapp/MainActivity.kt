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
import android.content.Context
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.exifinterface.media.ExifInterface
import android.hardware.camera2.CameraMetadata

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
    private lateinit var clearPointsIB: ImageButton
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

    // Пул для записи JPEG на диск (параллельно захвату следующих кадров)
    private val saverExecutor: java.util.concurrent.ExecutorService =
        java.util.concurrent.Executors.newFixedThreadPool(2)

    // Пул для сохранения снимка с точками (параллельно серии)
    private val pointsShotExecutor: java.util.concurrent.ExecutorService =
        java.util.concurrent.Executors.newSingleThreadExecutor()

    // Чтобы дождаться, что ВСЕ файлы дописались, прежде чем завершить серию
    private val pendingSaves = java.util.concurrent.atomic.AtomicInteger(0)
    private val allSavesDone = java.util.concurrent.CountDownLatch(0) // пересоздаётся на старте серии

    private var remoteConfig: RemoteProcessor.RemoteConfig? = null
    private var isServerAvailable = false

    private var camera: CameraX? = null
    private var imageCapture: ImageCapture? = null
    private var isCapturing = false
    // Последнее измеренное фокусное расстояние (диоптрии, 1/м)
    @Volatile
    private var lastFocusDistance: Float? = null

    // ===== РУЧНОЙ РЕЖИМ =====

    // Последние "живые" значения из CaptureResult (обновляются в setSessionCaptureCallback).
    // Используются при тапе для сохранения настроек точки.
    @Volatile private var lastExposureTimeNs: Long? = null
    @Volatile private var lastIso: Int? = null
    @Volatile private var lastAfState: Int? = null
    @Volatile private var lastAeState: Int? = null

    // Настройки точек в ручном режиме. Индекс = detectionIndex.
    // Заполняется при тапах пользователя.
    private val manualSettings = java.util.concurrent.ConcurrentHashMap<Int, ManualFrameSettings>()

    // Идёт ли сейчас конвергенция AF/AE по тапу (защита от повторных тапов).
    private val isMeteringForTap = java.util.concurrent.atomic.AtomicBoolean(false)

    // Текущий режим (кэш; читается из AppSettings при onResume/после настроек).
    @Volatile private var isManualMode: Boolean = false

    // ===== ДЕТЕКЦИЯ =====
    private var yoloDetector: YoloDetector? = null
    private var mlKitDetector: MlKitDetector? = null
    private var useYolo = false  // false = ML Kit, true = YOLO

    private val isAnalyzing = AtomicBoolean(false)
    private var lastFpsTime = 0L
    private var frameCount = 0

    @Volatile private var lastAnalysisWidth: Int = 0
    @Volatile private var lastAnalysisHeight: Int = 0

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
        object : ActivityResultContracts.OpenMultipleDocuments() {
            override fun createIntent(context: Context, input: Array<String>): Intent {
                val intent = super.createIntent(context, input)
                // Стартовая папка Pictures/CameraApp
                val initialUri = android.provider.DocumentsContract.buildDocumentUri(
                    "com.android.externalstorage.documents",
                    "primary:Pictures/CameraApp"
                )
                intent.putExtra(android.provider.DocumentsContract.EXTRA_INITIAL_URI, initialUri)
                return intent
            }
        }
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
        clearPointsIB = findViewById(R.id.clearPointsIB)

        cameraExecutor = Executors.newSingleThreadExecutor()
        processingExecutor = Executors.newFixedThreadPool(2)

        // Инициализация детекторов
        initDetectors()

        refreshRemoteStatus()

        // Подключаем обработчик ручных тапов (OverlayView сам обрабатывает касания
        // и отдаёт координаты в системе кадра анализа).
        overlayView.onManualTap = { cx, cy ->
            handleManualTap(cx, cy)
        }

        settingsIB.setOnClickListener {
            SettingsDialog(this) {
                refreshRemoteStatus()
                reinitDetectors()   // ← переинициализация при смене детектора
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

        clearPointsIB.setOnClickListener {
            if (isCapturing) return@setOnClickListener

            clearManualPointsAll()
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }
    private fun clearManualPointsAll() {
        manualSettings.clear()
        overlayView.clearManualPoints()
        updateCaptureButtonState()
    }
    // ======================== ДЕТЕКТОРЫ ========================

    private fun reinitDetectors() {
        // Освобождаем старые детекторы
        mlKitDetector?.close()
        mlKitDetector = null
        yoloDetector = null
        useYolo = false
        detectedObjects = emptyList()

        // Заново инициализируем по текущим настройкам
        initDetectors()
    }

    private fun initDetectors() {
        processingExecutor.execute {
            val detector = AppSettings.getDetector(this)

            if (detector == AppSettings.DETECTOR_YOLO) {
                try {
                    val yolo = YoloDetector(this)
                    if (yolo.initialize()) {
                        yoloDetector = yolo
                        useYolo = true
                        Log.d(TAG, "YOLO detector initialized")
                        runOnUiThread { fpsTV.text = "YOLO ready" }
                        return@execute
                    } else {
                        Log.w(TAG, "YOLO init failed, falling back to ML Kit")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "YOLO init error, falling back to ML Kit", e)
                }
            }

            // ML Kit (по умолчанию или fallback)
            mlKitDetector = MlKitDetector()
            useYolo = false
            Log.d(TAG, "Using ML Kit detector")
            runOnUiThread { fpsTV.text = "ML Kit ready" }
        }
    }

    // ======================== РУЧНОЙ / АВТО РЕЖИМ ========================

    /**
     * Считывает режим из настроек и применяет: глушит/включает детектор,
     * показывает/прячет кнопку очистки, сбрасывает ручные точки.
     */
    private fun applyCaptureMode() {
        val manual = AppSettings.isManualMode(this)
        val modeChanged = (manual != isManualMode)
        isManualMode = manual

        // ВСЕГДА синхронизируем состояние OverlayView (даже если режим не менялся),
        // иначе после съёмки manualMode в OverlayView может остаться рассинхронизированным
        // и ручные точки перестанут рисоваться.
        overlayView.setManualMode(manual)
        overlayView.setAnalysisSize(lastAnalysisWidth, lastAnalysisHeight)

        if (modeChanged) {
            // Сброс предыдущих ручных данных при переключении режима
            manualSettings.clear()
            overlayView.clearManualPoints()
        }

        if (manual) {
            detectedObjects = emptyList()
            clearPointsIB.visibility = View.VISIBLE
            fpsTV.text = "MANUAL"
        } else {
            clearPointsIB.visibility = View.GONE
            overlayView.clearManualPoints()
        }
        updateCaptureButtonState()
    }

    /** В ручном режиме кнопка съёмки активна только при >= 2 точках. */
    private fun updateCaptureButtonState() {
        if (isManualMode) {
            val enough = overlayView.manualPointCount() >= 2
            captureIB.isEnabled = enough
            captureIB.alpha = if (enough) 1.0f else 0.4f
        } else {
            captureIB.isEnabled = true
            captureIB.alpha = 1.0f
        }
    }

    /**
     * Обработка тапа в ручном режиме: добавляем точку, запускаем AF/AE
     * на ней, после сходимости считываем параметры и сохраняем в manualSettings.
     */
    private fun handleManualTap(cx: Float, cy: Float) {
        if (!isManualMode || isCapturing) return

        // Защита от параллельных замеров
        if (!isMeteringForTap.compareAndSet(false, true)) {
            Toast.makeText(this, "Focusing… wait", Toast.LENGTH_SHORT).show()
            return
        }

        // Индекс новой точки = текущее количество ручных точек
        val pointIndex = overlayView.manualPointCount()
        overlayView.addManualPoint(cx, cy)
        updateCaptureButtonState()

        // Предварительная запись настроек (ещё не настроена)
        val settings = ManualFrameSettings(
            detectionIndex = pointIndex,
            label = "P${pointIndex + 1}",
            cx = cx,
            cy = cy,
            rect = RectF(cx - 1f, cy - 1f, cx + 1f, cy + 1f)
        )
        manualSettings[pointIndex] = settings

        // Преобразование координат кадра анализа -> координаты previewView.
        val aw = lastAnalysisWidth.takeIf { it > 0 } ?: 1
        val ah = lastAnalysisHeight.takeIf { it > 0 } ?: 1
        val fx = (cx / aw) * previewView.width
        val fy = (cy / ah) * previewView.height

        Log.d(TAG, "TAP idx=$pointIndex cx=$cx cy=$cy aw=$aw ah=$ah -> fx=$fx fy=$fy preview=${previewView.width}x${previewView.height}")

        try {
            val meteringPoint = previewView.meteringPointFactory.createPoint(fx, fy)
            val action = FocusMeteringAction.Builder(
                meteringPoint,
                FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
            )
                .setAutoCancelDuration(5, java.util.concurrent.TimeUnit.SECONDS)
                .build()

            val future = camera?.cameraControl?.startFocusAndMetering(action)
            if (future == null) {
                isMeteringForTap.set(false)
                return
            }

            future.addListener({
                // AF/AE future завершился, НО линза/сенсор и preview-метаданные обновляются
                // с задержкой в несколько кадров. Даём время доехать и обновить
                // lastFocusDistance/lastIso/lastExposureTimeNs, иначе для разных точек
                // считаются устаревшие (одинаковые) значения.
                overlayView.postDelayed({
                    val focus = lastFocusDistance
                    val exp = lastExposureTimeNs
                    val iso = lastIso

                    settings.focusDistanceDiopters = focus
                    settings.exposureTimeNs = exp
                    settings.iso = iso
                    settings.configured = (focus != null || exp != null || iso != null)

                    Log.d(TAG, "POINT idx=$pointIndex configured focus=$focus iso=$iso exp=$exp")

                    overlayView.setManualPointConfigured(pointIndex, settings.configured)
                    val fm = focus?.let { if (it > 0f) "%.2fm".format(1f / it) else "∞" } ?: "?"
                    Toast.makeText(
                        this,
                        "Point ${pointIndex + 1}: focus=$fm iso=${iso ?: "?"}",
                        Toast.LENGTH_SHORT
                    ).show()

                    isMeteringForTap.set(false)
                }, 400)  // 400 мс на доезд линзы + обновление метаданных preview-потока

            }, ContextCompat.getMainExecutor(this))

        } catch (e: Exception) {
            Log.e(TAG, "Manual tap metering failed", e)
            isMeteringForTap.set(false)
        }
    }
    /**
     * Собирает список объектов для съёмки из ручных точек.
     * Порядок = порядок добавления (индекс точки), что совпадает
     * с ключами manualSettings и detectionIndex.
     */
    private fun buildManualObjects(): List<DetectedObject> {
        val count = manualSettings.size
        val list = ArrayList<DetectedObject>(count)
        for (idx in 0 until count) {
            val ms = manualSettings[idx] ?: continue
            list.add(
                DetectedObject(
                    label = ms.label,
                    cx = ms.cx,
                    cy = ms.cy,
                    rect = ms.rect
                )
            )
        }
        return list
    }
    // ======================== CAMERA SETUP с ДЕТЕКЦИЕЙ ========================

    @OptIn(ExperimentalCamera2Interop::class)
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // 1. Preview + чтение метаданных живого потока (для ручного режима)
            val previewBuilder = Preview.Builder()

            Camera2Interop.Extender(previewBuilder).setSessionCaptureCallback(
                object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(
                        session: CameraCaptureSession,
                        request: CaptureRequest,
                        result: TotalCaptureResult
                    ) {
                        // Обновляем "живые" значения параметров на каждом preview-кадре.
                        result.get(CaptureResult.LENS_FOCUS_DISTANCE)?.let {
                            lastFocusDistance = it
                        }
                        result.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let {
                            lastExposureTimeNs = it
                        }
                        result.get(CaptureResult.SENSOR_SENSITIVITY)?.let {
                            lastIso = it
                        }
                        lastAfState = result.get(CaptureResult.CONTROL_AF_STATE)
                        lastAeState = result.get(CaptureResult.CONTROL_AE_STATE)
                    }
                }
            )

            val preview = previewBuilder.build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // 2. ImageCapture + чтение LENS_FOCUS_DISTANCE из метаданных кадра
            val captureBuilder = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY) // CAPTURE_MODE_MAXIMIZE_QUALITY
                .setFlashMode(ImageCapture.FLASH_MODE_OFF)

            Camera2Interop.Extender(captureBuilder).setSessionCaptureCallback(
                object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(
                        session: CameraCaptureSession,
                        request: CaptureRequest,
                        result: TotalCaptureResult
                    ) {
                        result.get(CaptureResult.LENS_FOCUS_DISTANCE)?.let {
                            lastFocusDistance = it
                        }
                    }
                }
            )
            imageCapture = captureBuilder.build()

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

        // В ручном режиме детектор заглушён — только фиксируем размеры кадра для маппинга тапа.
        if (isManualMode) {
            lastAnalysisWidth = imageProxy.width
            lastAnalysisHeight = imageProxy.height
            // КРИТИЧНО: обновляем размеры в OverlayView, иначе тап маппится по source=1x1
            // и onTouchEvent отбрасывает касание (точка не добавится).
            overlayView.setAnalysisSize(imageProxy.width, imageProxy.height)
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
            lastAnalysisWidth = srcWidth
            lastAnalysisHeight = srcHeight

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
            PipelineLogger.measureStage("generate_masks") {
                for ((i, imgPath) in imagePaths.withIndex()) {
                    runOnUiThread { updateProgress("Detail mask ${i + 1}/${imagePaths.size}...") }
                    val maskPath = File(tempDir, "mask_$i.png").absolutePath
                    val success = DetailsMaskGenerator.generateMask(imgPath, maskPath)
                    if (success) maskPaths.add(maskPath) else maskPaths.add("")
                }
            }

            // Step 2: Filter duplicates
            runOnUiThread { updateProgress("Filtering duplicate masks...") }
            val uniqueIndices = PipelineLogger.measureStage("filter_duplicates") {
                filterDuplicateMasks(maskPaths)
            }

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
            val alignedImages = PipelineLogger.measureStage("align_images") {
                ImageAligner.alignImages(uImagePaths)
            }
            if (alignedImages.size < 2) {
                tempDir.deleteRecursively()
                return BitmapFactory.decodeFile(imagePaths[0])
            }

            // Step 4: Warp masks
            runOnUiThread { updateProgress("Warping masks...") }
            val refW = alignedImages[0].bitmap.width
            val refH = alignedImages[0].bitmap.height

            val alignedMasks = mutableListOf<Bitmap>()
            PipelineLogger.measureStage("warp_masks") {
                for ((i, aligned) in alignedImages.withIndex()) {
                    if (i < uMaskPaths.size && uMaskPaths[i].isNotEmpty()) {
                        val warpedMask = ImageAligner.warpMask(uMaskPaths[i], aligned.homography, refW, refH)
                        alignedMasks.add(warpedMask ?: Bitmap.createBitmap(refW, refH, Bitmap.Config.ARGB_8888))
                    } else {
                        alignedMasks.add(Bitmap.createBitmap(refW, refH, Bitmap.Config.ARGB_8888))
                    }
                }
            }
            // Step 5: Focus map
            runOnUiThread { updateProgress("Building focus map...") }
            val focusMap = PipelineLogger.measureStage("build_focus_map") {
                FocusMapBuilder.buildFocusMap(alignedMasks, uFocusPoints)
            }
            if (focusMap == null) {
                ImageAligner.releaseAlignedImages(alignedImages)
                tempDir.deleteRecursively()
                return BitmapFactory.decodeFile(imagePaths[0])
            }

            // Step 6: Composite
            runOnUiThread { updateProgress("Compositing...") }
            val alignedBitmaps = alignedImages.map { it.bitmap }
            val compositeResult = PipelineLogger.measureStage("composite") {
                FocusStackCompositor.compose(alignedBitmaps, focusMap)
            }
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

    /**
     * Рисует точки и их номера на копии bitmap.
     * Координаты объектов (cx, cy) заданы в пространстве кадра ImageAnalysis,
     * поэтому масштабируем их к размеру снимка.
     */
    private fun drawPointsOnBitmap(
        src: Bitmap,
        objects: List<DetectedObject>
    ): Bitmap {
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)

        // Масштаб от системы координат детекций к фактическому снимку.
        // detectedObjects заданы в координатах кадра анализа (analysisW x analysisH).
        val analysisW = lastAnalysisWidth.takeIf { it > 0 } ?: result.width
        val analysisH = lastAnalysisHeight.takeIf { it > 0 } ?: result.height
        val scaleX = result.width.toFloat() / analysisW
        val scaleY = result.height.toFloat() / analysisH

        // Масштаб элементов под разрешение снимка
        val baseScale = result.width / 1080f
        val pointR = 16f * baseScale
        val crossLen = 24f * baseScale

        val pointPaint = Paint().apply { color = Color.RED; style = Paint.Style.FILL; isAntiAlias = true }
        val pointStroke = Paint().apply {
            style = Paint.Style.STROKE; color = Color.WHITE
            strokeWidth = 4f * baseScale; isAntiAlias = true
        }
        val crossPaint = Paint().apply {
            color = Color.WHITE; strokeWidth = 3f * baseScale; isAntiAlias = true
        }
        val numberPaint = Paint().apply {
            color = Color.WHITE; textSize = 52f * baseScale; isAntiAlias = true
            typeface = Typeface.DEFAULT_BOLD; textAlign = Paint.Align.CENTER
        }
        val numberBgPaint = Paint().apply {
            color = Color.RED; style = Paint.Style.FILL; isAntiAlias = true
        }

        for ((index, obj) in objects.withIndex()) {
            val cx = obj.cx * scaleX
            val cy = obj.cy * scaleY

            canvas.drawCircle(cx, cy, pointR, pointPaint)
            canvas.drawCircle(cx, cy, pointR + 2f * baseScale, pointStroke)
            canvas.drawLine(cx - crossLen, cy, cx + crossLen, cy, crossPaint)
            canvas.drawLine(cx, cy - crossLen, cx, cy + crossLen, crossPaint)

            val numStr = "${index + 1}"
            val numW = numberPaint.measureText(numStr)
            val r = (maxOf(numW, 52f * baseScale) / 2f) + 8f * baseScale
            canvas.drawCircle(cx, cy - 50f * baseScale, r, numberBgPaint)
            canvas.drawText(numStr, cx, cy - 35f * baseScale, numberPaint)
        }

        return result
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun capturePointsPreview(
        objects: List<DetectedObject>,
        timestamp: String,
        onDone: (() -> Unit)? = null
    ) {
        val ic = imageCapture
        if (ic == null) {
            onDone?.invoke()
            return
        }

        ic.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val rotation = image.imageInfo.rotationDegrees
                    val bytes = image.toJpegBytes()
                    image.close()
                    // Кадр уже в памяти — можно освобождать вызывающего.
                    onDone?.invoke()

                    pointsShotExecutor.execute {
                        try {
                            var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                                ?: return@execute

                            if (rotation != 0) {
                                val m = Matrix().apply { postRotate(rotation.toFloat()) }
                                val rotated = Bitmap.createBitmap(
                                    bmp, 0, 0, bmp.width, bmp.height, m, true
                                )
                                if (rotated != bmp) bmp.recycle()
                                bmp = rotated
                            }

                            val annotated = drawPointsOnBitmap(bmp, objects)
                            if (annotated != bmp) bmp.recycle()

                            val file = File(outputDir, "POINTS_${timestamp}.jpg")
                            FileOutputStream(file).use {
                                annotated.compress(Bitmap.CompressFormat.JPEG, 95, it)
                            }
                            annotated.recycle()

                            val intent = Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE)
                            intent.data = Uri.fromFile(file)
                            sendBroadcast(intent)

                            Log.d(TAG, "Points preview saved: ${file.name}")
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed to save points preview", e)
                        }
                    }
                }

                override fun onError(e: ImageCaptureException) {
                    Log.e(TAG, "Points preview capture failed", e)
                    onDone?.invoke()
                }
            }
        )
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun captureSequenceManual(objects: List<DetectedObject>, sessionTimestamp: String) {
        val n = objects.size
        val pathSlots = arrayOfNulls<String>(n)
        val focusSlots = arrayOfNulls<Pair<Float, Float>>(n)
        val savesRemaining = java.util.concurrent.atomic.AtomicInteger(0)

        val timestamp = sessionTimestamp

        PipelineLogger.startSession()

        // ===== POINTS-кадр ПЕРВЫМ, пока камера ещё в авто-экспозиции =====
        // Дожидаемся завершения захвата, ЗАТЕМ выставляем ручные AE/AF.
        val pointsLatch = java.util.concurrent.CountDownLatch(1)
        capturePointsPreview(objects, timestamp) {
            pointsLatch.countDown()
        }
        pointsLatch.await(5, java.util.concurrent.TimeUnit.SECONDS)

        val cam2Control = camera?.cameraControl?.let { Camera2CameraControl.from(it) }

        for ((i, obj) in objects.withIndex()) {
            // Сопоставление точки и её ручных настроек.
            // ВНИМАНИЕ: предполагается detectionIndex == позиция i.
            val ms = manualSettings[i]
            if (ms == null || !ms.configured) {
                Log.w(TAG, "Manual settings missing for index $i, skipping")
                continue
            }

            runOnUiThread {
                updateProgress("Capturing ${i + 1}/${objects.size}: ${obj.label}")
            }

            try {
                // ===== Выставляем ручные параметры: AF_OFF + фокус, AE_OFF + выдержка/ISO =====
                val applyStart = System.currentTimeMillis()

                val optsBuilder = CaptureRequestOptions.Builder()

                // Фокус
                ms.focusDistanceDiopters?.let { fd ->
                    optsBuilder
                        .setCaptureRequestOption(
                            CaptureRequest.CONTROL_AF_MODE,
                            CameraMetadata.CONTROL_AF_MODE_OFF
                        )
                        .setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, fd)
                }

                // Экспозиция (только если есть оба параметра — иначе оставляем AE авто)
                val expNs = ms.exposureTimeNs
                val iso = ms.iso
                if (expNs != null && iso != null) {
                    optsBuilder
                        .setCaptureRequestOption(
                            CaptureRequest.CONTROL_AE_MODE,
                            CameraMetadata.CONTROL_AE_MODE_OFF
                        )
                        .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, expNs)
                        .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, iso)
                }

                cam2Control?.setCaptureRequestOptions(optsBuilder.build())

                // Даём сенсору/линзе время приехать на заданные значения.
                // Фокус-моторчику нужно ~200–400 мс на перемещение.
                Thread.sleep(350)
                val applyDurationMs = System.currentTimeMillis() - applyStart

                // ===== Захват =====
                val filename = "DET_${timestamp}_${i + 1}_${obj.cx.toInt()}x${obj.cy.toInt()}.jpg"
                val file = File(outputDir, filename)

                val captureStart = SystemClock.elapsedRealtime()
                val captureLatch = java.util.concurrent.CountDownLatch(1)

                savesRemaining.incrementAndGet()

                imageCapture?.takePicture(
                    cameraExecutor,
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            val capturedAt = SystemClock.elapsedRealtime()
                            val sensorEncodeMs = capturedAt - captureStart

                            val rotation = image.imageInfo.rotationDegrees
                            val bytes = image.toJpegBytes()
                            image.close()
                            captureLatch.countDown()

                            saverExecutor.execute {
                                val saveStart = SystemClock.elapsedRealtime()
                                try {
                                    file.outputStream().use { it.write(bytes) }
                                    applyExifRotation(file, rotation)

                                    pathSlots[i] = file.absolutePath
                                    focusSlots[i] = Pair(obj.cx, obj.cy)

                                    val intent = Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE)
                                    intent.data = Uri.fromFile(file)
                                    sendBroadcast(intent)

                                    val saveEnd = SystemClock.elapsedRealtime()
                                    PipelineLogger.stage(i + 1, "sensor_encode", sensorEncodeMs)
                                    PipelineLogger.stage(i + 1, "disk_write", saveEnd - saveStart)
                                } catch (e: Exception) {
                                    Log.e(TAG, "Save failed for $filename", e)
                                } finally {
                                    savesRemaining.decrementAndGet()
                                }
                            }
                        }

                        override fun onError(e: ImageCaptureException) {
                            Log.e(TAG, "Capture failed for $filename", e)
                            savesRemaining.decrementAndGet()
                            captureLatch.countDown()
                        }
                    }
                )

                captureLatch.await(10, java.util.concurrent.TimeUnit.SECONDS)
                val captureEnd = SystemClock.elapsedRealtime()
                val captureDurationMs = captureEnd - captureStart

                PipelineLogger.stage(i + 1, "manual_apply", applyDurationMs.toLong())

                // ===== Лог кадра (используем зафиксированные при тапе значения) =====
                val diopters = ms.focusDistanceDiopters
                val meters = diopters?.let { if (it > 0f) 1f / it else null }
                PipelineLogger.logFrame(
                    PipelineLogger.FrameLog(
                        index = i + 1,
                        label = obj.label,
                        focusDistanceDiopters = diopters,
                        focusDistanceMeters = meters,
                        focusDurationMs = applyDurationMs.toLong(),
                        captureDurationMs = captureDurationMs
                    )
                )

            } catch (e: Exception) {
                Log.e(TAG, "Error capturing manual frame $i", e)
            }
        }

        // ===== Снимаем ручные ограничения, возвращаем авто =====
        cam2Control?.setCaptureRequestOptions(
            CaptureRequestOptions.Builder()
                .setCaptureRequestOption(
                    CaptureRequest.CONTROL_AF_MODE,
                    CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                )
                .setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_MODE,
                    CameraMetadata.CONTROL_AE_MODE_ON
                )
                .build()
        )

        // ===== Дожидаемся записи всех файлов =====
        val waitStart = SystemClock.elapsedRealtime()
        while (savesRemaining.get() > 0) {
            Thread.sleep(10)
            if (SystemClock.elapsedRealtime() - waitStart > 15_000) {
                Log.w(TAG, "Timeout waiting for disk saves, remaining=${savesRemaining.get()}")
                break
            }
        }

        val capturedPaths = ArrayList<String>(n)
        val focusPoints = ArrayList<Pair<Float, Float>>(n)
        for (idx in 0 until n) {
            val p = pathSlots[idx] ?: continue
            capturedPaths.add(p)
            focusSlots[idx]?.let { focusPoints.add(it) }
        }

        PipelineLogger.writeJson(this)

        runOnUiThread {
            overlayView.setCapturingMode(false)
        }

        if (capturedPaths.size >= 2) {
            if (AppSettings.isCaptureOnly(this)) {
                // "Stop after capture": дальнейшей обработки нет — чистим точки здесь.
                runOnUiThread {
                    hideProgress()
                    captureIB.visibility = View.VISIBLE
                    folderIB.visibility = View.VISIBLE
                    overlayView.visibility = View.VISIBLE
                    isCapturing = false
                    clearManualPointsAll()
                    Toast.makeText(
                        this,
                        "Saved ${capturedPaths.size} frames to ${outputDir.name}. " +
                                "Process later via folder button.",
                        Toast.LENGTH_LONG
                    ).show()
                }
            } else {
                // Будет обработка (processImageStack) — точки больше не нужны, чистим сразу.
                runOnUiThread {
                    clearManualPointsAll()
                }
                processImageStack(capturedPaths, focusPoints)
            }
        } else {
            runOnUiThread {
                hideProgress()
                captureIB.visibility = View.VISIBLE
                folderIB.visibility = View.VISIBLE
                isCapturing = false
                clearManualPointsAll()
                Toast.makeText(this, "Not enough images captured", Toast.LENGTH_SHORT).show()
            }
        }

    }

    // ======================== CAPTURE ========================

    private fun startFocusStackCapture() {
        isManualMode = AppSettings.isManualMode(this)

        // Источник объектов зависит от режима
        val objectsSnapshot: List<DetectedObject> = if (isManualMode) {
            buildManualObjects()
        } else {
            detectedObjects
        }

        if (objectsSnapshot.isEmpty()) {
            Toast.makeText(this, "No points to focus on", Toast.LENGTH_SHORT).show()
            return
        }

        if (isManualMode) {
            val notConfigured = objectsSnapshot.indices.any { idx ->
                manualSettings[idx]?.configured != true
            }
            if (notConfigured) {
                Toast.makeText(
                    this,
                    "Не все точки настроены. Тапните по каждой и дождитесь фокусировки.",
                    Toast.LENGTH_LONG
                ).show()
                return
            }
            if (objectsSnapshot.size < 2) {
                Toast.makeText(this, "Нужно минимум 2 точки", Toast.LENGTH_SHORT).show()
                return
            }
        }

        isCapturing = true
        captureIB.visibility = View.GONE
        folderIB.visibility = View.GONE

        overlayView.setCapturingMode(true)
        showProgress(if (isManualMode) "Capturing (manual)..." else "Capturing focus stack...")

        val sessionTimestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

        if (isManualMode) {
            // POINTS-кадр снимается ВНУТРИ captureSequenceManual ПЕРВЫМ,
            // пока камера ещё в авто-экспозиции (иначе кадр будет чёрным
            // из-за уже выставленных AE_OFF/ручной выдержки).
            processingExecutor.execute {
                captureSequenceManual(objectsSnapshot, sessionTimestamp)
            }
        } else {
            capturePointsPreview(objectsSnapshot, sessionTimestamp)
            processingExecutor.execute {
                captureSequence(objectsSnapshot, sessionTimestamp)
            }
        }
    }
    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun captureSequence(objects: List<DetectedObject>, timestamp: String) {
        val n = objects.size
        val pathSlots = arrayOfNulls<String>(n)
        val focusSlots = arrayOfNulls<Pair<Float, Float>>(n)
        val savesRemaining = java.util.concurrent.atomic.AtomicInteger(0)

        PipelineLogger.startSession()

        for ((i, obj) in objects.withIndex()) {
            runOnUiThread {
                updateProgress("Capturing ${i + 1}/${objects.size}: ${obj.label}")
            }

            try {
                // ===== Фокусировка =====
                val focusStart = System.currentTimeMillis()
                val focusLatch = java.util.concurrent.CountDownLatch(1)

                runOnUiThread {
                    try {
                        val meteringPoint = previewView.meteringPointFactory
                            .createPoint(obj.cx, obj.cy)
                        val focusAction = FocusMeteringAction.Builder(meteringPoint)
                            .setAutoCancelDuration(3, java.util.concurrent.TimeUnit.SECONDS)
                            .build()

                        val future = camera?.cameraControl?.startFocusAndMetering(focusAction)
                        if (future != null) {
                            future.addListener({ focusLatch.countDown() }, cameraExecutor)
                        } else {
                            focusLatch.countDown()
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Focus metering error", e)
                        focusLatch.countDown()
                    }
                }
                focusLatch.await(2, java.util.concurrent.TimeUnit.SECONDS)
                val focusDurationMs = System.currentTimeMillis() - focusStart

                // ===== Фиксируем AE/AWB, чтобы takePicture не гонял precapture заново =====
                val cam2Control = camera?.cameraControl?.let { Camera2CameraControl.from(it) }
                cam2Control?.setCaptureRequestOptions(
                    CaptureRequestOptions.Builder()
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, true)
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, true)
                        .build()
                )

                // ===== Захват (камера освобождается, как только кадр в памяти) =====
                val filename = "DET_${timestamp}_${i + 1}_${obj.cx.toInt()}x${obj.cy.toInt()}.jpg"
                val file = File(outputDir, filename)

                val captureStart = SystemClock.elapsedRealtime()
                val captureLatch = java.util.concurrent.CountDownLatch(1)

                savesRemaining.incrementAndGet()

                imageCapture?.takePicture(
                    cameraExecutor,
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            // ВРЕМЯ: сенсор + JPEG-энкод, без диска
                            val capturedAt = SystemClock.elapsedRealtime()
                            val sensorEncodeMs = capturedAt - captureStart

                            // Копируем байты и сразу отпускаем буфер + камеру
                            val rotation = image.imageInfo.rotationDegrees
                            val bytes = image.toJpegBytes()
                            image.close()
                            captureLatch.countDown()

                            // Запись на диск — параллельно следующему захвату
                            saverExecutor.execute {
                                val saveStart = SystemClock.elapsedRealtime()
                                try {
                                    file.outputStream().use { it.write(bytes) }
                                    applyExifRotation(file, rotation)

                                    pathSlots[i] = file.absolutePath
                                    focusSlots[i] = Pair(obj.cx, obj.cy)

                                    val intent = Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE)
                                    intent.data = Uri.fromFile(file)
                                    sendBroadcast(intent)

                                    val saveEnd = SystemClock.elapsedRealtime()
                                    PipelineLogger.stage(i + 1, "sensor_encode", sensorEncodeMs)
                                    PipelineLogger.stage(i + 1, "disk_write", saveEnd - saveStart)
                                } catch (e: Exception) {
                                    Log.e(TAG, "Save failed for $filename", e)
                                } finally {
                                    savesRemaining.decrementAndGet()
                                }
                            }
                        }

                        override fun onError(e: ImageCaptureException) {
                            Log.e(TAG, "Capture failed for $filename", e)
                            savesRemaining.decrementAndGet()
                            captureLatch.countDown()
                        }
                    }
                )

                // Ждём только ЗАХВАТ, не запись на диск
                captureLatch.await(10, java.util.concurrent.TimeUnit.SECONDS)
                val captureEnd = SystemClock.elapsedRealtime()
                val captureDurationMs = captureEnd - captureStart

                // ===== Снимаем блокировку перед следующим кадром (другая сцена/дистанция) =====
                cam2Control?.setCaptureRequestOptions(
                    CaptureRequestOptions.Builder()
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, false)
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, false)
                        .build()
                )

                PipelineLogger.stage(i + 1, "focus_settle", focusDurationMs)

                // ===== Лог кадра =====
                val diopters = lastFocusDistance
                val meters = diopters?.let { if (it > 0f) 1f / it else null }
                PipelineLogger.logFrame(
                    PipelineLogger.FrameLog(
                        index = i + 1,
                        label = obj.label,
                        focusDistanceDiopters = diopters,
                        focusDistanceMeters = meters,
                        focusDurationMs = focusDurationMs,
                        captureDurationMs = captureDurationMs
                    )
                )

            } catch (e: Exception) {
                Log.e(TAG, "Error capturing frame $i", e)
            }
        }

        // ===== Дожидаемся, что ВСЕ файлы дописались на диск =====
        val waitStart = SystemClock.elapsedRealtime()
        while (savesRemaining.get() > 0) {
            Thread.sleep(10)
            if (SystemClock.elapsedRealtime() - waitStart > 15_000) {
                Log.w(TAG, "Timeout waiting for disk saves, remaining=${savesRemaining.get()}")
                break
            }
        }

        // Собираем результаты в порядке кадров, отбрасывая неудавшиеся
        val capturedPaths = ArrayList<String>(n)
        val focusPoints = ArrayList<Pair<Float, Float>>(n)
        for (idx in 0 until n) {
            val p = pathSlots[idx] ?: continue
            capturedPaths.add(p)
            focusSlots[idx]?.let { focusPoints.add(it) }
        }

        // Пишем JSON по завершении съёмки всех кадров
        PipelineLogger.writeJson(this)

        runOnUiThread {
            overlayView.setCapturingMode(false)
        }

        if (capturedPaths.size >= 2) {
            if (AppSettings.isCaptureOnly(this)) {
                runOnUiThread {
                    hideProgress()
                    captureIB.visibility = View.VISIBLE
                    folderIB.visibility = View.VISIBLE
                    overlayView.visibility = View.VISIBLE
                    isCapturing = false
                    Toast.makeText(
                        this,
                        "Saved ${capturedPaths.size} frames to ${outputDir.name}. " +
                                "Process later via folder button.",
                        Toast.LENGTH_LONG
                    ).show()
                }
            } else {
                processImageStack(capturedPaths, focusPoints)
            }
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
    private fun ImageProxy.toJpegBytes(): ByteArray {
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return bytes
    }

    private fun applyExifRotation(file: File, rotationDegrees: Int) {
        val orientation = when (rotationDegrees) {
            90 -> ExifInterface.ORIENTATION_ROTATE_90
            180 -> ExifInterface.ORIENTATION_ROTATE_180
            270 -> ExifInterface.ORIENTATION_ROTATE_270
            else -> ExifInterface.ORIENTATION_NORMAL
        }
        val exif = ExifInterface(file.absolutePath)
        exif.setAttribute(ExifInterface.TAG_ORIENTATION, orientation.toString())
        exif.saveAttributes()
    }
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
    override fun onResume() {
        super.onResume()
        applyCaptureMode()
    }
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        processingExecutor.shutdown()
        mlKitDetector?.close()
        saverExecutor.shutdown()
        pointsShotExecutor.shutdown()
    }
}