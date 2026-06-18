package com.example.cameraapp

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.app.ActivityManager
import android.os.Debug

/** Снимок использования памяти. Все значения в мегабайтах. */
data class MemoryLog(
    val tag: String,            // метка момента ("app_start", "after_processing", ...)
    // Память процесса (наиболее показательна)
    val totalPssMb: Float,      // Proportional Set Size — суммарная память процесса
    val dalvikPssMb: Float,     // Java/Kotlin часть
    val nativePssMb: Float,     // native (OpenCV, ONNX, ML Kit)
    // JVM heap
    val javaHeapUsedMb: Float,
    val javaHeapMaxMb: Float,   // лимит heap
    // Устройство в целом
    val deviceTotalMb: Float,
    val deviceAvailMb: Float,
    val deviceLowMemory: Boolean,
    val memoryClassMb: Int      // разрешённый размер heap для процесса
)


/**
 * Собирает метрики пайплайна (фокусное расстояние, длительности по участкам)
 * и пишет их в logcat + в JSON-файл в Pictures/CameraApp.
 */
object PipelineLogger {

    private const val TAG = "PipelineLog"

    /** Метрики одного кадра на этапе съёмки. */
    data class FrameLog(
        val index: Int,
        val label: String,
        val focusDistanceDiopters: Float?, // LENS_FOCUS_DISTANCE, 1/м; null = неизвестно
        val focusDistanceMeters: Float?,   // пересчёт в метры; null = бесконечность/неизвестно
        val focusDurationMs: Long,         // длительность фокусировки
        val captureDurationMs: Long        // длительность захвата (takePicture)
    )

    /** Метрики одного участка обработки. */
    data class StageLog(val stage: String, val durationMs: Long)

    private val frames = mutableListOf<FrameLog>()
    private val stages = mutableListOf<StageLog>()
    private var sessionStart = 0L

    private val memorySnapshots = mutableListOf<MemoryLog>()

    @Synchronized
    fun logMemory(context: Context, tag: String): MemoryLog {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

        // --- Память процесса (PSS) ---
        val debugInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(debugInfo)
        val totalPss  = debugInfo.totalPss / 1024f           // килобайты -> МБ
        val dalvikPss = debugInfo.dalvikPss / 1024f
        val nativePss = debugInfo.nativePss / 1024f

        // --- JVM heap ---
        val rt = Runtime.getRuntime()
        val javaUsed = (rt.totalMemory() - rt.freeMemory()) / (1024f * 1024f)
        val javaMax  = rt.maxMemory() / (1024f * 1024f)

        // --- Память устройства ---
        val devInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(devInfo)
        val devTotal = devInfo.totalMem / (1024f * 1024f)
        val devAvail = devInfo.availMem / (1024f * 1024f)

        val log = MemoryLog(
            tag = tag,
            totalPssMb = totalPss,
            dalvikPssMb = dalvikPss,
            nativePssMb = nativePss,
            javaHeapUsedMb = javaUsed,
            javaHeapMaxMb = javaMax,
            deviceTotalMb = devTotal,
            deviceAvailMb = devAvail,
            deviceLowMemory = devInfo.lowMemory,
            memoryClassMb = am.memoryClass
        )

        memorySnapshots.add(log)
        Log.d(
            TAG,
            "MEMORY [$tag]: PSS=%.1fMB (dalvik=%.1f native=%.1f), " .format(totalPss, dalvikPss, nativePss) +
                    "javaHeap=%.1f/%.1fMB, device avail=%.0f/%.0fMB%s"
                        .format(javaUsed, javaMax, devAvail, devTotal,
                            if (devInfo.lowMemory) " [LOW]" else "")
        )
        return log
    }

    @Synchronized
    fun startSession() {
        frames.clear()
        stages.clear()
        memorySnapshots.clear()
        sessionStart = System.currentTimeMillis()
        Log.d(TAG, "=== Session started ===")
    }

    @Synchronized
    fun logFrame(frame: FrameLog) {
        frames.add(frame)
        val meters = frame.focusDistanceMeters?.let { "%.3f m".format(it) } ?: "∞/unknown"
        Log.d(
            TAG,
            "FRAME #${frame.index} '${frame.label}': " +
                "focusDist=${frame.focusDistanceDiopters} dpt ($meters), " +
                "focus=${frame.focusDurationMs}ms, capture=${frame.captureDurationMs}ms"
        )
    }

    @Synchronized
    fun logStage(stage: String, durationMs: Long) {
        stages.add(StageLog(stage, durationMs))
        Log.d(TAG, "STAGE '$stage': ${durationMs}ms")
    }

    /** Замеряет длительность участка пайплайна и логирует её. */
    inline fun <T> measureStage(stage: String, block: () -> T): T {
        val start = System.currentTimeMillis()
        val result = block()
        logStage(stage, System.currentTimeMillis() - start)
        return result
    }

    @Synchronized
    fun writeJson(context: Context): String? {
        return try {
            val root = JSONObject().apply {
                put("session_start", sessionStart)
                put("session_end", System.currentTimeMillis())
                put("total_duration_ms", System.currentTimeMillis() - sessionStart)
            }

            val framesArr = JSONArray()
            for (f in frames) {
                framesArr.put(JSONObject().apply {
                    put("index", f.index)
                    put("label", f.label)
                    put("focus_distance_diopters", f.focusDistanceDiopters ?: JSONObject.NULL)
                    put("focus_distance_meters", f.focusDistanceMeters ?: JSONObject.NULL)
                    put("focus_duration_ms", f.focusDurationMs)
                    put("capture_duration_ms", f.captureDurationMs)
                })
            }
            root.put("frames", framesArr)

            val stagesArr = JSONArray()
            for (s in stages) {
                stagesArr.put(JSONObject().apply {
                    put("stage", s.stage)
                    put("duration_ms", s.durationMs)
                })
            }
            root.put("pipeline_stages", stagesArr)

            val memArr = JSONArray()
            for (m in memorySnapshots) {
                memArr.put(JSONObject().apply {
                    put("tag", m.tag)
                    put("total_pss_mb", m.totalPssMb)
                    put("dalvik_pss_mb", m.dalvikPssMb)
                    put("native_pss_mb", m.nativePssMb)
                    put("java_heap_used_mb", m.javaHeapUsedMb)
                    put("java_heap_max_mb", m.javaHeapMaxMb)
                    put("device_total_mb", m.deviceTotalMb)
                    put("device_avail_mb", m.deviceAvailMb)
                    put("device_low_memory", m.deviceLowMemory)
                    put("memory_class_mb", m.memoryClassMb)
                })
            }
            root.put("memory_snapshots", memArr)

            val json = root.toString(2)
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val name = "pipeline_log_$ts.json"

            val location = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                writeViaMediaStore(context, name, json)
            } else {
                writeLegacy(name, json)
            }

            Log.d(TAG, "JSON log written: $location")
            Log.d(TAG, json)
            location
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write JSON log", e)
            null
        }
    }

    private fun writeViaMediaStore(context: Context, name: String, json: String): String {
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, name)
            put(MediaStore.Downloads.MIME_TYPE, "application/json")
            put(MediaStore.Downloads.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/CameraApp")
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: error("MediaStore.insert returned null")
        resolver.openOutputStream(uri).use { out ->
            out!!.write(json.toByteArray())
        }
        return uri.toString()
    }

    private fun writeLegacy(name: String, json: String): String {
        val dir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
            "CameraApp"
        )
        if (!dir.exists()) dir.mkdirs()
        val file = File(dir, name)
        file.writeText(json)
        return file.absolutePath
    }
    @Synchronized
    fun stage(frameIndex: Int, name: String, durationMs: Long) {
        stages.add(StageLog("frame_${frameIndex}_$name", durationMs))
    }
}
