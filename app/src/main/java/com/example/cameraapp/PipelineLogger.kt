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

    @Synchronized
    fun startSession() {
        frames.clear()
        stages.clear()
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
