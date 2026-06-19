package com.example.cameraapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.json.JSONObject
import java.io.*
import java.net.HttpURLConnection
import java.net.InetSocketAddress
import java.net.Socket
import java.net.URL

/**
 * Handles remote focus stacking via async job queue:
 *   1. POST /api/focus-stack → получаем job_id (мгновенно)
 *   2. GET /api/status/<job_id> → polling пока status != "done"
 *   3. GET /api/result/<job_id> → скачиваем JPEG
 */
object RemoteProcessor {

    private const val TAG = "RemoteProcessor"
    private const val PING_TIMEOUT_MS = 3000
    private const val UPLOAD_TIMEOUT_MS = 60_000
    private const val POLL_TIMEOUT_MS = 10_000
    private const val DOWNLOAD_TIMEOUT_MS = 120_000
    private const val POLL_INTERVAL_MS = 2000L
    private const val MAX_POLL_ATTEMPTS = 300  // 300 * 2s = 10 минут макс

    data class RemoteConfig(
        val serverHost: String,
        val serverPort: Int,
        val enabled: Boolean
    )

    /**
     * Callback для обновления прогресса на UI.
     */
    interface ProgressCallback {
        fun onProgress(message: String)
    }

    /**
     * Check if remote server is reachable.
     */
    fun isServerAvailable(config: RemoteConfig): Boolean {
        if (!config.enabled) return false

        return try {
            val socket = Socket()
            socket.connect(
                InetSocketAddress(config.serverHost, config.serverPort),
                PING_TIMEOUT_MS
            )
            socket.close()

            val url = URL("http://${config.serverHost}:${config.serverPort}/ping")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = PING_TIMEOUT_MS
            conn.readTimeout = PING_TIMEOUT_MS
            conn.requestMethod = "GET"

            val code = conn.responseCode
            conn.disconnect()

            val available = code == 200
            Log.d(TAG, "Server ping: $available (HTTP $code)")
            available

        } catch (e: Exception) {
            Log.d(TAG, "Server unavailable: ${e.message}")
            false
        }
    }

    /**
     * Send images to remote server for focus stacking (async job queue).
     *
     * @param config server configuration
     * @param imagePaths list of image file paths
     * @param focusPoints list of (cx, cy) focus points
     * @param progressCallback optional callback for status updates
     * @return composite Bitmap or null if failed
     */
    fun processRemotely(
        config: RemoteConfig,
        imagePaths: List<String>,
        focusPoints: List<Pair<Float, Float>>,
        analysisWidth: Int,      // ДОБАВЛЕНО
        analysisHeight: Int,     // ДОБАВЛЕНО
        progressCallback: ProgressCallback? = null
    ): Bitmap? {
        if (!config.enabled) return null
        try {
            progressCallback?.onProgress("Uploading ${imagePaths.size} images...")
            val jobId = submitJob(config, imagePaths, focusPoints,
                analysisWidth, analysisHeight)

            if (jobId == null) {
                Log.e(TAG, "Failed to submit job")
                return null
            }

            Log.d(TAG, "Job submitted: $jobId")
            progressCallback?.onProgress("Job submitted: $jobId")

            // ========== STEP 2: Poll for completion ==========
            val success = pollUntilDone(config, jobId, progressCallback)

            if (!success) {
                Log.e(TAG, "Job failed or timed out: $jobId")
                return null
            }

            // ========== STEP 3: Download result ==========
            progressCallback?.onProgress("Downloading result...")
            val bitmap = downloadResult(config, jobId)

            if (bitmap != null) {
                Log.d(TAG, "Received composite: ${bitmap.width}x${bitmap.height}")
                progressCallback?.onProgress("Done! ${bitmap.width}x${bitmap.height}")
            } else {
                Log.e(TAG, "Failed to download result for job $jobId")
            }

            return bitmap

        } catch (e: Exception) {
            Log.e(TAG, "Remote processing failed", e)
            return null
        }
    }

    // ======================== STEP 1: SUBMIT ========================

    private fun submitJob(
        config: RemoteConfig,
        imagePaths: List<String>,
        focusPoints: List<Pair<Float, Float>>,
        analysisWidth: Int,
        analysisHeight: Int
    ): String? {
        val boundary = "----FocusStack${System.currentTimeMillis()}"
        val url = URL("http://${config.serverHost}:${config.serverPort}/api/focus-stack")
        val conn = url.openConnection() as HttpURLConnection

        try {
            conn.doOutput = true
            conn.doInput = true
            conn.requestMethod = "POST"
            conn.connectTimeout = UPLOAD_TIMEOUT_MS
            conn.readTimeout = UPLOAD_TIMEOUT_MS
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")

            Log.d(TAG, "Uploading ${imagePaths.size} images to ${config.serverHost}:${config.serverPort}")

            conn.outputStream.buffered().use { out ->
                val writer = PrintWriter(OutputStreamWriter(out, Charsets.UTF_8), true)

                for ((i, path) in imagePaths.withIndex()) {
                    val file = File(path)
                    if (!file.exists()) continue

                    writer.append("--$boundary\r\n")
                    writer.append("Content-Disposition: form-data; name=\"files\"; filename=\"${file.name}\"\r\n")
                    writer.append("Content-Type: image/jpeg\r\n")
                    writer.append("\r\n")
                    writer.flush()

                    FileInputStream(file).use { fis ->
                        fis.copyTo(out)
                    }
                    out.flush()
                    writer.append("\r\n")
                    writer.flush()

                    Log.d(TAG, "  Uploaded image ${i + 1}/${imagePaths.size}: ${file.name}")
                }

                // Focus points
                // Focus points — отправляем НОРМИРОВАННЫЕ координаты [0..1]
                val aw = if (analysisWidth > 0) analysisWidth.toFloat() else 1f
                val ah = if (analysisHeight > 0) analysisHeight.toFloat() else 1f
                val fpJson = focusPoints.joinToString(",", "[", "]") { (cx, cy) ->
                    val nx = (cx / aw).coerceIn(0f, 1f)
                    val ny = (cy / ah).coerceIn(0f, 1f)
                    """{"nx":$nx,"ny":$ny}"""
                }
                writer.append("--$boundary\r\n")
                writer.append("Content-Disposition: form-data; name=\"focus_points\"\r\n")
                writer.append("\r\n")
                writer.append(fpJson)
                writer.append("\r\n")

                writer.append("--$boundary--\r\n")
                writer.flush()
            }

            val responseCode = conn.responseCode
            Log.d(TAG, "Submit response: $responseCode")

            if (responseCode == 202 || responseCode == 200) {
                val responseBody = conn.inputStream.bufferedReader().readText()
                conn.disconnect()

                Log.d(TAG, "Submit response body: $responseBody")

                val json = JSONObject(responseBody)
                return json.optString("job_id", null)
            } else {
                val error = try {
                    conn.errorStream?.bufferedReader()?.readText() ?: "Unknown"
                } catch (_: Exception) { "Unknown" }
                Log.e(TAG, "Submit error $responseCode: $error")
                conn.disconnect()
                return null
            }

        } catch (e: Exception) {
            Log.e(TAG, "Submit failed", e)
            try { conn.disconnect() } catch (_: Exception) {}
            return null
        }
    }

    // ======================== STEP 2: POLL ========================

    private fun pollUntilDone(
        config: RemoteConfig,
        jobId: String,
        progressCallback: ProgressCallback?
    ): Boolean {
        val baseUrl = "http://${config.serverHost}:${config.serverPort}/api/status/$jobId"

        for (attempt in 1..MAX_POLL_ATTEMPTS) {
            try {
                Thread.sleep(POLL_INTERVAL_MS)

                val url = URL(baseUrl)
                val conn = url.openConnection() as HttpURLConnection
                conn.connectTimeout = POLL_TIMEOUT_MS
                conn.readTimeout = POLL_TIMEOUT_MS
                conn.requestMethod = "GET"

                val responseCode = conn.responseCode

                if (responseCode == 200) {
                    val body = conn.inputStream.bufferedReader().readText()
                    conn.disconnect()

                    val json = JSONObject(body)
                    val status = json.optString("status", "unknown")
                    val step = json.optString("step", "")

                    Log.d(TAG, "Poll #$attempt: status=$status, step=$step")

                    // Обновляем UI
                    val displayStep = when (step) {
                        "queued" -> "Queued..."
                        "saving_inputs" -> "Saving inputs..."
                        "generating_masks" -> "Generating detail masks..."
                        "aligning" -> "Aligning images..."
                        "building_focus_map" -> "Building focus map..."
                        "compositing" -> "Compositing..."
                        "blending_seams" -> "Blending seams..."
                        "complete" -> "Complete!"
                        "failed" -> "Failed"
                        else -> "Processing ($step)..."
                    }
                    progressCallback?.onProgress("Server: $displayStep")

                    when (status) {
                        "done" -> return true
                        "error" -> {
                            val error = json.optString("error", "Unknown error")
                            Log.e(TAG, "Job error: $error")
                            progressCallback?.onProgress("Server error: $error")
                            return false
                        }
                        "processing", "queued" -> {
                            // Продолжаем polling
                        }
                        else -> {
                            Log.w(TAG, "Unknown status: $status")
                        }
                    }
                } else if (responseCode == 404) {
                    conn.disconnect()
                    Log.e(TAG, "Job not found: $jobId")
                    return false
                } else {
                    conn.disconnect()
                    Log.w(TAG, "Poll got HTTP $responseCode, retrying...")
                }

            } catch (e: InterruptedException) {
                Log.w(TAG, "Polling interrupted")
                return false
            } catch (e: Exception) {
                Log.w(TAG, "Poll error (attempt $attempt): ${e.message}")
                // Продолжаем пробовать
            }
        }

        Log.e(TAG, "Polling timed out after $MAX_POLL_ATTEMPTS attempts")
        progressCallback?.onProgress("Server timeout")
        return false
    }

    // ======================== STEP 3: DOWNLOAD ========================

    private fun downloadResult(config: RemoteConfig, jobId: String): Bitmap? {
        val url = URL("http://${config.serverHost}:${config.serverPort}/api/result/$jobId")
        val conn = url.openConnection() as HttpURLConnection

        try {
            conn.connectTimeout = DOWNLOAD_TIMEOUT_MS
            conn.readTimeout = DOWNLOAD_TIMEOUT_MS
            conn.requestMethod = "GET"

            val responseCode = conn.responseCode
            Log.d(TAG, "Download response: $responseCode")

            if (responseCode == 200) {
                // Читаем в ByteArray чтобы избежать проблем с chunked streams
                val bytes = conn.inputStream.use { it.readBytes() }
                conn.disconnect()

                Log.d(TAG, "Downloaded ${bytes.size} bytes")

                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                if (bitmap == null) {
                    Log.e(TAG, "Failed to decode bitmap from ${bytes.size} bytes")
                }
                return bitmap

            } else if (responseCode == 409) {
                val body = try {
                    conn.errorStream?.bufferedReader()?.readText() ?: ""
                } catch (_: Exception) { "" }
                Log.e(TAG, "Result not ready: $body")
                conn.disconnect()
                return null
            } else {
                val error = try {
                    conn.errorStream?.bufferedReader()?.readText() ?: "Unknown"
                } catch (_: Exception) { "Unknown" }
                Log.e(TAG, "Download error $responseCode: $error")
                conn.disconnect()
                return null
            }

        } catch (e: Exception) {
            Log.e(TAG, "Download failed", e)
            try { conn.disconnect() } catch (_: Exception) {}
            return null
        }
    }
}