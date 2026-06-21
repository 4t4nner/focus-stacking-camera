package com.example.cameraapp

import android.content.Context
import android.content.SharedPreferences

object AppSettings {

    private const val PREFS_NAME = "cameraapp_settings"
    private const val KEY_REMOTE_SERVER = "remote_server"
    private const val KEY_REMOTE_PORT = "remote_port"
    private const val KEY_REMOTE_ENABLED = "remote_enabled"
    private const val KEY_DETECTOR = "detector_type"  // ← NEW
    private const val KEY_CAPTURE_ONLY = "capture_only"
    private const val KEY_CAPTURE_MODE = "capture_mode"

    // Значения режима съёмки
    const val MODE_AUTO = "auto"
    const val MODE_MANUAL = "manual"

    // Значения детектора
    const val DETECTOR_MLKIT = "mlkit"
    const val DETECTOR_YOLO = "yolo"

    private fun prefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    fun isCaptureOnly(context: Context): Boolean {
        return prefs(context).getBoolean(KEY_CAPTURE_ONLY, false)
    }

    fun setCaptureOnly(context: Context, value: Boolean) {
        prefs(context).edit().putBoolean(KEY_CAPTURE_ONLY, value).apply()
    }

    fun getRemoteHost(context: Context): String {
        return prefs(context).getString(KEY_REMOTE_SERVER, "") ?: ""
    }

    fun getRemotePort(context: Context): Int {
        return prefs(context).getInt(KEY_REMOTE_PORT, 5000)
    }

    fun isRemoteEnabled(context: Context): Boolean {
        return prefs(context).getBoolean(KEY_REMOTE_ENABLED, false)
    }

    // получить выбранный детектор (по умолчанию MLKit)
    fun getDetector(context: Context): String {
        return prefs(context).getString(KEY_DETECTOR, DETECTOR_MLKIT) ?: DETECTOR_MLKIT
    }

    //
    fun setDetector(context: Context, detector: String) {
        prefs(context).edit().putString(KEY_DETECTOR, detector).apply()
    }
    // Режим съёмки (по умолчанию AUTO — сохраняем текущее поведение)
    fun getCaptureMode(context: Context): String {
        return prefs(context).getString(KEY_CAPTURE_MODE, MODE_AUTO) ?: MODE_AUTO
    }

    fun setCaptureMode(context: Context, mode: String) {
        prefs(context).edit().putString(KEY_CAPTURE_MODE, mode).apply()
    }

    fun isManualMode(context: Context): Boolean {
        return getCaptureMode(context) == MODE_MANUAL
    }

    fun save(context: Context, host: String, port: Int, enabled: Boolean) {
        prefs(context).edit()
            .putString(KEY_REMOTE_SERVER, host.trim())
            .putInt(KEY_REMOTE_PORT, port)
            .putBoolean(KEY_REMOTE_ENABLED, enabled)
            .apply()
    }

    fun getRemoteConfig(context: Context): RemoteProcessor.RemoteConfig {
        val host = getRemoteHost(context)
        val port = getRemotePort(context)
        val enabled = isRemoteEnabled(context) && host.isNotBlank()
        return RemoteProcessor.RemoteConfig(host, port, enabled)
    }
}