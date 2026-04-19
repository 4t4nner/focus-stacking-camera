package com.example.cameraapp

import android.content.Context
import android.content.SharedPreferences

/**
 * Persistent settings stored in SharedPreferences.
 * Editable from Settings dialog in the app.
 */
object AppSettings {

    private const val PREFS_NAME = "cameraapp_settings"
    private const val KEY_REMOTE_SERVER = "remote_server"
    private const val KEY_REMOTE_PORT = "remote_port"
    private const val KEY_REMOTE_ENABLED = "remote_enabled"

    private fun prefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
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

    fun save(context: Context, host: String, port: Int, enabled: Boolean) {
        prefs(context).edit()
            .putString(KEY_REMOTE_SERVER, host.trim())
            .putInt(KEY_REMOTE_PORT, port)
            .putBoolean(KEY_REMOTE_ENABLED, enabled)
            .apply()
    }

    /**
     * Build RemoteConfig from saved settings.
     */
    fun getRemoteConfig(context: Context): RemoteProcessor.RemoteConfig {
        val host = getRemoteHost(context)
        val port = getRemotePort(context)
        val enabled = isRemoteEnabled(context) && host.isNotBlank()
        return RemoteProcessor.RemoteConfig(host, port, enabled)
    }
}