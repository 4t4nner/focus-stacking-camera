package com.example.cameraapp

import android.app.AlertDialog
import android.content.Context
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.widget.*
import com.google.android.material.textfield.TextInputEditText
import kotlin.concurrent.thread

/**
 * Dialog for editing remote server settings.
 * Settings are persisted in SharedPreferences via AppSettings.
 */
class SettingsDialog(
    private val context: Context,
    private val onSaved: () -> Unit
) {

    private val handler = Handler(Looper.getMainLooper())

    fun show() {
        val view = LayoutInflater.from(context)
            .inflate(R.layout.dialog_settings, null)

        val remoteSwitch = view.findViewById<Switch>(R.id.remoteEnabledSwitch)
        val hostET = view.findViewById<TextInputEditText>(R.id.serverHostET)
        val portET = view.findViewById<TextInputEditText>(R.id.serverPortET)
        val statusDot = view.findViewById<View>(R.id.statusDot)
        val statusTV = view.findViewById<TextView>(R.id.statusTV)
        val testBtn = view.findViewById<Button>(R.id.testConnectionBtn)

        // Load current values
        remoteSwitch.isChecked = AppSettings.isRemoteEnabled(context)
        hostET.setText(AppSettings.getRemoteHost(context))
        portET.setText(AppSettings.getRemotePort(context).toString())

        // Enable/disable fields based on switch
        fun updateFieldsEnabled() {
            val enabled = remoteSwitch.isChecked
            hostET.isEnabled = enabled
            portET.isEnabled = enabled
            testBtn.isEnabled = enabled
            hostET.alpha = if (enabled) 1f else 0.4f
            portET.alpha = if (enabled) 1f else 0.4f
        }
        updateFieldsEnabled()

        remoteSwitch.setOnCheckedChangeListener { _, _ ->
            updateFieldsEnabled()
        }

        // Test connection
        testBtn.setOnClickListener {
            val host = hostET.text.toString().trim()
            val port = portET.text.toString().toIntOrNull() ?: 5000

            if (host.isEmpty()) {
                statusTV.text = "Enter host first"
                statusDot.setBackgroundResource(R.drawable.status_dot_red)
                return@setOnClickListener
            }

            statusTV.text = "Testing..."
            statusDot.setBackgroundResource(R.drawable.status_dot_gray)
            testBtn.isEnabled = false

            thread {
                val config = RemoteProcessor.RemoteConfig(host, port, true)
                val available = RemoteProcessor.isServerAvailable(config)

                handler.post {
                    testBtn.isEnabled = true
                    if (available) {
                        statusDot.setBackgroundResource(R.drawable.status_dot_green)
                        statusTV.text = "Connected ✓"
                        statusTV.setTextColor(Color.parseColor("#4CAF50"))
                    } else {
                        statusDot.setBackgroundResource(R.drawable.status_dot_red)
                        statusTV.text = "Unreachable ✗"
                        statusTV.setTextColor(Color.parseColor("#F44336"))
                    }
                }
            }
        }

        // Build dialog
        AlertDialog.Builder(context)
            .setView(view)
            .setPositiveButton("Save") { _, _ ->
                val host = hostET.text.toString().trim()
                val port = portET.text.toString().toIntOrNull() ?: 5000
                val enabled = remoteSwitch.isChecked

                AppSettings.save(context, host, port, enabled)
                onSaved()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}