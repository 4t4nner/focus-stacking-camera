package com.example.cameraapp

import android.app.AlertDialog
import android.content.Context
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.widget.*
import com.google.android.material.textfield.TextInputEditText
import kotlin.concurrent.thread

class SettingsDialog(
    private val context: Context,
    private val onSaved: () -> Unit
) {

    private val handler = Handler(Looper.getMainLooper())

    fun show() {
        val view = LayoutInflater.from(context)
            .inflate(R.layout.dialog_settings, null)

        val captureOnlySwitch = view.findViewById<Switch>(R.id.captureOnlySwitch)
        captureOnlySwitch.isChecked = AppSettings.isCaptureOnly(context)

        val detectorRadioGroup = view.findViewById<RadioGroup>(R.id.detectorRadioGroup)
        val detectorMlKitRB = view.findViewById<RadioButton>(R.id.detectorMlKitRB)
        val detectorYoloRB = view.findViewById<RadioButton>(R.id.detectorYoloRB)

        val captureModeRadioGroup = view.findViewById<RadioGroup>(R.id.captureModeRadioGroup)
        val captureModeAutoRB = view.findViewById<RadioButton>(R.id.captureModeAutoRB)
        val captureModeManualRB = view.findViewById<RadioButton>(R.id.captureModeManualRB)
        val remoteSwitch = view.findViewById<Switch>(R.id.remoteEnabledSwitch)
        val hostET = view.findViewById<TextInputEditText>(R.id.serverHostET)
        val portET = view.findViewById<TextInputEditText>(R.id.serverPortET)
        val statusDot = view.findViewById<android.view.View>(R.id.statusDot)
        val statusTV = view.findViewById<TextView>(R.id.statusTV)
        val testBtn = view.findViewById<Button>(R.id.testConnectionBtn)

        // Load detector value ← NEW
        when (AppSettings.getDetector(context)) {
            AppSettings.DETECTOR_YOLO -> detectorYoloRB.isChecked = true
            else -> detectorMlKitRB.isChecked = true
        }

        // Load capture mode
        when (AppSettings.getCaptureMode(context)) {
            AppSettings.MODE_MANUAL -> captureModeManualRB.isChecked = true
            else -> captureModeAutoRB.isChecked = true
        }

        // Load current values
        remoteSwitch.isChecked = AppSettings.isRemoteEnabled(context)
        hostET.setText(AppSettings.getRemoteHost(context))
        portET.setText(AppSettings.getRemotePort(context).toString())

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

        AlertDialog.Builder(context)
            .setView(view)
            .setPositiveButton("Save") { _, _ ->
                AppSettings.setCaptureOnly(context, captureOnlySwitch.isChecked)

                // Save detector
                val detector = if (detectorYoloRB.isChecked)
                    AppSettings.DETECTOR_YOLO else AppSettings.DETECTOR_MLKIT
                AppSettings.setDetector(context, detector)

                // Save capture mode
                val mode = if (captureModeManualRB.isChecked)
                    AppSettings.MODE_MANUAL else AppSettings.MODE_AUTO
                AppSettings.setCaptureMode(context, mode)

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