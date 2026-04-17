package com.example.cameraapp

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.ObjectDetector
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions

class MlKitDetector {

    // Детектор для одиночных кадров (съёмка)
    private val singleDetector: ObjectDetector = ObjectDetection.getClient(
        ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
            .enableMultipleObjects()
            .enableClassification()
            .build()
    )

    // Детектор для потока (live preview)
    private val streamDetector: ObjectDetector = ObjectDetection.getClient(
        ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
            .enableMultipleObjects()
            .enableClassification()
            .build()
    )

    fun detect(bitmap: Bitmap): List<Detection> {
        return detectWith(singleDetector, bitmap)
    }

    fun detectStream(bitmap: Bitmap): List<Detection> {
        return detectWith(streamDetector, bitmap)
    }

    private fun detectWith(detector: ObjectDetector, bitmap: Bitmap): List<Detection> {
        val image = InputImage.fromBitmap(bitmap, 0)
        return try {
            val detectedObjects = Tasks.await(detector.process(image))
            detectedObjects.map { obj ->
                val box = obj.boundingBox
                val label = obj.labels.firstOrNull()
                Detection(
                    classId = label?.index ?: -1,
                    className = label?.text ?: "Unknown",
                    confidence = label?.confidence ?: 0f,
                    x1 = box.left.toFloat(),
                    y1 = box.top.toFloat(),
                    x2 = box.right.toFloat(),
                    y2 = box.bottom.toFloat()
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        }
    }

    fun close() {
        singleDetector.close()
        streamDetector.close()
    }
}