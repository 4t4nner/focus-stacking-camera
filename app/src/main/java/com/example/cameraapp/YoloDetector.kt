package com.example.cameraapp

import android.util.Log
import android.content.Context
import android.graphics.*
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

class YoloDetector(private val context: Context) {

    private lateinit var net: Net
    private var initialized = false

    private val inputSize = 640

    private val confThreshold = 0.4f
    private val nmsThreshold = 0.45f

    private val cocoNames = arrayOf(
        "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
        "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
        "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush"
    )

    fun initialize(): Boolean {
        if (initialized) return true
        if (!OpenCVLoader.initLocal()) return false

        // YOLOv8n — поддерживается OpenCV DNN (без attention блоков)
        val modelFile = copyAssetToFile("yolov10m.onnx") ?: return false
        net = Dnn.readNetFromONNX(modelFile.absolutePath)
        if (net.empty()) return false

        initialized = true
        return true
    }

    private fun copyAssetToFile(assetName: String): File? {
        return try {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) return file
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
            file
        } catch (e: Exception) { null }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        if (!initialized) return emptyList()

        val mat = Mat()
        val bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        Utils.bitmapToMat(bmp32, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)

        val imgW = mat.cols()
        val imgH = mat.rows()

        val scale = min(inputSize.toFloat() / imgW, inputSize.toFloat() / imgH)
        val newW = (imgW * scale).toInt()
        val newH = (imgH * scale).toInt()
        val padLeft = (inputSize - newW) / 2
        val padTop = (inputSize - newH) / 2

        val resized = Mat()
        Imgproc.resize(mat, resized, Size(newW.toDouble(), newH.toDouble()))

        val padded = Mat(inputSize, inputSize, resized.type(), Scalar(114.0, 114.0, 114.0))
        resized.copyTo(
            padded.submat(padTop, padTop + newH, padLeft, padLeft + newW)
        )

        val blob = Dnn.blobFromImage(
            padded,
            1.0 / 255.0,
            Size(inputSize.toDouble(), inputSize.toDouble()),
            Scalar(0.0),
            true,
            false
        )

        net.setInput(blob)
        val output = net.forward()
        Log.d("YoloDetector", "output shape: ${output.size(0)} x ${output.size(1)} x ${output.size(2)}")

        val detections = parseOutput(output, imgW, imgH, scale, padLeft, padTop)

        mat.release()
        resized.release()
        padded.release()
        blob.release()
        output.release()

        return detections
    }

    private fun parseOutput(
        output: Mat, origW: Int, origH: Int,
        scale: Float,
        padLeft: Int,
        padTop: Int
    ): List<Detection> {
        // YOLOv10 output: [1, 300, 6] -> [x1, y1, x2, y2, score, classId]
        // Координаты уже в пространстве inputSize (640x640), NMS не нужен.

        val numDet = output.size(1).toInt()   // 300
        val step = output.size(2).toInt()     // 6

        val data = FloatArray(numDet * step)
        output.reshape(1, 1)[0, 0, data]

        val result = mutableListOf<Detection>()

        for (i in 0 until numDet) {
            val base = i * step
            val score = data[base + 4]
            if (score < confThreshold) continue

            val classId = data[base + 5].toInt()

            // координаты в пространстве 640x640 -> в оригинал
            val x1 = ((data[base + 0] - padLeft) / scale).coerceIn(0f, origW.toFloat())
            val y1 = ((data[base + 1] - padTop) / scale).coerceIn(0f, origH.toFloat())
            val x2 = ((data[base + 2] - padLeft) / scale).coerceIn(0f, origW.toFloat())
            val y2 = ((data[base + 3] - padTop) / scale).coerceIn(0f, origH.toFloat())

            result.add(
                Detection(
                    classId = classId,
                    className = if (classId in cocoNames.indices) cocoNames[classId] else "class$classId",
                    confidence = score,
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2
                )
            )
        }

        return result
    }

    private fun nms(boxes: List<RectF>, scores: List<Float>, threshold: Float): List<Int> {
        val indices = scores.indices.sortedByDescending { scores[it] }.toMutableList()
        val result = mutableListOf<Int>()

        while (indices.isNotEmpty()) {
            val best = indices.removeAt(0)
            result.add(best)

            indices.removeAll { idx ->
                iou(boxes[best], boxes[idx]) > threshold
            }
        }
        return result
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)

        val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)

        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)

        return interArea / (aArea + bArea - interArea + 1e-6f)
    }
}
