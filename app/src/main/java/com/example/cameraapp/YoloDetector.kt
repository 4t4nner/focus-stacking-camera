package com.example.cameraapp

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
        // YOLOv8 output: [1, 84, N]
        // 84 = 4 (cx, cy, w, h) + 80 classes
        val numClasses = 80
        val rows = output.size(1).toInt()
        val cols = output.size(2).toInt()

        val data = FloatArray(rows * cols)
        output.reshape(1, 1)[0, 0, data]

        val boxes = mutableListOf<RectF>()
        val confidences = mutableListOf<Float>()
        val classIds = mutableListOf<Int>()

        for (i in 0 until cols) {
            val cx = data[0 * cols + i]
            val cy = data[1 * cols + i]
            val w  = data[2 * cols + i]
            val h  = data[3 * cols + i]

            var maxConf = 0f
            var maxIdx = 0
            for (c in 0 until numClasses) {
                val conf = data[(4 + c) * cols + i]
                if (conf > maxConf) {
                    maxConf = conf
                    maxIdx = c
                }
            }

            if (maxConf < confThreshold) continue

            val x1 = ((cx - w / 2f) - padLeft) / scale
            val y1 = ((cy - h / 2f) - padTop) / scale
            val x2 = ((cx + w / 2f) - padLeft) / scale
            val y2 = ((cy + h / 2f) - padTop) / scale

            boxes.add(RectF(
                max(0f, x1), max(0f, y1),
                min(origW.toFloat(), x2), min(origH.toFloat(), y2)
            ))
            confidences.add(maxConf)
            classIds.add(maxIdx)
        }

        val indices = nms(boxes, confidences, nmsThreshold)

        return indices.map { idx ->
            val b = boxes[idx]
            Detection(
                classId = classIds[idx],
                className = if (classIds[idx] < cocoNames.size) cocoNames[classIds[idx]] else "class${classIds[idx]}",
                confidence = confidences[idx],
                x1 = b.left, y1 = b.top,
                x2 = b.right, y2 = b.bottom
            )
        }
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
