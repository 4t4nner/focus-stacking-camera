package com.example.cameraapp

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var sourceWidth: Int = 1
    private var sourceHeight: Int = 1
    private var capturingMode: Boolean = false

    private val boxPaintGreen = Paint().apply {
        color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 6f; isAntiAlias = true
    }
    private val fillPaintGreen = Paint().apply { color = Color.GREEN; style = Paint.Style.FILL }

    private val textPaint = Paint().apply {
        color = Color.WHITE; textSize = 40f; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD
    }
    private val numberPaint = Paint().apply {
        color = Color.WHITE; textSize = 52f; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD
        textAlign = Paint.Align.CENTER
    }
    private val numberBgPaintGreen = Paint().apply {
        color = Color.GREEN; style = Paint.Style.FILL; isAntiAlias = true
    }
    private val numberBgPaintRed = Paint().apply {
        color = Color.RED; style = Paint.Style.FILL; isAntiAlias = true
    }

    private val pointPaintGreen = Paint().apply { color = Color.GREEN; style = Paint.Style.FILL; isAntiAlias = true }
    private val pointPaintRed = Paint().apply { color = Color.RED; style = Paint.Style.FILL; isAntiAlias = true }
    private val pointStroke = Paint().apply { style = Paint.Style.STROKE; color = Color.WHITE; strokeWidth = 4f; isAntiAlias = true }
    private val crossPaint = Paint().apply { color = Color.WHITE; strokeWidth = 3f; isAntiAlias = true }

    fun setDetections(dets: List<Detection>, srcWidth: Int, srcHeight: Int) {
        detections = dets
        sourceWidth = srcWidth
        sourceHeight = srcHeight
        postInvalidate()
    }

    fun setCapturingMode(capturing: Boolean) {
        capturingMode = capturing
        postInvalidate()
    }

    fun clear() {
        detections = emptyList()
        capturingMode = false
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val scaleX = viewW / sourceWidth.toFloat()
        val scaleY = viewH / sourceHeight.toFloat()
        val scale = maxOf(scaleX, scaleY)
        val offsetX = (viewW - sourceWidth * scale) / 2f
        val offsetY = (viewH - sourceHeight * scale) / 2f

        val pointPaint = if (capturingMode) pointPaintRed else pointPaintGreen
        val numberBgPaint = if (capturingMode) numberBgPaintRed else numberBgPaintGreen

        for ((index, det) in detections.withIndex()) {
            val left = det.x1 * scale + offsetX
            val top = det.y1 * scale + offsetY
            val right = det.x2 * scale + offsetX
            val bottom = det.y2 * scale + offsetY

            val cx = (left + right) / 2f
            val cy = (top + bottom) / 2f

            // Рамки — только в preview (не в режиме захвата)
            if (!capturingMode) {
                canvas.drawRect(left, top, right, bottom, boxPaintGreen)

                val label = "${det.className} ${(det.confidence * 100).toInt()}%"
                val textW = textPaint.measureText(label)
                val textH = 44f
                canvas.drawRect(left, top - textH - 8, left + textW + 12, top, fillPaintGreen)
                canvas.drawText(label, left + 6, top - 12, textPaint)
            }

            // Точка + крест — в обоих режимах
            canvas.drawCircle(cx, cy, 16f, pointPaint)
            canvas.drawCircle(cx, cy, 18f, pointStroke)
            canvas.drawLine(cx - 24, cy, cx + 24, cy, crossPaint)
            canvas.drawLine(cx, cy - 24, cx, cy + 24, crossPaint)

            // Номер рядом с точкой — в обоих режимах
            val numStr = "${index + 1}"
            val numW = numberPaint.measureText(numStr)
            val r = (maxOf(numW, 52f) / 2f) + 8f
            canvas.drawCircle(cx, cy - 50f, r, numberBgPaint)
            canvas.drawText(numStr, cx, cy - 35f, numberPaint)
        }
    }
}