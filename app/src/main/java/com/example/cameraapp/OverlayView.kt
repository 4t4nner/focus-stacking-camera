package com.example.cameraapp

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var sourceWidth: Int = 1
    private var sourceHeight: Int = 1
    private var capturingMode: Boolean = false

    // ===== РУЧНОЙ РЕЖИМ =====
    // Точка в системе координат кадра анализа (как DetectedObject.cx/cy).
    data class ManualPoint(val cx: Float, val cy: Float, var configured: Boolean = false)

    private var manualMode: Boolean = false
    private val manualPoints = mutableListOf<ManualPoint>()

    // Колбэк тапа: координаты в системе кадра анализа (sourceWidth x sourceHeight).
    var onManualTap: ((cx: Float, cy: Float) -> Unit)? = null

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
    private val numberBgPaintYellow = Paint().apply {
        color = Color.parseColor("#FFC107"); style = Paint.Style.FILL; isAntiAlias = true
    }

    private val pointPaintGreen = Paint().apply { color = Color.GREEN; style = Paint.Style.FILL; isAntiAlias = true }
    private val pointPaintRed = Paint().apply { color = Color.RED; style = Paint.Style.FILL; isAntiAlias = true }
    private val pointPaintYellow = Paint().apply { color = Color.parseColor("#FFC107"); style = Paint.Style.FILL; isAntiAlias = true }
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

    // ===== РУЧНОЙ РЕЖИМ: управление =====

    fun setManualMode(enabled: Boolean) {
        manualMode = enabled
        if (!enabled) manualPoints.clear()
        // В ручном режиме detections не приходят (детектор заглушён),
        // но source-размеры нужны для маппинга — задаём их извне через setAnalysisSize().
        postInvalidate()
    }

    /** Размеры кадра анализа для маппинга тапа (когда детектор заглушён и setDetections не вызывается). */
    fun setAnalysisSize(srcWidth: Int, srcHeight: Int) {
        if (srcWidth > 0 && srcHeight > 0) {
            sourceWidth = srcWidth
            sourceHeight = srcHeight
        }
    }

    fun addManualPoint(cx: Float, cy: Float) {
        android.util.Log.d("OverlayView", "addManualPoint $cx,$cy total=${manualPoints.size} manualMode=$manualMode")
        manualPoints.add(ManualPoint(cx, cy, configured = false))
        postInvalidate()
    }

    /** Отметить последнюю/конкретную точку как настроенную (по индексу). */
    fun setManualPointConfigured(index: Int, configured: Boolean) {
        if (index in manualPoints.indices) {
            manualPoints[index].configured = configured
            postInvalidate()
        }
    }

    fun clearManualPoints() {
        manualPoints.clear()
        postInvalidate()
    }

    fun manualPointCount(): Int = manualPoints.size

    fun clear() {
        detections = emptyList()
        capturingMode = false
        manualPoints.clear()
        postInvalidate()
    }

    // ===== Геометрия маппинга view <-> кадр анализа =====

    private fun currentScale(): Float {
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val scaleX = viewW / sourceWidth.toFloat()
        val scaleY = viewH / sourceHeight.toFloat()
        return maxOf(scaleX, scaleY)
    }

    private fun currentOffsetX(scale: Float): Float =
        (width.toFloat() - sourceWidth * scale) / 2f

    private fun currentOffsetY(scale: Float): Float =
        (height.toFloat() - sourceHeight * scale) / 2f

    // view-координаты -> координаты кадра анализа
    private fun viewToSource(vx: Float, vy: Float): Pair<Float, Float> {
        val scale = currentScale()
        val offX = currentOffsetX(scale)
        val offY = currentOffsetY(scale)
        val sx = (vx - offX) / scale
        val sy = (vy - offY) / scale
        return Pair(sx, sy)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!manualMode || capturingMode) return super.onTouchEvent(event)
        if (event.action == MotionEvent.ACTION_UP) {
            val (sx, sy) = viewToSource(event.x, event.y)
            // Игнорируем тапы вне области кадра
            if (sx in 0f..sourceWidth.toFloat() && sy in 0f..sourceHeight.toFloat()) {
                performClick()
                onManualTap?.invoke(sx, sy)
            }
            return true
        }
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val scale = currentScale()
        val offsetX = currentOffsetX(scale)
        val offsetY = currentOffsetY(scale)

        // ===== РУЧНЫЕ ТОЧКИ (рисуем всегда, независимо от детекций) =====
        if (manualMode && manualPoints.isNotEmpty()) {
            android.util.Log.d("OverlayView", "onDraw manual=$manualMode points=${manualPoints.size} src=${sourceWidth}x${sourceHeight} view=${width}x${height}")

            for ((index, p) in manualPoints.withIndex()) {
                val cx = p.cx * scale + offsetX
                val cy = p.cy * scale + offsetY

                val pPaint = if (p.configured) pointPaintGreen else pointPaintYellow
                val bgPaint = if (p.configured) numberBgPaintGreen else numberBgPaintYellow

                canvas.drawCircle(cx, cy, 16f, pPaint)
                canvas.drawCircle(cx, cy, 18f, pointStroke)
                canvas.drawLine(cx - 24, cy, cx + 24, cy, crossPaint)
                canvas.drawLine(cx, cy - 24, cx, cy + 24, crossPaint)

                val numStr = "${index + 1}"
                val numW = numberPaint.measureText(numStr)
                val r = (maxOf(numW, 52f) / 2f) + 8f
                canvas.drawCircle(cx, cy - 50f, r, bgPaint)
                canvas.drawText(numStr, cx, cy - 35f, numberPaint)
            }
            // В ручном режиме детекции не показываем.
            return
        }

        // ===== АВТО-РЕЖИМ: детекции (как раньше) =====
        if (detections.isEmpty()) return

        val pointPaint = if (capturingMode) pointPaintRed else pointPaintGreen
        val numberBgPaint = if (capturingMode) numberBgPaintRed else numberBgPaintGreen

        for ((index, det) in detections.withIndex()) {
            val left = det.x1 * scale + offsetX
            val top = det.y1 * scale + offsetY
            val right = det.x2 * scale + offsetX
            val bottom = det.y2 * scale + offsetY

            val cx = (left + right) / 2f
            val cy = (top + bottom) / 2f

            if (!capturingMode) {
                canvas.drawRect(left, top, right, bottom, boxPaintGreen)

                val label = "${det.className} ${(det.confidence * 100).toInt()}%"
                val textW = textPaint.measureText(label)
                val textH = 44f
                canvas.drawRect(left, top - textH - 8, left + textW + 12, top, fillPaintGreen)
                canvas.drawText(label, left + 6, top - 12, textPaint)
            }

            canvas.drawCircle(cx, cy, 16f, pointPaint)
            canvas.drawCircle(cx, cy, 18f, pointStroke)
            canvas.drawLine(cx - 24, cy, cx + 24, cy, crossPaint)
            canvas.drawLine(cx, cy - 24, cx, cy + 24, crossPaint)

            val numStr = "${index + 1}"
            val numW = numberPaint.measureText(numStr)
            val r = (maxOf(numW, 52f) / 2f) + 8f
            canvas.drawCircle(cx, cy - 50f, r, numberBgPaint)
            canvas.drawText(numStr, cx, cy - 35f, numberPaint)
        }
    }
}