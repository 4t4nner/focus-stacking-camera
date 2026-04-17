package com.example.cameraapp

import android.graphics.*

class FocusAnalyzer {

    /**
     * Анализирует детекции: рисует рамки и центры объектов.
     * Анализ резкости временно отключен, все объекты относятся к зоне 1.
     */
    fun analyze(bitmap: Bitmap, detections: List<Detection>): Pair<Bitmap, List<FocusPoint>> {
        if (detections.isEmpty()) return Pair(bitmap, emptyList())

        val result = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)

        val boxPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        val fillPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.FILL
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 36f
            isAntiAlias = true
            typeface = Typeface.DEFAULT_BOLD
        }
        val pointPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val pointStroke = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.WHITE
            strokeWidth = 4f
            isAntiAlias = true
        }
        val crossPaint = Paint().apply {
            color = Color.WHITE
            strokeWidth = 3f
            isAntiAlias = true
        }

        val focusPoints = mutableListOf<FocusPoint>()

        for (det in detections) {
            // Рисуем рамку
            canvas.drawRect(det.x1, det.y1, det.x2, det.y2, boxPaint)

            // Рисуем подпись (пока всегда Zone 1)
            val label = "${det.className} Z1"
            val textW = textPaint.measureText(label)
            val textH = 40f

            canvas.drawRect(det.x1, det.y1 - textH - 8, det.x1 + textW + 12, det.y1, fillPaint)
            canvas.drawText(label, det.x1 + 6, det.y1 - 10, textPaint)

            // Точка фокуса в центре объекта
            val cx = (det.x1 + det.x2) / 2f
            val cy = (det.y1 + det.y2) / 2f

            canvas.drawCircle(cx, cy, 20f, pointPaint)
            canvas.drawCircle(cx, cy, 22f, pointStroke)

            canvas.drawLine(cx - 30, cy, cx + 30, cy, crossPaint)
            canvas.drawLine(cx, cy - 30, cx, cy + 30, crossPaint)

            focusPoints.add(FocusPoint(1, cx, cy, det.className))
        }

        return Pair(result, focusPoints)
    }
}
