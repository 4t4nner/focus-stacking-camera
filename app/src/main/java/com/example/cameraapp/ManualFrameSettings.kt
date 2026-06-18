package com.example.cameraapp

import android.graphics.RectF

/**
 * Настройки одного кадра в РУЧНОМ режиме.
 * Заполняется при тапе пользователя по объекту:
 * после сходимости AF/AE считываются фактические значения из CaptureResult.
 *
 * При съёмке эти значения выставляются вручную (AF_OFF / AE_OFF),
 * чтобы кадр был снят ровно с теми фокусом/экспозицией, что выбрал пользователь.
 */
data class ManualFrameSettings(
    // Индекс объекта в списке детекций (стабильная привязка к точке).
    val detectionIndex: Int,

    val label: String,

    // Координаты точки в системе координат кадра ImageAnalysis
    // (как и DetectedObject.cx/cy) — для отрисовки и для имени файла.
    val cx: Float,
    val cy: Float,

    // Рамка объекта (на момент тапа) — для отрисовки/отладки.
    val rect: RectF,

    // ===== Замеренные параметры (заполняются после сходимости AF/AE) =====

    // Фокус в диоптриях (1/м), как LENS_FOCUS_DISTANCE. null = не удалось прочитать.
    var focusDistanceDiopters: Float? = null,

    // Выдержка в наносекундах (SENSOR_EXPOSURE_TIME). null = не прочитано.
    var exposureTimeNs: Long? = null,

    // ISO (SENSOR_SENSITIVITY). null = не прочитано.
    var iso: Int? = null,

    // true = точка успешно настроена (AF/AE сошлись, параметры считаны).
    var configured: Boolean = false
)
