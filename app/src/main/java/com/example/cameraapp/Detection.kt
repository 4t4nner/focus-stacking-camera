package com.example.cameraapp

data class Detection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float
)

data class FocusPoint(
    val zone: Int,
    val cx: Float,
    val cy: Float,
    val className: String
)

data class CapturedPhoto(
    val path: String,
    val objectName: String,
    val focusCx: Float,
    val focusCy: Float
)