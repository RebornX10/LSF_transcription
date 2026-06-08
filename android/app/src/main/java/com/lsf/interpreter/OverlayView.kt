package com.lsf.interpreter

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Draws the holistic skeleton over the camera preview from normalised
 * landmarks: hand + pose connections as lines, face mesh as light dots.
 * Mirrors horizontally for the front (selfie) camera.
 */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var frame: Frame? = null
    private var mirror = false

    private val handPaint = stroke(Color.parseColor("#2ee6a6"), 5f)
    private val handPaintR = stroke(Color.parseColor("#ffd166"), 5f)
    private val posePaint = stroke(Color.parseColor("#5ad1ff"), 5f)
    private val jointPaint = fill(Color.parseColor("#2ee6a6"))
    private val facePaint = fill(Color.parseColor("#88b4d2")).apply { alpha = 150 }

    private fun stroke(c: Int, w: Float) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = c; style = Paint.Style.STROKE; strokeWidth = w; strokeCap = Paint.Cap.ROUND
    }
    private fun fill(c: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = c; style = Paint.Style.FILL }

    fun setFrame(f: Frame?, mirror: Boolean) {
        this.frame = f
        this.mirror = mirror
        postInvalidate()
    }

    private fun mapX(nx: Float) = if (mirror) width - nx * width else nx * width
    private fun mapY(ny: Float) = ny * height

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val f = frame ?: return

        f.face?.let { face ->
            val n = Frame.count(face)
            for (i in 0 until n) {
                canvas.drawCircle(mapX(Frame.x(face, i)), mapY(Frame.y(face, i)), 1.6f, facePaint)
            }
        }
        f.pose?.let { drawConnections(canvas, it, POSE, posePaint) }
        f.leftHand?.let { drawHand(canvas, it, handPaint) }
        f.rightHand?.let { drawHand(canvas, it, handPaintR) }
    }

    private fun drawConnections(canvas: Canvas, pts: FloatArray, conns: Array<IntArray>, paint: Paint) {
        val n = Frame.count(pts)
        for (c in conns) {
            if (c[0] >= n || c[1] >= n) continue
            canvas.drawLine(
                mapX(Frame.x(pts, c[0])), mapY(Frame.y(pts, c[0])),
                mapX(Frame.x(pts, c[1])), mapY(Frame.y(pts, c[1])), paint,
            )
        }
    }

    private fun drawHand(canvas: Canvas, pts: FloatArray, paint: Paint) {
        drawConnections(canvas, pts, HAND, paint)
        val n = Frame.count(pts)
        for (i in 0 until n) canvas.drawCircle(mapX(Frame.x(pts, i)), mapY(Frame.y(pts, i)), 4f, jointPaint)
    }

    companion object {
        private val HAND = arrayOf(
            intArrayOf(0, 1), intArrayOf(1, 2), intArrayOf(2, 3), intArrayOf(3, 4),
            intArrayOf(0, 5), intArrayOf(5, 6), intArrayOf(6, 7), intArrayOf(7, 8),
            intArrayOf(5, 9), intArrayOf(9, 10), intArrayOf(10, 11), intArrayOf(11, 12),
            intArrayOf(9, 13), intArrayOf(13, 14), intArrayOf(14, 15), intArrayOf(15, 16),
            intArrayOf(13, 17), intArrayOf(17, 18), intArrayOf(18, 19), intArrayOf(19, 20),
            intArrayOf(0, 17),
        )
        private val POSE = arrayOf(
            intArrayOf(11, 12), intArrayOf(11, 13), intArrayOf(13, 15),
            intArrayOf(12, 14), intArrayOf(14, 16),
            intArrayOf(11, 23), intArrayOf(12, 24), intArrayOf(23, 24),
            intArrayOf(0, 11), intArrayOf(0, 12),
        )
    }
}
