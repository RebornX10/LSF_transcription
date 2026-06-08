package com.lsf.interpreter

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult

/**
 * CameraX analyzer that runs MediaPipe HolisticLandmarker on each frame
 * (RunningMode.VIDEO, synchronous) and emits a [Frame] of landmarks.
 *
 * The model file `holistic_landmarker.task` must be in app/src/main/assets/
 * (see android/fetch_model.sh).
 */
class HolisticAnalyzer(
    context: Context,
    private val onFrame: (Frame) -> Unit,
) : ImageAnalysis.Analyzer {

    private val landmarker: HolisticLandmarker = build(context)
    private var lastTimestamp = 0L

    private fun build(context: Context): HolisticLandmarker {
        val base = BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET)
            .build()
        val options = HolisticLandmarker.HolisticLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.VIDEO)
            .build()
        return HolisticLandmarker.createFromOptions(context, options)
    }

    override fun analyze(image: ImageProxy) {
        try {
            val bitmap = image.toBitmap()
            val upright = rotate(bitmap, image.imageInfo.rotationDegrees)
            val mpImage = BitmapImageBuilder(upright).build()
            var ts = SystemClock.uptimeMillis()
            if (ts <= lastTimestamp) ts = lastTimestamp + 1   // must strictly increase
            lastTimestamp = ts
            val result = landmarker.detectForVideo(mpImage, ts)
            onFrame(toFrame(result))
        } catch (_: Exception) {
            // Drop the occasional bad frame rather than crash the stream.
        } finally {
            image.close()
        }
    }

    private fun toFrame(r: HolisticLandmarkerResult) = Frame(
        pose = toArray(r.poseLandmarks()),
        face = toArray(r.faceLandmarks()),
        leftHand = toArray(r.leftHandLandmarks()),
        rightHand = toArray(r.rightHandLandmarks()),
    )

    private fun toArray(lms: List<NormalizedLandmark>?): FloatArray? {
        if (lms.isNullOrEmpty()) return null
        val out = FloatArray(lms.size * 3)
        for (i in lms.indices) {
            val lm = lms[i]
            out[i * 3] = lm.x()
            out[i * 3 + 1] = lm.y()
            out[i * 3 + 2] = lm.z()
        }
        return out
    }

    private fun rotate(bmp: Bitmap, deg: Int): Bitmap {
        if (deg == 0) return bmp
        val m = Matrix().apply { postRotate(deg.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    fun close() = landmarker.close()

    companion object {
        const val MODEL_ASSET = "holistic_landmarker.task"
    }
}
