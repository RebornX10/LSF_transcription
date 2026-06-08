package com.lsf.interpreter

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata

/**
 * Native dual-pixel / depth detection via Camera2.
 *
 * Devices that produce dual-pixel depth (e.g. Pixel phones) advertise the
 * DEPTH_OUTPUT capability on the relevant camera. This is the real device-level
 * signal — unlike the web version, which can only guess from labels.
 */
object DepthCapability {

    fun describe(context: Context): String {
        return try {
            val cm = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val facings = mutableListOf<String>()
            for (id in cm.cameraIdList) {
                val ch = cm.getCameraCharacteristics(id)
                val caps = ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: continue
                val hasDepth = caps.contains(
                    CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT
                )
                if (hasDepth) {
                    facings += when (ch.get(CameraCharacteristics.LENS_FACING)) {
                        CameraCharacteristics.LENS_FACING_FRONT -> "avant"
                        CameraCharacteristics.LENS_FACING_BACK -> "arrière"
                        else -> "externe"
                    }
                }
            }
            if (facings.isNotEmpty()) "depth: dual-pixel (${facings.joinToString()})"
            else "depth: mediapipe z (aucun capteur)"
        } catch (e: Exception) {
            "depth: indéterminé"
        }
    }
}
