package com.lsf.interpreter

import kotlin.math.hypot

/**
 * Port of the Python `features` "matching vector" (pose subset + both hands),
 * body-relative so it's invariant to where the signer stands and how far from
 * the camera. Layout matches the desktop app so DTW thresholds carry over:
 *   [pose(9*3) | left hand(21*3) | right hand(21*3)]  = 153 floats.
 */
object Features {
    // MediaPipe pose indices: nose, shoulders, elbows, wrists, hips.
    private val POSE_IDX = intArrayOf(0, 11, 12, 13, 14, 15, 16, 23, 24)
    const val N_POSE = 9
    const val N_HAND = 21
    const val DIM = (N_POSE + N_HAND + N_HAND) * 3   // 153

    // Slice offsets within the matching vector (used by motion gating).
    const val POSE_LEN = N_POSE * 3                  // 27
    const val LEFT_START = POSE_LEN                  // 27
    const val RIGHT_START = POSE_LEN + N_HAND * 3    // 90
    const val ARM_START = 9                          // elbows+wrists block (coords 9..21)
    const val ARM_END = 21

    private data class Ref(val ox: Float, val oy: Float, val oz: Float, val scale: Float)

    private fun reference(f: Frame): Ref {
        val p = f.pose
        if (p != null) {
            val lx = Frame.x(p, 11); val ly = Frame.y(p, 11); val lz = Frame.z(p, 11)
            val rx = Frame.x(p, 12); val ry = Frame.y(p, 12); val rz = Frame.z(p, 12)
            val scale = hypot((lx - rx).toDouble(), (ly - ry).toDouble()).toFloat()
            return Ref((lx + rx) / 2f, (ly + ry) / 2f, (lz + rz) / 2f, if (scale > 1e-3f) scale else 1e-3f)
        }
        // Fallback: centre on a hand wrist, arbitrary non-zero scale.
        val h = f.rightHand ?: f.leftHand
        return if (h != null) Ref(Frame.x(h, 0), Frame.y(h, 0), Frame.z(h, 0), 0.2f)
        else Ref(0f, 0f, 0f, 0.2f)
    }

    fun matchingVector(f: Frame): FloatArray {
        val ref = reference(f)
        val out = FloatArray(DIM)
        var o = 0
        // pose subset
        val p = f.pose
        for (idx in POSE_IDX) {
            if (p != null && idx < Frame.count(p)) {
                out[o] = (Frame.x(p, idx) - ref.ox) / ref.scale
                out[o + 1] = (Frame.y(p, idx) - ref.oy) / ref.scale
                out[o + 2] = (Frame.z(p, idx) - ref.oz) / ref.scale
            }
            o += 3
        }
        o = writeHand(out, o, f.leftHand, ref)
        writeHand(out, o, f.rightHand, ref)
        return out
    }

    private fun writeHand(out: FloatArray, offset: Int, hand: FloatArray?, ref: Ref): Int {
        var o = offset
        for (i in 0 until N_HAND) {
            if (hand != null && i < Frame.count(hand)) {
                out[o] = (Frame.x(hand, i) - ref.ox) / ref.scale
                out[o + 1] = (Frame.y(hand, i) - ref.oy) / ref.scale
                out[o + 2] = (Frame.z(hand, i) - ref.oz) / ref.scale
            }
            o += 3
        }
        return o
    }
}
