package com.lsf.interpreter

/**
 * One frame of holistic landmarks. Each group is a flat float array
 * [x0,y0,z0, x1,y1,z1, ...] of normalised image coords, or null if absent.
 */
class Frame(
    val pose: FloatArray?,        // 33 points
    val face: FloatArray?,        // ~478 points
    val leftHand: FloatArray?,    // 21 points
    val rightHand: FloatArray?,   // 21 points
) {
    val hasHands: Boolean get() = leftHand != null || rightHand != null

    companion object {
        const val STRIDE = 3
        fun x(a: FloatArray, i: Int) = a[i * STRIDE]
        fun y(a: FloatArray, i: Int) = a[i * STRIDE + 1]
        fun z(a: FloatArray, i: Int) = a[i * STRIDE + 2]
        fun count(a: FloatArray) = a.size / STRIDE
    }
}
