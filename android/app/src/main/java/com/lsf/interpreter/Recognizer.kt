package com.lsf.interpreter

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class Recognition(val gloss: String, val confidence: Float, val source: String = "dtw")

/**
 * On-device learn-by-example recogniser: motion segmentation + DTW against
 * recorded templates. Direct port of the Python `recognizer`, so behaviour and
 * thresholds match the desktop/web app. Templates persist to the app's files
 * dir as JSON.
 */
class Recognizer(private val store: File) {

    companion object {
        const val TEMPLATE_LEN = 32
        const val MIN_SEGMENT_FRAMES = 6
        const val MOTION_THRESHOLD = 0.025f
        const val SETTLE_FRAMES = 8
        const val DTW_BAND = 8
        const val REJECT_DISTANCE = 3.5
    }

    private val templates = LinkedHashMap<String, MutableList<Array<FloatArray>>>()

    private val buffer = ArrayList<FloatArray>()
    private var prevMatch: FloatArray? = null
    private var calmRun = 0
    private var active = false
    private var lastEmit = 0.0

    private var recording = false
    private var recordLabel: String? = null
    private val recordBuffer = ArrayList<FloatArray>()

    init {
        load()
    }

    val isRecording: Boolean get() = recording

    fun vocabulary(): List<String> = templates.keys.sorted()
    fun templateCounts(): Map<String, Int> = templates.mapValues { it.value.size }

    @Synchronized
    fun deleteGloss(gloss: String): Boolean {
        val existed = templates.remove(gloss) != null
        if (existed) save()
        return existed
    }

    // -- recording ------------------------------------------------------------
    @Synchronized
    fun startRecording(label: String) {
        recording = true; recordLabel = label; recordBuffer.clear()
    }

    @Synchronized
    fun cancelRecording() {
        recording = false; recordLabel = null; recordBuffer.clear()
    }

    @Synchronized
    fun stopRecording(): String? {
        recording = false
        val label = recordLabel
        if (label == null || recordBuffer.size < MIN_SEGMENT_FRAMES) {
            recordBuffer.clear(); recordLabel = null; return null
        }
        val seq = resample(recordBuffer, TEMPLATE_LEN)
        templates.getOrPut(label) { mutableListOf() }.add(seq)
        save()
        recordLabel = null; recordBuffer.clear()
        return label
    }

    // -- streaming ------------------------------------------------------------
    @Synchronized
    fun update(frame: Frame): Recognition? {
        val match = Features.matchingVector(frame)
        if (recording) recordBuffer.add(match)

        val motion = motionEnergy(frame, match)
        prevMatch = match
        val activeNow = motion > MOTION_THRESHOLD

        if (activeNow) {
            if (!active) { active = true; buffer.clear() }
            buffer.add(match); calmRun = 0
            return null
        }
        if (active) {
            calmRun++
            if (calmRun >= SETTLE_FRAMES) {
                val segment = ArrayList(buffer)
                active = false; buffer.clear(); calmRun = 0
                val now = System.currentTimeMillis() / 1000.0
                if (segment.size >= MIN_SEGMENT_FRAMES && now - lastEmit > 0.3) {
                    val rec = classify(segment)
                    if (rec != null) lastEmit = now
                    return rec
                }
            }
        }
        return null
    }

    private fun motionEnergy(frame: Frame, match: FloatArray): Float {
        val prev = prevMatch ?: return 0f
        val ranges = ArrayList<IntRange>()
        if (frame.leftHand != null) ranges.add(Features.LEFT_START until Features.LEFT_START + 63)
        if (frame.rightHand != null) ranges.add(Features.RIGHT_START until Features.RIGHT_START + 63)
        if (ranges.isEmpty()) ranges.add(Features.ARM_START until Features.ARM_END)
        var sumSq = 0.0; var n = 0
        for (r in ranges) for (i in r) { val d = match[i] - prev[i]; sumSq += (d * d).toDouble(); n++ }
        return if (n == 0) 0f else sqrt(sumSq / n).toFloat()
    }

    private fun classify(segment: List<FloatArray>): Recognition? {
        if (templates.isEmpty()) return null
        val seq = resample(segment, TEMPLATE_LEN)
        var bestGloss: String? = null
        var bestDist = Double.MAX_VALUE
        for ((gloss, samples) in templates) {
            for (t in samples) {
                val d = dtw(seq, t)
                if (d < bestDist) { bestDist = d; bestGloss = gloss }
            }
        }
        if (bestGloss == null || bestDist > REJECT_DISTANCE) return null
        val conf = max(0.0, 1.0 - bestDist / REJECT_DISTANCE).toFloat()
        return Recognition(bestGloss, conf)
    }

    // -- math -----------------------------------------------------------------
    private fun resample(seq: List<FloatArray>, n: Int): Array<FloatArray> {
        val t = seq.size
        if (t == n) return Array(n) { seq[it].copyOf() }
        if (t == 1) return Array(n) { seq[0].copyOf() }
        val dim = seq[0].size
        val out = Array(n) { FloatArray(dim) }
        for (k in 0 until n) {
            val pos = k.toDouble() / (n - 1) * (t - 1)
            val i0 = floor(pos).toInt()
            val i1 = min(i0 + 1, t - 1)
            val frac = (pos - i0).toFloat()
            val a = seq[i0]; val b = seq[i1]
            for (d in 0 until dim) out[k][d] = a[d] * (1 - frac) + b[d] * frac
        }
        return out
    }

    private fun dtw(a: Array<FloatArray>, b: Array<FloatArray>, band: Int = DTW_BAND): Double {
        val n = a.size; val m = b.size
        val inf = Double.MAX_VALUE / 4
        val cost = Array(n + 1) { DoubleArray(m + 1) { inf } }
        cost[0][0] = 0.0
        for (i in 1..n) {
            val jLo = max(1, i - band); val jHi = min(m, i + band)
            val ai = a[i - 1]
            for (j in jLo..jHi) {
                val d = dist(ai, b[j - 1])
                cost[i][j] = d + minOf(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])
            }
        }
        return cost[n][m] / (n + m)
    }

    private fun dist(a: FloatArray, b: FloatArray): Double {
        var s = 0.0
        for (i in a.indices) { val d = (a[i] - b[i]).toDouble(); s += d * d }
        return sqrt(s)
    }

    // -- persistence ----------------------------------------------------------
    private fun load() {
        if (!store.exists()) return
        try {
            val root = JSONObject(store.readText())
            for (gloss in root.keys()) {
                val samples = root.getJSONArray(gloss)
                val list = ArrayList<Array<FloatArray>>()
                for (s in 0 until samples.length()) {
                    val frames = samples.getJSONArray(s)
                    val seq = Array(frames.length()) { fi ->
                        val row = frames.getJSONArray(fi)
                        FloatArray(row.length()) { row.getDouble(it).toFloat() }
                    }
                    list.add(seq)
                }
                templates[gloss] = list
            }
        } catch (_: Exception) { /* corrupt store: start empty */ }
    }

    private fun save() {
        try {
            val root = JSONObject()
            for ((gloss, samples) in templates) {
                val arr = JSONArray()
                for (seq in samples) {
                    val frames = JSONArray()
                    for (row in seq) {
                        val r = JSONArray()
                        for (v in row) r.put(v.toDouble())
                        frames.put(r)
                    }
                    arr.put(frames)
                }
                root.put(gloss, arr)
            }
            store.writeText(root.toString())
        } catch (_: Exception) { /* ignore write failures */ }
    }
}
