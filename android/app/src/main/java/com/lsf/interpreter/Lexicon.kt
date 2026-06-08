package com.lsf.interpreter

import android.content.Context
import org.json.JSONObject

data class Word(val gloss: String, val fr: String, val tip: String?)

/** Loads the bundled LSF vocabulary catalogue from assets/lexicon.json. */
class Lexicon(context: Context) {
    val words: List<Word> = load(context)

    private fun load(context: Context): List<Word> = try {
        val text = context.assets.open("lexicon.json").bufferedReader().use { it.readText() }
        val arr = JSONObject(text).getJSONArray("words")
        (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            Word(
                gloss = o.getString("gloss"),
                fr = o.optString("fr", o.getString("gloss")),
                tip = if (o.has("tip")) o.optString("tip") else null,
            )
        }
    } catch (_: Exception) {
        emptyList()
    }
}
