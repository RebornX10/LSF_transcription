package com.lsf.interpreter

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.lsf.interpreter.databinding.ActivityMainBinding
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var recognizer: Recognizer
    private lateinit var analyzer: HolisticAnalyzer
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var lexicon: Lexicon
    private lateinit var glosses: List<String>

    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var recording = false
    private var hasTranscript = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        recognizer = Recognizer(File(filesDir, "lsf_signs.json"))
        lexicon = Lexicon(this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.depthStatus.text = DepthCapability.describe(this)
        setupSpinner()

        // Building the analyzer loads the MediaPipe model from assets; fail
        // loudly (not a crash) if fetch_model.sh wasn't run.
        try {
            analyzer = HolisticAnalyzer(this) { frame -> onFrame(frame) }
        } catch (e: Exception) {
            binding.transcript.text =
                "Modèle MediaPipe introuvable.\nLancez android/fetch_model.sh puis recompilez.\n\n${e.message}"
            return
        }

        binding.flipBtn.setOnClickListener { toggleCamera() }
        binding.recordBtn.setOnClickListener { toggleRecord() }

        if (hasCameraPermission()) startCamera()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
    }

    private fun setupSpinner() {
        glosses = lexicon.words.map { it.gloss }.ifEmpty { listOf("BONJOUR") }
        val labels = lexicon.words.map { "${it.fr}  (${it.gloss})" }.ifEmpty { glosses }
        binding.wordSpinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, labels
        )
    }

    private fun currentGloss(): String =
        glosses.getOrElse(binding.wordSpinner.selectedItemPosition) { glosses.first() }

    // -- camera ---------------------------------------------------------------
    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(binding.preview.surfaceProvider)
        }
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build().also { it.setAnalyzer(cameraExecutor, analyzer) }
        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis)
        } catch (e: Exception) {
            toast("Caméra indisponible : ${e.message}")
        }
    }

    private fun toggleCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
            CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
        bindCamera()
    }

    // -- per-frame (analyzer thread) -----------------------------------------
    private fun onFrame(frame: Frame) {
        binding.overlay.setFrame(frame, lensFacing == CameraSelector.LENS_FACING_FRONT)
        val rec = recognizer.update(frame)
        if (rec != null) runOnUiThread { appendTranscript(rec) }
    }

    private fun appendTranscript(rec: Recognition) {
        val line = "${rec.gloss}   ${(rec.confidence * 100).toInt()}%"
        if (!hasTranscript) { binding.transcript.text = ""; hasTranscript = true }
        binding.transcript.append(if (binding.transcript.text.isEmpty()) line else "\n$line")
        binding.transcriptScroll.post {
            binding.transcriptScroll.fullScroll(android.view.View.FOCUS_DOWN)
        }
    }

    // -- recording ------------------------------------------------------------
    private fun toggleRecord() {
        if (!recording) {
            val gloss = currentGloss()
            recognizer.startRecording(gloss)
            recording = true
            binding.recordBtn.setText(R.string.save)
            toast("Enregistrement : $gloss — signez puis Sauvegarder")
        } else {
            val saved = recognizer.stopRecording()
            recording = false
            binding.recordBtn.setText(R.string.record)
            toast(if (saved != null) "Signe appris : $saved" else "Trop court — réessayez")
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)
                startCamera()
            else toast("Permission caméra refusée.")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::cameraExecutor.isInitialized) cameraExecutor.shutdown()
        if (::analyzer.isInitialized) analyzer.close()
    }

    companion object {
        private const val REQ_CAMERA = 1001
    }
}
