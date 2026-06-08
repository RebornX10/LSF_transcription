# LSF Interpreter — native Android (Kotlin)

On-device French Sign Language interpreter for Android. Everything runs on the
phone — **no server, no internet**: CameraX feeds frames to **MediaPipe
HolisticLandmarker**, the skeleton is drawn over the preview, and signs are
recognised with the same **DTW learn-by-example** engine as the desktop/web app
(ported to Kotlin, identical thresholds).

## Features
- **Front / back camera selector** (selfie ⟲ main), with mirroring on the front camera.
- On-device holistic landmarks: **fingers, eyes, mouth, chest, arms, hands**.
- **Native dual-pixel / depth detection** via Camera2 `DEPTH_OUTPUT` capability
  (real device signal on Pixel-class phones) — shown in the top-left pill.
- Learn-by-example training: pick a word from the bundled ~80-word LSF
  catalogue, **Entraîner** → sign → **Sauvegarder**. Templates persist on-device.
- Live transcription at the bottom.

## Build (Android Studio)
> ⚠️ This project was written without an Android toolchain available, so it is
> **not compiler-verified**. Open it in Android Studio, which resolves the
> Gradle wrapper and dependencies; minor tweaks (e.g. a dependency version) may
> be needed.

1. **Get the model** (~13 MB, not committed):
   ```bash
   cd android && ./fetch_model.sh
   ```
   (Puts `holistic_landmarker.task` in `app/src/main/assets/`.)
2. Open the `android/` folder in **Android Studio** (Giraffe+), let it sync.
3. Run on a device/emulator, or build an APK:
   ```bash
   cd android && ./gradlew assembleDebug
   # → app/build/outputs/apk/debug/app-debug.apk
   ```
   (CLI build needs a local Android SDK + JDK 17; `local.properties` must point
   at the SDK — Android Studio writes this for you.)

## Layout
```
android/
  app/src/main/
    assets/lexicon.json              bundled LSF vocabulary (from ../signs)
    assets/holistic_landmarker.task  model (fetch_model.sh)
    java/com/lsf/interpreter/
      MainActivity.kt        CameraX + UI + camera selector + record/transcribe
      HolisticAnalyzer.kt    MediaPipe HolisticLandmarker per frame
      OverlayView.kt         skeleton overlay (hands/pose lines, face dots)
      Features.kt            body-relative matching vector (port of features.py)
      Recognizer.kt          motion segmentation + DTW + persistence (port)
      Lexicon.kt             loads the vocabulary catalogue
      DepthCapability.kt     Camera2 dual-pixel / depth detection
      Landmarks.kt           Frame data holder
    res/layout/activity_main.xml
  fetch_model.sh             downloads the .task model
```

## Notes / known rough edges
- **Overlay alignment**: landmarks are mapped to the preview assuming a
  fill/centre match; if the preview aspect ratio differs you may need to refine
  `OverlayView.mapX/mapY` (letterbox compensation).
- **HolisticLandmarker result accessors**: built against `tasks-vision:0.10.14`
  with `poseLandmarks()/faceLandmarks()/leftHandLandmarks()/rightHandLandmarks()`
  returning `List<NormalizedLandmark>`. If your tasks-vision version differs,
  adjust `HolisticAnalyzer.toFrame`.
- Uses CPU inference (`RunningMode.VIDEO`) for portability; switch the
  `BaseOptions` delegate to GPU for more speed on capable devices.
