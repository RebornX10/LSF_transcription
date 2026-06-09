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
- **Ships pre-trained**: ~72 signs (trained from Elix videos) are bundled and
  seeded into the on-device store on first launch; your own samples add to them.

## Build
✅ Verified building: Gradle 8.7, AGP 8.5.2, JDK 17, SDK 34 → a 64 MB
`app-debug.apk` (model + MediaPipe native libs bundled).

**Command line (no Android Studio):**
```bash
# JDK 17 + Android command-line tools required, e.g. via Homebrew:
#   brew install openjdk@17 && brew install --cask android-commandlinetools
#   sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools

cd android
./fetch_model.sh                 # downloads holistic_landmarker.task (~13 MB)
./gradlew assembleDebug          # → app/build/outputs/apk/debug/app-debug.apk
```
`local.properties` must contain `sdk.dir=$ANDROID_HOME` (already set locally).

**Install on a phone:** `adb install -r app/build/outputs/apk/debug/app-debug.apk`
(USB debugging on), or copy the APK to the device and tap it.

**Or no toolchain at all:** push to GitHub — the `Build Android APK` workflow
([../.github/workflows/android.yml](../.github/workflows/android.yml)) compiles
it in the cloud and uploads the APK as an artifact.

**Android Studio:** just open the `android/` folder and Run.

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

## Notes
- The app **compiles and packages** cleanly; the items below are runtime
  refinements to check on-device (they don't affect the build).
- **Overlay alignment**: landmarks are mapped to the preview assuming a
  fill/centre match; if the preview aspect ratio differs you may need to refine
  `OverlayView.mapX/mapY` (letterbox compensation).
- Built against `tasks-vision:0.10.14`; `HolisticLandmarkerResult` accessors
  `poseLandmarks()/faceLandmarks()/leftHandLandmarks()/rightHandLandmarks()`
  return `List<NormalizedLandmark>`.
- Uses CPU inference (`RunningMode.VIDEO`) for portability; switch the
  `BaseOptions` delegate to GPU for more speed on capable devices.
