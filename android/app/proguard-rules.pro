# Keep MediaPipe Tasks classes (they use JNI/reflection internally).
-keep class com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-dontwarn com.google.mediapipe.**
