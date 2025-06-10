package com.vmt.faceauth

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import kotlinx.coroutines.*

class MainActivity : ComponentActivity() {
    private lateinit var faceRecognitionHelper: FaceRecognitionHelper
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val TAG = "FaceAuthMain"
    private val recognitionResult = mutableStateOf("Đang chờ nhận diện...")
    private var lastAnalysisTime = 0L
    private val faceRect = mutableStateOf<Rect?>(null)
    private val capturedResult = mutableStateOf<RecognitionResult?>(null)
    private val isProcessing = mutableStateOf(false)
    private var cameraProvider: ProcessCameraProvider? = null
    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "MainActivity onCreate")
        try {
            faceRecognitionHelper = FaceRecognitionHelper(this)
            Log.d(TAG, "FaceRecognitionHelper initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing FaceRecognitionHelper: ${e.message}", e)
            Toast.makeText(this, "Lỗi khởi tạo nhận diện khuôn mặt", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        setContent {
            FaceAuthTheme {
                MainScreen(
                    recognitionResult = recognitionResult.value,
                    faceRect = faceRect.value,
                    capturedResult = capturedResult.value,
                    isProcessing = isProcessing.value,
                    onSave = {
                        capturedResult.value?.let { result ->
                            try {
                                faceRecognitionHelper.exportToCsv(result)
                                Toast.makeText(this, "Đã lưu MSSV ${result.mssv} vào CSV", Toast.LENGTH_SHORT).show()
                            } catch (e: Exception) {
                                Log.e(TAG, "Error saving CSV: ${e.message}", e)
                                Toast.makeText(this, "Lỗi lưu CSV: ${e.message}", Toast.LENGTH_SHORT).show()
                            }
                        }
                    },
                    onRestartCamera = {
                        capturedResult.value = null
                        recognitionResult.value = "Đang chờ nhận diện..."
                        faceRect.value = null
                        coroutineScope.launch {
                            delay(500)
                            startCamera()
                        }
                    },
                    onExit = {
                        try {
                            faceRecognitionHelper.cleanupOnExit()
                            Log.d(TAG, "Cleanup completed, exiting app")
                        } catch (e: Exception) {
                            Log.e(TAG, "Error during cleanup: ${e.message}", e)
                            Toast.makeText(this, "Lỗi khi thoát: ${e.message}", Toast.LENGTH_SHORT).show()
                        }
                        finish()
                    },
                    modifier = Modifier.fillMaxSize()
                )
            }
        }

        val requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            Log.d(TAG, "Camera permission granted: $isGranted")
            if (isGranted) {
                startCamera()
            } else {
                Log.w(TAG, "Camera permission denied")
                Toast.makeText(this, "Cần quyền camera để chạy ứng dụng", Toast.LENGTH_LONG).show()
                finish()
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Camera permission already granted")
            startCamera()
        } else {
            Log.d(TAG, "Requesting camera permission")
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        Log.d(TAG, "Starting camera")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                Log.d(TAG, "Camera provider obtained")

                val preview = Preview.Builder()
                    .setTargetResolution(android.util.Size(640, 480))
                    .build()
                    .also {
                        val previewView = findViewById<androidx.camera.view.PreviewView>(R.id.preview_view)
                        if (previewView != null) {
                            Log.d(TAG, "PreviewView found, setting surface provider")
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        } else {
                            Log.e(TAG, "PreviewView is null, retrying in 100ms")
                            coroutineScope.launch {
                                delay(100)
                                startCamera()
                            }
                            return@addListener
                        }
                    }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(android.util.Size(640, 480))
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor) { imageProxy ->
                            val currentTime = System.currentTimeMillis()
                            if (currentTime - lastAnalysisTime < 500) {
                                imageProxy.close()
                                return@setAnalyzer
                            }
                            lastAnalysisTime = currentTime
                            Log.d(TAG, "Analyzing frame: ${imageProxy.width}x${imageProxy.height}")
                            try {
                                isProcessing.value = true
                                val bitmap = imageProxy.toBitmap()
                                faceRecognitionHelper.processImage(bitmap) { result, rect, recResult ->
                                    recognitionResult.value = result
                                    faceRect.value = rect
                                    if (recResult != null) {
                                        capturedResult.value = recResult
                                        cameraProvider?.unbindAll()
                                        Log.d(TAG, "Camera unbound after capture")
                                    }
                                    isProcessing.value = false
                                }
                            } catch (e: Exception) {
                                Log.e(TAG, "Error processing frame: ${e.message}", e)
                                isProcessing.value = false
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                val cameraSelector = if (cameraProvider!!.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) {
                    Log.d(TAG, "Using front camera")
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    Log.w(TAG, "No front camera, using back camera")
                    CameraSelector.DEFAULT_BACK_CAMERA
                }

                Log.d(TAG, "Binding camera")
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
                Log.d(TAG, "Camera bound successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error starting camera: ${e.message}, cause: ${e.cause}", e)
                runOnUiThread {
                    Toast.makeText(this, "Lỗi khởi động camera: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "MainActivity onDestroy")
        cameraExecutor.shutdown()
        coroutineScope.cancel()
    }
}

fun androidx.camera.core.ImageProxy.toBitmap(): android.graphics.Bitmap {
    Log.d("FaceAuthMain", "Converting ImageProxy to Bitmap")
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    val bitmap = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    if (bitmap == null) {
        Log.e("FaceAuthMain", "Failed to decode bitmap from ImageProxy")
        throw IllegalStateException("Cannot decode bitmap")
    }
    return bitmap
}