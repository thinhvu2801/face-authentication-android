package com.vmt.faceauth

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.sqrt

data class RecognitionResult(val mssv: String, val name: String, val timestamp: String, val capturedImage: File? = null)

class FaceRecognitionHelper(private val context: Context) {
    private val tflite: Interpreter
    private val embeddings: List<Pair<String, FloatArray>>
    private val students: Map<String, String>
    private val recognitionResults = mutableListOf<RecognitionResult>()
    private val faceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(0.3f)
            .build()
    )
    private val TAG = "FaceRecognition"

    init {
        Log.d(TAG, "Initializing FaceRecognitionHelper")
        try {
            val model = FileUtil.loadMappedFile(context, "face_auth_model_facenet.tflite")
            tflite = Interpreter(model)
            Log.d(TAG, "TFLite model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model: ${e.message}", e)
            throw e
        }

        try {
            val embeddingsJson = context.assets.open("embeddings.json").bufferedReader().use { it.readText() }
            embeddings = JSONObject(embeddingsJson).let { json ->
                val array = json.getJSONArray("array")
                Log.d(TAG, "Loaded ${array.length()} embeddings")
                (0 until array.length()).map {
                    val obj = array.getJSONObject(it)
                    val mssv = obj.getString("mssv")
                    val emb = obj.getJSONArray("embedding").let { arr ->
                        FloatArray(arr.length()) { i -> arr.getDouble(i).toFloat() }
                    }
                    mssv to emb
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading embeddings.json: ${e.message}", e)
            throw e
        }

        try {
            val studentsJson = context.assets.open("students.json").bufferedReader().use { it.readText() }
            students = JSONObject(studentsJson).let { json ->
                val map = mutableMapOf<String, String>()
                json.keys().forEach { key ->
                    map[key] = json.getString(key)
                }
                Log.d(TAG, "Loaded ${map.size} students")
                map
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading students.json: ${e.message}", e)
            throw e
        }
    }

    fun processImage(bitmap: Bitmap, onResult: (String, Rect?, RecognitionResult?) -> Unit) {
        Log.d(TAG, "Processing image: ${bitmap.width}x${bitmap.height}")
        val rotatedBitmap = rotateBitmap(bitmap, 0f)
        val inputImage = InputImage.fromBitmap(rotatedBitmap, 0)
        faceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                Log.d(TAG, "Detected ${faces.size} faces")
                if (faces.isEmpty()) {
                    onResult("Không phát hiện khuôn mặt", null, null)
                    return@addOnSuccessListener
                }
                // Chỉ xử lý khuôn mặt đầu tiên
                val face = faces.first()
                val faceBitmap = cropFace(rotatedBitmap, face.boundingBox)
                if (faceBitmap != null) {
                    Log.d(TAG, "Cropped face: ${faceBitmap.width}x${faceBitmap.height}")
                    val embedding = getEmbedding(faceBitmap)
                    val (mssv, name) = findMatchingMssv(embedding)
                    Log.d(TAG, "Recognition result: MSSV=$mssv, Name=$name")
                    if (mssv != "Unknown") {
                        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
                        val imageFile = saveCapturedImage(rotatedBitmap, timestamp)
                        val result = RecognitionResult(mssv, name, timestamp, imageFile)
                        recognitionResults.add(result)
                        onResult("$mssv - $name", face.boundingBox, result)
                    } else {
                        onResult("$mssv - $name", face.boundingBox, null)
                    }
                } else {
                    Log.w(TAG, "Failed to crop face")
                    onResult("Không thể cắt khuôn mặt", null, null)
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Face detection failed: ${e.message}", e)
                onResult("Lỗi phát hiện khuôn mặt", null, null)
            }
    }

    private fun saveCapturedImage(bitmap: Bitmap, timestamp: String): File {
        val fileName = "capture_${timestamp.replace(":", "-")}.jpg"
        val file = File(context.getExternalFilesDir(null), fileName)
        val correctedBitmap = rotateBitmap(bitmap, 270f)
        FileOutputStream(file).use { out ->
            correctedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
        }
        Log.d(TAG, "Image saved to $file")
        return file
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = android.graphics.Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun cropFace(bitmap: Bitmap, boundingBox: Rect): Bitmap? {
        Log.d(TAG, "Cropping face: boundingBox=$boundingBox")
        val x = boundingBox.left.coerceAtLeast(0)
        val y = boundingBox.top.coerceAtLeast(0)
        val width = boundingBox.width().coerceAtMost(bitmap.width - x)
        val height = boundingBox.height().coerceAtMost(bitmap.height - y)
        return if (width > 0 && height > 0) {
            Bitmap.createBitmap(bitmap, x, y, width, height)
        } else {
            null
        }
    }

    private fun getEmbedding(faceBitmap: Bitmap): FloatArray {
        Log.d(TAG, "Getting embedding for face")
        val resizedBitmap = Bitmap.createScaledBitmap(faceBitmap, 224, 224, true)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val inputBuffer = tensorImage.tensorBuffer
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 128), DataType.FLOAT32)
        tflite.run(inputBuffer.buffer, outputBuffer.buffer)
        return outputBuffer.floatArray
    }

    private fun findMatchingMssv(embedding: FloatArray, threshold: Float = 1.5f): Pair<String, String> {
        Log.d(TAG, "Finding matching MSSV")
        var minDist = Float.MAX_VALUE
        var matchedMssv = "Unknown"
        for ((mssv, emb) in embeddings) {
            val dist = cosineDistance(embedding, emb)
            Log.d(TAG, "Comparing with MSSV=$mssv, distance=$dist")
            if (dist < minDist && dist < threshold) {
                minDist = dist
                matchedMssv = mssv
            }
        }
        val name = students.getOrDefault(matchedMssv, "Unknown")
        Log.d(TAG, "Final match: MSSV=$matchedMssv, Name=$name, MinDistance=$minDist")
        return matchedMssv to name
    }

    private fun cosineDistance(a: FloatArray, b: FloatArray): Float {
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        normA = sqrt(normA)
        normB = sqrt(normB)
        return if (normA > 0 && normB > 0) 1f - dotProduct / (normA * normB) else 1f
    }

    fun exportToCsv(result: RecognitionResult) {
        Log.d(TAG, "Exporting to CSV")
        val date = SimpleDateFormat("dd-MM-yyyy", Locale.getDefault()).format(Date())
        val file = File(context.getExternalFilesDir(null), "Attendance_$date.csv")
        val tempFile = File(context.getExternalFilesDir(null), "temp.csv")
        val existingRecords = mutableMapOf<String, String>()

        if (file.exists()) {
            file.readLines().forEach { line ->
                val parts = line.split(",")
                if (parts.size >= 3 && parts[0] != "MSSV") {
                    existingRecords[parts[0]] = line
                }
            }
        }

        if (!existingRecords.containsKey(result.mssv)) {
            FileWriter(file, true).use { writer ->
                if (file.length() == 0L) {
                    writer.append("MSSV,Name,Time\n")
                }
                writer.append("${result.mssv},${result.name},${result.timestamp}\n")
            }
            Log.d(TAG, "CSV appended to $file")
            // Xóa ảnh ngay sau khi lưu CSV
            result.capturedImage?.let { imageFile ->
                if (imageFile.exists() && imageFile.delete()) {
                    Log.d(TAG, "Deleted image after saving CSV: $imageFile")
                } else {
                    Log.w(TAG, "Failed to delete image: $imageFile")
                }
            }
        } else {
            Log.d(TAG, "MSSV ${result.mssv} already exists, skipping")
        }
    }

    fun cleanupOnExit() {
        Log.d(TAG, "Cleaning up on exit")
        val dir = context.getExternalFilesDir(null)
        val date = SimpleDateFormat("dd-MM-yyyy", Locale.getDefault()).format(Date())
        val csvFile = File(dir, "Attendance_$date.csv")
        val tempFile = File(dir, "temp.csv")

        dir?.listFiles { _, name -> name.endsWith(".jpg") }?.forEach { file ->
            if (file.delete()) {
                Log.d(TAG, "Deleted image: $file")
            } else {
                Log.w(TAG, "Failed to delete image: $file")
            }
        }

        if (csvFile.exists()) {
            val records = mutableMapOf<String, String>()
            csvFile.readLines().forEach { line ->
                val parts = line.split(",")
                if (parts.size >= 3 && parts[0] != "MSSV") {
                    records.putIfAbsent(parts[0], line)
                } else if (parts.size >= 3) {
                    records[parts[0]] = line
                }
            }

            FileWriter(tempFile).use { writer ->
                records.values.forEach { line ->
                    writer.append("$line\n")
                }
            }

            if (tempFile.renameTo(csvFile)) {
                Log.d(TAG, "CSV deduplicated: $csvFile")
            } else {
                Log.w(TAG, "Failed to rename temp CSV")
            }
        }
    }
}