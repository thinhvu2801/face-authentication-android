package com.vmt.faceauth

import android.graphics.Rect
import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.camera.view.PreviewView
import androidx.compose.ui.draw.alpha
import coil.compose.rememberAsyncImagePainter

@Composable
fun MainScreen(
    recognitionResult: String,
    faceRect: Rect?,
    capturedResult: RecognitionResult?,
    onSave: () -> Unit,
    onRestartCamera: () -> Unit,
    onExit: () -> Unit,
    modifier: Modifier = Modifier,
    isProcessing: Boolean = false
) {
    val context = LocalContext.current
    val showExitDialog = remember { mutableStateOf(false) }

    if (showExitDialog.value) {
        AlertDialog(
            onDismissRequest = { showExitDialog.value = false },
            title = { Text("Xác nhận thoát") },
            text = { Text("Bạn có muốn thoát ứng dụng?") },
            confirmButton = {
                Button(onClick = {
                    showExitDialog.value = false
                    onExit()
                }) {
                    Text("Thoát")
                }
            },
            dismissButton = {
                Button(onClick = { showExitDialog.value = false }) {
                    Text("Hủy")
                }
            }
        )
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color.White),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Box(modifier = Modifier.weight(1f)) {
            // Luôn giữ PreviewView, nhưng ẩn khi hiển thị ảnh
            AndroidView(
                factory = {
                    Log.d("FaceAuthMain", "Creating PreviewView")
                    PreviewView(context).apply {
                        layoutParams = android.view.ViewGroup.LayoutParams(
                            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                            android.view.ViewGroup.LayoutParams.MATCH_PARENT
                        )
                        id = R.id.preview_view
                    }
                },
                update = { previewView ->
                    Log.d("FaceAuthMain", "Updating PreviewView")
                    previewView.invalidate()
                },
                modifier = Modifier
                    .fillMaxSize()
                    .alpha(if (capturedResult == null) 1f else 0f)
            )
            // Vẽ bounding box cho PreviewView
            if (capturedResult == null) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    faceRect?.let { rect ->
                        val scaleX = size.width / 1024f
                        val scaleY = size.height / 768f
                        val left = rect.left * scaleX
                        val top = rect.top * scaleY
                        val width = rect.width() * scaleX
                        val height = rect.height() * scaleY
                        drawRect(
                            color = Color.Green,
                            topLeft = Offset(left, top),
                            size = Size(width, height),
                            style = Stroke(width = 4f)
                        )
                    }
                }
            }
            // Hiển thị ảnh chụp nếu có
            capturedResult?.capturedImage?.let { imageFile ->
                Image(
                    painter = rememberAsyncImagePainter(imageFile),
                    contentDescription = "Captured face",
                    modifier = Modifier.fillMaxSize()
                )
                Canvas(modifier = Modifier.fillMaxSize()) {
                    faceRect?.let { rect ->
                        val scaleX = size.width / 1024f
                        val scaleY = size.height / 768f
                        val left = rect.left * scaleX
                        val top = rect.top * scaleY
                        val width = rect.width() * scaleX
                        val height = rect.height() * scaleY
                        drawRect(
                            color = Color.Green,
                            topLeft = Offset(left, top),
                            size = Size(width, height),
                            style = Stroke(width = 4f)
                        )
                    }
                }
            }
            // Hiển thị loading indicator khi nhận diện
            if (isProcessing && capturedResult == null) {
                CircularProgressIndicator(
                    modifier = Modifier
                        .size(50.dp)
                        .align(Alignment.Center)
                )
            }
        }
        if (capturedResult == null) {
            Text(
                text = recognitionResult,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
                    .align(Alignment.CenterHorizontally)
            )
        } else {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "MSSV: ${capturedResult.mssv}",
                    style = MaterialTheme.typography.bodyLarge
                )
                Text(
                    text = "Tên: ${capturedResult.name}",
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(top = 8.dp)
                )
                Text(
                    text = "Thời gian: ${capturedResult.timestamp}",
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
        }
        Button(
            onClick = if (capturedResult == null) { { } } else { onSave },
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
                .align(Alignment.CenterHorizontally),
            enabled = capturedResult != null
        ) {
            Text(
                text = "Save",
                style = MaterialTheme.typography.bodyLarge
            )
        }
        if (capturedResult != null) {
            Button(
                onClick = onRestartCamera,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
                    .align(Alignment.CenterHorizontally)
            ) {
                Text(
                    text = "Chụp lại",
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        }
        Button(
            onClick = { showExitDialog.value = true },
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
                .align(Alignment.CenterHorizontally)
        ) {
            Text(
                text = "Thoát",
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }
}