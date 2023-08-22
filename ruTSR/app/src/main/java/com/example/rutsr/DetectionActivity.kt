package com.example.rutsr

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Executors


class DetectionActivity : AppCompatActivity() {
    private lateinit var detectionModel: Module

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val modelResultPreprocessor = ModelResultPreprocessor()
    val imageProxyHandler = ImageProxyHandler()
    private lateinit var detectionResultView: DetectionResultView


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection)

        initModel()
        checkAndRequestPermissions()
    }

    private fun checkAndRequestPermissions() {
        if (allPermissionsGranted()) {
            Log.d(TAG, "Camera initialization")
            startCamera()
        } else {
            Log.d(TAG, "Camera permissions request")
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        val previewView = findViewById<PreviewView>(R.id.previewView)

        detectionResultView = findViewById(R.id.resultView)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .apply {
                    setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                        imageProxy.close()
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera initialization error", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        Log.d(TAG, "${imageProxy.format}")
        val bitmap = imageProxyHandler.toBitmap(imageProxy)
        if (bitmap != null) {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, ModelResultPreprocessor.width, ModelResultPreprocessor.height, false)

            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                ModelResultPreprocessor.NO_MEAN_RGB,
                ModelResultPreprocessor.NO_STD_RGB
            )

            val output = detectionModel.forward(IValue.from(inputTensor)).toTuple()
            Log.d(TAG, "Detection Model output size: ${output.size}")
            val outputTensor: Tensor = output[0].toTensor()
            val outputs: FloatArray = outputTensor.getDataAsFloatArray()
            val imgScaleX: Float = detectionResultView.getWidth().toFloat() / ModelResultPreprocessor.width
            val imgScaleY: Float = detectionResultView.getHeight().toFloat() / ModelResultPreprocessor.height

            val results: ArrayList<ModelOutput> = modelResultPreprocessor.outputsToPredictions(
                outputs, imgScaleX, imgScaleY)
            Log.d(TAG, "Detection Model results: $results")
            runOnUiThread {
                detectionResultView.setOutput(results)
                detectionResultView.invalidate()
            }
        } else {
            Log.e(TAG, "Error converting ImageProxy to Bitmap")
        }
    }

    private fun initModel() {
        try {
            val modelPath = assetFilePath(this, "best.torchscript.pt")
            Log.d(TAG, "Model path: $modelPath")
            detectionModel = Module.load(modelPath)
        } catch (e: Exception) {
            Log.e(TAG, "PyTorch model loading error", e)
        }
    }

    @Throws(IOException::class)
    private fun assetFilePath(context: Context, assetName: String): String? {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        try {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
            return file.absolutePath
        } catch (e: IOException) {
            Log.e(TAG,"File Access error")
        }
        return null
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, R.string.camera_permission, Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        detectionModel.destroy()
    }

    companion object {
        private val TAG = "SignsDetection"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
