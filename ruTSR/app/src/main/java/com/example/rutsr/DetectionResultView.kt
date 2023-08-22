package com.example.rutsr

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class DetectionResultView : View {
    private var rect: Paint? = null
    private var signText: Paint? = null
    private var modelOutput: ArrayList<ModelOutput>? = null

    constructor(context: Context?) : super(context)
    constructor(context: Context?, attrs: AttributeSet?) : super(context, attrs) {
        rect = Paint()
        rect!!.color = Color.YELLOW
        rect!!.strokeWidth = 5f
        rect!!.style = Paint.Style.STROKE
        signText = Paint()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (modelOutput == null) return
        modelOutput?.forEach { result ->
            canvas.drawRect(result.box, rect!!)
            signText!!.color = Color.WHITE
            signText!!.strokeWidth = 0f
            signText!!.style = Paint.Style.FILL
            signText!!.textSize = 32f
            canvas.drawText(
                String.format(
                    "%s %.2f",
                    ModelResultPreprocessor.classNames[result.classIndex],
                    result.score
                ), result.box.left.toFloat() + TEXT_X, result.box.top.toFloat() + TEXT_Y,
                signText!!
            )
        }
    }

    fun setOutput(output: ArrayList<ModelOutput>?) {
        modelOutput = output
    }

    companion object {
        private const val TEXT_X = 40
        private const val TEXT_Y = 35
    }
}