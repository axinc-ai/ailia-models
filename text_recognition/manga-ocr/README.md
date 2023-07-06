# manga-ocr

### Input

![input_image](demo_img/demo_1.png)

Ailia input shape: (1, 1, 32, 100)  
Color Range: [-1.0, 1.0]

### Output

list of recognized text

```
demo_image/demo_1.png    	available
```

### Framework

Pytorch

### Model Format

ONNX opset=11

### Referemce

[manga-ocr](https://github.com/clovaai/deep-text-recognition-benchmark)

### Netron

[manga-ocr-encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deep-text-recognition-benchmark/None-ResNet-None-CTC.onnx.prototxt)
[manga-ocr-decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deep-text-recognition-benchmark/None-ResNet-None-CTC.onnx.prototxt)
