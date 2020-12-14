# deep-text-recognition-benchmark

### Input

![input_image](demo_image/demo_1.png)

Ailia input shape: (10, 1, 32, 100)  
Range: [0, 1]

(10 = batchsize)

### Output

recognized text

```
demo_image/demo_1.png    	available                	0.5123
```

### Framework

Pytorch

### Model Format

ONNX opset=11

### Referemce

[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

### Netron

[None-ResNet-None-CTC.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deep-text-recognition-benchmarj/None-ResNet-None-CTC.onnx.prototxt)
