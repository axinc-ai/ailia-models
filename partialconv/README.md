# Partial Convolution

### input
![input_image](https://github.com/sngyo/ailia-models/blob/master/partialconv/test_5735.JPEG)

Shape: (1, 3, 224, 224)   
Range: [[-2.11790393, -2.03571429, -1.80444444], [2.2489083 , 2.42857143, 2.64]]  
(Special image preprocessing is required, check partialconv.py)



### output
```
The predicted label is dumbbell
```

### Reference
[Partial Convolution Layer for Padding and Image Inpainting](https://github.com/NVIDIA/partialconv)

### Framework

PyTorch 1.2.0

### Model Format
ONNX opset = 10