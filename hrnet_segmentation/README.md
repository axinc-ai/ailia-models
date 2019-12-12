# HRNet segmentation

### input
![input_image](https://github.com/sngyo/ailia-models/blob/master/hrnet_segmentation/test.png)

Shape: (1, 3, 512, 1024) Range:[0, 1]

### output
![Result_image](https://github.com/sngyo/ailia-models/blob/master/hrnet_segmentation/result.png)

### Usage
We have three pretrained-model.
- HRNetV2-W48
- HRNetV2-W18-Small-v1
- HRNetV2-W18-Small-v2 (default)

```bash
python3 hrnet_segmentation.py -a HRNetV2-W48
```

### Reference
[High-resolution networks (HRNets) for Semantic Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)

### Framework
PyTorch 0.4.1

### Model Format
ONNX opset = 1

