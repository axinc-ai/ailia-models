# HRNet segmentation

### input
![input_image](test.png)

Shape: (1, 3, 512, 1024) Range:[0, 1]

### output
- Normal output
![Result_image](result.png)
  
- Smoothed output
![Smoothed_result_image](result_smooth.png)


### Usage
We have three pretrained-model.
- HRNetV2-W48
- HRNetV2-W18-Small-v1
- HRNetV2-W18-Small-v2 (default)

```bash
python3 hrnet_segmentation.py -a HRNetV2-W48
```

If you want the segmentated image to be smooth, use `--smooth` / `-s` option.  
By applying resize method `interpolaation=cv2.INTER_LINEAR`, the visualisation will be more smooth.
```bash
python3 hrnet_segmentation.py -a HRNetV2-W48 --smooth
```



### Reference

[High-resolution networks (HRNets) for Semantic Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)

### Framework
PyTorch 0.4.1

### Model Format
ONNX opset = 10

### Netron

[HRNetV2-W18-Small-v1.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/hrnet/HRNetV2-W18-Small-v1.onnx.prototxt)

[HRNetV2-W18-Small-v2.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/hrnet/HRNetV2-W18-Small-v2.onnx.prototxt)

