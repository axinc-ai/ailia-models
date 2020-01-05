# RotNet

### input
![input_image](input.jpg)

(from https://github.com/d4nst/RotNet/tree/master/data/test_examples)

Shape(1, 3, 224, 224) (automatically croped in the script)

### output
- Original: original image (after cropped)
- Rotated: input image (randomly rotated)
- Corrected: output image (model output is predicted angle, therefore we rotated the "rotated image" to visualize our output)
![output_image](output.png)

### Reference
[RotNet](https://github.com/d4nst/RotNet)

### Framework
Keras

### Model Format
ONNX opset = 10
