# ResNet Facial Feature Detection

### input
![input_image](https://github.com/sngyo/ailia-models/blob/master/facial_feature/test.png)

Shape: (1, 1, 96, 96) Range:[0, 1]

### output
![Result_image](https://github.com/sngyo/ailia-models/blob/master/facial_feature/result.png)
 
The raw output of our model is a list of 15 coordinates of feature points.  
In the `facial_feature.py`, we place these points on the input image and resize it to 300pixels * 300pixels for easier understanding.

### Reference

[kaggle-facial-keypoints](https://github.com/axinc-ai/kaggle-facial-keypoints)

### Framework
PyTorch 1.2.0

### Model Format
ONNX opset = 10
