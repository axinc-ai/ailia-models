# PolyLaneNet: Lane Estimation via Deep Polynomial Regression

### Input
Images with the same aspect ratio as 360×640. This model detect lane from inputs.
- input image    
![入力画像](./raws/original.jpg)    

### Output
Images which the detected lane in input images is colored in green.
- output image    
Four green lines mean each detected lane which model predicted and numbers attached to each line mean line number.
![出力画像](./raws/output.jpg)

### Usage    
- Image mode (image to image)   
You run sample script as below if your desired file is in {Path to polylanenet}/input/image/8.jpg.
```bash
$python3 polylanenet.py --input ./input/image/8.jpg
```

- Video mode (video to video)   
You run sample script as below if your desired file is in {Path to polylanenet}/input/video/video1.mp4
```bash
$python3 polylanenet.py --video ./input/video/video1.mp4
```



## Reference
- Repository    
[PolyLaneNet](https://github.com/lucastabelini/PolyLaneNet)

- Input images and videos    
Input images are part of [TuSimple](https://github.com/TuSimple/tusimple-benchmark) dataset and input videos are created by using TuSimple images.

## Framework

PyTorch 1.9.0


## Model Format

ONNX opset = 11

## Netron

[polylanenet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/polylanenet/polylanenet.onnx.prototxt)
