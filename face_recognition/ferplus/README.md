# FER+

## Input

<table>
<tr>
<td>
  <img src="img/fer0032227.png" width="96px">
</td>
<td>
  <img src="img/fer0032328.png" width="96px">
</td>
<td>
  <img src="img/fer0032363.png" width="96px">
</td>
<td>
  <img src="img/fer0032285.png" width="96px">
</td>
<td>
  <img src="img/fer0033915.png" width="96px">
</td>
<td>
  <img src="img/fer0033721.png" width="96px">
</td>
<td>
  <img src="img/fer0033894.png" width="96px">
</td>
</tr>
</table>

(Image
from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

Shape: (1, 1, 64, 64)

## Output

- Estimating emotion
```bash
### Estimating emotion ###
 emotion: happiness
```

### Example

<table>
<tr>
<td>
  <img src="img/fer0032227.png" width="96px">
  <div>happiness</div>
</td>
<td>
  <img src="img/fer0032328.png" width="96px">
  <div>surprise</div>
</td>
<td>
  <img src="img/fer0032363.png" width="96px">
  <div>sadness</div>
</td>
<td>
  <img src="img/fer0032285.png" width="96px">
  <div>anger</div>
</td>
<td>
  <img src="img/fer0033915.png" width="96px">
  <div>disgust</div>
</td>
<td>
  <img src="img/fer0033721.png" width="96px">
  <div>fear</div>
</td>
<td>
  <img src="img/fer0033894.png" width="96px">
  <div>contempt</div>
</td>
</tr>
</table>

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,
``` bash
$ python3 ferplus.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 ferplus.py --input IMAGE_PATH
```

If you want to perform face detection in preprocessing, use the `--detection` option.
```bash
$ python3 ferplus.py --input IMAGE_PATH --detection
```

By adding the `--video` option, you can input the video.  
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.  
You can use --savepath option to specify the output file to save.
```bash
$ python3 ferplus.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

By adding the `--model_name` option, you can specify model name which is selected from "majority", "probability", "crossentropy" "multi_target". (default is majority)
```bash
$ python3 ferplus.py --model_name majority
```

## Reference

- [FER+](https://github.com/microsoft/FERPlus)

## Framework

MS Cognitive Toolkit

## Model Format

ONNX opset = 9

## Netron

[VGG13_majority.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ferplus/VGG13_majority.onnx.prototxt)  
[VGG13_probability.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ferplus/VGG13_probability.onnx.prototxt)  
[VGG13_crossentropy.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ferplus/VGG13_crossentropy.onnx.prototxt)  
[VGG13_multi_target.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ferplus/VGG13_multi_target.onnx.prototxt)
