# DeepFaceLive

## Input

- Source image

  <img src="Kim_Chen_Yin.png" width="320px"/>

(Image from https://github.com/iperov/DeepFaceLive/tree/master/build/animatables)

- Driving image

  <table>
  <tr>
  <td><img src="sample/frame_000001.png" width="320px"/></td>
  <td><img src="sample/frame_000053.png" width="320px"/></td>
  <td><img src="sample/frame_000157.png" width="320px"/></td>
  </tr>
  </table>

(Image from https://github.com/iperov/DeepFaceLive/tree/master/build/samples)

## Output

<img src="sample_results/frame_000001_res.png" width="320px">
<img src="sample_results/frame_000053_res.png" width="320px">
<img src="sample_results/frame_000157_res.png" width="320px">

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 deepfacelive.py
```

If you want to specify the driving images, put the image directory path after the `--input` option.  
The source image can be specified with the `--source` option.  
You can use `--savepath` option to change the name of the output directory to save.
```bash
$ python3 deepfacelive.py --input DRIVING_IMAGE_DIR --source SOURCE_IMAGE --savepath SAVE_IMAGE_DIR
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 deepfacelive.py --video VIDEO_PATH
```

By adding the `--detector` option, you can specify detector type which is selected from "yolov5", "centerface", "s3fd". (default is yolov5)
```bash
$ python3 deepfacelive.py --detector yolov5
```

By adding the `--marker` option, you can specify marker type which is selected from "facemesh", "insightface". (default is facemesh)
```bash
$ python3 deepfacelive.py --marker facemesh
```

## Reference

- [DeepFaceLive](https://github.com/iperov/DeepFaceLive)

## Framework

Onnxruntime

## Model Format

ONNX opset=13

## Netron

[generator.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepfacelive/generator.onnx.prototxt)  
