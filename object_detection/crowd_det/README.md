# Detection in Crowded Scenes

## Input

![Input](demo.jpg)

(Image from http://www.crowdhuman.org/)

## Output

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 crowd_det.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 crowd_det.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 crowd_det.py --video VIDEO_PATH
```

By adding the `--model_type` option, you can specify model type which is selected from "rcnn_fpn_baseline", "rcnn_emd_simple", "rcnn_emd_refine". (default is rcnn_fpn_baseline)
```bash
$ python3 crowd_det.py --model_type rcnn_fpn_baseline
```

## Reference

- [Detection in Crowded Scenes](https://github.com/Purkialo/CrowdDet)

## Framework

Pytorch

## Model Format

ONNX opset=16

## Netron

[rcnn_fpn_baseline_mge.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pixel-link/rcnn_fpn_baseline_mge.onnx.prototxt)  
[rcnn_emd_simple_mge.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pixel-link/rcnn_emd_simple_mge.onnx.prototxt)  
[rcnn_emd_refine_mge.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pixel-link/rcnn_emd_refine_mge.onnx.prototxt)
