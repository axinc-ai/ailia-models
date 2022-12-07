# DID M3D

## Input

![Input](000005.png)

(Image from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

## Output

![Output](output.png)

## Data Preparation

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 did_m3d.py
```

if you want to --savepath option to change the name of the output file to save.
```bash
$ python3 did_m3d.py --savepath SAVE_IMAGE_PATH
```


If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 did_m3d.py --input IMAGE_PATH
```

The depth file and calib file are automatically found in the predetermined path.  
If you want to specify directory or directly file path, put the path after the `--calib_path` option.
```bash
$ python3 did_m3d.py --input IMAGE_PATH --calib_path CALIB_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 egonet.py --video VIDEO_PATH
```


## Reference

- [DID M3D](https://github.com/SPengLiang/DID-M3D)

## Framework

Pytorch

## Model Format

ONNX opset=12

## Netron

[did_m3d.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/did_m3d/did_m3d.onnx.prototxt)  
