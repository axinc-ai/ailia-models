# Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach

## Input

![Input](input.png)

(Image from https://github.com/skhadem/3D-BoundingBox/tree/master/eval/image_2)

Ailia input shape: (1, 3, 256, 192)

## Output

![Output](output.png)

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 pose-hg-3d.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 pose-hg-3d.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 pose-hg-3d.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the --normal option.

## Reference

[Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach](https://github.com/xingyizhou/pytorch-pose-hg-3d)

## Framework

Pytorch 0.4.1.

## Model Format

ONNX opset = 10

## Netron

[pose_hg_3d.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pose_hg_3d/pose_hg_3d.onnx.prototxt)
