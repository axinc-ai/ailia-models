# 3D Bounding Box Estimation Using Deep Learning and Geometry

## Input

![Input](input.png)

(Image from https://github.com/skhadem/3D-BoundingBox/tree/master/eval/image_2)

Ailia input shape: (1, 3, 224, 224)

## Output

![Output](output.png)

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 3d_bbox.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 3d_bbox.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 3d_bbox.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the --normal option.

## Reference

[3D Bounding Box Estimation Using Deep Learning and Geometry](https://github.com/skhadem/3D-BoundingBox)

## Framework

Pytorch

## Model Format

ONNX opset = 10

## Netron

[3d_bbox.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/3d_bbox/3d_bbox.onnx.prototxt)
