# MMPose - 2D animal pose estimation

## Input

![Input](input.jpg)

(Image from https://pixabay.com/ja/photos/%e7%89%9b-%e5%ae%b6%e7%95%9c-%e4%b9%b3%e7%89%9b-%e4%b9%b3%e7%94%a8%e7%89%9b-%e5%8b%95%e7%89%a9-5717276/)

Shape : (1, 3, 256, 256)

## Output

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 animalpose.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 animalpose.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 animalpose.py --video VIDEO_PATH
```

By default, yolov3 and hrnet32 are used. Yolox_m and hrnet48 can also be used for accuracy.
```bash
$ python3 animalpose.py -d yolox_m -m hrnet48
```

## Reference

- [MMPose](https://github.com/open-mmlab/mmpose) 

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[hrnet_w32_256x256.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animalpose/hrnet_w32_256x256.onnx.prototxt)  
[hrnet_w48_256x256.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animalpose/hrnet_w48_256x256.onnx.prototxt)  
[res50_256x256.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animalpose/res50_256x256.onnx.prototxt)  
[res101_256x256.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animalpose/res101_256x256.onnx.prototxt)  
[res152_256x256.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animalpose/res152_256x256.onnx.prototxt)  
[yolov3.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov3/yolov3.opt.onnx.prototxt)  
