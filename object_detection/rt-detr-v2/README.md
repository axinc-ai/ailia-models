# RT-DETR: DETRs Beat YOLOs on Real-time Object Detection


## Input

![Input](demo.jpg)

(Image from http://www.crowdhuman.org/)

Ailia input shape : (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)  

## Output

![Output](output.png)

Ailia output shape : (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

## Usage
Automatically downloads the onnx and prototxt files when running.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 rt-detr-v2.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 rt-detr-v2.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--arch` option, you can specify architecture type which is selected from "rtdetrv2_r18vd_120e","rtdetrv2_r34vd_120e","rtdetrv2_r50vd_6x","rtdetrv2_r50vd_m_7x","rtdetrv2_r101vd_6x" , (default: rtdetrv2_r18vd_120e)

```bash
$ python3 rt-detr-v2.py --arch rtdetrv2_r18vd_120e
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 rt-detr-v2.py --video VIDEO_PATH
```

## Reference

[RT-DETR](https://github.com/lyuwenyu/RT-DETR)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron


[rtdetrv2_r18vd_120e_coco.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rt-detr-v2/rtdetrv2_r18vd_120e_coco.onnx.prototxt)

[rtdetrv2_r34vd_120e_coco.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rt-detr-v2/rtdetrv2_r34vd_120e_coco.onnx.prototxt)

[rtdetrv2_r50vd_6x_coco.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rt-detr-v2/rtdetrv2_r50vd_6x_coco.onnx.prototxt)

[rtdetrv2_r50vd_m_7x_coco.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rt-detr-v2/rtdetrv2_r50vd_m_7x_coco.onnx.prototxt)

[rtdetrv2_r101vd_6x_coco.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rt-detr-v2/rtdetrv2_r101vd_6x_coco.onnx.prototxt)
