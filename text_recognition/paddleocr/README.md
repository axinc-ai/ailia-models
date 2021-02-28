# PoseNet of "Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image"

## Input

![input image](input.jpg)

(from https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/doc/imgs)

<br/>

## Output

![output_image](output.png)

<br/>

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python paddleocr.py
(ex on CPU)  $ python paddleocr.py -e 0
(ex on BLAS) $ python paddleocr.py -e 1
(ex on GPU)  $ python paddleocr.py -e 2
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 paddleocr.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
$ python3 paddleocr.py -i IMAGE_PATH -s SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.
```bash
$ python3 paddleocr.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
$ python3 paddleocr.py -v VIDEO_PATH -s SAVE_VIDEO_PATH
(ex) $ python3 paddleocr.py --video input.mp4 --savepath output.mp4
```

<br/>

## About font file

Requires Japanese font.
The default specification for each OS is as follows.

```
    if sys.platform == "win32":
        # Windows
        dc['vis_font_path'] = 'C:/windows/Fonts/meiryo.ttc'
    elif sys.platform == "darwin":
        # Mac OS
        dc['vis_font_path'] = '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc'
    else:
        # Linux
        dc['vis_font_path'] = '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf'
```

If the Japanese font of the above path does not exist, please prepare it as described or change the specification in the program.
In case of linux, you can get the target font by executing the following command.

```
apt install fonts-ipaexfont
```

<br/>

## Reference

[PaddleOCR : Awesome multilingual OCR toolkits based on PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/doc/imgs)
[PaddleOCR2Pytorch : PaddleOCR推理的pytorch实现和模型转换](https://github.com/frotms/PaddleOCR2Pytorch)

<br/>

## Framework

Pytorch

<br/>

## Model Format

ONNX opset = 10

<br/>

## Netron

[ch_ppocr_server_v2.0_det_train.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/paddle_ocr/ch_ppocr_server_v2.0_det_train.onnx.prototxt)
[ch_ppocr_mobile_v2.0_cls_train.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/paddle_ocr/ch_ppocr_mobile_v2.0_cls_train.onnx.prototxt)
[japan_mobile_v2.0_rec_infer.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/paddle_ocr/japan_mobile_v2.0_rec_infer.onnx.prototxt)
