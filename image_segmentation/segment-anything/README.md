# Segment Anything

## Input

![Input](truck.jpg)

(Image from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/truck.jpg)

## Output

- 1 positive point
![Output](example/output1.png)

- 2 positive points
![Output](example/output2.png)

- 1 positive point, 1 negative point
![Output](example/output3.png)

- box
![Output](example/output4.png)

- box, 1 negative point
![Output](example/output5.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 segment-anything.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 segment-anything.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

If you want to specify the positive point, put the coordinates(x,y) after the `--pos` option.
```bash
$ python3 segment-anything.py --pos 500 375
```

And if you want to specify the negative point, put the coordinates after the `--neg` option.
```bash
$ python3 segment-anything.py --pos 500 375 --neg 360 405
```

If you want to specify the box, put the coordinates(x1,y1,x2,y2) after the `--box` option.
```bash
$ python3 segment-anything.py --box 425 600 700 875
```

These options can be combined.
```bash
$ python3 segment-anything.py --pos 500 375 --pos 1125 625
```

```bash
$ python3 segment-anything.py --box 425 600 700 875 --neg 575 750
```

By adding the `--model_type` option, you can specify model type which is selected from "sam_h", "sam_l", "sam_b". (default is sam_h)
```bash
$ python3 segment-anything.py --model_type sam_l
```

## Reference

- [Segment Anything](https://github.com/facebookresearch/segment-anything)

## Framework

Pytorch

## Model Format

ONNX opset=12

## Netron

[sam_h_4b8939.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything/sam_h_4b8939.onnx.prototxt)  
[sam_l_0b3195.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything/sam_l_0b3195.onnx.prototxt)  
[sam_b_01ec64.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything/sam_b_01ec64.onnx.prototxt)  
[vit_h_4b8939.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything/vit_h_4b8939.onnx.prototxt)  
[vit_l_0b3195.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything/vit_l_0b3195.onnx.prototxt)  
[vit_b_01ec64.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything/vit_b_01ec64.onnx.prototxt)
