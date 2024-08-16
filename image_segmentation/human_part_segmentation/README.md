# Human-Part-Segmentation

## Input

![Input](demo.jpg)

(Image from https://github.com/PeikeLi/Self-Correction-Human-Parsing/blob/master/demo/demo.jpg)

#### LIP
Shape : (1, 3, 473, 473)  
#### ATR
Shape : (1, 3, 512, 512)  
#### PASCAL
Shape : (1, 3, 512, 512)

## Output

#### LIP
![Output](output.png)

- parsing shape : (1, 20, 119, 119)
- fusion shape : (1, 20, 119, 119)
- edge shape : (1, 2, 119, 119)

#### ATR
- parsing shape : (1, 18, 128, 128)
- fusion shape : (1, 18, 128, 128)
- edge shape : (1, 2, 128, 128)
#### PASCAL
- parsing shape : (1, 7, 128, 128)
- fusion shape : (1, 7, 128, 128)
- edge shape : (1, 2, 128, 128)

### Category

#### LIP
```
CATEGORY = (
    'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
    'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
    'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'
)
```
#### ATR
```
CATEGORY = (
    'Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
    'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'
)
```
#### PASCAL
```
CATEGORY = (
    'Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'
)
```

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 human_part_segmentation.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 human_part_segmentation.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 human_part_segmentation.py --video VIDEO_PATH
```

By adding the `--arch` option, you can specify architecture type which is selected from "lip", "atr", "pascal".  
(default is lip)
```bash
$ python3 human_part_segmentation --arch lip
```

## Reference

- [Self Correction for Human Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing)
- [Human-Part-Segmentation](https://github.com/mayankgrwl97/human-part-segmentation)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[resnet-lip.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/human_part_segmentation/resnet-lip.onnx.prototxt)  
[resnet-atr.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/human_part_segmentation/resnet-atr.onnx.prototxt)  
[resnet-pascal.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/human_part_segmentation/resnet-pascal.onnx.prototxt)
