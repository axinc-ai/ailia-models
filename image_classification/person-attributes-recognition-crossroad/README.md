# MMFashion

## Input

![Input](input.jpg)

(Image from https://github.com/open-mmlab/mmfashion/blob/master/demo/imgs/01_4_full.jpg)

Ailia input shape: (1, 3, 576, 576)

## Output

```
class_count=7
  category=[0][is_male ]
  prob=0.7977050542831421
  category=[1][has_bag ]
  prob=0.7185143232345581
  category=[2][has_backpack ]
  prob=0.9199745655059814
  category=[3][has_hat ]
  prob=0.3727506697177887
  category=[4][has_longsleeves ]
  prob=0.7904267907142639
  category=[5][has_longpants ]
  prob=0.7581638097763062
  category=[6][has_longhair ]
  prob=0.8155472278594971

```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 person-attributes-recognition-crossroad.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 person-attributes-recognition-crossroad.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 person-attributes-recognition-crossroad.py --video VIDEO_PATH
```

By adding the --model option, you can specify model type which is selected from "0230", "0234".

```bash
$ python3 yolov.py --model MODELNAME
```

## Reference

[person-attributes-recognition-crossroad-0230](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-attributes-recognition-crossroad-0230)

[person-attributes-recognition-crossroad-0234](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-attributes-recognition-crossroad-0234)

## Framework

openvino

## Model Format

ONNX opset=13

## Netron

[person-attributes-recognition-crossroad-0230.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/person-attributes-recognition-crossroad/person-attributes-recognition-crossroad-0230.onnx.prototxt) 

[person-attributes-recognition-crossroad-0234.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/person-attributes-recognition-crossroad/person-attributes-recognition-crossroad-0234.onnx.prototxt) 
