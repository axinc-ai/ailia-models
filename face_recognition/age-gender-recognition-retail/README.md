# age-gender-recognition

## Input

![Input](demo.jpg)

(Image
from https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/age-gender-recognition-retail-0013/assets/age-gender-recognition-retail-0001.jpg)

Shape: (1, 62, 62, 3) BGR channel order

## Output

- Estimating gender and age
```bash
### Estimating gender and age ###
gender is: Female (98.75)
age is: 25
```

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,
``` bash
$ python3 age-gender-recognition-retail.py 
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 age-gender-recognition-retail.py --input IMAGE_PATH
```

If you want to perform face detection in preprocessing, use the `--detection` option.
```bash
$ python3 age-gender-recognition-retail.py --input IMAGE_PATH --detection
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.  
You can use --savepath option to specify the output file to save.
```bash
$ python3 age-gender-recognition-retail.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

## Reference

- [OpenVINO - Open Model Zoo repository - age-gender-recognition-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/age-gender-recognition-retail-0013)
- [OpenVINO - age-gender-recognition-retail-0013](https://docs.openvinotoolkit.org/latest/omz_models_model_age_gender_recognition_retail_0013.html)

## Framework

OpenVINO

## Model Format

ONNX opset = 11

## Netron

[age-gender-recognition-retail-0013.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/age-gender-recognition-retail/age-gender-recognition-retail-0013.onnx.prototxt)
