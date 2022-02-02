# Landmarks classifier_asia_V1.1

## Input

![Input](image_1.jpg)

(Image from https://pixabay.com/photos/japan-tokyo-tower-landmark-343444/)

Shape : (1, 321, 321, 3)  

## Output

Shape : (1, 98960)

- Top-K prediction
```bash
TopK predictions:
  Tokyo Tower: 92.34%
  Sapporo TV Tower: 84.53%
  Yokohama Marine Tower: 81.77%
  Hakata Port Tower: 81.05%
  Nagoya TV Tower: 74.36%
  Tamsui Fisherman's Wharf: 70.55%
  Guangzhou TV Tower: 70.32%
  Kobe Port Tower: 66.71%
  Chikugo River Lift bridge: 65.11%
  Oasis 21: 64.98%
  Yamashita Park: 64.60%
  Tokyo Skytree: 62.34%
  Wakato Ohashi Bridge: 62.05%
  Diamond Exchange District: 61.20%
  Wat Arun: 58.97%
  Zhongyuan Tower: 58.24%
  Corregidor Island: 57.99%
  田川市煤炭·历史博物馆: 57.88%
  Kobe Maritime Museum: 57.17%
  Zero Carbon Building: 56.78%
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 landmarks_classifier_asia.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 landmarks_classifier_asia.py --input IMAGE_PATH
```

## Reference

- [Landmarks classifier_asia_V1.1](https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1)

## Framework

TensorFlow Hub

## Model Format

ONNX opset=11

## Netron

[landmarks_classifier_asia_V1_1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/landmarks_classifier_asia/landmarks_classifier_asia_V1_1.onnx.prototxt)
