# Landmarks classifier_asia_V1.1

## Input

![Input](image_1.jpg)

(Image from https://storage.googleapis.com/tfhub-visualizers/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1/image_1.jpg)

Shape : (1, 321, 321, 3)  

## Output

Shape : (1, 98960)

- Top-K prediction
```bash
### predicts the most likely topK labels ###
   Asagirikogen Rest Area: 87.68%
   Lake Yamanaka: 83.80%
   Lake Ashi: 80.77%
   Mt. Omuro: 79.26%
   Mount Fuji: 79.07%
   Koryaksky: 77.78%
   Daisen: 77.55%
   Khor Virab: 77.34%
   Mayon Volcano: 76.84%
   Taiseki-ji: 74.64%
   Lake Shikotsu: 71.92%
   Mishima Sky Walk: 71.39%
   Little Ararat: 66.83%
   Kaimondake volcano: 66.65%
   Semeru: 65.02%
   Kagoshima Bay: 64.06%
   Mount Nantai: 62.84%
   Niseko Mt. Resort Grand Hirafu: 62.70%
   Avachinsky: 62.58%
   Mount Arayat: 61.73%
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
